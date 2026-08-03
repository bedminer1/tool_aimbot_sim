"""MuJoCo-headless training environment — identical physics to C++ mujocoaim.

Replicates src/cli/main.cpp physics loop exactly:
  - gimbal.xml model loaded via MuJoCo Python bindings
  - Analytical gimbal update + mj_forward (matching C++ sub-step loop)
  - Target motion via mocap body (analytical, matching C++ target models)
  - Bullet pool via mocap bodies (analytical ballistic trajectory)
  - Hit detection via MuJoCo armor geom world positions
  - Detection lag buffer, shooting delay, barrel heat

This IS the C++ sim, minus the GLFW window. Zero sim2sim gap.

Usage:
  env = MujocoTrainingEnv(difficulty="medium")
  # drop-in replacement for GimbalEnv — same obs/act/reward interface
"""

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import Env, spaces

from training.ppo_env import (
    DT, BULLET_SPEED, GRAVITY, BULLET_RADIUS,
    MAX_YAW_VEL, MAX_PITCH_VEL, MAX_YAW_ACCEL, MAX_PITCH_ACCEL,
    YAW_LIMIT, PITCH_MIN, PITCH_MAX,
    GIMBAL_HEIGHT, MUZZLE_LENGTH,
    HEAT_LIMIT, HEAT_PER_SHOT, HEAT_COOLING,
    MAX_SHOTS, TIME_LIMIT,
    DETECTION_LAG, SHOOT_DELAY, AUTO_COOLDOWN,
    ORBIT_RADIUS, ORBIT_OMEGA,
    TARGET_HALF_X, TARGET_HALF_Y, TARGET_HALF_Z,
    OBS_STACK, PI,
    POS_SCALE, ANGLE_SCALE, VEL_SCALE, HEAT_SCALE,
    LINEAR_X, LINEAR_Y_MIN, LINEAR_Y_MAX, LINEAR_SPEED,
    HARD_RANGE_LO, HARD_RANGE_HI, HARD_ANGLE_MAX,
    HARD_SPEED_LO, HARD_SPEED_HI,
)

MUJOCO_DT = 0.002   # MuJoCo physics timestep (gimbal.xml)
SUBSTEPS = int(DT / MUJOCO_DT)  # 30 substeps per 1/60s frame


class MujocoTrainingEnv(Env):
    """MuJoCo-headless env matching C++ mujocoaim physics exactly."""

    def __init__(self, difficulty="medium", render_mode=None):
        super().__init__()
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown difficulty: {difficulty}")
        self.difficulty = difficulty

        # ── Load MuJoCo model ──────────────────────────────────────────
        xml_path = os.path.join(os.path.dirname(__file__), "..", "gimbal.xml")
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        # Look up IDs matching C++ main.cpp.
        self._yaw_jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
        self._pitch_jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")
        self._yaw_motor = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
        self._pitch_motor = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
        self._target_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap = self._model.body_mocapid[self._target_body]
        self._muzzle_site = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "muzzle_site")
        self._armor_geoms = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"armor_{i}")
            for i in range(4)
        ]
        self._bullet_mocaps = []
        for i in range(100):
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"bullet_{i}")
            self._bullet_mocaps.append(self._model.body_mocapid[bid])

        self._yaw_qpos = self._model.jnt_qposadr[self._yaw_jid]
        self._pitch_qpos = self._model.jnt_qposadr[self._pitch_jid]
        self._yaw_qvel = self._model.jnt_dofadr[self._yaw_jid]
        self._pitch_qvel = self._model.jnt_dofadr[self._pitch_jid]

        # ── Observation / action space ──────────────────────────────────
        obs_dim = OBS_STACK * 3 + 4 + 1 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        # ── Viewer (optional) ──────────────────────────────────────────
        self._viewer = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self._model, self._data, show_left_ui=False, show_right_ui=False)

        self.reset()

    # ══ Target motion (same as ppo_env.py) ════════════════════════════════

    def _update_target_linear(self):
        dt = self._target_prev_time_linear
        dt = self._time - self._target_prev_time_linear
        if dt <= 0.0:
            return self._target_center
        ny = self._target_center[1] + self._linear_dir * LINEAR_SPEED * dt
        if ny > LINEAR_Y_MAX:
            ny = LINEAR_Y_MAX; self._linear_dir = -1
        elif ny < LINEAR_Y_MIN:
            ny = LINEAR_Y_MIN; self._linear_dir = 1
        self._target_center = np.array([LINEAR_X, ny, 0.015])
        self._target_prev_time_linear = self._time
        return self._target_center

    @staticmethod
    def _smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _random_waypoint(self):
        r = self._rng.uniform(HARD_RANGE_LO, HARD_RANGE_HI)
        a = self._rng.uniform(-HARD_ANGLE_MAX, HARD_ANGLE_MAX)
        return np.array([r * np.cos(a), r * np.sin(a), 0.015])

    def _update_target_hard(self):
        elapsed = self._time - self._target_state_start
        if self._target_state == "IDLE":
            if elapsed >= self._target_duration:
                self._target_state = "MOVING"; self._target_state_start = self._time
                self._target_start_pos = self._target_center.copy()
                self._target_waypoint = self._random_waypoint()
                dist = np.linalg.norm(self._target_waypoint - self._target_start_pos)
                speed = self._rng.uniform(HARD_SPEED_LO, HARD_SPEED_HI)
                self._target_duration = dist / max(0.1, speed)
        else:
            t = elapsed / self._target_duration
            if t >= 1.0:
                self._target_center = self._target_waypoint.copy()
                self._target_state = "IDLE"; self._target_state_start = self._time
                self._target_duration = self._rng.uniform(0.5, 2.0)
            else:
                st = self._smoothstep(t)
                self._target_center = self._target_start_pos + (
                    self._target_waypoint - self._target_start_pos) * st
        return self._target_center

    def _update_target(self):
        if self.difficulty == "hard":
            return self._update_target_hard()
        return self._update_target_linear()

    def _plate_positions(self):
        center = self._target_center
        spin = ORBIT_OMEGA * self._time if self.difficulty != "easy" else 0.0
        c, s = np.cos(spin), np.sin(spin)
        return [
            center + np.array([-s * 0.245,  c * 0.245, 0.02]),
            center + np.array([ c * 0.245,  s * 0.245, 0.02]),
            center + np.array([ s * 0.245, -c * 0.245, 0.02]),
            center + np.array([-c * 0.245, -s * 0.245, 0.02]),
        ]

    def _closest_plate(self):
        plates = self._plate_positions()
        best, best_d2 = None, float("inf")
        gz = GIMBAL_HEIGHT
        for p in plates:
            d2 = p[0]**2 + p[1]**2 + (p[2] - gz)**2
            if d2 < best_d2:
                best_d2 = d2; best = p
        return best

    # ══ Detection lag (ring buffer, matches C++ DetectLagBuffer) ══════════

    def _push_lag(self, pos):
        self._lag_times[self._lag_head] = self._time
        self._lag_pos[self._lag_head] = pos
        self._lag_head = (self._lag_head + 1) % 60
        self._lag_count = min(self._lag_count + 1, 60)

    def _get_lagged(self):
        query_t = self._time - DETECTION_LAG
        best = None
        for i in range(min(self._lag_count, 60)):
            idx = (self._lag_head - 1 - i) % 60
            if self._lag_times[idx] <= query_t:
                best = self._lag_pos[idx].copy()
                break
        if best is None and self._lag_count > 0:
            best = self._lag_pos[(self._lag_head - self._lag_count) % 60].copy()
        return best if best is not None else self._target_center.copy()

    # ══ Bullet physics ════════════════════════════════════════════════════

    def _bullet_pos_at(self, st, sp, sd, t):
        dt = t - st
        vel = sd * BULLET_SPEED
        return np.array([
            sp[0] + vel[0] * dt,
            sp[1] + vel[1] * dt,
            sp[2] + vel[2] * dt - 0.5 * GRAVITY * dt * dt,
        ])

    def _check_hit(self, st, sp, sd):
        """Segment-vs-box against MuJoCo armor geoms (exact C++ match)."""
        m, d = self._model, self._data
        for geom_id in self._armor_geoms:
            body_id = m.geom_bodyid[geom_id]
            bp = d.xpos[body_id]
            bx = d.xmat[body_id].reshape(3, 3)
            half = m.geom_size[geom_id] + BULLET_RADIUS

            to_target = np.array([bp[0], bp[1], bp[2]]) - sp
            h_xy = np.hypot(to_target[0], to_target[1])
            if h_xy < 1e-6:
                continue
            pitch_to_target = np.arctan2(to_target[2], h_xy)
            tof = h_xy / (BULLET_SPEED * np.cos(pitch_to_target))

            n_samples = 8
            a = sp.copy()
            for i in range(1, n_samples + 1):
                t_frac = tof * i / n_samples
                b = self._bullet_pos_at(st, sp, sd, st + t_frac)

                da = a - bp; db = b - bp
                al = np.array([np.dot(bx[0], da), np.dot(bx[1], da), np.dot(bx[2], da)])
                bl = np.array([np.dot(bx[0], db), np.dot(bx[1], db), np.dot(bx[2], db)])

                t_min, t_max = 0.0, 1.0
                hit = True
                for j in range(3):
                    direction = bl[j] - al[j]
                    if abs(direction) < 1e-9:
                        if al[j] < -half[j] or al[j] > half[j]:
                            hit = False; break
                        continue
                    t1 = (-half[j] - al[j]) / direction
                    t2 = (half[j] - al[j]) / direction
                    if t1 > t2:
                        t1, t2 = t2, t1
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)
                    if t_min > t_max:
                        hit = False; break
                if hit and t_max >= 0.0:
                    return True
                a = b
        return False

    # ══ Observation ═══════════════════════════════════════════════════════

    def _build_obs(self):
        d = self._data
        obs = np.zeros(OBS_STACK * 3 + 6, dtype=np.float32)
        mz = d.site_xpos[self._muzzle_site]
        muzzle_pos = np.array([mz[0], mz[1], mz[2]])
        for i in range(OBS_STACK):
            obs[i * 3:(i + 1) * 3] = (self._pos_history[i] - muzzle_pos) * POS_SCALE
        obs[24] = d.qpos[self._yaw_qpos] * ANGLE_SCALE
        obs[25] = d.qpos[self._pitch_qpos] * ANGLE_SCALE
        obs[26] = d.qvel[self._yaw_qvel] * VEL_SCALE
        obs[27] = d.qvel[self._pitch_qvel] * VEL_SCALE
        obs[28] = self._heat * HEAT_SCALE
        last = 0.0 if self._last_shot_time < 0 else min(self._time - self._last_shot_time, 1.0)
        obs[29] = last
        return obs

    # ══ Reset ═════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed) if seed is not None else np.random

        mujoco.mj_resetData(self._model, self._data)

        # Target state.
        if self.difficulty == "hard":
            self._target_center = np.array([self._rng.uniform(3.0, 5.0), 0.0, 0.0])
            self._target_state = "IDLE"
            self._target_state_start = 0.0
            self._target_duration = self._rng.uniform(0.1, 0.8)
            self._target_start_pos = self._target_center.copy()
            self._target_waypoint = self._target_center.copy()
        else:
            self._target_center = np.array([LINEAR_X, 0.0, 0.015])
            self._linear_dir = 1
            self._target_prev_time_linear = 0.0

        # Time / heat / bullets.
        self._time = 0.0
        self._shots_fired = 0
        self._score = 0
        self._heat = 0.0
        self._last_heat_update = 0.0
        self._last_shot_time = -1.0
        self._pending_shots = []
        self._bullet_times = []
        self._bullet_scored = []

        # Lag buffer.
        self._lag_times = np.zeros(60)
        self._lag_pos = np.zeros((60, 3))
        self._lag_head = 0
        self._lag_count = 0

        # Position history.
        self._pos_history = np.zeros((OBS_STACK, 3))

        # Hide all bullets underground.
        for mid in self._bullet_mocaps:
            self._data.mocap_pos[mid, 2] = -10.0

        # Seed initial observation.
        self._push_lag(self._closest_plate())
        self._pos_history[0] = self._closest_plate()

        if self._viewer is not None:
            self._viewer.sync()

        return self._build_obs(), {}

    # ══ Step ═══════════════════════════════════════════════════════════════

    def step(self, action):
        d = self._data
        yaw_vel = float(np.clip(action[0], -1.0, 1.0) * MAX_YAW_VEL)
        pitch_vel = float(np.clip(action[1], -1.0, 1.0) * MAX_PITCH_VEL)
        fire = float(action[2]) > 0.0

        # ── Accel limits (matching C++ static clamp) ─────────────────
        if not hasattr(self, '_prev_yaw_vel'):
            self._prev_yaw_vel = 0.0
            self._prev_pitch_vel = 0.0
        lim = MAX_YAW_ACCEL * DT
        yaw_vel = np.clip(yaw_vel, self._prev_yaw_vel - lim, self._prev_yaw_vel + lim)
        pitch_vel = np.clip(pitch_vel, self._prev_pitch_vel - lim, self._prev_pitch_vel + lim)
        self._prev_yaw_vel = yaw_vel
        self._prev_pitch_vel = pitch_vel

        reward = 0.0
        terminated = False

        # ── Update target (analytical, matching C++) ─────────────────
        self._update_target()

        # ── Sub-step loop (matching C++ while(d->time-fs<kRenderDt)) ─
        for _ in range(SUBSTEPS):
            d.ctrl[self._yaw_motor] = yaw_vel
            d.ctrl[self._pitch_motor] = pitch_vel
            d.qpos[self._yaw_qpos] = np.clip(
                (d.qpos[self._yaw_qpos] + yaw_vel * MUJOCO_DT + PI) % (2 * PI) - PI,
                -YAW_LIMIT, YAW_LIMIT)
            d.qpos[self._pitch_qpos] = np.clip(
                d.qpos[self._pitch_qpos] + pitch_vel * MUJOCO_DT, PITCH_MIN, PITCH_MAX)
            d.qvel[self._yaw_qvel] = yaw_vel
            d.qvel[self._pitch_qvel] = pitch_vel
            d.time += MUJOCO_DT
            mujoco.mj_forward(self._model, d)

        # ── Cool heat ────────────────────────────────────────────────
        dt_heat = max(0.0, d.time - self._last_heat_update)
        self._heat = max(0.0, self._heat - HEAT_COOLING * dt_heat)
        self._last_heat_update = d.time

        # ── Set target mocap body ────────────────────────────────────
        center = self._target_center
        d.mocap_pos[self._target_mocap] = (float(center[0]), float(center[1]), float(center[2]))
        spin_yaw = 0.0 if self.difficulty == "easy" else ORBIT_OMEGA * d.time
        half = spin_yaw * 0.5
        d.mocap_quat[self._target_mocap] = (np.cos(half), 0.0, 0.0, np.sin(half))
        mujoco.mj_forward(self._model, d)

        # ── Detection lag ────────────────────────────────────────────
        self._push_lag(self._closest_plate())

        # ── Process pending shots (30ms delay) ───────────────────────
        remaining = []
        for trigger_t, mz_pos, mz_dir in self._pending_shots:
            if d.time >= trigger_t + SHOOT_DELAY:
                self._bullet_times.append((d.time, mz_pos.copy(), mz_dir.copy()))
                self._bullet_scored.append(False)
            else:
                remaining.append((trigger_t, mz_pos, mz_dir))
        self._pending_shots = remaining

        # ── Fire ─────────────────────────────────────────────────────
        mz = d.site_xpos[self._muzzle_site]
        muzzle_pos = np.array([mz[0], mz[1], mz[2]])
        # MuJoCo xmat is column-major — x-axis = elements [0,3,6] (matching C++ main.cpp)
        mx_flat = d.site_xmat[self._muzzle_site]
        muzzle_dir = np.array([mx_flat[0], mx_flat[3], mx_flat[6]])

        can_fire = (fire and self._shots_fired < MAX_SHOTS
                    and self._heat + HEAT_PER_SHOT <= HEAT_LIMIT
                    and (self._last_shot_time < 0
                         or d.time - self._last_shot_time >= AUTO_COOLDOWN))

        if can_fire:
            if SHOOT_DELAY > 0:
                self._pending_shots.append((d.time, muzzle_pos.copy(), muzzle_dir.copy()))
            else:
                self._bullet_times.append((d.time, muzzle_pos.copy(), muzzle_dir.copy()))
                self._bullet_scored.append(False)
            self._shots_fired += 1
            self._last_shot_time = d.time
            self._heat += HEAT_PER_SHOT

        # ── Check hits ───────────────────────────────────────────────
        for i, (st, sp, sd) in enumerate(self._bullet_times):
            if not self._bullet_scored[i]:
                if self._check_hit(st, sp, sd):
                    self._bullet_scored[i] = True
                    self._score += 1
                    reward += 5.0

        # ── Tracking reward ──────────────────────────────────────────
        delayed_pos = self._get_lagged()
        delta = delayed_pos - muzzle_pos
        target_yaw = np.arctan2(delta[1], delta[0])
        target_pitch = np.arctan2(delta[2], np.hypot(delta[0], delta[1]))
        yaw_err = target_yaw - d.qpos[self._yaw_qpos]
        yaw_err = (yaw_err + PI) % (2 * PI) - PI
        pitch_err = target_pitch - d.qpos[self._pitch_qpos]
        reward -= 0.5 * (yaw_err**2 + pitch_err**2)

        # ── Update bullet mocap positions (visual debug only) ────────
        for i, (st, sp, sd) in enumerate(self._bullet_times):
            if i < 100:
                cur = self._bullet_pos_at(st, sp, sd, d.time)
                self._data.mocap_pos[self._bullet_mocaps[i]] = (
                    float(cur[0]), float(cur[1]), float(cur[2]))
        for i in range(len(self._bullet_times), 100):
            self._data.mocap_pos[self._bullet_mocaps[i], 2] = -10.0

        # ── Position history ─────────────────────────────────────────
        self._pos_history = np.roll(self._pos_history, 1, axis=0)
        self._pos_history[0] = delayed_pos

        # ── Advance time ─────────────────────────────────────────────
        self._time = d.time
        reward -= 0.005

        # ── Termination ──────────────────────────────────────────────
        if self._shots_fired >= MAX_SHOTS or d.time >= TIME_LIMIT:
            unfired = MAX_SHOTS - self._shots_fired
            if unfired > 0:
                reward -= 0.5 * unfired
            terminated = True

        if self._viewer is not None:
            self._viewer.sync()

        return self._build_obs(), reward, terminated, False, {
            "score": self._score,
            "shots": self._shots_fired,
            "time": d.time,
            "heat": self._heat,
        }

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            if sys.platform == "darwin":
                import subprocess
                subprocess.run(["killall", "-9", "mjpython"], capture_output=True)
