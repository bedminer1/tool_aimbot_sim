"""
@file    training/ppo_env.py
@brief   Pure-Python replica of the MuJoCo C++ sim for PPO training

@details
Gymnasium environment matching the C++ sim (src/cli/main.cpp) exactly:
same target motion, detection lag, ballistic bullets, barrel heat.

Why a replica instead of wrapping MuJoCo?
  Speed: >100k steps/s in pure Python vs ~60 fps in the viewer.
  The policy trains on 500k steps in ~5 minutes instead of ~2 hours.

Architecture (layered to match C++):
  1. Target motion:   IDLE/MOVING state machine + smoothstep (no orbit wobble
                       since the Python env doesn't render the chassis spin —
                       the C++ spin is handled by the multi-plate hit detection
                       which the PPO predictor doesn't need to model)
  2. Detection lag:   ring buffer (60 entries), 15 ms delay
  3. Shooting delay:  30 ms between trigger and muzzle exit
  4. Ballistic bullets: analytic trajectory, gravity, segment-vs-box hits
  5. Barrel heat:      RMUC model (260 J / +10 per shot / -30 J/s)
  6. Auto-fire:        policy outputs fire_logit; threshold at 0

Observation (30-D, fixed-scale normalized):
  8-frame pos history (24) + gimbal yaw/pitch (2) + gimbal vel (2)
  + barrel heat (1) + time since last shot (1)

Action (3-D):
  [yaw_vel, pitch_vel, fire_logit]  — all in [-1, 1]

Reward structure:
  +5.0    per hit (sparse)
  -0.01 × (|yaw_err| + |pitch_err|)   tracking gradient
  -0.005  per timestep (time pressure)
  -0.5 ×  unfired_shots at timeout (waste penalty)

The tracking reward is deliberately small relative to hits — a 3:1 ratio
ensures the policy prioritizes shooting over perfect tracking.

@see training/train.py, src/cli/main.cpp, aim_models/aim_predictor_ppo.hpp
@author  bedminer1
@date    2026-08-03
"""

import numpy as np
from gymnasium import Env, spaces


# ── Constants (matching C++ main.cpp) ──────────────────────────────────────
PI = np.pi
DT = 1.0 / 60.0
BULLET_SPEED = 24.8
GRAVITY = 9.81
BULLET_RADIUS = 0.017 / 2.0
MAX_YAW_VEL = 4.0
MAX_PITCH_VEL = 4.0
MAX_YAW_ACCEL = 20.0     # rad/s² — ~5× motor rating, smooths jitter
MAX_PITCH_ACCEL = 20.0
YAW_LIMIT = 2.8
PITCH_MIN = -0.8
PITCH_MAX = 0.8
GIMBAL_HEIGHT = 0.34    # pitch joint world Z (base 0.30 + yaw_link 0.04)
MUZZLE_LENGTH = 0.76     # muzzle site X in pitch_link frame
HEAT_LIMIT = 260.0
HEAT_PER_SHOT = 10.0
HEAT_COOLING = 30.0
MAX_SHOTS = 50
TIME_LIMIT = 40.0  # episode timeout — penalize unfired shots
DETECTION_LAG = 0.015
SHOOT_DELAY = 0.030
AUTO_COOLDOWN = 0.12
AUTO_FIRE_THRESH = 0.20  # radians — fire when aimed within ~11°
ORBIT_RADIUS = 0.04
ORBIT_PERIOD = 0.6
ORBIT_OMEGA = 2.0 * PI / ORBIT_PERIOD
TARGET_HALF_Y = 0.0675   # 135 mm / 2 (RMUL spec)
TARGET_HALF_Z = 0.0625  # 125 mm / 2
TARGET_HALF_X = 0.0085  # thickness / 2
OBS_STACK = 8            # number of position history frames

# ── Difficulty-specific constants ───────────────────────────────────────────
# Medium / Easy: linear Y back-and-forth (matching C++ TargetEasy/Medium)
LINEAR_X = 4.0
LINEAR_Y_MIN = -1.5
LINEAR_Y_MAX = 1.5
LINEAR_SPEED = 0.5
# Hard: random waypoints (matching C++ TargetHard)
HARD_RANGE_LO = 3.0
HARD_RANGE_HI = 4.5
HARD_ANGLE_MAX = 0.4
HARD_SPEED_LO = 0.5
HARD_SPEED_HI = 1.5

# ── Observation normalization constants ─────────────────────────────────────
POS_SCALE = 1.0 / 5.0
ANGLE_SCALE = 1.0 / PI
VEL_SCALE = 1.0 / 4.0
HEAT_SCALE = 1.0 / HEAT_LIMIT


class GimbalEnv(Env):
    """PPO training environment matching the C++ aimbot sim."""

    def __init__(self, difficulty="hard", render_mode=None):
        super().__init__()
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown difficulty: {difficulty}")
        self.difficulty = difficulty
        obs_dim = OBS_STACK * 3 + 4 + 1 + 1  # 30
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed) if seed is not None else np.random

        # ── Gimbal state ──
        self.yaw = 0.0
        self.pitch = 0.0
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0

        # ── Target state (matching C++ TargetBehavior) ──
        if self.difficulty in ("easy", "medium"):
            # Linear Y back-and-forth at fixed X.
            self._target_center = np.array([LINEAR_X, 0.0, 0.015])
            self._target_vel = np.zeros(3)
            self._linear_dir = 1  # +Y or -Y
            self._target_prev_composite = self._target_center.copy()
            self._target_prev_time = 0.0
        else:  # hard
            self._target_center = np.array([rng.uniform(3.0, 5.0), 0.0, 0.0])
            self._target_state = "IDLE"
            self._target_state_start = 0.0
            self._target_duration = rng.uniform(0.1, 0.8)
            self._target_start_pos = self._target_center.copy()
            self._target_waypoint = self._target_center.copy()
            self._target_vel = np.zeros(3)
            self._target_prev_composite = self._target_center.copy()
            self._target_prev_time = 0.0
        self._rng = rng

        # ── Bullets / heat ──
        self.time = 0.0
        self.shots_fired = 0
        self.score = 0
        self.heat = 0.0
        self.last_heat_update = 0.0
        self.last_auto_shot = -1.0
        self._pending_shots = []       # list of (trigger_time, muzzle_xy, muzzle_dir)
        self._bullet_times = []        # list of (spawn_time, spawn_pos, spawn_dir)
        self._bullet_scored = []

        # ── Detection lag buffer (ring buffer of (time, pos)) ──
        self._obs_times = np.zeros(60)
        self._obs_positions = np.zeros((60, 3))
        self._obs_head = 0
        self._obs_count = 0

        # ── Position history for observation ──
        self._pos_history = np.zeros((OBS_STACK, 3))

        self._first_shot_time = -1.0
        self._last_shot_time = -1.0
        self._muzzle_pos = np.array([MUZZLE_LENGTH, 0.0, GIMBAL_HEIGHT])

        return self._build_obs(), {}

    # ══ Target motion (matching C++ target models) ═══════════════════════════

    def _update_target_linear(self):
        """Easy/Medium: linear Y back-and-forth at fixed X (TargetEasy/Medium)."""
        dt = self.time - self._target_prev_time
        if dt <= 0.0:
            return self._target_center

        ny = self._target_center[1] + self._linear_dir * LINEAR_SPEED * dt
        if ny > LINEAR_Y_MAX:
            ny = LINEAR_Y_MAX
            self._linear_dir = -1
        elif ny < LINEAR_Y_MIN:
            ny = LINEAR_Y_MIN
            self._linear_dir = 1
        self._target_center = np.array([LINEAR_X, ny, 0.015])

        composite = self._target_center
        if self.difficulty == "medium":
            # Spin/orbit wobble (matching C++ kSpinRads = 2π/0.6s)
            phase = ORBIT_OMEGA * self.time
            orbit = np.array([ORBIT_RADIUS * np.sin(phase),
                              ORBIT_RADIUS * np.cos(phase), 0.0])
            composite = self._target_center + orbit

        if dt > 1e-6:
            self._target_vel = (composite - self._target_prev_composite) / dt
        self._target_prev_composite = composite
        self._target_prev_time = self.time
        return composite

    @staticmethod
    def _smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _random_waypoint(self):
        r = self._rng.uniform(HARD_RANGE_LO, HARD_RANGE_HI)
        a = self._rng.uniform(-HARD_ANGLE_MAX, HARD_ANGLE_MAX)
        return np.array([r * np.cos(a), r * np.sin(a), 0.015])

    def _update_target(self):
        if self.difficulty in ("easy", "medium"):
            return self._update_target_linear()

        # Hard: random waypoints + orbit wobble
        elapsed = self.time - self._target_state_start
        if self._target_state == "IDLE":
            self._target_vel[:] = 0.0
            if elapsed >= self._target_duration:
                self._target_state = "MOVING"
                self._target_state_start = self.time
                self._target_start_pos = self._target_center.copy()
                self._target_waypoint = self._random_waypoint()
                dist = np.linalg.norm(self._target_waypoint - self._target_start_pos)
                speed = self._rng.uniform(HARD_SPEED_LO, HARD_SPEED_HI)
                self._target_duration = dist / max(0.1, speed)
        else:  # MOVING
            t = elapsed / self._target_duration
            if t >= 1.0:
                self._target_center = self._target_waypoint.copy()
                self._target_vel[:] = 0.0
                self._target_state = "IDLE"
                self._target_state_start = self.time
                self._target_duration = self._rng.uniform(0.5, 2.0)
            else:
                st = self._smoothstep(t)
                self._target_center = self._target_start_pos + (
                    self._target_waypoint - self._target_start_pos
                ) * st

        # Orbit wobble
        phase = ORBIT_OMEGA * self.time
        orbit = np.array([ORBIT_RADIUS * np.sin(phase), ORBIT_RADIUS * np.cos(phase), 0.0])
        composite = self._target_center + orbit

        # Velocity from finite difference
        dt_v = self.time - self._target_prev_time
        if dt_v > 1e-6:
            self._target_vel = (composite - self._target_prev_composite) / dt_v
        self._target_prev_composite = composite
        self._target_prev_time = self.time

        return composite

    # ══ Detection lag buffer ════════════════════════════════════════════════

    def _push_observation(self, pos):
        self._obs_times[self._obs_head] = self.time
        self._obs_positions[self._obs_head] = pos
        self._obs_head = (self._obs_head + 1) % 60
        self._obs_count = min(self._obs_count + 1, 60)

    def _get_delayed_pos(self):
        query_t = self.time - DETECTION_LAG
        best = None
        for i in range(min(self._obs_count, 60)):
            idx = (self._obs_head - 1 - i) % 60
            if self._obs_times[idx] <= query_t:
                best = self._obs_positions[idx].copy()
                break
        if best is None and self._obs_count > 0:
            idx = (self._obs_head - self._obs_count) % 60
            best = self._obs_positions[idx].copy()
        return best if best is not None else self._target_center.copy()

    # ══ Bullet physics ══════════════════════════════════════════════════════

    def _bullet_pos_at(self, spawn_time, spawn_pos, spawn_dir, t):
        dt = t - spawn_time
        vel = spawn_dir * BULLET_SPEED
        return np.array([
            spawn_pos[0] + vel[0] * dt,
            spawn_pos[1] + vel[1] * dt,
            spawn_pos[2] + vel[2] * dt - 0.5 * GRAVITY * dt * dt,
        ])

    def _check_hit(self, spawn_time, spawn_pos, spawn_dir, target_composite):
        """Check if bullet segment hits any of 4 armor plates (matching C++)."""
        center = self._target_center
        spin = ORBIT_OMEGA * self.time if self.difficulty != "easy" else 0.0
        c, s = np.cos(spin), np.sin(spin)

        # 4 plate positions in world frame (matching gimbal.xml armor_0..3).
        # Local positions: N(0,+0.245), E(+0.245,0), S(0,-0.245), W(-0.245,0)
        # at z=0.02 above chassis center (z=0.015 + half plate height).
        plates = [
            center + np.array([-s * 0.245,  c * 0.245, 0.02]),   # N
            center + np.array([ c * 0.245,  s * 0.245, 0.02]),   # E
            center + np.array([ s * 0.245, -c * 0.245, 0.02]),   # S
            center + np.array([-c * 0.245, -s * 0.245, 0.02]),   # W
        ]

        for plate_pos in plates:
            if self._segment_hits_box(spawn_pos, spawn_dir, plate_pos,
                                       spawn_time, target_composite):
                return True
        return False

    def _segment_hits_box(self, a_start, a_dir, box_center, spawn_time, _unused):
        """Segment-vs-box: bullet trajectory against an armor plate box.

        Uses analytic bullet trajectory with gravity."""
        to_target = box_center - a_start
        h_xy = np.hypot(to_target[0], to_target[1])
        if h_xy < 1e-6:
            return False
        pitch_to_target = np.arctan2(to_target[2], h_xy)
        tof = h_xy / (BULLET_SPEED * np.cos(pitch_to_target))

        # Direction from box to origin (gimbal near origin).
        to_origin = -box_center
        d_norm = np.linalg.norm(to_origin)
        if d_norm < 1e-6:
            return False
        local_x = to_origin / d_norm
        local_y = np.array([-local_x[1], local_x[0], 0.0])
        local_z = np.array([0.0, 0.0, 1.0])

        half = np.array([TARGET_HALF_X + BULLET_RADIUS,
                         TARGET_HALF_Y + BULLET_RADIUS,
                         TARGET_HALF_Z + BULLET_RADIUS])

        n_samples = 8
        a = a_start.copy()
        for i in range(1, n_samples + 1):
            t_frac = tof * i / n_samples
            b = self._bullet_pos_at(spawn_time, a_start, a_dir,
                                     spawn_time + t_frac)
            da = a - box_center
            db = b - box_center
            al = np.array([np.dot(local_x, da), np.dot(local_y, da), np.dot(local_z, da)])
            bl = np.array([np.dot(local_x, db), np.dot(local_y, db), np.dot(local_z, db)])

            t_min, t_max = 0.0, 1.0
            hit = True
            for j in range(3):
                direction = bl[j] - al[j]
                if abs(direction) < 1e-9:
                    if al[j] < -half[j] or al[j] > half[j]:
                        hit = False
                        break
                    continue
                t1 = (-half[j] - al[j]) / direction
                t2 = (half[j] - al[j]) / direction
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    hit = False
                    break
            if hit and t_max >= 0.0:
                return True
            a = b
        return False

    # ══ Step ═══════════════════════════════════════════════════════════════

    def _build_obs(self):
        obs = np.zeros(OBS_STACK * 3 + 6, dtype=np.float32)
        for i in range(OBS_STACK):
            obs[i * 3:(i + 1) * 3] = (self._pos_history[i] - self._muzzle_pos) * POS_SCALE
        obs[24] = self.yaw * ANGLE_SCALE
        obs[25] = self.pitch * ANGLE_SCALE
        obs[26] = self.yaw_vel * VEL_SCALE
        obs[27] = self.pitch_vel * VEL_SCALE
        obs[28] = self.heat * HEAT_SCALE
        last = 0.0 if self._last_shot_time < 0 else min(self.time - self._last_shot_time, 1.0)
        obs[29] = last
        return obs

    def step(self, action):
        yaw_vel = float(np.clip(action[0], -1.0, 1.0) * MAX_YAW_VEL)
        pitch_vel = float(np.clip(action[1], -1.0, 1.0) * MAX_PITCH_VEL)
        fire = float(action[2]) > 0.0

        reward = 0.0
        terminated = False

        # ── Update target ──
        target_pos = self._update_target()
        self._push_observation(self._target_center)  # center only (matching C++ observer)

        # ── Cool heat ──
        dt_heat = max(0.0, self.time - self.last_heat_update)
        self.heat = max(0.0, self.heat - HEAT_COOLING * dt_heat)
        self.last_heat_update = self.time

        # ── Update gimbal kinematics ──
        # Clamp acceleration for realistic motor behavior.
        yaw_vel = np.clip(yaw_vel, self.yaw_vel - MAX_YAW_ACCEL * DT,
                          self.yaw_vel + MAX_YAW_ACCEL * DT)
        pitch_vel = np.clip(pitch_vel, self.pitch_vel - MAX_PITCH_ACCEL * DT,
                            self.pitch_vel + MAX_PITCH_ACCEL * DT)
        self.yaw += yaw_vel * DT
        self.pitch += pitch_vel * DT
        self.yaw = (self.yaw + PI) % (2 * PI) - PI
        self.yaw = np.clip(self.yaw, -YAW_LIMIT, YAW_LIMIT)
        self.pitch = np.clip(self.pitch, PITCH_MIN, PITCH_MAX)
        self.yaw_vel = yaw_vel
        self.pitch_vel = pitch_vel

        # ── Compute muzzle direction ──
        muzzle_dir = np.array([np.cos(self.pitch) * np.cos(self.yaw),
                               np.cos(self.pitch) * np.sin(self.yaw),
                               np.sin(self.pitch)])

        # ── Compute muzzle world position from forward kinematics ──
        self._muzzle_pos[0] = MUZZLE_LENGTH * muzzle_dir[0]
        self._muzzle_pos[1] = MUZZLE_LENGTH * muzzle_dir[1]
        self._muzzle_pos[2] = GIMBAL_HEIGHT + MUZZLE_LENGTH * muzzle_dir[2]

        # ── Process pending shots ──
        remaining = []
        for trigger_t, mz_pos, mz_dir in self._pending_shots:
            if self.time >= trigger_t + SHOOT_DELAY:
                self._bullet_times.append((self.time, mz_pos.copy(), mz_dir.copy()))
                self._bullet_scored.append(False)
            else:
                remaining.append((trigger_t, mz_pos, mz_dir))
        self._pending_shots = remaining

        # ── Fire (policy decides) ──
        can_fire = (fire and self.shots_fired < MAX_SHOTS
                    and self.heat + HEAT_PER_SHOT <= HEAT_LIMIT
                    and (self._last_shot_time < 0
                         or self.time - self._last_shot_time >= AUTO_COOLDOWN))

        if can_fire:
            if SHOOT_DELAY > 0:
                self._pending_shots.append((self.time, self._muzzle_pos.copy(), muzzle_dir.copy()))
            else:
                self._bullet_times.append((self.time, self._muzzle_pos.copy(), muzzle_dir.copy()))
                self._bullet_scored.append(False)

            if self.shots_fired == 0:
                self._first_shot_time = self.time
            self.shots_fired += 1
            self._last_shot_time = self.time
            self.heat += HEAT_PER_SHOT

        # ── Check hits ──
        for i, (st, sp, sd) in enumerate(self._bullet_times):
            if not self._bullet_scored[i]:
                if self._check_hit(st, sp, sd, target_pos):
                    self._bullet_scored[i] = True
                    self.score += 1
                    reward += 5.0  # hit reward

        # Tracking gradient.
        delayed_pos = self._get_delayed_pos()
        delta = delayed_pos - self._muzzle_pos
        target_yaw = np.arctan2(delta[1], delta[0])
        target_pitch = np.arctan2(delta[2], np.hypot(delta[0], delta[1]))
        yaw_err = target_yaw - self.yaw
        yaw_err = (yaw_err + PI) % (2 * PI) - PI
        pitch_err = target_pitch - self.pitch
        reward -= 0.5 * (yaw_err**2 + pitch_err**2)  # squared: small cheap, large punished

        # Position history for next frame.
        self._pos_history = np.roll(self._pos_history, 1, axis=0)
        self._pos_history[0] = delayed_pos

        # Advance time.
        self.time += DT
        reward -= 0.005  # time penalty — finish or bleed

        # Terminate on shots fired or time limit.
        if self.shots_fired >= MAX_SHOTS or self.time >= TIME_LIMIT:
            # Penalize unfired shots at timeout.
            unfired = MAX_SHOTS - self.shots_fired
            if unfired > 0:
                reward -= 0.5 * unfired
            terminated = True

        return self._build_obs(), reward, terminated, False, {
            "score": self.score,
            "shots": self.shots_fired,
            "time": self.time,
            "heat": self.heat,
        }
