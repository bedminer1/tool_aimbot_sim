"""Fast pure-Python replica of the MuJoCo C++ sim for PPO training.

Matches the C++ sim exactly: random waypoints, smoothstep, orbit wobble,
kinematic gimbal, ballistic bullets, detection/shooting delays, barrel heat.
No rendering — designed for >100k steps/s during training.
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
YAW_LIMIT = 2.8
PITCH_MIN = -0.8
PITCH_MAX = 0.8
HEAT_LIMIT = 260.0
HEAT_PER_SHOT = 10.0
HEAT_COOLING = 30.0
MAX_SHOTS = 100
DETECTION_LAG = 0.015
SHOOT_DELAY = 0.030
AUTO_COOLDOWN = 0.12
ORBIT_RADIUS = 0.04
ORBIT_PERIOD = 0.6
ORBIT_OMEGA = 2.0 * PI / ORBIT_PERIOD
TARGET_HALF_Y = 0.07    # 140 mm / 2
TARGET_HALF_Z = 0.0625  # 125 mm / 2
TARGET_HALF_X = 0.0085  # thickness / 2
OBS_STACK = 8            # number of position history frames

# ── Observation normalization constants ─────────────────────────────────────
POS_SCALE = 1.0 / 5.0
ANGLE_SCALE = 1.0 / PI
VEL_SCALE = 1.0 / 4.0
HEAT_SCALE = 1.0 / HEAT_LIMIT


class GimbalEnv(Env):
    """PPO training environment matching the C++ aimbot sim."""

    def __init__(self, render_mode=None):
        super().__init__()
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
        self._target_center = np.array([rng.uniform(3.0, 5.0), 0.0, 0.43])
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
        self._muzzle_pos = np.array([0.0, 0.0, 0.76])

        return self._build_obs(), {}

    # ══ Target motion (exact replica of C++ update_target_motion) ══════════

    def _random_waypoint(self):
        r = self._rng.uniform(3.0, 5.0)
        a = self._rng.uniform(-0.55, 0.55)
        z = self._rng.uniform(0.35, 0.55)
        return np.array([r * np.cos(a), r * np.sin(a), z])

    @staticmethod
    def _smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _update_target(self):
        elapsed = self.time - self._target_state_start
        if self._target_state == "IDLE":
            self._target_vel[:] = 0.0
            if elapsed >= self._target_duration:
                self._target_state = "MOVING"
                self._target_state_start = self.time
                self._target_start_pos = self._target_center.copy()
                self._target_waypoint = self._random_waypoint()
                dist = np.linalg.norm(self._target_waypoint - self._target_start_pos)
                speed = self._rng.uniform(1.0, 2.5)
                self._target_duration = dist / max(0.1, speed)
        else:  # MOVING
            t = elapsed / self._target_duration
            if t >= 1.0:
                self._target_center = self._target_waypoint.copy()
                self._target_vel[:] = 0.0
                self._target_state = "IDLE"
                self._target_state_start = self.time
                self._target_duration = self._rng.uniform(0.1, 0.8)
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
        """Segment-vs-box: bullet path from spawn to impact vs target box."""
        # Impact time: when bullet reaches target's XY plane distance.
        to_target = target_composite - spawn_pos
        h_xy = np.hypot(to_target[0], to_target[1])
        pitch = np.arctan2(to_target[2], h_xy)
        tof = h_xy / (BULLET_SPEED * np.cos(pitch))

        # Bullet positions at sample points along trajectory.
        n_samples = 8
        prev = spawn_pos.copy()
        for i in range(1, n_samples + 1):
            t_frac = tof * i / n_samples
            curr = self._bullet_pos_at(spawn_time, spawn_pos, spawn_dir,
                                       spawn_time + t_frac)
            if self._segment_hits_box(prev, curr, target_composite):
                return True
            prev = curr
        return False

    def _segment_hits_box(self, a, b, target_center):
        """Check if segment a→b intersects the target box."""
        # Transform to target-local frame (X=thickness, Y=width, Z=height).
        # Center of box is target_center.
        # Local axes: +X = outward from orbit (toward +gimbal direction approx).
        # For simplicity: box is oriented with its face toward origin.

        # Direction from target to origin (muzzle is near origin).
        to_origin = -target_center
        h_xy = np.hypot(to_origin[0], to_origin[1])
        if h_xy < 1e-6:
            return False
        local_x = to_origin / np.linalg.norm(to_origin)  # points toward origin
        local_y = np.array([-local_x[1], local_x[0], 0.0])
        local_z = np.array([0.0, 0.0, 1.0])

        # Expand half-extents by bullet radius.
        half = np.array([TARGET_HALF_X + BULLET_RADIUS,
                         TARGET_HALF_Y + BULLET_RADIUS,
                         TARGET_HALF_Z + BULLET_RADIUS])

        # Transform a and b to local frame.
        da = a - target_center
        db = b - target_center
        al = np.array([np.dot(local_x, da), np.dot(local_y, da), np.dot(local_z, da)])
        bl = np.array([np.dot(local_x, db), np.dot(local_y, db), np.dot(local_z, db)])

        t_min, t_max = 0.0, 1.0
        for i in range(3):
            direction = bl[i] - al[i]
            if abs(direction) < 1e-9:
                if al[i] < -half[i] or al[i] > half[i]:
                    return False
                continue
            t1 = (-half[i] - al[i]) / direction
            t2 = (half[i] - al[i]) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        return t_max >= 0.0

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
        self._push_observation(target_pos)

        # ── Cool heat ──
        dt_heat = max(0.0, self.time - self.last_heat_update)
        self.heat = max(0.0, self.heat - HEAT_COOLING * dt_heat)
        self.last_heat_update = self.time

        # ── Update gimbal kinematics ──
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

        # ── Process pending shots ──
        remaining = []
        for trigger_t, mz_pos, mz_dir in self._pending_shots:
            if self.time >= trigger_t + SHOOT_DELAY:
                self._bullet_times.append((self.time, mz_pos.copy(), mz_dir.copy()))
                self._bullet_scored.append(False)
            else:
                remaining.append((trigger_t, mz_pos, mz_dir))
        self._pending_shots = remaining

        # ── Fire ──
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
            reward -= 0.05  # spray penalty (low so shooting isn't punished early)

        # ── Check hits ──
        for i, (st, sp, sd) in enumerate(self._bullet_times):
            if not self._bullet_scored[i]:
                if self._check_hit(st, sp, sd, target_pos):
                    self._bullet_scored[i] = True
                    self.score += 1
                    reward += 3.0  # hit reward (strong signal to learn shooting)

        # ── Tracking reward ──
        delayed_pos = self._get_delayed_pos()
        delta = delayed_pos - self._muzzle_pos
        target_yaw = np.arctan2(delta[1], delta[0])
        target_pitch = np.arctan2(delta[2], np.hypot(delta[0], delta[1]))
        yaw_err = target_yaw - self.yaw
        yaw_err = (yaw_err + PI) % (2 * PI) - PI
        pitch_err = target_pitch - self.pitch
        reward -= 0.002 * (yaw_err * yaw_err + pitch_err * pitch_err)

        # ── Update position history ──
        self._pos_history = np.roll(self._pos_history, 1, axis=0)
        self._pos_history[0] = delayed_pos

        # ── Advance time ──
        self.time += DT

        # ── Terminate ──
        if self.shots_fired >= MAX_SHOTS:
            terminated = True

        return self._build_obs(), reward, terminated, False, {
            "score": self.score,
            "shots": self.shots_fired,
            "time": self.time,
            "heat": self.heat,
        }
