"""MuJoCo-rendered GimbalEnv for visual PPO training.

Inherits GimbalEnv for fast analytical physics; adds MuJoCo passive
viewer for real-time 3D visualization of gimbal, target, and bullets.

Usage:
    env = MujocoGimbalEnv(difficulty="medium", render_mode="human")
    # training proceeds normally — viewer auto-updates each step
"""

import os
import subprocess
import sys
import numpy as np
import mujoco
import mujoco.viewer  # explicit submodule import for launch_passive
from training.ppo_env import GimbalEnv, ORBIT_OMEGA, BULLET_SPEED, GRAVITY


class MujocoGimbalEnv(GimbalEnv):
    """GimbalEnv with MuJoCo passive viewer for visual training."""

    def __init__(self, difficulty="medium", render_mode=None):
        super().__init__(difficulty=difficulty)

        xml_path = os.path.join(os.path.dirname(__file__), "..", "gimbal.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found: {xml_path}")

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        # Look up mocap body IDs.
        target_bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap = self._mj_model.body_mocapid[target_bid]

        self._bullet_mocaps = []
        for i in range(100):
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, f"bullet_{i}")
            self._bullet_mocaps.append(self._mj_model.body_mocapid[bid])

        # Joint qpos addresses.
        yaw_jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
        pitch_jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")
        self._yaw_qpos = self._mj_model.jnt_qposadr[yaw_jid]
        self._pitch_qpos = self._mj_model.jnt_qposadr[pitch_jid]

        self._viewer = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self._mj_model, self._mj_data,
                show_left_ui=False, show_right_ui=False,
            )

    # ── Render sync ─────────────────────────────────────────────────────────

    def _sync_to_mujoco(self):
        """Copy analytical env state to MuJoCo model for rendering."""
        d = self._mj_data

        # Target mocap body: center position + spin yaw.
        center = self._target_center
        mid = self._target_mocap
        d.mocap_pos[mid] = (float(center[0]), float(center[1]), float(center[2]))

        spin_yaw = 0.0 if self.difficulty == "easy" else ORBIT_OMEGA * self.time
        half = spin_yaw * 0.5
        d.mocap_quat[mid] = (np.cos(half), 0.0, 0.0, np.sin(half))

        # Gimbal joints.
        d.qpos[self._yaw_qpos] = self.yaw
        d.qpos[self._pitch_qpos] = self.pitch

        # Bullet visible positions.
        active = len(self._bullet_times)
        for i in range(min(active, 100)):
            st, sp, sd = self._bullet_times[i]
            cur = self._bullet_pos_at(st, sp, sd, self.time)
            mid = self._bullet_mocaps[i]
            d.mocap_pos[mid] = (float(cur[0]), float(cur[1]), float(cur[2]))
        # Hide unused bullets underground.
        for i in range(active, 100):
            mid = self._bullet_mocaps[i]
            d.mocap_pos[mid, 2] = -10.0

        mujoco.mj_forward(self._mj_model, d)

    # ── Step override ───────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._viewer is not None:
            self._sync_to_mujoco()
            self._viewer.sync()
        return obs, reward, terminated, truncated, info

    # ── Reset override ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._viewer is not None:
            self._sync_to_mujoco()
            self._viewer.sync()
        return obs, info

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            # Handle.close() sets exitrequest flag but the mjpython
            # NSApplication survives the render loop. Force-kill so the
            # next launch doesn't hit "already open".
            if sys.platform == "darwin":
                subprocess.run(["killall", "-9", "mjpython"],
                               capture_output=True)
        super().close()

    def __del__(self):
        """Safety net: close viewer if GC'd without explicit close()."""
        try:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
        except Exception:
            pass
