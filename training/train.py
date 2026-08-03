"""PPO training for RoboMaster gimbal aimbot.

Trains a policy to track and shoot a target at configurable difficulty.
Two env backends:
  headless: GimbalEnv       — pure Python, >100k steps/s
  rendered: MujocoGimbalEnv  — same physics + MuJoCo 3D viewer

Outputs:
  - src/aim_predictor.onnx     ← ONNX policy for C++ inference
  - training/ppo_checkpoint    ← SB3 checkpoint (optional resume)
  - training/eval_log.csv      ← eval metrics for graphing

Usage:
  python training/train.py                              # headless, medium
  python training/train.py --render                      # with MuJoCo viewer
  python training/train.py -d hard --timesteps 1000000   # hard, 1M steps
  python training/train.py --resume                      # continue checkpoint
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.ppo_env import GimbalEnv, MAX_SHOTS  # noqa: E402


def _unwrap_info(info):
    """VecNormalize wraps info in a list of dicts."""
    if isinstance(info, (list, tuple)):
        return info[0] if len(info) > 0 else {}
    return info or {}


class EvalCallback(BaseCallback):
    """Periodic evaluation: runs episodes, logs metrics to CSV."""

    def __init__(self, eval_freq=10000, n_episodes=5, csv_path=None,
                 save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.best_accuracy = 0.0
        self.best_time = float("inf")
        self._csv_path = csv_path
        self._save_path = save_path
        if self._csv_path:
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "accuracy", "time", "shots_landed"])

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        scores, times_, shot_counts = [], [], []
        obs = self.training_env.reset()
        n_envs = self.training_env.num_envs
        dones = np.zeros(n_envs, dtype=bool)
        while not np.all(dones):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done_arr, info = self.training_env.step(action)
            for i in range(n_envs):
                if done_arr[i] and not dones[i]:
                    dones[i] = True
                    inf = info[i] if isinstance(info, (list, tuple)) else info
                    scores.append(inf.get("score", 0))
                    times_.append(inf.get("time", 0))
                    shot_counts.append(inf.get("shots", 0))

        avg_score = np.mean(scores)
        avg_time = np.mean(times_)
        avg_shots = np.mean(shot_counts)
        accuracy = avg_score / MAX_SHOTS * 100

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            if self._save_path:
                self.model.save(self._save_path)
                print(f"  → saved best model ({accuracy:.1f}%)")
        if avg_time < self.best_time:
            self.best_time = avg_time

        print(f"\n[Eval {self.n_calls:7d}] "
              f"acc={accuracy:5.1f}% (best={self.best_accuracy:5.1f}%)  "
              f"time={avg_time:6.2f}s (best={self.best_time:6.2f}s)  "
              f"shots={avg_shots:.0f}")

        if self._csv_path:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow([self.n_calls, f"{accuracy:.1f}",
                                        f"{avg_time:.2f}", f"{avg_shots:.0f}"])
        return True


def export_onnx(model, path):
    """Export SB3 PPO policy to ONNX for C++ inference (3D: yaw_vel, pitch_vel, fire)."""
    obs_dim = model.observation_space.shape[0]
    dummy = th.randn(1, obs_dim)

    th.onnx.export(
        model.policy,
        (dummy,),
        path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX: {path}")

    session = ort.InferenceSession(path)
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    outputs = session.run(None, {"obs": test_input})
    assert outputs[0].shape == (1, 3), f"Bad output shape: {outputs[0].shape}"
    print(f"ONNX validation OK — output shape: {outputs[0].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--difficulty", "-d", default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--render", action="store_true",
                        help="Use MuJoCo passive viewer for 3D visualization")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(root, "training", "ppo_checkpoint.zip")
    best_path = os.path.join(root, "training", "ppo_best.zip")
    onnx_path = os.path.join(root, "src", "aim_predictor.onnx")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    log_dir = os.path.join(root, "training_log")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"eval_log_{timestamp}.csv")

    if args.render:
        from training.mujoco_training_env import MujocoTrainingEnv  # noqa: F811

        _viewer_created = False

        def make_env():
            nonlocal _viewer_created
            rm = "human" if not _viewer_created else None
            _viewer_created = True
            return MujocoTrainingEnv(difficulty=args.difficulty, render_mode=rm)
    else:
        from training.mujoco_training_env import MujocoTrainingEnv  # noqa: F811

        def make_env():
            return MujocoTrainingEnv(difficulty=args.difficulty)

    env = DummyVecEnv([make_env for _ in range(args.n_envs)])

    try:
        if args.resume and os.path.exists(checkpoint_path):
            print(f"Resuming from {checkpoint_path}")
            model = PPO.load(checkpoint_path, env=env)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [128, 128]},
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
                verbose=1,
                device="cpu",
            )

        eval_cb = EvalCallback(eval_freq=args.eval_freq, n_episodes=8,
                               csv_path=csv_path, save_path=best_path)

        t0 = time.time()
        model.learn(total_timesteps=args.timesteps, callback=eval_cb)
        elapsed = time.time() - t0
        print(f"\nTraining finished in {elapsed:.0f}s ({args.timesteps} steps)")

        model.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Final eval.
        scores, times_, shot_counts = [], [], []
        obs = env.reset()
        n_envs = env.num_envs
        dones = np.zeros(n_envs, dtype=bool)
        while not np.all(dones):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, info = env.step(action)
            for i in range(n_envs):
                if done_arr[i] and not dones[i]:
                    dones[i] = True
                    inf = info[i] if isinstance(info, (list, tuple)) else info
                    scores.append(inf["score"])
                    times_.append(inf["time"])
                    shot_counts.append(inf.get("shots", 0))
        print(f"\nFinal ({len(scores)} episodes): "
              f"acc={np.mean(scores):.1f}%  time={np.mean(times_):.2f}s  "
              f"shots={np.mean(shot_counts):.0f}")

        # Export best model (not final — PPO can degrade).
        if os.path.exists(best_path):
            print(f"Loading best model ({eval_cb.best_accuracy:.1f}%) for ONNX export")
            model = PPO.load(best_path, env=env)
        export_onnx(model, onnx_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
