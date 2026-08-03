"""PPO training for RoboMaster gimbal aimbot.

Trains a policy to track and shoot a randomly-moving target.
Primary metric: accuracy (hit %). Secondary: time to fire 100 shots.

Outputs:
  - src/aim_predictor.onnx    ← ONNX policy for C++ inference
  - training/ppo_checkpoint   ← SB3 checkpoint (optional resume)

Usage:
  python training/train.py                     # train from scratch
  python training/train.py --resume            # resume from checkpoint
  python training/train.py --timesteps 500000  # custom budget
"""

import argparse
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.ppo_env import GimbalEnv  # noqa: E402


def _unwrap_info(info):
    """VecNormalize wraps info in a list of dicts."""
    if isinstance(info, (list, tuple)):
        return info[0] if len(info) > 0 else {}
    return info or {}


class EvalCallback(BaseCallback):
    """Periodic evaluation: runs N episodes, reports accuracy and time."""

    def __init__(self, eval_freq=10000, n_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.best_accuracy = 0.0
        self.best_time = float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        scores, times_ = [], []
        obs = self.training_env.reset()
        # DummyVecEnv: track which envs are done.
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

        avg_score = np.mean(scores)
        avg_time = np.mean(times_)
        accuracy = avg_score / 100.0 * 100

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        if avg_time < self.best_time:
            self.best_time = avg_time

        print(f"\n[Eval {self.n_calls:7d}] "
              f"acc={accuracy:5.1f}% (best={self.best_accuracy:5.1f}%)  "
              f"time={avg_time:6.2f}s (best={self.best_time:6.2f}s)")
        return True


def export_onnx(model, path):
    """Export SB3 PPO policy to ONNX for C++ inference (2D: yaw_vel, pitch_vel)."""
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
    assert outputs[0].shape == (1, 2), f"Bad output shape: {outputs[0].shape}"
    print(f"ONNX validation OK — output shape: {outputs[0].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(root, "training", "ppo_checkpoint.zip")
    onnx_path = os.path.join(root, "src", "aim_predictor.onnx")

    def make_env():
        return GimbalEnv()

    env = DummyVecEnv([make_env for _ in range(args.n_envs)])

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
            ent_coef=0.01,
            verbose=1,
            device="cpu",
        )

    eval_cb = EvalCallback(eval_freq=args.eval_freq, n_episodes=8)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.0f}s ({args.timesteps} steps)")

    model.save(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Final eval.
    scores, times_ = [], []
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
    print(f"\nFinal ({len(scores)} episodes): acc={np.mean(scores):.1f}%  time={np.mean(times_):.2f}s")

    export_onnx(model, onnx_path)


if __name__ == "__main__":
    main()
