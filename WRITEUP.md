# Unmissable

## Problem

In RoboMaster, sentry robots auto-aim moving armor plates using 24.8 m/s projectiles with ballistic drop, barrel heat limits, and 15 ms detection lag. Our team's current solution is a hand-tuned Kalman filter — it works but requires per-target parameter tuning and leaves accuracy on the table. Beyond competition robotics, the core problem — a moving platform tracking and precisely targeting a mobile object under latency constraints — generalizes to firefighting drones aiming water cannons, surgical robots tracking tumors during respiration, and agricultural sprayers targeting individual weeds. A sim-trained aiming policy that learns shot timing and trajectory prediction transfers across domains.

**Success criteria, defined before building:**

1. A sim that reproduces the real robot's I/O, latency, and ballistics.
2. A deterministic baseline ≥50% accuracy against random waypoint motion.
3. An RL pipeline that trains policies and exports to the real firmware interface.
4. Measure failure modes honestly — speed, range, direction changes — not best-case runs.

## Approach

**Simulation.** MuJoCo, not Gazebo. Gazebo is our navigation sim and the obvious choice for consistency, but MuJoCo's C API, single-XML model, and headless speed mattered more for a project iterating on aiming algorithms rather than full robot dynamics. The sim models our DM4310 gimbal motors (4 rad/s max, ±2.8 rad yaw, ±0.8 rad pitch), 140×125 mm target plate, ballistic 17 mm projectiles, and barrel heat from the RMUC rulebook (+10/shot, −30/s cooling, soft-lock at 260). Target motion: random polar waypoints at 3–5 m, 1.0–2.5 m/s, smoothstep interpolation, plus 0.04 m orbit wobble at 0.6 s period. Exports the same `hw::Command` struct as our STM32 firmware.

**Baseline: velocity extrapolation.** Delayed position → extrapolate with velocity + acceleration over bullet time-of-flight plus 30 ms shoot delay → 4-iteration refinement with `z = x·tan(θ) − (g·x²)/(2·v²·cos²(θ))`. Fifty lines, no learning, deterministic. Chosen because if RL can't beat this, deployment isn't justified.

**PPO.** I cloned the C++ sim as a pure-Python Gymnasium env — identical physics, no rendering, 100K+ steps/s. Observation: 30 floats (8-frame position history, gimbal state, barrel heat, time since last shot). Action: yaw velocity, pitch velocity, fire logit. Reward: +3/hit, −0.05/shot, −0.002/step for tracking error. Training: 4 parallel envs, 2048-step rollouts, 10 epochs at batch 256, 500K timesteps, CPU only.

**Chose MLP over LSTM.** Deliberate. Eight frames of explicit position history give the MLP sufficient temporal context, and a feedforward ONNX model runs <0.1 ms on M1 — estimated <1 ms on STM32 with CMSIS-NN. An LSTM would add deployment complexity for gains I haven't yet proven this task needs.

## Evidence

Vel extrapolation: 50–70% over 10 runs of 100 shots, seeds recorded. The range reflects target speed — accuracy drops when the target changes direction mid-flight, since constant-velocity extrapolation has no orbit model.

PPO: pipeline verified end-to-end (train → ONNX export → C++ inference loads correctly, output shape [1,3] validated). No accuracy numbers — still mid-training. Claiming only that the infrastructure works.

Manual aiming demonstrated in video as a lower bound: sub-30%.

All three claims are modest and measured. No generalizations beyond the tested range.

## Constraints

**Latency.** 15 ms detection lag + 30 ms shoot delay + inference. Vel extrapolation: negligible. PPO ONNX: <0.1 ms on M1. Both fit in a 1 ms control cycle. STM32 path acknowledged but not yet benchmarked on hardware.

**Heat.** The 260 soft-lock means an agent spraying on cooldown overheats at ~26 shots. The PPO reward structure (−0.05/shot, +3/hit) creates a 60:1 hit-to-miss incentive ratio, but whether this is sufficient to learn shot discipline is an open question.

**Range.** Validated 3–5 m. Beyond 5 m, time-of-flight exceeds 250 ms and constant-velocity prediction breaks down. Not tested.

**Model size.** [128,128] MLP: ~0.5 MB ONNX, fits STM32 flash budget.

## Honesty & Trajectory

**What fails.** Velocity extrapolation loses the target on direction changes — a 2.5 m/s perpendicular target displaces 0.6 m during time-of-flight, well outside the 140 mm plate. The solver has no model of target intent. This is the gap PPO should close.

**PPO is not done.** No trained policy exists. The pipeline is built but untested. I claim no PPO results.

**What I'm not modeling yet.** No vision pipeline noise (ground-truth position through a delay buffer — real YOLO + PnP would add false negatives). Static gimbal base (no chassis motion). Single target only.

**This matters because.** If PPO learns to hold fire during unpredictable maneuvers and shoot when the target trajectory is committed — timing that no analytical solver anticipates — it would justify the RL approach over hand-tuned solutions. That outcome is not guaranteed. The experiment is designed to find out.

**Next steps, ordered:**

1. Complete PPO training. Produce training curve + 50 seeded eval episodes vs vel-extrap baseline.
2. Benchmark the intercept + EKF + MPC approach (already coded, not yet measured) as a third comparison.
3. Add adversarial target motion — a human-controlled dodging mode.
4. Inject vision noise (Gaussian position error, 5–10% false negatives) and measure degradation.

---

## Appendix A: Reproducibility

```bash
git clone <repo> && cd tool_aimbot_sim
cmake -S . -B build && cmake --build build
./build/headless_check && ./build/sim

# Training
python -m venv .venv && source .venv/bin/activate
pip install -e .
python training/train.py --timesteps 500000
```

Deps: stable-baselines3≥2.0, onnx≥1.14, onnxruntime≥1.14, gymnasium≥1.0. Seeds fixed at 42. macOS + MuJoCo framework at `/Library/Frameworks/mujoco.framework`.
