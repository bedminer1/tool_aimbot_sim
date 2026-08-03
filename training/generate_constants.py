#!/usr/bin/env python3
"""Generate C++ constant headers from ppo_env.py (single source of truth).

Run this whenever you change constants in training/ppo_env.py.
Regenerates:
  include/common/sim_constants.hpp
  include/target_models/target_params.hpp
  include/aim_models/aim_constants.hpp

Usage:
  python training/generate_constants.py          # dry-run (print to stdout)
  python training/generate_constants.py --write  # write the files
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.ppo_env import (
    DT, BULLET_SPEED, GRAVITY, BULLET_RADIUS,
    MAX_YAW_VEL, MAX_PITCH_VEL, YAW_LIMIT, PITCH_MIN, PITCH_MAX,
    GIMBAL_HEIGHT, MUZZLE_LENGTH,
    HEAT_LIMIT, HEAT_PER_SHOT, HEAT_COOLING,
    MAX_SHOTS, TIME_LIMIT,
    DETECTION_LAG, SHOOT_DELAY, AUTO_COOLDOWN,
    ORBIT_RADIUS, ORBIT_OMEGA,
    TARGET_HALF_X, TARGET_HALF_Y, TARGET_HALF_Z,
    OBS_STACK,
    LINEAR_X, LINEAR_Y_MIN, LINEAR_Y_MAX, LINEAR_SPEED,
    HARD_RANGE_LO, HARD_RANGE_HI, HARD_ANGLE_MAX,
    HARD_SPEED_LO, HARD_SPEED_HI,
)


def cpp(v):
    """Format a Python value as a C++ literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s if "." in s else f"{s}.0"
    return str(v)


def generate():
    """Return dict of {rel_path: content}."""

    sim = f"""\
// Auto-generated from training/ppo_env.py — DO NOT EDIT BY HAND.
// Regenerate: python training/generate_constants.py --write

#pragma once

// ── Time ───────────────────────────────────────────────────────────
constexpr double kRenderDt   = {cpp(DT)};
constexpr double kTimeLimit  = {cpp(TIME_LIMIT)};

// ── Bullet physics ─────────────────────────────────────────────────
constexpr double kBulletSpeed  = {cpp(BULLET_SPEED)};
constexpr double kGravity      = {cpp(GRAVITY)};
constexpr double kBulletRadius = {cpp(BULLET_RADIUS)};

// ── Gimbal ─────────────────────────────────────────────────────────
constexpr double kMaxYawVel    = {cpp(MAX_YAW_VEL)};
constexpr double kMaxPitchVel  = {cpp(MAX_PITCH_VEL)};
constexpr double kYawLimit     = {cpp(YAW_LIMIT)};
constexpr double kPitchMin     = {cpp(PITCH_MIN)};
constexpr double kPitchMax     = {cpp(PITCH_MAX)};
constexpr double kGimbalHeight = {cpp(GIMBAL_HEIGHT)};
constexpr double kMuzzleLength = {cpp(MUZZLE_LENGTH)};

// ── Barrel heat ────────────────────────────────────────────────────
constexpr double kHeatLimit   = {cpp(HEAT_LIMIT)};
constexpr double kHeatPerShot = {cpp(HEAT_PER_SHOT)};
constexpr double kHeatCooling = {cpp(HEAT_COOLING)};

// ── Shooting ───────────────────────────────────────────────────────
constexpr int    kMaxShots      = {cpp(MAX_SHOTS)};
constexpr int    kMaxBullets    = kMaxShots;
constexpr double kAutoCooldown  = {cpp(AUTO_COOLDOWN)};
constexpr double kDetectionLagS = {cpp(DETECTION_LAG)};
constexpr double kShootDelayS   = {cpp(SHOOT_DELAY)};

// ── Target geometry ────────────────────────────────────────────────
constexpr double kTargetHalfX = {cpp(TARGET_HALF_X)};
constexpr double kTargetHalfY = {cpp(TARGET_HALF_Y)};
constexpr double kTargetHalfZ = {cpp(TARGET_HALF_Z)};
constexpr int    kNumArmorPlates = 4;

// ── Spin (orbit wobble on medium/hard) ─────────────────────────────
constexpr double kOrbitRadius = {cpp(ORBIT_RADIUS)};
constexpr double kSpinRads    = {cpp(ORBIT_OMEGA)};
"""

    params = f"""\
// Auto-generated from training/ppo_env.py — DO NOT EDIT BY HAND.
// Regenerate: python training/generate_constants.py --write

#pragma once

// ── Easy / Medium: linear Y back-and-forth ─────────────────────────
constexpr double kEasyX     = {cpp(LINEAR_X)};
constexpr double kEasyYMin  = {cpp(LINEAR_Y_MIN)};
constexpr double kEasyYMax  = {cpp(LINEAR_Y_MAX)};
constexpr double kEasySpeed = {cpp(LINEAR_SPEED)};

// ── Hard: random waypoints ─────────────────────────────────────────
constexpr double kHardRangeLo  = {cpp(HARD_RANGE_LO)};
constexpr double kHardRangeHi  = {cpp(HARD_RANGE_HI)};
constexpr double kHardAngleMax = {cpp(HARD_ANGLE_MAX)};
constexpr double kHardSpeedLo  = {cpp(HARD_SPEED_LO)};
constexpr double kHardSpeedHi  = {cpp(HARD_SPEED_HI)};
constexpr double kHardPauseLo  = 0.1;
constexpr double kHardPauseHi  = 0.8;
"""

    aim = f"""\
// Auto-generated from training/ppo_env.py — DO NOT EDIT BY HAND.
// Regenerate: python training/generate_constants.py --write

#pragma once

// ── PPO observation ────────────────────────────────────────────────
constexpr int kObsStack = {cpp(OBS_STACK)};
constexpr int kObsDim   = kObsStack * 3 + 4 + 1 + 1;
constexpr int kActDim   = 3;
"""

    return {
        "include/common/sim_constants.hpp": sim,
        "include/target_models/target_params.hpp": params,
        "include/aim_models/aim_constants.hpp": aim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write files to disk")
    args = parser.parse_args()

    files = generate()
    for rel_path, content in files.items():
        abs_path = os.path.join(ROOT, rel_path)
        if args.write:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w") as f:
                f.write(content)
            print(f"Wrote {rel_path}")
        else:
            print(f"=== {rel_path} ===")
            print(content)


if __name__ == "__main__":
    main()
