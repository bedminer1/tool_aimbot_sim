// Auto-generated from training/ppo_env.py — DO NOT EDIT BY HAND.
// Regenerate: python training/generate_constants.py --write

#pragma once

// ── Easy / Medium: linear Y back-and-forth ─────────────────────────
constexpr double kEasyX     = 4.0;
constexpr double kEasyYMin  = -1.5;
constexpr double kEasyYMax  = 1.5;
constexpr double kEasySpeed = 0.5;

// ── Hard: random waypoints ─────────────────────────────────────────
constexpr double kHardRangeLo  = 3.0;
constexpr double kHardRangeHi  = 4.5;
constexpr double kHardAngleMax = 0.4;
constexpr double kHardSpeedLo  = 0.5;
constexpr double kHardSpeedHi  = 1.5;
constexpr double kHardPauseLo  = 0.1;
constexpr double kHardPauseHi  = 0.8;
