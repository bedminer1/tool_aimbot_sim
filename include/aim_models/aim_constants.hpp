// Auto-generated from training/ppo_env.py — DO NOT EDIT BY HAND.
// Regenerate: python training/generate_constants.py --write

#pragma once

// ── PPO observation ────────────────────────────────────────────────
constexpr int kObsStack = 8;
constexpr int kObsDim   = kObsStack * 3 + 4 + 1 + 1;
constexpr int kActDim   = 3;
