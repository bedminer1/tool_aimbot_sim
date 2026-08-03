// Auto-generated from training/ppo_env.py — DO NOT EDIT BY HAND.
// Regenerate: python training/generate_constants.py --write

#pragma once

// ── Time ───────────────────────────────────────────────────────────
constexpr double kRenderDt   = 0.0166666667;
constexpr double kTimeLimit  = 40.0;

// ── Bullet physics ─────────────────────────────────────────────────
constexpr double kBulletSpeed  = 24.8;
constexpr double kGravity      = 9.81;
constexpr double kBulletRadius = 0.0085;

// ── Gimbal ─────────────────────────────────────────────────────────
constexpr double kMaxYawVel    = 4.0;
constexpr double kMaxPitchVel  = 4.0;
constexpr double kMaxYawAccel  = 20.0;
constexpr double kMaxPitchAccel = 20.0;
constexpr double kYawLimit     = 2.8;
constexpr double kPitchMin     = -0.8;
constexpr double kPitchMax     = 0.8;
constexpr double kGimbalHeight = 0.34;
constexpr double kMuzzleLength = 0.76;

// ── Barrel heat ────────────────────────────────────────────────────
constexpr double kHeatLimit   = 260.0;
constexpr double kHeatPerShot = 10.0;
constexpr double kHeatCooling = 30.0;

// ── Shooting ───────────────────────────────────────────────────────
constexpr int    kMaxShots      = 50;
constexpr int    kMaxBullets    = kMaxShots;
constexpr double kAutoCooldown  = 0.12;
constexpr double kDetectionLagS = 0.015;
constexpr double kShootDelayS   = 0.03;

// ── Target geometry ────────────────────────────────────────────────
constexpr double kTargetHalfX = 0.0085;
constexpr double kTargetHalfY = 0.0675;
constexpr double kTargetHalfZ = 0.0625;
constexpr int    kNumArmorPlates = 4;

// ── Spin (orbit wobble on medium/hard) ─────────────────────────────
constexpr double kOrbitRadius = 0.04;
constexpr double kSpinRads    = 10.471975512;
