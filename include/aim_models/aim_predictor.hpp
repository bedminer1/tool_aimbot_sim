#pragma once

#include "common/types.hpp"

#include <algorithm>
#include <cmath>
#include <deque>

// ── Ballistic / predictor constants ────────────────────────────────────────

constexpr double kBulletSpeed = 24.8;  // m/s
constexpr double kGravity = 9.81;
constexpr double kPitchMin = -0.8;
constexpr double kPitchMax = 0.8;

// ── Aim prediction output ──────────────────────────────────────────────────

struct AimPrediction {
    double target_yaw = 0.0;
    double target_pitch = 0.0;
    double yaw_error = 0.0;
    double pitch_error = 0.0;
    Vec3 predicted_target_pos{};
};

// ── Ballistic pitch solver ─────────────────────────────────────────────────

bool ballistic_pitch(double horizontal_distance, double dz, double& pitch);

// ── Velocity extrapolation predictor ───────────────────────────────────────

AimPrediction predict_aim(
    const Vec3& muzzle_pos,
    const Vec3& target_pos,
    const Vec3& target_vel,
    const Vec3& target_accel,
    double system_delay,
    double current_yaw,
    double current_pitch);

// ── Detection lag ring buffer ──────────────────────────────────────────────

class DetectLagBuffer {
public:
    explicit DetectLagBuffer(size_t max_entries = 300);
    void push(double t, const Vec3& pos, const Vec3& vel);
    bool lookup(double t, Vec3& pos, Vec3& vel) const;
    void clear();
private:
    struct Entry { double t; Vec3 pos; Vec3 vel; };
    std::deque<Entry> buf_;
    size_t max_;
};
