#pragma once

#include <algorithm>
#include <cmath>
#include <deque>

// ── Shared types ───────────────────────────────────────────────────────────

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(const Vec3& v, double s) { return {v.x * s, v.y * s, v.z * s}; }
inline double norm_xy(const Vec3& v) { return std::hypot(v.x, v.y); }
inline double norm(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// ── Ballistic / predictor constants ────────────────────────────────────────

constexpr double kPi = 3.14159265358979323846;
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
// Low-arc solution to: z = x·tan(θ) − (g·x²)/(2·v²·cos²(θ)).
// Returns false if discriminant < 0 (unreachable with gravity).

bool ballistic_pitch(double horizontal_distance, double dz, double& pitch);

// ── Velocity extrapolation predictor ───────────────────────────────────────
// 4-iteration refinement with acceleration and system-delay compensation.
// target_pos should already be lag-extrapolated to current time.
// system_delay = detection_lag + shoot_delay — added to time-of-flight so
// the prediction accounts for where the target will be when the bullet exits.

AimPrediction predict_aim(
    const Vec3& muzzle_pos,
    const Vec3& target_pos,
    const Vec3& target_vel,
    const Vec3& target_accel,
    double system_delay,
    double current_yaw,
    double current_pitch);

// ── Detection lag ring buffer ──────────────────────────────────────────────
// Stores (timestamp, pos, vel) tuples. Query by timestamp returns the nearest
// entry at or before the requested time.

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
