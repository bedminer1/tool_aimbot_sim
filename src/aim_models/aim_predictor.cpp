/**
 * @file    src/aim_models/aim_predictor.cpp
 * @brief   Ballistic solver + velocity extrapolation + detection lag buffer
 *
 * @details
 * ballistic_pitch:
 *   Closed-form solution to the projectile equation:
 *     d_h·tan(θ) - (g·d_h²) / (2·v²·cos²(θ)) = dz
 *   Rearranged to a quadratic in tan(θ) and solved with the quadratic formula.
 *   Uses the lower trajectory (shorter time-of-flight).
 *
 * predict_aim:
 *   Iterative constant-velocity lead prediction with ballistic pitch refinement.
 *   4 iterations of: compute pitch → compute TOF → predict forward → repeat.
 *   This converges because pitch changes only slightly between iterations
 *   (the ballistic correction is small at typical engagement ranges).
 *
 * @see aim_predictor.hpp
 * @author  bedminer1
 * @date    2026-08-03
 */

#include "aim_models/aim_predictor.hpp"

#include <cmath>

// ── Ballistic pitch ────────────────────────────────────────────────────────

bool ballistic_pitch(double horizontal_distance, double dz, double& pitch)
{
    const double v2 = kBulletSpeed * kBulletSpeed;
    const double discriminant =
      v2 * v2 - kGravity * (kGravity * horizontal_distance * horizontal_distance + 2.0 * dz * v2);
    if (discriminant < 0.0 || horizontal_distance < 1e-6) return false;
    pitch = std::atan((v2 - std::sqrt(discriminant)) / (kGravity * horizontal_distance));
    pitch = std::clamp(pitch, kPitchMin, kPitchMax);
    return true;
}

// ── Velocity extrapolation with acceleration ───────────────────────────────

AimPrediction predict_aim(
    const Vec3& muzzle_pos,
    const Vec3& target_pos,
    const Vec3& target_vel,
    const Vec3& target_accel,
    double system_delay,
    double current_yaw,
    double current_pitch)
{
    Vec3 aim = target_pos;
    double pitch = 0.0;

    for (int i = 0; i < 4; ++i) {
        const Vec3 delta = aim - muzzle_pos;
        const double horizontal = norm_xy(delta);
        if (!ballistic_pitch(horizontal, delta.z, pitch)) {
            pitch = std::atan2(delta.z, horizontal);
        }
        const double tof = horizontal / std::max(0.1, kBulletSpeed * std::cos(pitch));
        const double t_total = tof + system_delay;
        // Kinematic prediction: p + v·t + ½a·t²
        aim = target_pos + target_vel * t_total + target_accel * (0.5 * t_total * t_total);
    }

    const Vec3 delta = aim - muzzle_pos;
    const double yaw = std::atan2(delta.y, delta.x);
    const double horizontal = norm_xy(delta);
    if (!ballistic_pitch(horizontal, delta.z, pitch)) pitch = std::atan2(delta.z, horizontal);

    AimPrediction out{};
    out.target_yaw = yaw;
    out.target_pitch = pitch;
    out.yaw_error = wrap_pi(yaw - current_yaw);
    out.pitch_error = pitch - current_pitch;
    out.predicted_target_pos = aim;
    return out;
}

// ── Detection lag ring buffer ──────────────────────────────────────────────

DetectLagBuffer::DetectLagBuffer(size_t max_entries) : max_(max_entries) {}

void DetectLagBuffer::push(double t, const Vec3& pos, const Vec3& vel)
{
    buf_.push_back({t, pos, vel});
    while (buf_.size() > max_) buf_.pop_front();
}

bool DetectLagBuffer::lookup(double t, Vec3& pos, Vec3& vel) const
{
    if (buf_.empty()) return false;

    // Find the entry with largest timestamp <= t.
    // Linear scan is fine for ~300 entries at 60 fps.
    const Entry* best = nullptr;
    for (const auto& e : buf_) {
        if (e.t <= t) best = &e;
        else break;  // buf_ is sorted by timestamp
    }
    if (!best) {
        // No entry at or before t — return oldest.
        best = &buf_.front();
    }
    pos = best->pos;
    vel = best->vel;
    return true;
}

void DetectLagBuffer::clear() { buf_.clear(); }
