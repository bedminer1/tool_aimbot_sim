/**
 * @file    aim_models/aim_predictor.hpp / aim_predictor.cpp
 * @brief   Ballistic solver, velocity-extrapolation predictor, detection-lag buffer
 *
 * @details
 * Foundation layer shared by all aiming approaches.
 *
 * Ballistic pitch solver:
 *   Solves for the elevation angle required to hit a target at
 *   (horizontal_distance, dz) with projectile speed v_b = 24.8 m/s.
 *   Uses iterative Newton refinement on the trajectory equation:
 *     dz = v_b·sin(θ)·t - ½g·t²,  where t = d_h / (v_b·cos(θ))
 *
 * Velocity extrapolation predictor (predict_aim):
 *   Simplest aimbot approach. Projects target position forward by
 *   system_delay seconds assuming constant velocity:
 *     p_pred = p_obs + v_obs · (Δt + t_delay)
 *   Then computes ballistic pitch to hit p_pred.
 *   Fails when the target accelerates (waypoint transitions) or orbits
 *   (circular motion) — those require the Intercept+MPC approach.
 *
 * DetectLagBuffer:
 *   Ring buffer simulating the CV pipeline delay (~15 ms). Observations
 *   are timestamped on push(); the aimbot queries with a lagged timestamp
 *   on lookup() to get the position/velocity the CV system would have
 *   reported at that instant. This prevents the aimbot from "seeing
 *   the future" — a common sim-reality gap.
 *
 * @see aim_predictor_intercept.hpp, aim_predictor_ppo.hpp
 * @author  bedminer1
 * @date    2026-08-03
 */

#pragma once

#include "common/types.hpp"
#include "common/sim_constants.hpp"

#include <algorithm>
#include <cmath>
#include <deque>

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
