#pragma once

#include "aim_predictor.hpp"  // Vec3, AimPrediction, kPi, kBulletSpeed, kGravity, kPitchMin, kPitchMax

// ══════════════════════════════════════════════════════════════════════════════
// Model-based aiming pipeline: EKF → circular motion model → intercept solver
// → MPC feedforward+feedback controller.  All in one file because it's one
// approach — separating them just scatters the coupling.
// ══════════════════════════════════════════════════════════════════════════════

// ── Circular motion model ──────────────────────────────────────────────────
// Target = chassis center (translating) + armor plate (spinning).

struct CircularModel {
    Vec3 p_c0;    // chassis center at current time (world frame)
    Vec3 v_c;     // chassis velocity (world frame)
    double r;     // armor plate radius (m)
    double theta0;// phase (rad)
    double omega; // angular velocity (rad/s)
};

Vec3 model_position_at(const CircularModel& m, double t);
Vec3 model_velocity_at(const CircularModel& m, double t);

// ── 8-D EKF ────────────────────────────────────────────────────────────────
// State: [p_x, p_y, p_z, v_x, v_y, v_z, θ, ω]
// Measures armor plate position: z = [p_x + r·cos(θ), p_y + r·sin(θ), p_z]
// High process noise on v, ω → tracks maneuvers aggressively.

class CircularEkf {
public:
    static constexpr int N = 8, M = 3;
    static constexpr double kArmorR = 0.04;

    CircularEkf();
    void update(double t, const Vec3& z);
    void reset();

    Vec3 center_pos() const { return {x_[0], x_[1], x_[2]}; }
    Vec3 center_vel() const { return {x_[3], x_[4], x_[5]}; }
    double phase()   const { return x_[6]; }
    double omega()   const { return x_[7]; }
    double radius()  const { return kArmorR; }
    bool initialized() const { return init_; }

private:
    double x_[N]{}, P_[N*N]{}, t_ = 0;
    bool init_ = false;
    static constexpr double q_p_=0.01, q_v_=50.0, q_th_=1.0, q_w_=20.0;
    static constexpr double r_xyz_ = 0.001;
    void init(double t, const Vec3& z);
    void predict(double dt);
    void correct(const Vec3& z);
    static void mat_mult(int r, int inner, int c, const double* A, const double* B, double* C);
    static void mat_transpose(int r, int c, const double* A, double* AT);
    static bool mat_inv_3x3(const double* A, double* Ainv);
};

// ── Analytical intercept time solver ───────────────────────────────────────
// Solves |p(t)|² = (v_b·t)² for smallest positive t.
// Algorithm: quadratic seed → damped Newton (≤6 iters) → bisection fallback.
// Flat trajectory — ballistic (gravity) pitch correction applied separately.

struct InterceptResult { bool valid; double t; };

InterceptResult solve_intercept_time(
    const Vec3& p_c0, const Vec3& v_c, double r, double theta0, double omega,
    double v_b, double t_max = 1.5);

// ── EKF-based predictor ────────────────────────────────────────────────────
// Feeds EKF with delayed observations, runs intercept solver on EKF state.
// No warmup — produces predictions from the first observation.

class InterceptPredictor {
public:
    void observe(double t, const Vec3& pos) { ekf_.update(t, pos); }
    void clear() { ekf_.reset(); last_intercept_valid_ = false; }

    AimPrediction predict(const Vec3& muzzle_pos, double current_yaw, double current_pitch);

    const CircularModel& last_model()   const { return last_model_; }
    double last_intercept_t()           const { return last_intercept_t_; }
    bool   last_intercept_valid()       const { return last_intercept_valid_; }

private:
    CircularEkf ekf_;
    CircularModel last_model_{};
    double last_intercept_t_ = 0;
    bool last_intercept_valid_ = false;
};

// ── MPC feedforward+feedback controller ────────────────────────────────────
// Feedforward: predicted target angular velocity at intercept (from model).
// Feedback: P-controller on residual error.

struct MpcConfig {
    double kp_yaw = 12.0, kp_pitch = 12.0;
    double max_yaw_vel = 4.0, max_pitch_vel = 4.0;
    double dt = 1.0 / 60.0;
};

struct MpcCommand {
    double yaw, yaw_vel, yaw_accel, pitch, pitch_vel, pitch_accel;
};

MpcCommand make_mpc_command(
    const Vec3& muzzle_pos, const CircularModel& model, const AimPrediction& pred,
    double current_yaw, double current_pitch,
    double current_yaw_vel, double current_pitch_vel,
    double intercept_t, const MpcConfig& cfg = MpcConfig{});
