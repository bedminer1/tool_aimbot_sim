#include "target_models/target_hard.hpp"
#include <cmath>
#include <cstdlib>

static double urand(double lo, double hi) {
    return lo + (double)std::rand() / RAND_MAX * (hi - lo);
}

Vec3 TargetHard::random_waypoint() {
    double r = urand(kRangeLo, kRangeHi);
    double a = urand(-kAngleMax, kAngleMax);
    double z = urand(kZLo, kZHi);
    return {r * std::cos(a), r * std::sin(a), z};
}

Vec3 TargetHard::update(double time) {
    double dt = time - prev_time_;
    if (dt <= 0.0) return composite_;
    double elapsed = time - state_start_;

    if (state_ == IDLE) {
        if (elapsed >= state_duration_) {
            state_ = MOVING;
            state_start_ = time;
            start_pos_ = center_;
            target_pos_ = random_waypoint();
            double dist = norm(target_pos_ - start_pos_);
            double speed = urand(kSpeedLo, kSpeedHi);
            state_duration_ = dist / std::max(0.1, speed);
        }
    } else {
        double t = elapsed / state_duration_;
        if (t >= 1.0) {
            center_ = target_pos_;
            state_ = IDLE;
            state_start_ = time;
            state_duration_ = urand(kPauseLo, kPauseHi);
        } else {
            double st = smoothstep(t);
            center_ = start_pos_ + (target_pos_ - start_pos_) * st;
        }
    }

    double phase = 2.0 * kPi * time / kOrbitT;
    composite_ = {center_.x + kOrbitR * std::sin(phase),
                  center_.y + kOrbitR * std::cos(phase),
                  center_.z};

    if (dt > 1e-6) vel_ = (composite_ - prev_composite_) * (1.0 / dt);
    prev_composite_ = composite_;
    prev_time_ = time;
    return composite_;
}

void TargetHard::reset() {
    state_ = IDLE;
    state_start_ = 0.0;
    state_duration_ = 0.5;
    center_ = start_pos_ = target_pos_ = {4.0, 0.0, 0.43};
    composite_ = {4.0, 0.0, 0.43};
    vel_ = {0.0, 0.0, 0.0};
    prev_composite_ = composite_;
    prev_time_ = 0.0;
}
