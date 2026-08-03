#include "target_models/target_hard.hpp"
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

TargetState TargetHard::update(double time) {
    double dt = time - prev_t_;
    if (dt <= 0.0) return {pos_, 0.0};
    double elapsed = time - st_start_;

    if (st_ == IDLE) {
        if (elapsed >= st_dur_) {
            st_ = MOVING; st_start_ = time;
            start_p_ = pos_; targ_p_ = random_waypoint();
            double dist = norm(targ_p_ - start_p_);
            double speed = urand(kSpeedLo, kSpeedHi);
            st_dur_ = dist / std::max(0.1, speed);
        }
    } else {
        double t = elapsed / st_dur_;
        if (t >= 1.0) { pos_ = targ_p_; st_ = IDLE; st_start_ = time;
            st_dur_ = urand(kPauseLo, kPauseHi); }
        else pos_ = start_p_ + (targ_p_ - start_p_) * smoothstep(t);
    }

    if (dt > 1e-6) vel_ = (pos_ - prev_p_) * (1.0 / dt);
    prev_p_ = pos_; prev_t_ = time;
    return {pos_, kSpinRads * time};  // clockwise spin
}

void TargetHard::reset() {
    st_ = IDLE; st_start_ = 0.0; st_dur_ = 0.5;
    pos_ = start_p_ = targ_p_ = {4.0, 0.0, 0.43};
    vel_ = {}; prev_p_ = pos_; prev_t_ = 0.0;
}
