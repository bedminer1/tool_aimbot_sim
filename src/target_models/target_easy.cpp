#include "target_models/target_easy.hpp"

TargetState TargetEasy::update(double time) {
    double dt = time - prev_time_;
    if (dt <= 0.0) return {pos_, 0.0};

    double ny = pos_.y + dir_ * kSpeed * dt;
    if (ny > kYMax) { ny = kYMax; dir_ = -1; }
    else if (ny < kYMin) { ny = kYMin; dir_ = 1; }
    pos_ = {kX, ny, 0.43};
    if (dt > 1e-6) vel_ = (pos_ - prev_pos_) * (1.0 / dt);
    prev_pos_ = pos_;
    prev_time_ = time;
    return {pos_, 0.0};  // no spin
}

void TargetEasy::reset() {
    pos_ = {4.0, 0.0, 0.43};
    vel_ = {};
    prev_pos_ = pos_;
    prev_time_ = 0.0;
    dir_ = 1;
}
