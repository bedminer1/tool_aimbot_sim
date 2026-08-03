#include "target_models/target_easy.hpp"

Vec3 TargetEasy::update(double time) {
    double dt = time - prev_time_;
    if (dt <= 0.0) return pos_;

    double new_y = pos_.y + direction_ * kSpeed * dt;
    if (new_y > kYMax) { new_y = kYMax; direction_ = -1; }
    else if (new_y < kYMin) { new_y = kYMin; direction_ = 1; }
    pos_ = {kX, new_y, 0.43};
    vel_ = {0.0, direction_ * kSpeed, 0.0};
    if (dt > 1e-6) vel_ = (pos_ - prev_pos_) * (1.0 / dt);
    prev_pos_ = pos_;
    prev_time_ = time;
    return pos_;
}

void TargetEasy::reset() {
    pos_ = {4.0, 0.0, 0.43};
    vel_ = {0.0, 0.0, 0.0};
    prev_pos_ = pos_;
    prev_time_ = 0.0;
    direction_ = 1;
}
