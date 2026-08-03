#include "target_models/target_medium.hpp"
#include <cmath>

Vec3 TargetMedium::update(double time) {
    double dt = time - prev_time_;
    if (dt <= 0.0) return composite_;

    double new_y = center_.y + direction_ * kSpeed * dt;
    if (new_y > kYMax) { new_y = kYMax; direction_ = -1; }
    else if (new_y < kYMin) { new_y = kYMin; direction_ = 1; }
    center_ = {kX, new_y, 0.43};

    double phase = 2.0 * kPi * time / kOrbitT;
    composite_ = {center_.x + kOrbitR * std::sin(phase),
                  center_.y + kOrbitR * std::cos(phase),
                  center_.z};

    if (dt > 1e-6) vel_ = (composite_ - prev_composite_) * (1.0 / dt);
    prev_composite_ = composite_;
    prev_time_ = time;
    return composite_;
}

void TargetMedium::reset() {
    center_ = {4.0, 0.0, 0.43};
    composite_ = center_;
    vel_ = {0.0, 0.0, 0.0};
    prev_composite_ = composite_;
    prev_time_ = 0.0;
    direction_ = 1;
}
