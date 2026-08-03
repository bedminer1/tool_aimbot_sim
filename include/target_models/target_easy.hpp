#pragma once

#include <cmath>
#include "target_models/target_interface.hpp"
#include "target_models/target_params.hpp"

class TargetEasy : public ITarget {
public:
    TargetState update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 position() const override { return pos_; }
    void reset() override;

private:
    Vec3 pos_{kEasyX, 0.0, 0.015};
    Vec3 vel_{};
    Vec3 prev_pos_{kEasyX, 0.0, 0.015};
    double prev_time_ = 0.0;
    int dir_ = 1;
};
