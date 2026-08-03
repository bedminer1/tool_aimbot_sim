#pragma once

#include "target_models/target_interface.hpp"

// ── Easy: left-right translate, no spin ───────────────────────────────────

class TargetEasy : public ITarget {
public:
    TargetState update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 position() const override { return pos_; }
    void reset() override;

private:
    Vec3 pos_{4.0, 0.0, 0.43};
    Vec3 vel_{};
    Vec3 prev_pos_{4.0, 0.0, 0.43};
    double prev_time_ = 0.0;
    int dir_ = 1;
    static constexpr double kSpeed = 0.5;
    static constexpr double kYMin = -1.5, kYMax = 1.5;
    static constexpr double kX = 4.0;
};
