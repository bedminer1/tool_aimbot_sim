#pragma once

#include "target_models/target_interface.hpp"

// ── Medium target: translates left-to-right + orbit wobble ─────────────────
// Same linear path as Easy, plus 0.04 m radius orbit at 0.6 s period.

class TargetMedium : public ITarget {
public:
    Vec3 update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 composite_pos() const override { return composite_; }
    void reset() override;

private:
    Vec3 center_{4.0, 0.0, 0.43};
    Vec3 composite_{4.0, 0.0, 0.43};
    Vec3 vel_{0.0, 0.0, 0.0};
    Vec3 prev_composite_{4.0, 0.0, 0.43};
    double prev_time_ = 0.0;
    int direction_ = 1;
    static constexpr double kSpeed = 0.5;
    static constexpr double kYMin = -1.5, kYMax = 1.5;
    static constexpr double kX = 4.0;
    static constexpr double kOrbitR = 0.04;
    static constexpr double kOrbitT = 0.6;
};
