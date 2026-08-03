#pragma once

#include "target_models/target_interface.hpp"

// ── Easy target: translates left-to-right, no orbit ────────────────────────
// Center moves linearly between (3, -1.5, 0.43) and (4.5, 1.5, 0.43) at 0.5 m/s.
// No orbit wobble, no random behavior.

class TargetEasy : public ITarget {
public:
    Vec3 update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 composite_pos() const override { return pos_; }
    void orbit_normal(double& ox, double& oy) const override { ox=0; oy=1; }  // face +Y
    void reset() override;

private:
    Vec3 pos_{3.75, 0.0, 0.43};
    Vec3 vel_{0.0, 0.0, 0.0};
    Vec3 prev_pos_{3.75, 0.0, 0.43};
    double prev_time_ = 0.0;
    int direction_ = 1;
    static constexpr double kSpeed = 0.5;
    static constexpr double kYMin = -1.5, kYMax = 1.5;
    static constexpr double kX = 4.0;
};
