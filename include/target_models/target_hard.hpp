#pragma once

#include <cmath>
#include "target_models/target_interface.hpp"

// ── Hard: random waypoints + spin ──────────────────────────────────────────

class TargetHard : public ITarget {
public:
    TargetState update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 position() const override { return pos_; }
    void reset() override;

private:
    enum State { IDLE, MOVING };
    State st_ = IDLE;
    double st_start_ = 0.0, st_dur_ = 0.5;
    Vec3 pos_{4.0, 0.0, 0.43};
    Vec3 start_p_{4.0, 0.0, 0.43}, targ_p_{4.0, 0.0, 0.43};
    Vec3 vel_{}, prev_p_{4.0, 0.0, 0.43};
    double prev_t_ = 0.0;

    static constexpr double kSpinRads = 10.472;
    static constexpr double kSpeedLo = 1.0, kSpeedHi = 2.5;
    static constexpr double kPauseLo = 0.1, kPauseHi = 0.8;
    static constexpr double kRangeLo = 3.0, kRangeHi = 5.0;
    static constexpr double kAngleMax = 0.55;
    static constexpr double kZLo = 0.35, kZHi = 0.55;

    Vec3 random_waypoint();
};
