#pragma once

#include "target_models/target_interface.hpp"

// ── Hard target: random waypoints + orbit wobble ───────────────────────────
// Randomly picks waypoints at 3–5 m range, pathfinds with smoothstep,
// pauses 0.1–0.8 s, orbits at 0.04 m / 0.6 s period.

class TargetHard : public ITarget {
public:
    Vec3 update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 composite_pos() const override { return composite_; }
    void reset() override;

private:
    enum State { IDLE, MOVING };
    State state_ = IDLE;
    double state_start_ = 0.0;
    double state_duration_ = 0.5;
    Vec3 center_{4.0, 0.0, 0.43};
    Vec3 start_pos_{4.0, 0.0, 0.43};
    Vec3 target_pos_{4.0, 0.0, 0.43};
    Vec3 composite_{4.0, 0.0, 0.43};
    Vec3 vel_{0.0, 0.0, 0.0};
    Vec3 prev_composite_{4.0, 0.0, 0.43};
    double prev_time_ = 0.0;

    static constexpr double kOrbitR = 0.04;
    static constexpr double kOrbitT = 0.6;
    static constexpr double kSpeedLo = 1.0, kSpeedHi = 2.5;
    static constexpr double kPauseLo = 0.1, kPauseHi = 0.8;
    static constexpr double kRangeLo = 3.0, kRangeHi = 5.0;
    static constexpr double kAngleMax = 0.55;
    static constexpr double kZLo = 0.35, kZHi = 0.55;

    Vec3 random_waypoint();
};
