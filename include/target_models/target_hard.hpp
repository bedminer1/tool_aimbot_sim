#pragma once

#include <cmath>
#include "target_models/target_interface.hpp"
#include "target_models/target_params.hpp"
#include "common/sim_constants.hpp"

class TargetHard : public ITarget {
public:
    TargetState update(double time) override;
    Vec3 velocity() const override { return vel_; }
    Vec3 position() const override { return pos_; }
    void reset() override;

private:
    enum State { IDLE, MOVING };
    State st_ = IDLE;
    double st_start_ = 0.0, st_dur_ = kHardPauseLo;
    Vec3 pos_{kHardRangeLo + 1.0, 0.0, 0.015};
    Vec3 start_p_{kHardRangeLo + 1.0, 0.0, 0.015};
    Vec3 targ_p_{kHardRangeLo + 1.0, 0.0, 0.015};
    Vec3 vel_{}, prev_p_{kHardRangeLo + 1.0, 0.0, 0.015};
    double prev_t_ = 0.0;

    Vec3 random_waypoint();
};
