#pragma once

#include "common/types.hpp"

// ── Target state (set each frame by the target model) ──────────────────────

struct TargetState {
    Vec3 pos;    // chassis center position (world frame)
    double yaw;  // chassis yaw (spin angle, rad)
};

// ── Target interface ───────────────────────────────────────────────────────

struct ITarget {
    virtual ~ITarget() = default;

    // Advance simulation and return chassis state.
    virtual TargetState update(double time) = 0;

    // Observed velocity of the chassis center (for aimbot prediction).
    virtual Vec3 velocity() const = 0;

    // Current position.
    virtual Vec3 position() const = 0;

    // Reset to initial state.
    virtual void reset() = 0;
};
