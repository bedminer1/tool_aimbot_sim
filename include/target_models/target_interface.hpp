/**
 * @file    target_models/target_interface.hpp
 * @brief   Polymorphic target motion interface
 *
 * @details
 * Strategy pattern for swappable target difficulties. The aimbot observes
 * only position() and velocity() — it doesn't know about chassis spin,
 * waypoint machines, or difficulty-specific motion models.
 *
 * TargetState conveys the chassis center world position + yaw (spin angle)
 * to the MuJoCo mocap body each frame.
 *
 * Concrete implementations:
 *   - TargetEasy   — left-right translate, no spin
 *   - TargetMedium — left-right translate + clockwise spin (10.472 rad/s)
 *   - TargetHard   — random waypoints with smoothstep + spin
 *
 * @author  bedminer1
 * @date    2026-08-03
 */

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
