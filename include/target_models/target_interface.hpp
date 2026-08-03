#pragma once

#include "common/types.hpp"

// ── Target interface ───────────────────────────────────────────────────────

struct ITarget {
    virtual ~ITarget() = default;

    // Advance simulation and return composite (center + orbit) position.
    virtual Vec3 update(double time) = 0;

    // Current observed velocity (finite-difference of composite position).
    virtual Vec3 velocity() const = 0;

    // Current composite position.
    virtual Vec3 composite_pos() const = 0;

    // Reset to initial state.
    virtual void reset() = 0;
};
