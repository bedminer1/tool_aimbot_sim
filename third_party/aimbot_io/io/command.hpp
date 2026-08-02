#pragma once

// Minimal mirror of 27_aimbot_software/io/command.hpp.
// CMake prefers the real 27_aimbot_software include path when it is present.
namespace hw
{
struct Command
{
    bool control = false;
    bool found = false;
    bool shoot = false;
    double yaw = 0.0;
    double pitch = 0.0;
    double horizon_distance = 0.0;
    double yaw_vel = 0.0;
    double yaw_accel = 0.0;
    double pitch_vel = 0.0;
    double pitch_accel = 0.0;
};
}  // namespace hw
