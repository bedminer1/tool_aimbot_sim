# tool_aimbot_sim

MuJoCo prototype for a RoboMaster-style yaw gimbal and moving target.

Current scope is intentionally small:

- yaw-only gimbal
- red square target moving left/right in front of the gimbal
- manual A/D yaw control
- aimbot-style `hw::Command` output fields for bridging to `27_aimbot_software`
- on-screen telemetry for target yaw, yaw error, command yaw/yaw velocity, and simulated gimbal status

This is the first sim scaffold for the Griffin Labs RL-from-scratch challenge. It is not the RL environment yet.

## Build

macOS setup assumes the MuJoCo framework is installed at `/Library/Frameworks/mujoco.framework` and GLFW is available through Homebrew.

```bash
cmake -S . -B build
cmake --build build
./build/headless_check
```

Run:

```bash
./build/sim
```

Optional explicit XML path:

```bash
./build/sim gimbal.xml
```

## Controls

- `A`: yaw left
- `D`: yaw right
- `R`: reset
- `Esc`: quit
- mouse drag: rotate camera
- shift + drag: zoom

Manual control is kinematic velocity control for responsiveness:

- while A/D is held, `command.yaw_vel` is nonzero
- when released, `command.yaw_vel` becomes zero immediately
- `command.yaw` resets to current yaw on release so the sim does not chase a stale setpoint

Later baseline/RL work can swap this for actuator dynamics behind a mode flag.

## Aimbot I/O bridge

`main.cpp` includes `io/command.hpp` and emits the same basic command shape used by `27_aimbot_software`:

```cpp
hw::Command command;
command.control = true;
command.found = true;
command.yaw = ...;
command.yaw_vel = ...;
command.pitch = 0.0;
```

CMake prefers the real sibling checkout:

`../27_aimbot_software/io/command.hpp`

If that repo is not present, it falls back to:

`third_party/aimbot_io/io/command.hpp`

The fallback is a minimal mirror of the command struct so this repo still builds standalone.

## Files

- `gimbal.xml`: MuJoCo yaw gimbal, target body, sensors, yaw actuator
- `main.cpp`: GLFW viewer, A/D input, moving target, aimbot-style command/status overlay
- `test_headless.cpp`: non-visual check for XML names, target motion, snappy A/D yaw, and brake-on-release
- `third_party/aimbot_io/io/command.hpp`: standalone fallback command struct

## Current limitations

- yaw-only, no pitch
- no projectile model
- no vision detector or tracker loop yet
- manual control uses kinematic velocity, not realistic torque dynamics
- no RL environment API yet

Next step: add an auto-aim baseline mode where target yaw drives a PID/lead controller, then measure tracking error against the moving square.
