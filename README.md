# tool_aimbot_sim

MuJoCo prototype for a RoboMaster-style yaw/pitch gimbal and moving target.

Current scope is intentionally small:

- yaw/pitch gimbal
- red square target moving left/right in front of the gimbal
- manual A/D yaw control and W/S pitch control
- starts in a fixed gimbal POV camera mounted slightly above the barrel and angled down so the sight dot is visible
- aimbot-style `hw::Command` output fields for bridging to `27_aimbot_software`
- on-screen telemetry for target yaw/pitch, yaw/pitch error, command velocities, and simulated gimbal status

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
- `W`: pitch up
- `S`: pitch down
- `F`: toggle gimbal POV / free camera
- `R`: reset
- `Esc`: quit
- mouse drag: rotate free camera
- shift + drag: zoom free camera

Manual control is kinematic velocity control for responsiveness:

- while A/D or W/S is held, `command.yaw_vel` / `command.pitch_vel` is nonzero
- when released, the corresponding velocity becomes zero immediately
- `command.yaw` / `command.pitch` reset to current gimbal angles on release so the sim does not chase stale setpoints

Later baseline/RL work can swap this for actuator dynamics behind a mode flag.

## Aimbot I/O bridge

`main.cpp` includes `io/command.hpp` and emits the same basic command shape used by `27_aimbot_software`:

```cpp
hw::Command command;
command.control = true;
command.found = true;
command.yaw = ...;
command.yaw_vel = ...;
command.pitch = ...;
command.pitch_vel = ...;
```

CMake prefers the real sibling checkout:

`../27_aimbot_software/io/command.hpp`

If that repo is not present, it falls back to:

`third_party/aimbot_io/io/command.hpp`

The fallback is a minimal mirror of the command struct so this repo still builds standalone.

## Files

- `gimbal.xml`: MuJoCo yaw/pitch gimbal, target body, gimbal POV camera, sensors, yaw/pitch actuators
- `main.cpp`: GLFW viewer, A/D/W/S input, moving target, aimbot-style command/status overlay
- `test_headless.cpp`: non-visual check for XML names, target motion, snappy yaw/pitch control, and brake-on-release
- `third_party/aimbot_io/io/command.hpp`: standalone fallback command struct

## Current limitations

- no projectile model
- no vision detector or tracker loop yet
- manual control uses kinematic velocity, not realistic torque dynamics
- no RL environment API yet

Next step: add an auto-aim baseline mode where target yaw drives a PID/lead controller, then measure tracking error against the moving square.
