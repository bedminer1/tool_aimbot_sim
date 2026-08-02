# tool_aimbot_sim

MuJoCo prototype for a RoboMaster-style yaw/pitch gimbal and moving target.

Current scope is intentionally small:

- yaw/pitch gimbal
- 140 x 125 mm red target plate, 5 m away when straight on
- target orientation follows the top-down clockwise orbit so one face points inward and the other outward
- gimbal base is raised so its bottom sits just above the top of the target
- mouse-controlled yaw/pitch aiming from the gimbal POV
- left-click shooting with 10 scored 17 mm bullets per run
- `T` toggles a simple aimbot that emits the same `hw::Command` fields and fires automatically
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

- move mouse: aim yaw/pitch
- left click: fire one shot
- `T`: toggle aimbot mode and reset score/timer
- `F`: toggle gimbal POV / free camera
- `R`: reset
- `Esc`: quit
- mouse drag: rotate free camera
- shift + drag: zoom free camera

Manual aiming is kinematic velocity control for responsiveness:

- mouse delta becomes `command.yaw_vel` / `command.pitch_vel`
- when the mouse stops, both velocities become zero immediately
- `command.yaw` / `command.pitch` are reset to current gimbal angles every frame so the sim does not chase stale setpoints

Later baseline/RL work can swap this for actuator dynamics behind a mode flag.

Scoring:

- each run has 10 shots
- left-click fires one shot from the green laser ray
- bullets travel at 24 km/h on a ballistic trajectory
- score increments when the bullet intersects the moving target plate
- timer starts on the first shot and stops on the tenth shot
- `R` resets the counter and score

Aimbot mode:

- uses the same simulated inputs as the real pipeline: target yaw/pitch error plus gimbal yaw/pitch status
- outputs `hw::Command{control, found, shoot, yaw, pitch, yaw_vel, pitch_vel, yaw_accel, pitch_accel}`
- aims with a low-arc ballistic solution and fires automatically with a short shot cooldown

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
- `main.cpp`: GLFW viewer, mouse aim input, moving target, aimbot-style command/status overlay
- `test_headless.cpp`: non-visual check for XML names, target motion, snappy yaw/pitch control, and brake-on-release
- `third_party/aimbot_io/io/command.hpp`: standalone fallback command struct

## Current limitations

- no vision detector or tracker loop yet
- manual control uses kinematic velocity, not realistic torque dynamics
- no RL environment API yet

Next step: add an auto-aim baseline mode where target yaw drives a PID/lead controller, then measure tracking error against the moving square.
