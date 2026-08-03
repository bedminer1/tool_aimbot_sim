# justfile — shorthand commands for tool_aimbot_sim
# Install: brew install just

default:
    @just --list

# Build the mujocoaim binary
build:
    cmake --build build --target mujocoaim -j$(sysctl -n hw.ncpu)

# Run with easy target
run-easy:
    mujocoaim -d easy

# Run with medium target
run-medium:
    mujocoaim -d medium

# Run with hard target
run-hard:
    mujocoaim -d hard

# Train PPO policy (500k steps)
train:
    cd training && python train.py

# Train PPO policy (longer)
train-long:
    cd training && python train.py --timesteps 1000000

# Regenerate compile_commands.json
compile-db:
    cd build && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    ln -sf build/compile_commands.json compile_commands.json
