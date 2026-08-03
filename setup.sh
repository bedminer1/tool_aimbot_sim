#!/usr/bin/env bash
# setup.sh — one-command environment setup for tool_aimbot_sim
#
# Usage:  bash setup.sh
#
# Installs: MuJoCo framework, glfw3, ONNX Runtime (macOS Homebrew),
#           Python training deps (uv + pip), CMake build directory.
#
# After running:  cmake --build build --target mujocoaim -j$(sysctl -n hw.ncpu)
#                 mujocoaim -d easy

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
say()  { printf "${GREEN}==> %s${NC}\n" "$*"; }
warn() { printf "${RED}==> %s${NC}\n" "$*"; }

# ── 1. System deps (macOS Homebrew) ────────────────────────────────────────

say "Checking Homebrew..."
command -v brew >/dev/null 2>&1 || { warn "Install Homebrew: https://brew.sh"; exit 1; }

brew list glfw >/dev/null 2>&1    || brew install glfw
brew list onnxruntime >/dev/null 2>&1 || brew install onnxruntime

# ── 2. MuJoCo framework (manual DMG install) ───────────────────────────────

MUJOCO_VER="${MUJOCO_VERSION:-3.3.0}"
MUJOCO_FW="/Library/Frameworks/mujoco.framework"

if [ ! -d "$MUJOCO_FW" ]; then
    say "Installing MuJoCo ${MUJOCO_VER}..."
    DMG="mujoco-${MUJOCO_VER}-macos-universal2.dmg"
    URL="https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VER}/${DMG}"
    curl -L -o "/tmp/${DMG}" "$URL"
    hdiutil attach "/tmp/${DMG}" -nobrowse -quiet
    sudo cp -R "/Volumes/MuJoCo/mujoco.framework" /Library/Frameworks/
    hdiutil detach "/Volumes/MuJoCo" -quiet
    rm "/tmp/${DMG}"

    # Fix header layout for #include <mujoco/mujoco.h>
    cd "$MUJOCO_FW/Headers/mujoco"
    for f in ../mj*.h ../mujoco.h ../mjsan.h ../mjspec*.h ../mjxmacro.h; do
        [ -f "$f" ] && sudo ln -sf "$f" .
    done

    # Clear Gatekeeper quarantine
    sudo xattr -r -d com.apple.quarantine "$MUJOCO_FW"
    say "MuJoCo installed"
else
    say "MuJoCo already installed at $MUJOCO_FW"
fi

# ── 3. Python training deps (uv) ───────────────────────────────────────────

say "Setting up Python environment..."
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.12 2>/dev/null || true
uv pip install mujoco robot_descriptions pynput numpy gymnasium \
    stable-baselines3 torch onnx onnxruntime 2>/dev/null || true

# ── 4. CMake build directory ───────────────────────────────────────────────

say "Configuring CMake..."
mkdir -p build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
ln -sf build/compile_commands.json ../compile_commands.json
cd ..

# ── 5. Symlink binary ──────────────────────────────────────────────────────

say "Building..."
cmake --build build --target mujocoaim -j"$(sysctl -n hw.ncpu)"

BIN="build/mujocoaim"
if [ -f "$BIN" ] && [ ! -L "/usr/local/bin/mujocoaim" ]; then
    sudo ln -sf "$(pwd)/$BIN" /usr/local/bin/mujocoaim
    say "Linked: /usr/local/bin/mujocoaim -> $(pwd)/$BIN"
fi

say "Setup complete. Run:  mujocoaim -d easy"
