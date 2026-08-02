#pragma once

#include "aim_predictor.hpp"   // Vec3, AimPrediction
#include <array>
#include <memory>
#include <string>

// Forward-declare ONNX Runtime types (avoid header pollution).
namespace Ort { class Env; class Session; class MemoryInfo; class Allocator; }

// ── PPO predictor ──────────────────────────────────────────────────────────
// Loads a trained ONNX policy and runs inference each frame.
// Observation: 30 floats.  Action: yaw_vel, pitch_vel, fire_prob.

class PpoPredictor {
public:
    static constexpr int kObsDim = 30;
    static constexpr int kActDim = 3;
    static constexpr int kHistoryLen = 8;

    PpoPredictor();
    ~PpoPredictor();

    // Load ONNX model from disk. Returns false on failure.
    bool load(const std::string& path);

    // Build observation vector from sim state and run inference.
    // Returns: yaw_vel [-4,4], pitch_vel [-4,4], fire [0 or 1].
    struct PpoAction { double yaw_vel, pitch_vel; bool fire; };
    PpoAction predict(
        const Vec3& muzzle_pos,
        const Vec3& delayed_pos,
        double gimbal_yaw, double gimbal_pitch,
        double gimbal_yaw_vel, double gimbal_pitch_vel,
        double barrel_heat,
        double time_since_last_shot);

    bool loaded() const { return session_ != nullptr; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::string input_name_;
    std::string output_name_;

    // Position history (ring buffer for last kHistoryLen observations).
    Vec3 history_[kHistoryLen]{};
    int hist_head_ = 0;
};
