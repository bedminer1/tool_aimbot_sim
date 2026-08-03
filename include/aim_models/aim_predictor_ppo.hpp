/**
 * @file    aim_models/aim_predictor_ppo.hpp / aim_predictor_ppo.cpp
 * @brief   PPO-trained neural network predictor (ONNX Runtime inference)
 *
 * @details
 * Loads an ONNX policy exported from training/train.py and runs inference
 * in the C++ sim. The policy was trained via PPO (stable-baselines3) on
 * a pure-Python replica of the sim (training/ppo_env.py).
 *
 * Observation (30-D):
 *   8-frame position history (24) + gimbal yaw/pitch (2) + gimbal velocity (2)
 *   + barrel heat (1) + time since last shot (1)
 *
 * Action (3-D):
 *   [yaw_vel, pitch_vel, fire_logit]  — fire when logit > 0
 *
 * Scale matching:
 *   The policy was trained with fixed-scale normalization (pos/5, angle/π,
 *   vel/4, heat/260). The C++ predictor applies the same scaling. If the
 *   scales drift between Python training and C++ inference, the policy
 *   sees a different input distribution and produces garbage.
 *
 * Graceful degradation:
 *   If the .onnx file is missing or fails to load, loaded() returns false
 *   and the aimbot falls back to VelExtrap without crashing.
 *
 * @see training/ppo_env.py, training/train.py
 * @author  bedminer1
 * @date    2026-08-03
 */

#pragma once

#include "aim_models/aim_predictor.hpp"   // Vec3
#include <array>
#include <memory>
#include <string>

namespace Ort { class Env; class Session; class MemoryInfo; }

// 3D action: yaw_vel, pitch_vel, fire_logit.

class PpoPredictor {
public:
    static constexpr int kObsDim = 30;
    static constexpr int kActDim = 3;
    static constexpr int kHistoryLen = 8;

    PpoPredictor();
    ~PpoPredictor();

    bool load(const std::string& path);

    struct PpoAction { double yaw_vel, pitch_vel; bool fire; };
    PpoAction predict(
        const Vec3& muzzle_pos, const Vec3& delayed_pos,
        double gimbal_yaw, double gimbal_pitch,
        double gimbal_yaw_vel, double gimbal_pitch_vel,
        double barrel_heat, double time_since_last_shot);

    bool loaded() const { return session_ != nullptr; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::string input_name_, output_name_;
    Vec3 history_[kHistoryLen]{};
    int hist_head_ = 0;
};
