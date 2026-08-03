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
