#include "aim_predictor_ppo.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

PpoPredictor::PpoPredictor()
{
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ppo");
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    } catch (const std::exception&) {
        // ONNX Runtime not available — loaded() will return false.
    }
}

PpoPredictor::~PpoPredictor() = default;

bool PpoPredictor::load(const std::string& path)
{
    if (!env_) return false;
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);

        // Cache input/output names.
        Ort::AllocatorWithDefaultOptions alloc;
        input_name_ = session_->GetInputNameAllocated(0, alloc).get();
        output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

        // Verify shapes.
        auto input_info = session_->GetInputTypeInfo(0);
        auto input_shape = input_info.GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() != 2 || input_shape[1] != kObsDim) {
            std::fprintf(stderr, "PpoPredictor: bad input shape (expected [?, %d])\n", kObsDim);
            session_.reset();
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PpoPredictor: failed to load %s: %s\n", path.c_str(), e.what());
        session_.reset();
        return false;
    }
}

PpoPredictor::PpoAction PpoPredictor::predict(
    const Vec3& muzzle_pos,
    const Vec3& delayed_pos,
    double gimbal_yaw, double gimbal_pitch,
    double gimbal_yaw_vel, double gimbal_pitch_vel,
    double barrel_heat,
    double time_since_last_shot)
{
    PpoAction out{0.0, 0.0, false};
    if (!session_) return out;

    // ── Update position history ────────────────────────────────────────
    history_[hist_head_] = delayed_pos;
    hist_head_ = (hist_head_ + 1) % kHistoryLen;

    // ── Build observation vector (30 floats, normalized) ───────────────
    float obs[kObsDim]{};
    // Position history (relative to muzzle, scaled by 1/5).
    for (int i = 0; i < kHistoryLen; ++i) {
        int idx = (hist_head_ - 1 - i + kHistoryLen) % kHistoryLen;
        float rx = (float)((history_[idx].x - muzzle_pos.x) * (1.0 / 5.0));
        float ry = (float)((history_[idx].y - muzzle_pos.y) * (1.0 / 5.0));
        float rz = (float)((history_[idx].z - muzzle_pos.z) * (1.0 / 5.0));
        obs[i * 3 + 0] = rx;
        obs[i * 3 + 1] = ry;
        obs[i * 3 + 2] = rz;
    }
    // Gimbal state.
    obs[24] = (float)(gimbal_yaw       * (1.0 / kPi));
    obs[25] = (float)(gimbal_pitch     * (1.0 / kPi));
    obs[26] = (float)(gimbal_yaw_vel   * (1.0 / 4.0));
    obs[27] = (float)(gimbal_pitch_vel * (1.0 / 4.0));
    obs[28] = (float)(barrel_heat      * (1.0 / 260.0));
    obs[29] = (float)std::min(time_since_last_shot, 1.0);

    // ── Run inference ──────────────────────────────────────────────────
    try {
        std::vector<int64_t> shape = {1, kObsDim};
        auto input = Ort::Value::CreateTensor<float>(
            *memory_info_, obs, kObsDim, shape.data(), shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_names, &input, 1,
                                     output_names, 1);

        const float* action = outputs[0].GetTensorData<float>();
        // SB3 PPO outputs: yaw_vel, pitch_vel, fire_logit (all in [-1,1] from tanh).
        out.yaw_vel   = std::clamp((double)action[0], -1.0, 1.0) * 4.0;
        out.pitch_vel = std::clamp((double)action[1], -1.0, 1.0) * 4.0;
        out.fire      = action[2] > 0.0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PpoPredictor: inference error: %s\n", e.what());
    }

    return out;
}
