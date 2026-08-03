/**
 * @file    src/aim_models/aim_predictor_ppo.cpp
 * @brief   ONNX Runtime inference wrapper for PPO-trained policy
 *
 * @details
 * Loads an ONNX model, normalizes inputs (fixed-scale, matching the Python
 * training env), runs forward pass, and denormalizes outputs.
 *
 * Input normalization (must match training/ppo_env.py):
 *   pos    /= 5.0
 *   angle  /= π
 *   vel    /= 4.0
 *   heat   /= 260.0
 *
 * Action denormalization:
 *   yaw_vel  = action[0] × 4.0
 *   pitch_vel = action[1] × 4.0
 *   fire     = action[2] > 0.0
 *
 * History buffer:
 *   Maintains an 8-frame ring buffer of delayed target positions.
 *   On each predict() call, the latest delayed position is pushed and
 *   the buffer is serialized into the observation vector.
 *
 * @see aim_predictor_ppo.hpp, training/train.py
 * @author  bedminer1
 * @date    2026-08-03
 */

#include "aim_models/aim_predictor_ppo.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>

PpoPredictor::PpoPredictor()
{
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ppo");
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    } catch (const std::exception&) {}
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
        Ort::AllocatorWithDefaultOptions alloc;
        input_name_ = session_->GetInputNameAllocated(0, alloc).get();
        output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
        auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 2 || shape[1] != kObsDim) {
            std::fprintf(stderr, "PpoPredictor: bad input shape\n");
            session_.reset(); return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PpoPredictor: %s\n", e.what());
        session_.reset(); return false;
    }
}

PpoPredictor::PpoAction PpoPredictor::predict(
    const Vec3& muzzle_pos, const Vec3& delayed_pos,
    double gy, double gp, double gyv, double gpv,
    double heat, double tsls)
{
    PpoAction out{0, 0, false};
    if (!session_) return out;

    history_[hist_head_] = delayed_pos;
    hist_head_ = (hist_head_ + 1) % kHistoryLen;

    float obs[kObsDim]{};
    for (int i = 0; i < kHistoryLen; ++i) {
        int idx = (hist_head_ - 1 - i + kHistoryLen) % kHistoryLen;
        obs[i*3+0] = (float)((history_[idx].x - muzzle_pos.x) * (1.0/5.0));
        obs[i*3+1] = (float)((history_[idx].y - muzzle_pos.y) * (1.0/5.0));
        obs[i*3+2] = (float)((history_[idx].z - muzzle_pos.z) * (1.0/5.0));
    }
    obs[24] = (float)(gy  * (1.0/kPi));
    obs[25] = (float)(gp  * (1.0/kPi));
    obs[26] = (float)(gyv * (1.0/4.0));
    obs[27] = (float)(gpv * (1.0/4.0));
    obs[28] = (float)(heat * (1.0/260.0));
    obs[29] = (float)std::min(tsls, 1.0);

    try {
        std::vector<int64_t> shape = {1, kObsDim};
        auto input = Ort::Value::CreateTensor<float>(*memory_info_, obs, kObsDim, shape.data(), shape.size());
        const char* in[] = {input_name_.c_str()};
        const char* out_n[] = {output_name_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out_n, 1);
        const float* act = outputs[0].GetTensorData<float>();
        out.yaw_vel   = std::clamp((double)act[0], -1.0, 1.0) * 4.0;
        out.pitch_vel = std::clamp((double)act[1], -1.0, 1.0) * 4.0;
        out.fire      = act[2] > 0.0f;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PpoPredictor: %s\n", e.what());
    }
    return out;
}
