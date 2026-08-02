#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "io/command.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kManualYawRate = 3.2;
constexpr double kManualPitchRate = 2.4;
constexpr double kPitchMin = -0.8;
constexpr double kPitchMax = 0.8;

double wrap_pi(double x)
{
    while (x > kPi) x -= 2.0 * kPi;
    while (x < -kPi) x += 2.0 * kPi;
    return x;
}
}  // namespace

int main()
{
    char err[1024] = {0};
    mjModel* m = mj_loadXML("gimbal.xml", nullptr, err, sizeof(err));
    if (!m) {
        std::printf("FAIL load: %s\n", err);
        return 1;
    }

    mjData* d = mj_makeData(m);
    const int yaw_joint = mj_name2id(m, mjOBJ_JOINT, "yaw");
    const int pitch_joint = mj_name2id(m, mjOBJ_JOINT, "pitch");
    const int yaw_motor = mj_name2id(m, mjOBJ_ACTUATOR, "yaw_motor");
    const int pitch_motor = mj_name2id(m, mjOBJ_ACTUATOR, "pitch_motor");
    const int target_body = mj_name2id(m, mjOBJ_BODY, "target");
    const int target_site = mj_name2id(m, mjOBJ_SITE, "target_site");
    const int muzzle_site = mj_name2id(m, mjOBJ_SITE, "muzzle_site");
    const int gimbal_camera = mj_name2id(m, mjOBJ_CAMERA, "gimbal_pov");

    if (yaw_joint < 0 || pitch_joint < 0 || yaw_motor < 0 || pitch_motor < 0 ||
        target_body < 0 || target_site < 0 || muzzle_site < 0 || gimbal_camera < 0) {
        std::printf("FAIL required MJCF name missing\n");
        return 1;
    }

    const int target_mocap = m->body_mocapid[target_body];
    if (target_mocap < 0) {
        std::printf("FAIL target is not mocap\n");
        return 1;
    }

    const int yaw_qpos = m->jnt_qposadr[yaw_joint];
    const int pitch_qpos = m->jnt_qposadr[pitch_joint];
    const int yaw_qvel = m->jnt_dofadr[yaw_joint];
    const int pitch_qvel = m->jnt_dofadr[pitch_joint];
    hw::Command cmd{};
    cmd.control = true;
    cmd.found = true;

    d->mocap_pos[3 * target_mocap + 0] = 3.0;
    d->mocap_pos[3 * target_mocap + 1] = 0.0;
    d->mocap_pos[3 * target_mocap + 2] = 0.43;
    d->mocap_quat[4 * target_mocap + 0] = 1.0;
    mj_forward(m, d);

    const mjtNum* initial_camera = d->cam_xpos + 3 * gimbal_camera;
    const mjtNum* initial_target = d->site_xpos + 3 * target_site;
    double view_dir[3] = {
      -d->cam_xmat[9 * gimbal_camera + 2],
      -d->cam_xmat[9 * gimbal_camera + 5],
      -d->cam_xmat[9 * gimbal_camera + 8],
    };
    double to_target[3] = {
      initial_target[0] - initial_camera[0],
      initial_target[1] - initial_camera[1],
      initial_target[2] - initial_camera[2],
    };
    const double to_target_norm =
      std::sqrt(to_target[0] * to_target[0] + to_target[1] * to_target[1] + to_target[2] * to_target[2]);
    const double camera_alignment =
      (view_dir[0] * to_target[0] + view_dir[1] * to_target[1] + view_dir[2] * to_target[2]) /
      to_target_norm;

    for (int i = 0; i < 150; ++i) {
        d->mocap_pos[3 * target_mocap + 0] = 3.0;
        d->mocap_pos[3 * target_mocap + 1] = std::sin(d->time);
        d->mocap_pos[3 * target_mocap + 2] = 0.43;
        d->mocap_quat[4 * target_mocap + 0] = 1.0;

        cmd.yaw_vel = kManualYawRate;
        cmd.pitch_vel = kManualPitchRate;
        cmd.yaw = wrap_pi(cmd.yaw + cmd.yaw_vel * m->opt.timestep);
        cmd.pitch = std::clamp(cmd.pitch + cmd.pitch_vel * m->opt.timestep, kPitchMin, kPitchMax);

        d->ctrl[yaw_motor] = cmd.yaw_vel;
        d->ctrl[pitch_motor] = cmd.pitch_vel;
        d->qpos[yaw_qpos] = std::clamp(
          wrap_pi(d->qpos[yaw_qpos] + d->ctrl[yaw_motor] * m->opt.timestep), -2.8, 2.8);
        d->qpos[pitch_qpos] = std::clamp(
          d->qpos[pitch_qpos] + d->ctrl[pitch_motor] * m->opt.timestep, kPitchMin, kPitchMax);
        d->qvel[yaw_qvel] = d->ctrl[yaw_motor];
        d->qvel[pitch_qvel] = d->ctrl[pitch_motor];
        d->time += m->opt.timestep;
        mj_forward(m, d);
    }

    const double yaw_after_press = d->qpos[yaw_qpos];
    const double pitch_after_press = d->qpos[pitch_qpos];
    const double yaw_vel_after_press = d->qvel[yaw_qvel];
    const double pitch_vel_after_press = d->qvel[pitch_qvel];

    for (int i = 0; i < 150; ++i) {
        cmd.yaw_vel = 0.0;
        cmd.pitch_vel = 0.0;
        cmd.yaw = d->qpos[yaw_qpos];
        cmd.pitch = d->qpos[pitch_qpos];
        d->ctrl[yaw_motor] = 0.0;
        d->ctrl[pitch_motor] = 0.0;
        d->qvel[yaw_qvel] = 0.0;
        d->qvel[pitch_qvel] = 0.0;
        d->time += m->opt.timestep;
        mj_forward(m, d);
    }

    const double yaw_after_release = d->qpos[yaw_qpos];
    const double pitch_after_release = d->qpos[pitch_qpos];
    const double yaw_vel_after_release = d->qvel[yaw_qvel];
    const double pitch_vel_after_release = d->qvel[pitch_qvel];
    const double yaw_coast = std::abs(yaw_after_release - yaw_after_press);
    const double pitch_coast = std::abs(pitch_after_release - pitch_after_press);

    const mjtNum* muzzle = d->site_xpos + 3 * muzzle_site;
    const mjtNum* target = d->site_xpos + 3 * target_site;
    const double dx = target[0] - muzzle[0];
    const double dy = target[1] - muzzle[1];
    const double dz = target[2] - muzzle[2];
    const double target_yaw = std::atan2(dy, dx);
    const double target_pitch = std::atan2(dz, std::hypot(dx, dy));
    const double yaw_error = wrap_pi(target_yaw - yaw_after_release);
    const double pitch_error = target_pitch - pitch_after_release;

    std::printf(
      "yaw=%.3f yaw_vel=%.3f pitch=%.3f pitch_vel=%.3f yaw_coast=%.3f pitch_coast=%.3f "
      "target_yaw=%.3f target_pitch=%.3f yaw_error=%.3f pitch_error=%.3f camera_alignment=%.3f\n",
      yaw_after_press, yaw_vel_after_press, pitch_after_press, pitch_vel_after_press, yaw_coast,
      pitch_coast, target_yaw, target_pitch, yaw_error, pitch_error, camera_alignment);

    mj_deleteData(d);
    mj_deleteModel(m);

    if (yaw_after_press < 0.90 || yaw_vel_after_press < 3.1) {
        std::printf("FAIL yaw not responsive enough\n");
        return 1;
    }
    if (pitch_after_press < 0.70 || pitch_vel_after_press < 2.3) {
        std::printf("FAIL pitch not responsive enough\n");
        return 1;
    }
    if (std::abs(yaw_vel_after_release) > 0.001 || std::abs(pitch_vel_after_release) > 0.001) {
        std::printf("FAIL gimbal did not brake on release\n");
        return 1;
    }
    if (yaw_coast > 0.001 || pitch_coast > 0.001) {
        std::printf("FAIL excessive coast after release\n");
        return 1;
    }
    if (!std::isfinite(yaw_error) || !std::isfinite(pitch_error)) {
        std::printf("FAIL aim error invalid\n");
        return 1;
    }
    if (camera_alignment < 0.99) {
        std::printf("FAIL gimbal POV camera is not aimed at the initial target\n");
        return 1;
    }

    std::printf("PASS headless yaw/pitch gimbal input/output check\n");
    return 0;
}
