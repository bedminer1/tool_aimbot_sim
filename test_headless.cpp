#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "io/command.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kManualYawRate = 3.2;

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
    const int yaw_motor = mj_name2id(m, mjOBJ_ACTUATOR, "yaw_motor");
    const int target_body = mj_name2id(m, mjOBJ_BODY, "target");
    const int target_site = mj_name2id(m, mjOBJ_SITE, "target_site");
    const int muzzle_site = mj_name2id(m, mjOBJ_SITE, "muzzle_site");

    if (yaw_joint < 0 || yaw_motor < 0 || target_body < 0 || target_site < 0 || muzzle_site < 0) {
        std::printf("FAIL required MJCF name missing\n");
        return 1;
    }

    const int target_mocap = m->body_mocapid[target_body];
    if (target_mocap < 0) {
        std::printf("FAIL target is not mocap\n");
        return 1;
    }

    const int yaw_qpos = m->jnt_qposadr[yaw_joint];
    const int yaw_qvel = m->jnt_dofadr[yaw_joint];
    hw::Command cmd{};
    cmd.control = true;
    cmd.found = true;

    for (int i = 0; i < 150; ++i) {
        d->mocap_pos[3 * target_mocap + 0] = 3.0;
        d->mocap_pos[3 * target_mocap + 1] = std::sin(d->time);
        d->mocap_pos[3 * target_mocap + 2] = 0.43;
        d->mocap_quat[4 * target_mocap + 0] = 1.0;

        cmd.yaw_vel = kManualYawRate;
        cmd.yaw = wrap_pi(cmd.yaw + cmd.yaw_vel * m->opt.timestep);
        d->ctrl[yaw_motor] = cmd.yaw_vel;
        d->qpos[yaw_qpos] = std::clamp(
          wrap_pi(d->qpos[yaw_qpos] + d->ctrl[yaw_motor] * m->opt.timestep), -2.8, 2.8);
        d->qvel[yaw_qvel] = d->ctrl[yaw_motor];
        d->time += m->opt.timestep;
        mj_forward(m, d);
    }

    const double yaw_after_press = d->qpos[yaw_qpos];
    const double vel_after_press = d->qvel[yaw_qvel];

    for (int i = 0; i < 150; ++i) {
        cmd.yaw_vel = 0.0;
        cmd.yaw = d->qpos[yaw_qpos];
        d->ctrl[yaw_motor] = cmd.yaw_vel;
        d->qvel[yaw_qvel] = 0.0;
        d->time += m->opt.timestep;
        mj_forward(m, d);
    }

    const double yaw_after_release = d->qpos[yaw_qpos];
    const double vel_after_release = d->qvel[yaw_qvel];
    const double coast = std::abs(yaw_after_release - yaw_after_press);

    const mjtNum* muzzle = d->site_xpos + 3 * muzzle_site;
    const mjtNum* target = d->site_xpos + 3 * target_site;
    const double target_yaw = std::atan2(target[1] - muzzle[1], target[0] - muzzle[0]);
    const double yaw_error = wrap_pi(target_yaw - yaw_after_release);

    std::printf(
      "yaw_after_press=%.3f vel_after_press=%.3f yaw_after_release=%.3f "
      "vel_after_release=%.3f coast=%.3f target_yaw=%.3f yaw_error=%.3f\n",
      yaw_after_press, vel_after_press, yaw_after_release, vel_after_release, coast, target_yaw,
      yaw_error);

    mj_deleteData(d);
    mj_deleteModel(m);

    if (yaw_after_press < 0.90) {
        std::printf("FAIL yaw not responsive enough\n");
        return 1;
    }
    if (vel_after_press < 3.1) {
        std::printf("FAIL yaw velocity not responsive enough\n");
        return 1;
    }
    if (std::abs(vel_after_release) > 0.001) {
        std::printf("FAIL yaw did not brake on release\n");
        return 1;
    }
    if (coast > 0.001) {
        std::printf("FAIL excessive coast after release\n");
        return 1;
    }
    if (!std::isfinite(yaw_error)) {
        std::printf("FAIL yaw error invalid\n");
        return 1;
    }

    std::printf("PASS headless gimbal input/output check\n");
    return 0;
}
