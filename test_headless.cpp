#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "io/command.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kTargetX = 3.0;
constexpr double kTargetZ = 0.43;
constexpr double kTargetLateralAmplitudeY = 0.85;
constexpr double kTargetLateralPeriodS = 5.0;
constexpr double kTargetOrbitRadius = 0.04;
constexpr double kTargetOrbitPeriodS = 1.2;
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

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

double dot(const mjtNum* axis, const Vec3& v)
{
    return axis[0] * v.x + axis[1] * v.y + axis[2] * v.z;
}

bool ray_hits_box(const mjModel* m, const mjData* d, const Vec3& origin, const Vec3& dir,
                  int target_body, int target_geom)
{
    const mjtNum* body_pos = d->xpos + 3 * target_body;
    const mjtNum* body_xmat = d->xmat + 9 * target_body;
    const mjtNum* half = m->geom_size + 3 * target_geom;
    const Vec3 delta{origin.x - body_pos[0], origin.y - body_pos[1], origin.z - body_pos[2]};
    const double origin_vals[3] = {dot(body_xmat + 0, delta), dot(body_xmat + 3, delta),
                                   dot(body_xmat + 6, delta)};
    const double dir_vals[3] = {dot(body_xmat + 0, dir), dot(body_xmat + 3, dir),
                                dot(body_xmat + 6, dir)};
    const double half_vals[3] = {half[0], half[1], half[2]};

    double t_min = 0.0;
    double t_max = 1e9;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(dir_vals[i]) < 1e-9) {
            if (origin_vals[i] < -half_vals[i] || origin_vals[i] > half_vals[i]) return false;
            continue;
        }
        double t1 = (-half_vals[i] - origin_vals[i]) / dir_vals[i];
        double t2 = (half_vals[i] - origin_vals[i]) / dir_vals[i];
        if (t1 > t2) std::swap(t1, t2);
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        if (t_min > t_max) return false;
    }
    return t_max >= 0.0;
}

void set_orbiting_target_pose(mjData* d, int target_mocap)
{
    const double lateral_phase = 2.0 * kPi * d->time / kTargetLateralPeriodS;
    const double orbit_phase = 2.0 * kPi * d->time / kTargetOrbitPeriodS;
    const double center_y = kTargetLateralAmplitudeY * std::sin(lateral_phase);

    // Clockwise when viewed from top-down (+Z looking down): start at +Y and move toward +X.
    const double orbit_x = kTargetOrbitRadius * std::sin(orbit_phase);
    const double orbit_y = kTargetOrbitRadius * std::cos(orbit_phase);

    d->mocap_pos[3 * target_mocap + 0] = kTargetX + orbit_x;
    d->mocap_pos[3 * target_mocap + 1] = center_y + orbit_y;
    d->mocap_pos[3 * target_mocap + 2] = kTargetZ;

    const mjtNum outward_x = orbit_x / kTargetOrbitRadius;
    const mjtNum outward_y = orbit_y / kTargetOrbitRadius;
    const mjtNum rot[9] = {
      outward_x, outward_y, 0.0,
      -outward_y, outward_x, 0.0,
      0.0, 0.0, 1.0,
    };
    mju_mat2Quat(d->mocap_quat + 4 * target_mocap, rot);
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
    const int gimbal_base_body = mj_name2id(m, mjOBJ_BODY, "gimbal_base");
    const int target_body = mj_name2id(m, mjOBJ_BODY, "target");
    const int target_geom = mj_name2id(m, mjOBJ_GEOM, "target_square");
    const int target_site = mj_name2id(m, mjOBJ_SITE, "target_site");
    const int muzzle_site = mj_name2id(m, mjOBJ_SITE, "muzzle_site");
    const int aim_ray_site = mj_name2id(m, mjOBJ_SITE, "aim_ray");
    const int gimbal_camera = mj_name2id(m, mjOBJ_CAMERA, "gimbal_pov");

    if (yaw_joint < 0 || pitch_joint < 0 || yaw_motor < 0 || pitch_motor < 0 ||
        gimbal_base_body < 0 || target_body < 0 || target_geom < 0 || target_site < 0 ||
        muzzle_site < 0 || aim_ray_site < 0 ||
        gimbal_camera < 0) {
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

    set_orbiting_target_pose(d, target_mocap);
    mj_forward(m, d);

    const double gimbal_bottom_z = d->xpos[3 * gimbal_base_body + 2] - 0.13;
    const double target_top_z = d->xpos[3 * target_body + 2] + m->geom_size[3 * target_geom + 2];
    const bool target_is_twice_previous_size =
      std::abs(m->geom_size[3 * target_geom + 1] - 0.11) < 1e-9 &&
      std::abs(m->geom_size[3 * target_geom + 2] - 0.11) < 1e-9;
    const bool gimbal_bottom_just_above_target =
      gimbal_bottom_z > target_top_z && (gimbal_bottom_z - target_top_z) < 0.08;
    const double initial_target_y = d->xpos[3 * target_body + 1];
    const double initial_target_z = d->xpos[3 * target_body + 2];
    const double initial_target_x = d->xpos[3 * target_body + 0];
    const double initial_target_xaxis_dot_outward = d->xmat[9 * target_body + 1];

    const mjtNum* initial_camera = d->cam_xpos + 3 * gimbal_camera;
    const mjtNum* initial_target = d->site_xpos + 3 * target_site;
    const mjtNum* initial_aim_ray = d->site_xpos + 3 * aim_ray_site;
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
    const double initial_camera_z = initial_camera[2];
    const double initial_aim_ray_z = initial_aim_ray[2];
    const double initial_view_z = view_dir[2];
    const bool camera_above_aim_ray = initial_camera[2] > initial_aim_ray[2] + 0.05;
    const bool camera_angled_down = view_dir[2] < -0.10;

    for (int i = 0; i < 150; ++i) {
        set_orbiting_target_pose(d, target_mocap);

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
    const double target_travel_yz = std::hypot(
      d->xpos[3 * target_body + 1] - initial_target_y,
      d->xpos[3 * target_body + 2] - initial_target_z);
    const double target_topdown_x_delta = d->xpos[3 * target_body + 0] - initial_target_x;

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

    const mjtNum* target_pos = d->xpos + 3 * target_body;
    const mjtNum* target_xaxis = d->xmat + 9 * target_body;
    const Vec3 hit_origin{
      target_pos[0] - target_xaxis[0] * 0.5,
      target_pos[1] - target_xaxis[1] * 0.5,
      target_pos[2] - target_xaxis[2] * 0.5,
    };
    const Vec3 hit_dir{target_xaxis[0], target_xaxis[1], target_xaxis[2]};
    const Vec3 miss_origin{hit_origin.x, hit_origin.y, hit_origin.z + 0.5};
    const bool shot_hit = ray_hits_box(m, d, hit_origin, hit_dir, target_body, target_geom);
    const bool shot_miss = ray_hits_box(m, d, miss_origin, hit_dir, target_body, target_geom);

    std::printf(
      "yaw=%.3f yaw_vel=%.3f pitch=%.3f pitch_vel=%.3f yaw_coast=%.3f pitch_coast=%.3f "
      "target_yaw=%.3f target_pitch=%.3f yaw_error=%.3f pitch_error=%.3f camera_alignment=%.3f "
      "camera_z=%.3f aim_z=%.3f view_z=%.3f target_travel=%.3f target_xdelta=%.3f target_xdot=%.3f "
      "shot_hit=%d shot_miss=%d\n",
      yaw_after_press, yaw_vel_after_press, pitch_after_press, pitch_vel_after_press, yaw_coast,
      pitch_coast, target_yaw, target_pitch, yaw_error, pitch_error, camera_alignment,
      initial_camera_z, initial_aim_ray_z, initial_view_z, target_travel_yz, target_topdown_x_delta,
      initial_target_xaxis_dot_outward, shot_hit ? 1 : 0, shot_miss ? 1 : 0);

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
    if (!camera_above_aim_ray || !camera_angled_down) {
        std::printf("FAIL gimbal POV camera is not mounted higher and angled down\n");
        return 1;
    }
    if (!target_is_twice_previous_size) {
        std::printf("FAIL target square is not twice the previous size\n");
        return 1;
    }
    if (!gimbal_bottom_just_above_target) {
        std::printf("FAIL gimbal bottom is not just above target top\n");
        return 1;
    }
    if (target_travel_yz < 0.05 || target_topdown_x_delta < 0.02 ||
        initial_target_xaxis_dot_outward < 0.99) {
        std::printf("FAIL target does not orbit/faces the wrong way\n");
        return 1;
    }
    if (!shot_hit || shot_miss) {
        std::printf("FAIL shot hit/miss classifier is wrong\n");
        return 1;
    }

    std::printf("PASS headless yaw/pitch gimbal input/output check\n");
    return 0;
}
