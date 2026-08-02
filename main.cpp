#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "io/command.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRenderDt = 1.0 / 60.0;
constexpr double kTargetX = 3.0;
constexpr double kTargetZ = 0.43;
constexpr double kTargetAmplitudeY = 1.0;
constexpr double kTargetPeriodS = 4.0;
constexpr double kMouseYawSensitivity = 0.0025;    // rad/pixel
constexpr double kMousePitchSensitivity = 0.0020;  // rad/pixel
constexpr double kMaxYawVel = 4.0;        // matches yaw_motor ctrlrange
constexpr double kMaxPitchVel = 4.0;      // matches pitch_motor ctrlrange
constexpr double kPitchMin = -0.8;
constexpr double kPitchMax = 0.8;

struct AimbotInput
{
    double target_x = kTargetX;
    double target_y = 0.0;
    double target_z = kTargetZ;
    double target_yaw = 0.0;
    double target_pitch = 0.0;
    double yaw_error = 0.0;
    double pitch_error = 0.0;
};

struct SimGimbalStatus
{
    double yaw = 0.0;
    double yaw_vel = 0.0;
    double pitch = 0.0;
    double pitch_vel = 0.0;
};

double wrap_pi(double x)
{
    while (x > kPi) x -= 2.0 * kPi;
    while (x < -kPi) x += 2.0 * kPi;
    return x;
}

bool file_exists(const char* path)
{
    if (FILE* f = std::fopen(path, "r")) {
        std::fclose(f);
        return true;
    }
    return false;
}

std::string xml_path_from_args(int argc, char** argv)
{
    if (argc >= 2) return argv[1];
    if (file_exists("gimbal.xml")) return "gimbal.xml";
    if (file_exists("../gimbal.xml")) return "../gimbal.xml";
    return "gimbal.xml";
}

void reset_command_to_current_gimbal(
  hw::Command& cmd, const mjData* d, int yaw_qpos_addr, int pitch_qpos_addr)
{
    cmd.control = true;
    cmd.found = true;
    cmd.shoot = false;
    cmd.yaw = d->qpos[yaw_qpos_addr];
    cmd.pitch = d->qpos[pitch_qpos_addr];
    cmd.yaw_vel = 0.0;
    cmd.pitch_vel = 0.0;
}

void setup_free_camera(mjvCamera& cam)
{
    cam.type = mjCAMERA_FREE;
    cam.fixedcamid = -1;
    cam.distance = 4.4;
    cam.lookat[0] = 1.2;
    cam.lookat[1] = 0.0;
    cam.lookat[2] = 0.40;
    cam.elevation = -18;
    cam.azimuth = 135;
}

}  // namespace

int main(int argc, char** argv)
{
    char error[1024] = {0};
    const std::string xml_path = xml_path_from_args(argc, argv);
    mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!m) {
        std::fprintf(stderr, "Failed to load %s\n%s\n", xml_path.c_str(), error);
        return 1;
    }
    mjData* d = mj_makeData(m);

    const int yaw_joint = mj_name2id(m, mjOBJ_JOINT, "yaw");
    const int pitch_joint = mj_name2id(m, mjOBJ_JOINT, "pitch");
    const int yaw_motor = mj_name2id(m, mjOBJ_ACTUATOR, "yaw_motor");
    const int pitch_motor = mj_name2id(m, mjOBJ_ACTUATOR, "pitch_motor");
    const int target_body = mj_name2id(m, mjOBJ_BODY, "target");
    const int target_mocap = (target_body >= 0) ? m->body_mocapid[target_body] : -1;
    const int muzzle_site = mj_name2id(m, mjOBJ_SITE, "muzzle_site");
    const int target_site = mj_name2id(m, mjOBJ_SITE, "target_site");
    const int gimbal_camera = mj_name2id(m, mjOBJ_CAMERA, "gimbal_pov");

    if (yaw_joint < 0 || pitch_joint < 0 || yaw_motor < 0 || pitch_motor < 0 ||
        target_mocap < 0 || muzzle_site < 0 || target_site < 0 || gimbal_camera < 0) {
        std::fprintf(stderr, "Missing required MJCF names in %s\n", xml_path.c_str());
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }

    const int yaw_qpos_addr = m->jnt_qposadr[yaw_joint];
    const int pitch_qpos_addr = m->jnt_qposadr[pitch_joint];
    const int yaw_qvel_addr = m->jnt_dofadr[yaw_joint];
    const int pitch_qvel_addr = m->jnt_dofadr[pitch_joint];

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }

    GLFWwindow* w = glfwCreateWindow(1200, 900, "Yaw/Pitch Gimbal Aimbot I/O", nullptr, nullptr);
    if (!w) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }
    glfwMakeContextCurrent(w);
    glfwSwapInterval(1);

    mjvCamera cam = {};
    mjvOption opt = {};
    mjvScene scn = {};
    mjrContext con = {};
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = gimbal_camera;

    double mouse_x = 0.0;
    double mouse_y = 0.0;
    bool mouse_initialized = false;
    bool reset_prev = false;
    bool camera_prev = false;
    hw::Command command{};
    reset_command_to_current_gimbal(command, d, yaw_qpos_addr, pitch_qpos_addr);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    while (!glfwWindowShouldClose(w)) {
        const bool reset_now = glfwGetKey(w, GLFW_KEY_R) == GLFW_PRESS;
        if (reset_now && !reset_prev) {
            mj_resetData(m, d);
            reset_command_to_current_gimbal(command, d, yaw_qpos_addr, pitch_qpos_addr);
            cam.type = mjCAMERA_FIXED;
            cam.fixedcamid = gimbal_camera;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            mouse_initialized = false;
        }
        reset_prev = reset_now;

        const bool camera_now = glfwGetKey(w, GLFW_KEY_F) == GLFW_PRESS;
        if (camera_now && !camera_prev) {
            if (cam.type == mjCAMERA_FIXED) {
                setup_free_camera(cam);
                glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            } else {
                cam.type = mjCAMERA_FIXED;
                cam.fixedcamid = gimbal_camera;
                glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                mouse_initialized = false;
            }
        }
        camera_prev = camera_now;

        if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }

        // Move square target left/right in front of the gimbal.
        const double target_y = kTargetAmplitudeY * std::sin(2.0 * kPi * d->time / kTargetPeriodS);
        d->mocap_pos[3 * target_mocap + 0] = kTargetX;
        d->mocap_pos[3 * target_mocap + 1] = target_y;
        d->mocap_pos[3 * target_mocap + 2] = kTargetZ;
        d->mocap_quat[4 * target_mocap + 0] = 1.0;
        d->mocap_quat[4 * target_mocap + 1] = 0.0;
        d->mocap_quat[4 * target_mocap + 2] = 0.0;
        d->mocap_quat[4 * target_mocap + 3] = 0.0;

        double nx, ny;
        glfwGetCursorPos(w, &nx, &ny);
        if (!mouse_initialized) {
            mouse_x = nx;
            mouse_y = ny;
            mouse_initialized = true;
        }

        const double dx_mouse = nx - mouse_x;
        const double dy_mouse = ny - mouse_y;
        mouse_x = nx;
        mouse_y = ny;

        double desired_yaw_vel = 0.0;
        double desired_pitch_vel = 0.0;
        if (cam.type == mjCAMERA_FIXED) {
            const double yaw_delta = -dx_mouse * kMouseYawSensitivity;
            const double pitch_delta = -dy_mouse * kMousePitchSensitivity;
            const double next_yaw = wrap_pi(d->qpos[yaw_qpos_addr] + yaw_delta);
            const double next_pitch = std::clamp(d->qpos[pitch_qpos_addr] + pitch_delta, kPitchMin, kPitchMax);
            desired_yaw_vel = std::clamp(yaw_delta / kRenderDt, -kMaxYawVel, kMaxYawVel);
            desired_pitch_vel = std::clamp(pitch_delta / kRenderDt, -kMaxPitchVel, kMaxPitchVel);

            command.control = true;
            command.found = true;
            command.yaw = next_yaw;
            command.pitch = next_pitch;
            command.yaw_vel = desired_yaw_vel;
            command.pitch_vel = desired_pitch_vel;
        }

        const double frame_start = d->time;
        while (d->time - frame_start < kRenderDt) {
            // Manual prototype: kinematic velocity command so mouse aiming feels immediate.
            // Later RL/baseline code can swap this for actuator dynamics when needed.
            d->ctrl[yaw_motor] = desired_yaw_vel;
            d->ctrl[pitch_motor] = desired_pitch_vel;
            d->qpos[yaw_qpos_addr] = std::clamp(
              wrap_pi(d->qpos[yaw_qpos_addr] + d->ctrl[yaw_motor] * m->opt.timestep), -2.8, 2.8);
            d->qpos[pitch_qpos_addr] = std::clamp(
              d->qpos[pitch_qpos_addr] + d->ctrl[pitch_motor] * m->opt.timestep, kPitchMin,
              kPitchMax);
            d->qvel[yaw_qvel_addr] = d->ctrl[yaw_motor];
            d->qvel[pitch_qvel_addr] = d->ctrl[pitch_motor];
            d->time += m->opt.timestep;
            mj_forward(m, d);
        }

        mj_forward(m, d);

        SimGimbalStatus status{};
        status.yaw = d->qpos[yaw_qpos_addr];
        status.yaw_vel = d->qvel[yaw_qvel_addr];
        status.pitch = d->qpos[pitch_qpos_addr];
        status.pitch_vel = d->qvel[pitch_qvel_addr];

        const mjtNum* muzzle = d->site_xpos + 3 * muzzle_site;
        const mjtNum* target = d->site_xpos + 3 * target_site;
        const double dx = target[0] - muzzle[0];
        const double dy = target[1] - muzzle[1];
        const double dz = target[2] - muzzle[2];
        AimbotInput input{};
        input.target_x = target[0];
        input.target_y = target[1];
        input.target_z = target[2];
        input.target_yaw = std::atan2(dy, dx);
        input.target_pitch = std::atan2(dz, std::hypot(dx, dy));
        input.yaw_error = wrap_pi(input.target_yaw - status.yaw);
        input.pitch_error = input.target_pitch - status.pitch;

        if (cam.type == mjCAMERA_FIXED) {
            command.yaw = d->qpos[yaw_qpos_addr];
            command.pitch = d->qpos[pitch_qpos_addr];
        }

        // Free-camera mouse controls are only active after pressing F.
        if (cam.type == mjCAMERA_FREE && glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                mjv_moveCamera(m, mjMOUSE_ZOOM, 0, dy_mouse * 0.01, &cam);
            } else {
                mjv_moveCamera(m, mjMOUSE_ROTATE_H, dx_mouse * 0.005, dy_mouse * 0.005, &cam);
            }
        }

        mjrRect vp = {};
        glfwGetFramebufferSize(w, &vp.width, &vp.height);
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(vp, &scn, &con);

        char left[1200];
        std::snprintf(
          left,
          sizeof(left),
          "Mouse:aim F:camera POV/free R:reset Esc:quit\n"
          "input target=(%.2f, %.2f, %.2f) target_yaw=%+.3f target_pitch=%+.3f\n"
          "error yaw=%+.3f pitch=%+.3f\n"
          "Command{yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f}\n"
          "status yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f ctrl=(%+.2f,%+.2f)",
          input.target_x, input.target_y, input.target_z, input.target_yaw, input.target_pitch,
          input.yaw_error, input.pitch_error, command.yaw, command.yaw_vel, command.pitch,
          command.pitch_vel, status.yaw, status.yaw_vel, status.pitch, status.pitch_vel,
          d->ctrl[yaw_motor], d->ctrl[pitch_motor]);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, left, nullptr, &con);

        glfwSwapBuffers(w);
        glfwPollEvents();
    }

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwDestroyWindow(w);
    glfwTerminate();
    return 0;
}
