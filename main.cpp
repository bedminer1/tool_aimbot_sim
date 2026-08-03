#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "src/aim_predictor.hpp"
#include "src/aim_predictor_intercept.hpp"
#include "src/aim_predictor_ppo.hpp"

// ── Aiming approach selector (runtime, press Y to cycle) ───────────────────
enum AimApproach { AIM_VEL_EXTRAP = 0, AIM_INTERCEPT, AIM_PPO, AIM_COUNT };
constexpr const char* kAimApproachNames[] = {"VelExtrap", "Intercept+MPC", "PPO"};

namespace
{
// ── Sim / control constants ────────────────────────────────────────────────

constexpr double kRenderDt = 1.0 / 60.0;
constexpr double kMouseYawSensitivity = 0.0025;
constexpr double kMousePitchSensitivity = 0.0020;
constexpr double kMaxYawVel = 4.0;
constexpr double kMaxPitchVel = 4.0;
constexpr double kAutoYawKp = 9.0;
constexpr double kAutoPitchKp = 9.0;
constexpr double kAutoShotCooldownS = 0.12;
constexpr double kBulletRadius = 0.017 / 2.0;
constexpr int kMaxShots = 100;
constexpr int kMaxBullets = kMaxShots;
// RMUC 2026 sentry barrel heat (Rule Manual §3.5 / §Launching Mechanisms).
// Q0 = heat limit (soft lock). Q2 = Q0 + 100 = hard lock.
// Heat: +10 per 17 mm projectile, −30/s.
constexpr double kHeatLimit = 260.0;
constexpr double kHeatPerShot = 10.0;
constexpr double kHeatCoolingPerSecond = 30.0;

// ── Environment realism knobs ──────────────────────────────────────────────
// Detection lag: simulates CV pipeline latency (capture → YOLO → PnP → EKF).
constexpr double kDetectionLagS = 0.015;   // 15 ms

// Shooting delay: trigger solenoid → barrel exit.
constexpr double kShootDelayS = 0.030;     // 30 ms

// Fire discipline: don't waste bullets on hopeless shots.
constexpr double kMinRange = 3.0;          // m — too close, can't track
constexpr double kMaxRange = 7.0;          // m — beyond effective ballistic range
constexpr double kMaxTargetSpeed = 3.5;    // m/s — too fast to track reliably

// ── Types ──────────────────────────────────────────────────────────────────

struct SimGimbalStatus
{
    double yaw = 0.0;
    double yaw_vel = 0.0;
    double pitch = 0.0;
    double pitch_vel = 0.0;
};

struct Bullet
{
    bool active = false;
    bool scored = false;
    double spawn_time = 0.0;
    int mocap_id = -1;
    Vec3 prev_pos{};
    Vec3 pos{};
    Vec3 vel{};
};

struct PendingShot
{
    double trigger_time;
    Vec3 muzzle_pos;
    Vec3 muzzle_dir;
    int shot_index;
};

double wrap_pi(double x)
{
    while (x > kPi) x -= 2.0 * kPi;
    while (x < -kPi) x += 2.0 * kPi;
    return x;
}

bool file_exists(const char* path)
{
    if (FILE* f = std::fopen(path, "r")) { std::fclose(f); return true; }
    return false;
}

// ── Random waypoint target motion (no oracle) ──────────────────────────────

constexpr double kTargetOrbitRadius = 0.04;
constexpr double kTargetOrbitPeriodS = 0.6;

struct TargetBehavior
{
    enum State { IDLE, MOVING };
    State state = IDLE;
    double state_start_time = 0.0;
    double state_duration = 1.0;
    Vec3 start_pos{};
    Vec3 target_pos{};
    Vec3 center_pos{};
    Vec3 composite_pos{};
    Vec3 velocity{};
    Vec3 prev_composite{};
    double prev_time = 0.0;
};

TargetBehavior g_target;

double uniform_rand(double lo, double hi)
{
    return lo + (static_cast<double>(std::rand()) / RAND_MAX) * (hi - lo);
}

Vec3 random_waypoint()
{
    // Stay 3–5 m from gimbal. Polar with angle constrained to keep Y bounded.
    const double r = uniform_rand(3.0, 5.0);
    const double angle = uniform_rand(-0.55, 0.55);
    return {r * std::cos(angle), r * std::sin(angle), 0.015};
}

double smoothstep(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

void update_target_motion(mjData* d, int target_mocap, TargetBehavior& tb)
{
    const double now = d->time;
    const double elapsed = now - tb.state_start_time;

    if (tb.state == TargetBehavior::IDLE) {
        if (elapsed >= tb.state_duration) {
            tb.state = TargetBehavior::MOVING;
            tb.state_start_time = now;
            tb.start_pos = tb.center_pos;
            tb.target_pos = random_waypoint();
            const double dist = norm(tb.target_pos - tb.start_pos);
            const double speed = uniform_rand(1.0, 2.5);
            tb.state_duration = dist / std::max(0.1, speed);
        }
    } else {
        const double t = elapsed / tb.state_duration;
        if (t >= 1.0) {
            tb.center_pos = tb.target_pos;
            tb.state = TargetBehavior::IDLE;
            tb.state_start_time = now;
            tb.state_duration = uniform_rand(0.1, 0.8);
        } else {
            tb.center_pos = tb.start_pos + (tb.target_pos - tb.start_pos) * smoothstep(t);
        }
    }

    const double orbit_phase = 2.0 * kPi * now / kTargetOrbitPeriodS;
    const double ox = kTargetOrbitRadius * std::sin(orbit_phase);
    const double oy = kTargetOrbitRadius * std::cos(orbit_phase);
    const Vec3 composite{tb.center_pos.x + ox, tb.center_pos.y + oy, tb.center_pos.z};
    tb.composite_pos = composite;

    d->mocap_pos[3 * target_mocap + 0] = composite.x;
    d->mocap_pos[3 * target_mocap + 1] = composite.y;
    d->mocap_pos[3 * target_mocap + 2] = composite.z;

    const double r = kTargetOrbitRadius;
    const mjtNum rot[9] = {ox / r, oy / r, 0.0, -oy / r, ox / r, 0.0, 0.0, 0.0, 1.0};
    mju_mat2Quat(d->mocap_quat + 4 * target_mocap, rot);

    const double dt = now - tb.prev_time;
    if (dt > 1e-6) tb.velocity = (composite - tb.prev_composite) * (1.0 / dt);
    tb.prev_composite = composite;
    tb.prev_time = now;
}

// ── Hit detection ──────────────────────────────────────────────────────────

double dot(const mjtNum* axis, const Vec3& v)
{
    return axis[0] * v.x + axis[1] * v.y + axis[2] * v.z;
}

bool ray_hits_target_box(const mjModel* m, const mjData* d, int muzzle_site, int target_body,
                         int target_geom)
{
    const mjtNum* origin = d->site_xpos + 3 * muzzle_site;
    const mjtNum* site_xmat = d->site_xmat + 9 * muzzle_site;
    const Vec3 dir_world{site_xmat[0], site_xmat[3], site_xmat[6]};
    const mjtNum* body_pos = d->xpos + 3 * target_body;
    const mjtNum* body_xmat = d->xmat + 9 * target_body;
    const mjtNum* half = m->geom_size + 3 * target_geom;

    const Vec3 delta{origin[0] - body_pos[0], origin[1] - body_pos[1], origin[2] - body_pos[2]};
    const double ov[3] = {dot(body_xmat + 0, delta), dot(body_xmat + 3, delta),
                          dot(body_xmat + 6, delta)};
    const double dv[3] = {dot(body_xmat + 0, dir_world), dot(body_xmat + 3, dir_world),
                          dot(body_xmat + 6, dir_world)};
    const double hv[3] = {half[0], half[1], half[2]};

    double t_min = 0.0, t_max = 1e9;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(dv[i]) < 1e-9) {
            if (ov[i] < -hv[i] || ov[i] > hv[i]) return false;
            continue;
        }
        double t1 = (-hv[i] - ov[i]) / dv[i], t2 = (hv[i] - ov[i]) / dv[i];
        if (t1 > t2) std::swap(t1, t2);
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        if (t_min > t_max) return false;
    }
    return t_max >= 0.0;
}

bool segment_hits_expanded_target_box(const mjModel* m, const mjData* d, const Vec3& a,
                                      const Vec3& b, int target_body, int target_geom)
{
    const mjtNum* bp = d->xpos + 3 * target_body;
    const mjtNum* bx = d->xmat + 9 * target_body;
    const mjtNum* half = m->geom_size + 3 * target_geom;
    const Vec3 da{a.x - bp[0], a.y - bp[1], a.z - bp[2]};
    const Vec3 db{b.x - bp[0], b.y - bp[1], b.z - bp[2]};
    const double av[3] = {dot(bx + 0, da), dot(bx + 3, da), dot(bx + 6, da)};
    const double bv[3] = {dot(bx + 0, db), dot(bx + 3, db), dot(bx + 6, db)};
    const double hv[3] = {half[0] + kBulletRadius, half[1] + kBulletRadius,
                          half[2] + kBulletRadius};

    double t_min = 0.0, t_max = 1.0;
    for (int i = 0; i < 3; ++i) {
        const double dir = bv[i] - av[i];
        if (std::abs(dir) < 1e-9) {
            if (av[i] < -hv[i] || av[i] > hv[i]) return false;
            continue;
        }
        double t1 = (-hv[i] - av[i]) / dir, t2 = (hv[i] - av[i]) / dir;
        if (t1 > t2) std::swap(t1, t2);
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        if (t_min > t_max) return false;
    }
    return true;
}

// ── Bullets ────────────────────────────────────────────────────────────────

void hide_bullet(mjData* d, int mocap_id)
{
    d->mocap_pos[3 * mocap_id + 0] = 0.0;
    d->mocap_pos[3 * mocap_id + 1] = 0.0;
    d->mocap_pos[3 * mocap_id + 2] = -10.0;
    d->mocap_quat[4 * mocap_id + 0] = 1.0;
    d->mocap_quat[4 * mocap_id + 1] = 0.0;
    d->mocap_quat[4 * mocap_id + 2] = 0.0;
    d->mocap_quat[4 * mocap_id + 3] = 0.0;
}

void reset_bullets(std::array<Bullet, kMaxBullets>& bullets, mjData* d)
{
    for (auto& b : bullets) {
        b.active = b.scored = false;
        b.spawn_time = 0.0;
        b.prev_pos = b.pos = b.vel = {};
        if (b.mocap_id >= 0) hide_bullet(d, b.mocap_id);
    }
}

Vec3 bullet_position_at(const Bullet& bullet, double time)
{
    const double t = time - bullet.spawn_time;
    return {bullet.pos.x + bullet.vel.x * t, bullet.pos.y + bullet.vel.y * t,
            bullet.pos.z + bullet.vel.z * t - 0.5 * kGravity * t * t};
}

void spawn_bullet(std::array<Bullet, kMaxBullets>& bullets, const mjData* d, int muzzle_site,
                  int shot_index)
{
    Bullet& b = bullets[shot_index % kMaxBullets];
    const mjtNum* mz = d->site_xpos + 3 * muzzle_site;
    const mjtNum* mx = d->site_xmat + 9 * muzzle_site;
    const Vec3 dir{mx[0], mx[3], mx[6]};
    b.active = true;
    b.scored = false;
    b.spawn_time = d->time;
    b.pos = {mz[0], mz[1], mz[2]};
    b.prev_pos = b.pos;
    b.vel = dir * kBulletSpeed;
}

// Delayed-fire variant: explicit muzzle pos/dir/time (shooting delay).
void spawn_bullet_at(std::array<Bullet, kMaxBullets>& bullets, const Vec3& muzzle_pos,
                     const Vec3& muzzle_dir, double spawn_time, int shot_index)
{
    Bullet& b = bullets[shot_index % kMaxBullets];
    b.active = true;
    b.scored = false;
    b.spawn_time = spawn_time;
    b.pos = muzzle_pos;
    b.prev_pos = b.pos;
    b.vel = muzzle_dir * kBulletSpeed;
}

void update_bullets(std::array<Bullet, kMaxBullets>& bullets, const mjModel* m, mjData* d,
                    int target_body, int target_geom, int& score, bool& last_shot_hit)
{
    for (auto& b : bullets) {
        if (!b.active) continue;
        const Vec3 current = bullet_position_at(b, d->time);
        if (!b.scored &&
            segment_hits_expanded_target_box(m, d, b.prev_pos, current, target_body, target_geom)) {
            b.scored = true;
            b.active = false;
            ++score;
            last_shot_hit = true;
            if (b.mocap_id >= 0) hide_bullet(d, b.mocap_id);
            continue;
        }
        b.prev_pos = current;
        if (b.mocap_id >= 0) {
            d->mocap_pos[3 * b.mocap_id + 0] = current.x;
            d->mocap_pos[3 * b.mocap_id + 1] = current.y;
            d->mocap_pos[3 * b.mocap_id + 2] = current.z;
        }
        if (current.z < -1.0 || d->time - b.spawn_time > 3.0) {
            b.active = false;
            if (b.mocap_id >= 0) hide_bullet(d, b.mocap_id);
        }
    }
}

// ── Heat ───────────────────────────────────────────────────────────────────

void cool_heat(double& heat, double now, double& last_update)
{
    const double dt = std::max(0.0, now - last_update);
    heat = std::max(0.0, heat - kHeatCoolingPerSecond * dt);
    last_update = now;
}

// ── Gimbal helpers ─────────────────────────────────────────────────────────

std::string xml_path_from_args(int argc, char** argv)
{
    if (argc >= 2) return argv[1];
    if (file_exists("gimbal.xml")) return "gimbal.xml";
    if (file_exists("../gimbal.xml")) return "../gimbal.xml";
    return "gimbal.xml";
}

void reset_command_to_current_gimbal(hw::Command& cmd, const mjData* d, int yaw_qpos_addr,
                                     int pitch_qpos_addr)
{
    cmd.control = true;
    cmd.found = true;
    cmd.shoot = false;
    cmd.yaw = d->qpos[yaw_qpos_addr];
    cmd.pitch = d->qpos[pitch_qpos_addr];
    cmd.yaw_vel = 0.0;
    cmd.pitch_vel = 0.0;
    cmd.yaw_accel = 0.0;
    cmd.pitch_accel = 0.0;
}

hw::Command make_aimbot_command(const AimPrediction& pred, const SimGimbalStatus& status)
{
    hw::Command cmd;
    cmd.control = true;
    cmd.found = true;
    cmd.shoot = false;
    cmd.yaw = pred.target_yaw;
    cmd.pitch = std::clamp(pred.target_pitch, kPitchMin, kPitchMax);
    cmd.yaw_vel = std::clamp(kAutoYawKp * pred.yaw_error, -kMaxYawVel, kMaxYawVel);
    cmd.pitch_vel = std::clamp(kAutoPitchKp * pred.pitch_error, -kMaxPitchVel, kMaxPitchVel);
    cmd.yaw_accel = (cmd.yaw_vel - status.yaw_vel) / kRenderDt;
    cmd.pitch_accel = (cmd.pitch_vel - status.pitch_vel) / kRenderDt;
    return cmd;
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

// ── main ───────────────────────────────────────────────────────────────────

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
    const int target_geom = mj_name2id(m, mjOBJ_GEOM, "target_square");
    const int target_mocap = (target_body >= 0) ? m->body_mocapid[target_body] : -1;
    const int muzzle_site = mj_name2id(m, mjOBJ_SITE, "muzzle_site");
    const int target_site = mj_name2id(m, mjOBJ_SITE, "target_site");
    const int gimbal_camera = mj_name2id(m, mjOBJ_CAMERA, "gimbal_pov");
    std::array<int, kMaxBullets> bullet_mocaps{};

    if (yaw_joint < 0 || pitch_joint < 0 || yaw_motor < 0 || pitch_motor < 0 || target_geom < 0 ||
        target_mocap < 0 || muzzle_site < 0 || target_site < 0 || gimbal_camera < 0) {
        std::fprintf(stderr, "Missing required MJCF names in %s\n", xml_path.c_str());
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }

    const int yaw_qpos = m->jnt_qposadr[yaw_joint];
    const int pitch_qpos = m->jnt_qposadr[pitch_joint];
    const int yaw_qvel = m->jnt_dofadr[yaw_joint];
    const int pitch_qvel = m->jnt_dofadr[pitch_joint];

    for (int i = 0; i < kMaxBullets; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "bullet_%d", i);
        const int body_id = mj_name2id(m, mjOBJ_BODY, name);
        if (body_id < 0 || m->body_mocapid[body_id] < 0) {
            std::fprintf(stderr, "Missing required bullet mocap body %s\n", name);
            mj_deleteData(d);
            mj_deleteModel(m);
            return 1;
        }
        bullet_mocaps[i] = m->body_mocapid[body_id];
    }

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

    double mouse_x = 0.0, mouse_y = 0.0;
    bool mouse_initialized = false;
    bool reset_prev = false, camera_prev = false, aimbot_prev = false;
    bool aimbot_enabled = false, shot_prev = false;
    int shots_fired = 0, score = 0;
    bool last_shot_hit = false;
    double first_shot_time = -1.0, last_shot_time = -1.0;
    double last_auto_shot_time = -1.0;
    double barrel_heat = 0.0, last_heat_update_time = 0.0;

    std::array<Bullet, kMaxBullets> bullets{};
    for (int i = 0; i < kMaxBullets; ++i) bullets[i].mocap_id = bullet_mocaps[i];
    reset_bullets(bullets, d);

    hw::Command command{};
    reset_command_to_current_gimbal(command, d, yaw_qpos, pitch_qpos);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Environment state.
    DetectLagBuffer detect_buf(300);
    InterceptPredictor ipred;
    PpoPredictor ppred;
    ppred.load("src/aim_predictor.onnx");  // trained model — optional, fails gracefully
    AimApproach aim_approach = AIM_VEL_EXTRAP;
    std::vector<PendingShot> pending_shots;
    size_t pending_fired = 0;  // total shots queued via delay path

    g_target.center_pos = {5.0, 0.0, 0.0};
    g_target.prev_composite = g_target.center_pos;
    g_target.state = TargetBehavior::IDLE;
    g_target.state_start_time = 0.0;
    g_target.state_duration = 0.5;

    while (!glfwWindowShouldClose(w)) {
        // ── Reset ──
        const bool reset_now = glfwGetKey(w, GLFW_KEY_R) == GLFW_PRESS;
        if (reset_now && !reset_prev) {
            mj_resetData(m, d);
            reset_command_to_current_gimbal(command, d, yaw_qpos, pitch_qpos);
            cam.type = mjCAMERA_FIXED;
            cam.fixedcamid = gimbal_camera;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            mouse_initialized = false;
            aimbot_enabled = false;
            aimbot_prev = false;
            shot_prev = false;
            shots_fired = 0;
            score = 0;
            last_shot_hit = false;
            first_shot_time = last_shot_time = -1.0;
            last_auto_shot_time = -1.0;
            barrel_heat = 0.0;
            last_heat_update_time = 0.0;
            reset_bullets(bullets, d);
            detect_buf.clear();
            ipred.clear();
            pending_shots.clear();
            pending_fired = 0;
            g_target = TargetBehavior{};
            g_target.center_pos = {5.0, 0.0, 0.0};
            g_target.prev_composite = g_target.center_pos;
            g_target.state = TargetBehavior::IDLE;
            g_target.state_start_time = d->time;
            g_target.state_duration = 0.5;
        }
        reset_prev = reset_now;

        // ── Camera toggle ──
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

        // ── Aimbot toggle ──
        const bool aimbot_now = glfwGetKey(w, GLFW_KEY_T) == GLFW_PRESS;
        if (aimbot_now && !aimbot_prev) {
            aimbot_enabled = !aimbot_enabled;
            shots_fired = 0;
            score = 0;
            last_shot_hit = false;
            first_shot_time = last_shot_time = -1.0;
            last_auto_shot_time = -1.0;
            shot_prev = false;
            reset_bullets(bullets, d);
            detect_buf.clear();
            ipred.clear();
            pending_shots.clear();
            pending_fired = 0;
            if (aimbot_enabled) {
                cam.type = mjCAMERA_FIXED;
                cam.fixedcamid = gimbal_camera;
                glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                mouse_initialized = false;
            }
        }
        aimbot_prev = aimbot_now;

        if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(w, GLFW_TRUE);

        // ── Target physics ──
        update_target_motion(d, target_mocap, g_target);
        mj_forward(m, d);

        // Record perfect observation in history (aimbot queries with lag).
        detect_buf.push(d->time, g_target.composite_pos, g_target.velocity);

        cool_heat(barrel_heat, d->time, last_heat_update_time);

        SimGimbalStatus pre_status{};
        pre_status.yaw = d->qpos[yaw_qpos];
        pre_status.yaw_vel = d->qvel[yaw_qvel];
        pre_status.pitch = d->qpos[pitch_qpos];
        pre_status.pitch_vel = d->qvel[pitch_qvel];

        // Aimbot queries delayed observation.
        Vec3 obs_pos = g_target.composite_pos;
        Vec3 obs_vel = g_target.velocity;
        detect_buf.lookup(d->time - kDetectionLagS, obs_pos, obs_vel);

        const mjtNum* mz = d->site_xpos + 3 * muzzle_site;
        const Vec3 muzzle_pos{mz[0], mz[1], mz[2]};

        // ── Prediction (approach-dependent) ────────────────────────────
        AimPrediction pred{};
        if (aim_approach == AIM_INTERCEPT) {
            ipred.observe(d->time - kDetectionLagS, obs_pos);
            pred = ipred.predict(muzzle_pos, pre_status.yaw, pre_status.pitch);
        } else if (aim_approach == AIM_PPO) {
            // PPO handles prediction + control internally (see command section).
            pred.target_yaw = std::atan2(obs_pos.y - muzzle_pos.y, obs_pos.x - muzzle_pos.x);
            pred.target_pitch = std::atan2(obs_pos.z - muzzle_pos.z,
                                           norm_xy(obs_pos - muzzle_pos));
        } else {
            // VelExtrap: lag-compensated position, acceleration estimate, system delay.
            Vec3 cur_pos = obs_pos + obs_vel * kDetectionLagS;  // extrapolate to now
            static Vec3 prev_vel = obs_vel;
            static double prev_t = 0;
            double dt_accel = d->time - prev_t;
            Vec3 accel{0,0,0};
            if (dt_accel > 1e-6) {
                accel = (obs_vel - prev_vel) * (1.0 / dt_accel);
                // Clamp to reasonable acceleration (~5 m/s² max for ground robots).
                double a = norm(accel);
                if (a > 5.0) accel = accel * (5.0 / a);
            }
            prev_vel = obs_vel;
            prev_t = d->time;
            const double sys_delay = kDetectionLagS + kShootDelayS;
            pred = predict_aim(muzzle_pos, cur_pos, obs_vel, accel, sys_delay,
                               pre_status.yaw, pre_status.pitch);
        }

        // ── Mouse / aimbot input ──
        double nx, ny;
        glfwGetCursorPos(w, &nx, &ny);
        if (!mouse_initialized) { mouse_x = nx; mouse_y = ny; mouse_initialized = true; }
        const double dx_mouse = nx - mouse_x, dy_mouse = ny - mouse_y;
        mouse_x = nx; mouse_y = ny;

        // ── Approach toggle ──
        const bool y_now = glfwGetKey(w, GLFW_KEY_Y) == GLFW_PRESS;
        static bool y_prev = false;
        if (y_now && !y_prev) {
            aim_approach = static_cast<AimApproach>((aim_approach + 1) % AIM_COUNT);
            detect_buf.clear();
            ipred.clear();
        }
        y_prev = y_now;

        double desired_yaw_vel = 0.0, desired_pitch_vel = 0.0;
        if (aimbot_enabled) {
            // ── Command (approach-dependent) ───────────────────────────
            if (aim_approach == AIM_PPO && ppred.loaded()) {
                double tsls = (last_auto_shot_time < 0.0) ? 1.0
                              : d->time - last_auto_shot_time;
                auto pa = ppred.predict(muzzle_pos, obs_pos,
                                        pre_status.yaw, pre_status.pitch,
                                        pre_status.yaw_vel, pre_status.pitch_vel,
                                        barrel_heat, tsls);
                command.control = true;
                command.found = true;
                command.yaw = pre_status.yaw;
                command.pitch = pre_status.pitch;
                command.yaw_vel = pa.yaw_vel;
                command.pitch_vel = pa.pitch_vel;
                command.yaw_accel = (pa.yaw_vel - pre_status.yaw_vel) / kRenderDt;
                command.pitch_accel = (pa.pitch_vel - pre_status.pitch_vel) / kRenderDt;
                command.shoot = pa.fire;
                desired_yaw_vel = pa.yaw_vel;
                desired_pitch_vel = pa.pitch_vel;
            } else if (aim_approach == AIM_INTERCEPT && ipred.last_intercept_valid()) {
                MpcCommand mc = make_mpc_command(
                    muzzle_pos, ipred.last_model(), pred,
                    pre_status.yaw, pre_status.pitch,
                    pre_status.yaw_vel, pre_status.pitch_vel,
                    ipred.last_intercept_t());
                command.control = true;
                command.found = true;
                command.yaw = mc.yaw;
                command.yaw_vel = mc.yaw_vel;
                command.yaw_accel = mc.yaw_accel;
                command.pitch = mc.pitch;
                command.pitch_vel = mc.pitch_vel;
                command.pitch_accel = mc.pitch_accel;
                desired_yaw_vel = mc.yaw_vel;
                desired_pitch_vel = mc.pitch_vel;
            } else {
                // VelExtrap: P-controller + velocity feedforward from observed motion.
                const Vec3 delta = obs_pos - muzzle_pos;
                const double h2 = delta.x*delta.x + delta.y*delta.y;
                double ff_yaw = 0, ff_pitch = 0;
                if (h2 > 1e-6) {
                    ff_yaw = (delta.x*obs_vel.y - delta.y*obs_vel.x) / h2;
                    double h = std::sqrt(h2), r2 = h2 + delta.z*delta.z;
                    if (r2 > 1e-6) {
                        double hdot = (delta.x*obs_vel.x + delta.y*obs_vel.y) / h;
                        ff_pitch = (h*obs_vel.z - delta.z*hdot) / r2;
                    }
                }
                command = make_aimbot_command(pred, pre_status);
                command.yaw_vel = std::clamp(ff_yaw + command.yaw_vel, -kMaxYawVel, kMaxYawVel);
                command.pitch_vel = std::clamp(ff_pitch + command.pitch_vel, -kMaxPitchVel, kMaxPitchVel);
                desired_yaw_vel = command.yaw_vel;
                desired_pitch_vel = command.pitch_vel;
            }
        } else if (cam.type == mjCAMERA_FIXED) {
            const double yd = -dx_mouse * kMouseYawSensitivity;
            const double pd = -dy_mouse * kMousePitchSensitivity;
            const double next_yaw = wrap_pi(d->qpos[yaw_qpos] + yd);
            const double next_pitch = std::clamp(d->qpos[pitch_qpos] + pd, kPitchMin, kPitchMax);
            desired_yaw_vel = std::clamp(yd / kRenderDt, -kMaxYawVel, kMaxYawVel);
            desired_pitch_vel = std::clamp(pd / kRenderDt, -kMaxPitchVel, kMaxPitchVel);
            command.control = true;
            command.found = true;
            command.yaw = next_yaw;
            command.pitch = next_pitch;
            command.yaw_vel = desired_yaw_vel;
            command.pitch_vel = desired_pitch_vel;
        }

        // ── Physics step ──
        const double frame_start = d->time;
        while (d->time - frame_start < kRenderDt) {
            d->ctrl[yaw_motor] = desired_yaw_vel;
            d->ctrl[pitch_motor] = desired_pitch_vel;
            d->qpos[yaw_qpos] =
              std::clamp(wrap_pi(d->qpos[yaw_qpos] + d->ctrl[yaw_motor] * m->opt.timestep),
                         -2.8, 2.8);
            d->qpos[pitch_qpos] = std::clamp(
              d->qpos[pitch_qpos] + d->ctrl[pitch_motor] * m->opt.timestep, kPitchMin, kPitchMax);
            d->qvel[yaw_qvel] = d->ctrl[yaw_motor];
            d->qvel[pitch_qvel] = d->ctrl[pitch_motor];
            d->time += m->opt.timestep;
            mj_forward(m, d);
        }
        mj_forward(m, d);

        // ── Process pending delayed shots ──
        for (size_t i = 0; i < pending_shots.size(); ) {
            const auto& ps = pending_shots[i];
            if (d->time >= ps.trigger_time + kShootDelayS) {
                spawn_bullet_at(bullets, ps.muzzle_pos, ps.muzzle_dir, d->time, ps.shot_index);
                last_shot_hit = false;
                pending_shots.erase(pending_shots.begin() + i);
                // Don't advance i — erase shifted elements.
            } else {
                ++i;
            }
        }

        // ── Fire decision ──
        const bool manual_now = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool manual_edge = cam.type == mjCAMERA_FIXED && manual_now && !shot_prev;
        const bool auto_aimed =
          aimbot_enabled && std::abs(wrap_pi(command.yaw - d->qpos[yaw_qpos])) < 0.015 &&
          std::abs(command.pitch - d->qpos[pitch_qpos]) < 0.015;
        const double target_range = norm_xy(obs_pos - muzzle_pos);
        const double target_speed = norm(g_target.velocity);
        const bool in_range =
          target_range >= kMinRange && target_range <= kMaxRange &&
          target_speed <= kMaxTargetSpeed;
        const bool auto_ready =
          auto_aimed && in_range &&
          shots_fired + pending_shots.size() < static_cast<size_t>(kMaxShots) &&
          barrel_heat + kHeatPerShot <= kHeatLimit &&
          (last_auto_shot_time < 0.0 || d->time - last_auto_shot_time >= kAutoShotCooldownS);
        const bool fire_now =
          shots_fired + pending_shots.size() < static_cast<size_t>(kMaxShots) &&
          (manual_edge || auto_ready);
        command.shoot = fire_now;

        if (fire_now) {
            const mjtNum* fmz = d->site_xpos + 3 * muzzle_site;
            const mjtNum* fmx = d->site_xmat + 9 * muzzle_site;
            const Vec3 fmz_pos{fmz[0], fmz[1], fmz[2]};
            const Vec3 fmz_dir{fmx[0], fmx[3], fmx[6]};

            if (kShootDelayS > 0.0) {
                pending_shots.push_back({d->time, fmz_pos, fmz_dir, shots_fired});
                ++pending_fired;
            } else {
                spawn_bullet(bullets, d, muzzle_site, shots_fired);
                last_shot_hit = false;
            }

            if (shots_fired == 0) first_shot_time = d->time;
            ++shots_fired;
            if (shots_fired == kMaxShots) last_shot_time = d->time;
            barrel_heat += kHeatPerShot;
            if (auto_ready) last_auto_shot_time = d->time;
        }
        shot_prev = manual_now;
        update_bullets(bullets, m, d, target_body, target_geom, score, last_shot_hit);

        if (cam.type == mjCAMERA_FIXED && !aimbot_enabled) {
            command.yaw = d->qpos[yaw_qpos];
            command.pitch = d->qpos[pitch_qpos];
        }

        // ── Free camera ──
        if (cam.type == mjCAMERA_FREE &&
            glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                mjv_moveCamera(m, mjMOUSE_ZOOM, 0, dy_mouse * 0.01, &cam);
            } else {
                mjv_moveCamera(m, mjMOUSE_ROTATE_H, dx_mouse * 0.005, dy_mouse * 0.005, &cam);
            }
        }

        // ── Render ──
        mjrRect vp = {};
        glfwGetFramebufferSize(w, &vp.width, &vp.height);
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(vp, &scn, &con);

        const double shot_elapsed =
          (first_shot_time < 0.0)
            ? 0.0
            : ((shots_fired >= kMaxShots ? last_shot_time : d->time) - first_shot_time);

        char left[1500];
        std::snprintf(
          left, sizeof(left),
          "Mouse:aim T:aimbot %s Y:approach[%s] F:camera POV/free R:reset Esc:quit\n"
          "shots %d/100 score %d last=%s heat %.0f+%.0f/%.0f time %.2fs lag=%.0fms delay=%.0fms pending=%zu\n"
          "input target=(%.2f, %.2f, %.2f) target_yaw=%+.3f target_pitch=%+.3f\n"
          "error yaw=%+.3f pitch=%+.3f\n"
          "Command{shoot=%d yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f}\n"
          "status yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f ctrl=(%+.2f,%+.2f)",
          aimbot_enabled ? "ON" : "OFF", kAimApproachNames[aim_approach],
          shots_fired, score,
          last_shot_hit ? "HIT" : pending_shots.empty() ? "MISS" : "...",
          barrel_heat, kHeatPerShot, kHeatLimit, shot_elapsed,
          kDetectionLagS * 1000.0, kShootDelayS * 1000.0, pending_shots.size(),
          obs_pos.x, obs_pos.y, obs_pos.z, pred.target_yaw, pred.target_pitch,
          pred.yaw_error, pred.pitch_error, command.shoot ? 1 : 0,
          command.yaw, command.yaw_vel, command.pitch, command.pitch_vel,
          d->qpos[yaw_qpos], d->qvel[yaw_qvel], d->qpos[pitch_qpos], d->qvel[pitch_qvel],
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
