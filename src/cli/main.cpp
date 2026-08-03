// mujocoaim — RoboMaster gimbal aimbot sim with swappable targets and aimers.
//
// Usage: mujocoaim [--difficulty easy|medium|hard] [gimbal.xml]

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "common/types.hpp"
#include "aim_models/aim_predictor.hpp"
#include "aim_models/aim_predictor_intercept.hpp"
#include "aim_models/aim_predictor_ppo.hpp"
#include "target_models/target_interface.hpp"
#include "target_models/target_easy.hpp"
#include "target_models/target_medium.hpp"
#include "target_models/target_hard.hpp"

namespace
{
// ── Constants ──────────────────────────────────────────────────────────────

constexpr double kRenderDt = 1.0 / 60.0;
constexpr double kMouseYawSens = 0.0025, kMousePitchSens = 0.0020;
constexpr double kMaxYawVel = 4.0, kMaxPitchVel = 4.0;
constexpr double kAutoYawKp = 9.0, kAutoPitchKp = 9.0;
constexpr double kAutoCooldown = 0.12;
constexpr double kBulletRadius = 0.017 / 2.0;
constexpr int kMaxShots = 100, kMaxBullets = kMaxShots;
constexpr double kHeatLimit = 260.0, kHeatPerShot = 10.0, kHeatCooling = 30.0;
constexpr double kDetectionLagS = 0.015, kShootDelayS = 0.030;
constexpr double kMinRange = 3.0, kMaxRange = 7.0, kMaxTargetSpeed = 3.5;
constexpr double kOrbitPeriodS = 0.6;

// ── Aim approach ───────────────────────────────────────────────────────────

enum AimApproach { AIM_VEL_EXTRAP = 0, AIM_INTERCEPT, AIM_PPO, AIM_COUNT };
constexpr const char* kAimNames[] = {"VelExtrap", "Intercept+MPC", "PPO"};

// ── Types ──────────────────────────────────────────────────────────────────

struct SimGimbalStatus { double yaw, yaw_vel, pitch, pitch_vel; };

struct Bullet {
    bool active = false, scored = false;
    double spawn_time = 0;
    int mocap_id = -1;
    Vec3 prev_pos{}, pos{}, vel{};
};

struct PendingShot {
    double trigger_time;
    Vec3 muzzle_pos, muzzle_dir;
    int shot_index;
};

// ── Bullet physics ─────────────────────────────────────────────────────────

void hide_bullet(mjData* d, int id) {
    d->mocap_pos[3*id+0]=0; d->mocap_pos[3*id+1]=0; d->mocap_pos[3*id+2]=-10;
    d->mocap_quat[4*id+0]=1; d->mocap_quat[4*id+1]=0;
    d->mocap_quat[4*id+2]=0; d->mocap_quat[4*id+3]=0;
}

void reset_bullets(std::array<Bullet,kMaxBullets>& bullets, mjData* d) {
    for (auto& b : bullets) { b.active=b.scored=false; b.spawn_time=0;
        b.prev_pos=b.pos=b.vel={}; if (b.mocap_id>=0) hide_bullet(d,b.mocap_id); }
}

Vec3 bullet_pos_at(const Bullet& b, double t) {
    double dt = t - b.spawn_time;
    return {b.pos.x+b.vel.x*dt, b.pos.y+b.vel.y*dt, b.pos.z+b.vel.z*dt-0.5*kGravity*dt*dt};
}

void spawn_bullet(std::array<Bullet,kMaxBullets>& bullets, const mjData* d,
                  int muzzle_site, int idx) {
    Bullet& b = bullets[idx%kMaxBullets];
    const auto* mz=d->site_xpos+3*muzzle_site, *mx=d->site_xmat+9*muzzle_site;
    Vec3 dir{mx[0],mx[3],mx[6]};
    b.active=true; b.scored=false; b.spawn_time=d->time;
    b.pos={mz[0],mz[1],mz[2]}; b.prev_pos=b.pos; b.vel=dir*kBulletSpeed;
}

void spawn_bullet_at(std::array<Bullet,kMaxBullets>& bullets, const Vec3& pos,
                     const Vec3& dir, double t, int idx) {
    Bullet& b = bullets[idx%kMaxBullets];
    b.active=true; b.scored=false; b.spawn_time=t; b.pos=pos; b.prev_pos=pos;
    b.vel=dir*kBulletSpeed;
}

// ── Hit detection ──────────────────────────────────────────────────────────

double ddot(const double* a, const Vec3& v) { return a[0]*v.x+a[1]*v.y+a[2]*v.z; }

bool segment_hits_box(const mjModel* m, const mjData* d, const Vec3& a, const Vec3& b,
                      int tb, int tg) {
    const auto* bp=d->xpos+3*tb, *bx=d->xmat+9*tb, *half=m->geom_size+3*tg;
    Vec3 da{a.x-bp[0],a.y-bp[1],a.z-bp[2]}, db{b.x-bp[0],b.y-bp[1],b.z-bp[2]};
    double av[3]={ddot(bx,da),ddot(bx+3,da),ddot(bx+6,da)};
    double bv[3]={ddot(bx,db),ddot(bx+3,db),ddot(bx+6,db)};
    double hv[3]={half[0]+kBulletRadius,half[1]+kBulletRadius,half[2]+kBulletRadius};
    double tmin=0,tmax=1;
    for (int i=0;i<3;++i) {
        double dir=bv[i]-av[i];
        if (std::abs(dir)<1e-9) { if (av[i]<-hv[i]||av[i]>hv[i]) return false; continue; }
        double t1=(-hv[i]-av[i])/dir, t2=(hv[i]-av[i])/dir;
        if (t1>t2) std::swap(t1,t2);
        tmin=std::max(tmin,t1); tmax=std::min(tmax,t2);
        if (tmin>tmax) return false;
    }
    return true;
}

void update_bullets(std::array<Bullet,kMaxBullets>& bullets, const mjModel* m, mjData* d,
                    int tb, int tg, int& score, bool& last_hit) {
    for (auto& b : bullets) {
        if (!b.active) continue;
        Vec3 cur = bullet_pos_at(b, d->time);
        if (!b.scored && segment_hits_box(m,d,b.prev_pos,cur,tb,tg)) {
            b.scored=true; b.active=false; ++score; last_hit=true;
            if (b.mocap_id>=0) hide_bullet(d,b.mocap_id); continue;
        }
        b.prev_pos=cur;
        if (b.mocap_id>=0) { d->mocap_pos[3*b.mocap_id+0]=cur.x;
            d->mocap_pos[3*b.mocap_id+1]=cur.y; d->mocap_pos[3*b.mocap_id+2]=cur.z; }
        if (cur.z<-1||d->time-b.spawn_time>3) { b.active=false;
            if (b.mocap_id>=0) hide_bullet(d,b.mocap_id); }
    }
}

// ── Target rendering ───────────────────────────────────────────────────────

void write_target_mocap(mjData* d, int mocap_id, const Vec3& pos) {
    d->mocap_pos[3*mocap_id+0]=pos.x; d->mocap_pos[3*mocap_id+1]=pos.y;
    d->mocap_pos[3*mocap_id+2]=pos.z;
    // Orient plate: face origin (gimbal).
    Vec3 to_origin{-pos.x, -pos.y, -pos.z};
    double len = norm(to_origin);
    if (len > 1e-6) {
        Vec3 out = to_origin*(-1.0/len);
        Vec3 ly{-out.y, out.x, 0}; double yl=norm(ly);
        if (yl<1e-6) ly={0,1,0}; else ly=ly*(1.0/yl);
        double rot[9]={out.x,out.y,out.z, ly.x,ly.y,ly.z, 0,0,1};
        mju_mat2Quat(d->mocap_quat+4*mocap_id, rot);
    }
}

// ── Heat ───────────────────────────────────────────────────────────────────

void cool_heat(double& h, double now, double& last) {
    h=std::max(0.0,h-kHeatCooling*std::max(0.0,now-last)); last=now;
}

// ── Gimbal helpers ─────────────────────────────────────────────────────────

void reset_cmd(hw::Command& cmd, const mjData* d, int yqp, int pqp) {
    cmd.control=true; cmd.found=true; cmd.shoot=false;
    cmd.yaw=d->qpos[yqp]; cmd.pitch=d->qpos[pqp];
    cmd.yaw_vel=cmd.pitch_vel=cmd.yaw_accel=cmd.pitch_accel=0;
}

hw::Command make_p_cmd(const AimPrediction& pred, const SimGimbalStatus& st) {
    hw::Command cmd; cmd.control=true; cmd.found=true; cmd.shoot=false;
    cmd.yaw=pred.target_yaw; cmd.pitch=std::clamp(pred.target_pitch,kPitchMin,kPitchMax);
    cmd.yaw_vel=std::clamp(kAutoYawKp*pred.yaw_error,-kMaxYawVel,kMaxYawVel);
    cmd.pitch_vel=std::clamp(kAutoPitchKp*pred.pitch_error,-kMaxPitchVel,kMaxPitchVel);
    cmd.yaw_accel=(cmd.yaw_vel-st.yaw_vel)/kRenderDt;
    cmd.pitch_accel=(cmd.pitch_vel-st.pitch_vel)/kRenderDt;
    return cmd;
}

} // namespace

// ══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    // ── CLI ─────────────────────────────────────────────────────────────
    std::string difficulty = "hard";
    std::string xml_path = "gimbal.xml";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--difficulty" || a == "-d") && i+1 < argc) difficulty = argv[++i];
        else if (a[0] != '-') xml_path = a;
    }

    if (difficulty != "easy" && difficulty != "medium" && difficulty != "hard") {
        std::fprintf(stderr, "Usage: mujocoaim [-d easy|medium|hard] [model.xml]\n");
        return 1;
    }

    // ── Create target ───────────────────────────────────────────────────
    std::unique_ptr<ITarget> target;
    if (difficulty == "easy") target = std::make_unique<TargetEasy>();
    else if (difficulty == "medium") target = std::make_unique<TargetMedium>();
    else target = std::make_unique<TargetHard>();

    // ── MuJoCo ──────────────────────────────────────────────────────────
    char err[1024]={};
    mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
    if (!m) { std::fprintf(stderr,"%s\n",err); return 1; }
    mjData* d = mj_makeData(m);

    int yj=mj_name2id(m,mjOBJ_JOINT,"yaw"), pj=mj_name2id(m,mjOBJ_JOINT,"pitch");
    int ym=mj_name2id(m,mjOBJ_ACTUATOR,"yaw_motor"), pm=mj_name2id(m,mjOBJ_ACTUATOR,"pitch_motor");
    int tb=mj_name2id(m,mjOBJ_BODY,"target"), tg=mj_name2id(m,mjOBJ_GEOM,"target_square");
    int tm=(tb>=0)?m->body_mocapid[tb]:-1;
    int ms=mj_name2id(m,mjOBJ_SITE,"muzzle_site"), ts=mj_name2id(m,mjOBJ_SITE,"target_site");
    int gc=mj_name2id(m,mjOBJ_CAMERA,"gimbal_pov");
    std::array<int,kMaxBullets> bm{};
    if (yj<0||pj<0||ym<0||pm<0||tg<0||tm<0||ms<0||ts<0||gc<0) {
        std::fprintf(stderr,"Missing MJCF names\n"); return 1; }

    int yqp=m->jnt_qposadr[yj], pqp=m->jnt_qposadr[pj];
    int yqv=m->jnt_dofadr[yj], pqv=m->jnt_dofadr[pj];
    for (int i=0;i<kMaxBullets;++i) {
        char nm[32]; std::snprintf(nm,sizeof(nm),"bullet_%d",i);
        int bi=mj_name2id(m,mjOBJ_BODY,nm);
        if (bi<0||m->body_mocapid[bi]<0) { std::fprintf(stderr,"Missing bullet %s\n",nm); return 1; }
        bm[i]=m->body_mocapid[bi];
    }

    if (!glfwInit()) { std::fprintf(stderr,"GLFW init failed\n"); return 1; }
    GLFWwindow* w=glfwCreateWindow(1200,900,"mujocoaim",nullptr,nullptr);
    if (!w) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(w); glfwSwapInterval(1);

    mjvCamera cam{}; mjvOption opt{}; mjvScene scn{}; mjrContext con{};
    mjv_defaultCamera(&cam); mjv_defaultOption(&opt);
    mjv_makeScene(m,&scn,1000); mjr_makeContext(m,&con,mjFONTSCALE_150);
    cam.type=mjCAMERA_FIXED; cam.fixedcamid=gc;

    // ── State ───────────────────────────────────────────────────────────
    double mx=0,my=0; bool mi=false;
    bool rp=false,cp=false,ap=false,ab=false,sp=false;
    int sf=0,sc=0; bool lh=false;
    double ft=-1,lt=-1,lat=-1,bh=0,lhu=0;
    std::array<Bullet,kMaxBullets> bullets{};
    for (int i=0;i<kMaxBullets;++i) bullets[i].mocap_id=bm[i];
    reset_bullets(bullets,d);
    hw::Command cmd{}; reset_cmd(cmd,d,yqp,pqp);
    glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);

    DetectLagBuffer dbuf(300);
    InterceptPredictor ipred;
    PpoPredictor ppred;
    ppred.load("src/aim_predictor.onnx");
    AimApproach aim_ap = AIM_VEL_EXTRAP;
    std::vector<PendingShot> pshots;

    target->reset();

    while (!glfwWindowShouldClose(w)) {
        // Reset
        bool rn=glfwGetKey(w,GLFW_KEY_R)==GLFW_PRESS;
        if (rn&&!rp) {
            mj_resetData(m,d); reset_cmd(cmd,d,yqp,pqp);
            cam.type=mjCAMERA_FIXED; cam.fixedcamid=gc;
            glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);
            mi=false; ab=false; ap=false; sp=false; sf=0; sc=0; lh=false;
            ft=lt=lat=-1; bh=0; lhu=0; reset_bullets(bullets,d);
            dbuf.clear(); ipred.clear(); pshots.clear();
            target->reset();
        } rp=rn;

        // Camera
        bool cn=glfwGetKey(w,GLFW_KEY_F)==GLFW_PRESS;
        if (cn&&!cp) {
            if (cam.type==mjCAMERA_FIXED) {
                cam.type=mjCAMERA_FREE; cam.fixedcamid=-1; cam.distance=4.4;
                cam.lookat[0]=1.2; cam.lookat[1]=0; cam.lookat[2]=0.4;
                cam.elevation=-18; cam.azimuth=135;
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
            } else { cam.type=mjCAMERA_FIXED; cam.fixedcamid=gc;
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED); mi=false; }
        } cp=cn;

        // Aimbot toggle
        bool an=glfwGetKey(w,GLFW_KEY_T)==GLFW_PRESS;
        if (an&&!ap) {
            ab=!ab; sf=0; sc=0; lh=false; ft=lt=lat=-1; sp=false;
            reset_bullets(bullets,d); dbuf.clear(); ipred.clear(); pshots.clear();
            if (ab) { cam.type=mjCAMERA_FIXED; cam.fixedcamid=gc;
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED); mi=false; }
        } ap=an;

        if (glfwGetKey(w,GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(w,true);

        // ── Target ──
        Vec3 tpos = target->update(d->time);
        write_target_mocap(d, tm, tpos);
        mj_forward(m,d);

        dbuf.push(d->time, target->composite_pos(), target->velocity());
        cool_heat(bh,d->time,lhu);

        SimGimbalStatus gs{};
        gs.yaw=d->qpos[yqp]; gs.yaw_vel=d->qvel[yqv];
        gs.pitch=d->qpos[pqp]; gs.pitch_vel=d->qvel[pqv];

        Vec3 obs_pos=target->composite_pos(), obs_vel=target->velocity();
        dbuf.lookup(d->time-kDetectionLagS,obs_pos,obs_vel);
        auto* mz=d->site_xpos+3*ms;
        Vec3 mpos{mz[0],mz[1],mz[2]};

        // ── Prediction ──
        AimPrediction pred{};
        if (aim_ap==AIM_INTERCEPT) {
            ipred.observe(d->time-kDetectionLagS,obs_pos);
            pred=ipred.predict(mpos,gs.yaw,gs.pitch);
        } else if (aim_ap==AIM_PPO) {
            pred.target_yaw=std::atan2(obs_pos.y-mpos.y,obs_pos.x-mpos.x);
            pred.target_pitch=std::atan2(obs_pos.z-mpos.z,norm_xy(obs_pos-mpos));
        } else {
            Vec3 cp=obs_pos+obs_vel*kDetectionLagS;
            static Vec3 pv=obs_vel; static double pt=0;
            double da=d->time-pt; Vec3 acc{};
            if (da>1e-6) { acc=(obs_vel-pv)*(1.0/da); double a=norm(acc); if(a>5)acc=acc*(5.0/a); }
            pv=obs_vel; pt=d->time;
            pred=predict_aim(mpos,cp,obs_vel,acc,kDetectionLagS+kShootDelayS,gs.yaw,gs.pitch);
        }

        // ── Mouse ──
        double nx,ny; glfwGetCursorPos(w,&nx,&ny);
        if (!mi) { mx=nx; my=ny; mi=true; }
        double dxm=nx-mx, dym=ny-my; mx=nx; my=ny;

        // Approach toggle
        { static bool yp=false; bool yn=glfwGetKey(w,GLFW_KEY_Y)==GLFW_PRESS;
          if (yn&&!yp) { aim_ap=static_cast<AimApproach>((aim_ap+1)%AIM_COUNT);
              dbuf.clear(); ipred.clear(); } yp=yn; }

        // ── Command ──
        double dyv=0, dpv=0;
        if (ab) {
            if (aim_ap==AIM_PPO&&ppred.loaded()) {
                double ts=(lat<0)?1.0:d->time-lat;
                auto pa=ppred.predict(mpos,obs_pos,gs.yaw,gs.pitch,gs.yaw_vel,gs.pitch_vel,bh,ts);
                cmd.control=true; cmd.found=true; cmd.yaw=gs.yaw; cmd.pitch=gs.pitch;
                cmd.yaw_vel=pa.yaw_vel; cmd.pitch_vel=pa.pitch_vel;
                cmd.yaw_accel=(pa.yaw_vel-gs.yaw_vel)/kRenderDt;
                cmd.pitch_accel=(pa.pitch_vel-gs.pitch_vel)/kRenderDt;
                cmd.shoot=pa.fire; dyv=pa.yaw_vel; dpv=pa.pitch_vel;
            } else if (aim_ap==AIM_INTERCEPT&&ipred.last_intercept_valid()) {
                auto mc=make_mpc_command(mpos,ipred.last_model(),pred,gs.yaw,gs.pitch,
                                         gs.yaw_vel,gs.pitch_vel,ipred.last_intercept_t());
                cmd.control=true; cmd.found=true;
                cmd.yaw=mc.yaw; cmd.yaw_vel=mc.yaw_vel; cmd.yaw_accel=mc.yaw_accel;
                cmd.pitch=mc.pitch; cmd.pitch_vel=mc.pitch_vel; cmd.pitch_accel=mc.pitch_accel;
                dyv=mc.yaw_vel; dpv=mc.pitch_vel;
            } else {
                Vec3 delta=obs_pos-mpos; double h2=delta.x*delta.x+delta.y*delta.y;
                double ffy=0,ffp=0;
                if(h2>1e-6){ ffy=(delta.x*obs_vel.y-delta.y*obs_vel.x)/h2;
                    double h=std::sqrt(h2),r2=h2+delta.z*delta.z;
                    if(r2>1e-6){ double hd=(delta.x*obs_vel.x+delta.y*obs_vel.y)/h;
                        ffp=(h*obs_vel.z-delta.z*hd)/r2; }}
                cmd=make_p_cmd(pred,gs);
                cmd.yaw_vel=std::clamp(ffy+cmd.yaw_vel,-kMaxYawVel,kMaxYawVel);
                cmd.pitch_vel=std::clamp(ffp+cmd.pitch_vel,-kMaxPitchVel,kMaxPitchVel);
                dyv=cmd.yaw_vel; dpv=cmd.pitch_vel;
            }
        } else if (cam.type==mjCAMERA_FIXED) {
            double yd=-dxm*kMouseYawSens, pd=-dym*kMousePitchSens;
            dyv=std::clamp(yd/kRenderDt,-kMaxYawVel,kMaxYawVel);
            dpv=std::clamp(pd/kRenderDt,-kMaxPitchVel,kMaxPitchVel);
            cmd.control=true; cmd.found=true;
            cmd.yaw=wrap_pi(d->qpos[yqp]+yd);
            cmd.pitch=std::clamp(d->qpos[pqp]+pd,kPitchMin,kPitchMax);
            cmd.yaw_vel=dyv; cmd.pitch_vel=dpv;
        }

        // ── Physics ──
        double fs=d->time;
        while (d->time-fs<kRenderDt) {
            d->ctrl[ym]=dyv; d->ctrl[pm]=dpv;
            d->qpos[yqp]=std::clamp(wrap_pi(d->qpos[yqp]+d->ctrl[ym]*m->opt.timestep),-2.8,2.8);
            d->qpos[pqp]=std::clamp(d->qpos[pqp]+d->ctrl[pm]*m->opt.timestep,kPitchMin,kPitchMax);
            d->qvel[yqv]=d->ctrl[ym]; d->qvel[pqv]=d->ctrl[pm];
            d->time+=m->opt.timestep; mj_forward(m,d);
        } mj_forward(m,d);

        // ── Delayed shots ──
        for (size_t i=0;i<pshots.size();) {
            auto& ps=pshots[i];
            if (d->time>=ps.trigger_time+kShootDelayS) {
                spawn_bullet_at(bullets,ps.muzzle_pos,ps.muzzle_dir,d->time,ps.shot_index);
                lh=false; pshots.erase(pshots.begin()+i);
            } else ++i;
        }

        // ── Fire ──
        bool mn=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
        bool me=cam.type==mjCAMERA_FIXED&&mn&&!sp;
        bool aa=ab&&std::abs(wrap_pi(cmd.yaw-d->qpos[yqp]))<0.015
                  &&std::abs(cmd.pitch-d->qpos[pqp])<0.015;
        double tr=norm_xy(obs_pos-mpos), tspeed=norm(target->velocity());
        bool ir=tr>=kMinRange&&tr<=kMaxRange&&tspeed<=kMaxTargetSpeed;
        bool ar=aa&&ir&&sf+pshots.size()<(size_t)kMaxShots&&bh+kHeatPerShot<=kHeatLimit
                &&(lat<0||d->time-lat>=kAutoCooldown);
        bool fn=sf+pshots.size()<(size_t)kMaxShots&&(me||ar);
        cmd.shoot=fn;
        if (fn) {
            auto* fmz=d->site_xpos+3*ms,*fmx=d->site_xmat+9*ms;
            Vec3 fpos{fmz[0],fmz[1],fmz[2]}, fdir{fmx[0],fmx[3],fmx[6]};
            if (kShootDelayS>0) pshots.push_back({d->time,fpos,fdir,sf});
            else { spawn_bullet(bullets,d,ms,sf); lh=false; }
            if (sf==0) ft=d->time; ++sf;
            if (sf==kMaxShots) lt=d->time;
            bh+=kHeatPerShot; if (ar) lat=d->time;
        } sp=mn;
        update_bullets(bullets,m,d,tb,tg,sc,lh);

        if (cam.type==mjCAMERA_FIXED&&!ab) {
            cmd.yaw=d->qpos[yqp]; cmd.pitch=d->qpos[pqp]; }

        // Free camera
        if (cam.type==mjCAMERA_FREE&&glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS) {
            if (glfwGetKey(w,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS||glfwGetKey(w,GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS)
                mjv_moveCamera(m,mjMOUSE_ZOOM,0,dym*0.01,&cam);
            else mjv_moveCamera(m,mjMOUSE_ROTATE_H,dxm*0.005,dym*0.005,&cam);
        }

        // ── Render ──
        mjrRect vp{}; glfwGetFramebufferSize(w,&vp.width,&vp.height);
        mjv_updateScene(m,d,&opt,nullptr,&cam,mjCAT_ALL,&scn);
        mjr_render(vp,&scn,&con);

        double se=(ft<0)?0:((sf>=kMaxShots?lt:d->time)-ft);
        char left[1500];
        std::snprintf(left,sizeof(left),
            "mujocoaim -d %s | Mouse:aim T:aimbot %s Y:approach[%s] F:POV/free R:reset Esc:quit\n"
            "shots %d/100 score %d last=%s heat %.0f/%.0f time %.2fs lag=%.0fms delay=%.0fms pending=%zu\n"
            "target=(%.2f,%.2f,%.2f) yaw=%+.3f pitch=%+.3f err yaw=%+.3f pitch=%+.3f\n"
            "cmd{shoot=%d yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f}\n"
            "status yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f ctrl=(%+.2f,%+.2f)",
            difficulty.c_str(), ab?"ON":"OFF", kAimNames[aim_ap], sf,sc,
            lh?"HIT":pshots.empty()?"MISS":"...", bh,kHeatLimit,se,
            kDetectionLagS*1000,kShootDelayS*1000,pshots.size(),
            obs_pos.x,obs_pos.y,obs_pos.z, pred.target_yaw,pred.target_pitch,
            pred.yaw_error,pred.pitch_error, cmd.shoot?1:0,
            cmd.yaw,cmd.yaw_vel,cmd.pitch,cmd.pitch_vel,
            d->qpos[yqp],d->qvel[yqv],d->qpos[pqp],d->qvel[pqv],
            d->ctrl[ym],d->ctrl[pm]);
        mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,left,nullptr,&con);
        glfwSwapBuffers(w); glfwPollEvents();
    }

    mjv_freeScene(&scn); mjr_freeContext(&con);
    mj_deleteData(d); mj_deleteModel(m);
    glfwDestroyWindow(w); glfwTerminate();
    return 0;
}
