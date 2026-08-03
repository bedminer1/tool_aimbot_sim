/**
 * @file    src/cli/main.cpp
 * @brief   mujocoaim — RoboMaster gimbal aimbot simulation (MuJoCo + GLFW)
 *
 * @details
 * Interactive simulation for developing and benchmarking gimbal aimbot
 * approaches against a RoboMaster-style chassis target.
 *
 * Architecture:
 *   - MuJoCo physics (gimbal.xml): yaw/pitch gimbal + 4-plate chassis target
 *   - Swappable target models (TargetEasy/Medium/Hard): runtime toggled with G key
 *   - Three aiming approaches (VelExtrap, Intercept+MPC, PPO): toggled with Y key
 *   - GLFW window: mouse-aimed free camera or gimbal-POV fixed camera (F key)
 *   - Bullet pool: 100 pre-allocated mocap bodies, ballistic physics
 *   - Barrel heat: RMUC sentry heat model (260 J limit, +10/shot, −30 J/s)
 *   - Hit detection: segment-vs-expanded-box across all 4 armor plates
 *   - Detection lag buffer + shooting delay (15 ms + 30 ms realism knobs)
 *
 * Key bindings:
 *   T     — toggle aimbot on/off
 *   Y     — cycle aiming approach (VelExtrap → Intercept+MPC → PPO)
 *   G     — cycle target difficulty (easy → medium → hard)
 *   F     — toggle between gimbal POV and free camera
 *   R     — reset (score, heat, bullets, target)
 *   Esc   — quit
 *   Mouse — free-look (free camera) or aim (gimbal POV)
 *
 * Launch:
 *   mujocoaim -d easy|medium|hard [gimbal.xml]
 *
 * @see gimbal.xml, target_models/, aim_models/
 * @author  bedminer1
 * @date    2026-08-03
 */

// mujocoaim — RoboMaster gimbal aimbot sim with swappable targets and aimers.
//
// Usage: mujocoaim [-d easy|medium|hard] [gimbal.xml]
// Keys:  T=aimbot, Y=approach, G=difficulty, F=camera, R=reset, Esc=quit

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
#include "common/sim_constants.hpp"
#include "aim_models/aim_constants.hpp"
#include "aim_models/aim_predictor.hpp"
#include "aim_models/aim_predictor_intercept.hpp"
#include "aim_models/aim_predictor_ppo.hpp"
#include "target_models/target_interface.hpp"
#include "target_models/target_easy.hpp"
#include "target_models/target_medium.hpp"
#include "target_models/target_hard.hpp"

namespace
{
// ── UI / mouse (not in sim_constants) ──────────────────────────────────────
constexpr double kMouseYawSens = 0.0025, kMousePitchSens = 0.0020;
constexpr double kAutoYawKp = 9.0, kAutoPitchKp = 9.0;
constexpr double kMinRange = 3.0, kMaxRange = 7.0, kMaxTargetSpeed = 3.5;

enum AimApproach { AIM_VEL_EXTRAP=0, AIM_INTERCEPT, AIM_PPO, AIM_COUNT };
constexpr const char* kAimNames[] = {"VelExtrap","Intercept+MPC","PPO"};
constexpr const char* kDiffNames[] = {"easy","medium","hard"};

struct SimGimbalStatus { double yaw, yaw_vel, pitch, pitch_vel; };

struct Bullet {
    bool active=false, scored=false; double spawn_time=0; int mocap_id=-1;
    Vec3 prev_pos{}, pos{}, vel{};
};
struct PendingShot { double trigger_time; Vec3 muzzle_pos, muzzle_dir; int shot_index; };

void hide_bullet(mjData* d, int id) {
    d->mocap_pos[3*id+0]=0;d->mocap_pos[3*id+1]=0;d->mocap_pos[3*id+2]=-10;
    d->mocap_quat[4*id+0]=1;d->mocap_quat[4*id+1]=0;d->mocap_quat[4*id+2]=0;d->mocap_quat[4*id+3]=0;
}
void reset_bullets(std::array<Bullet,kMaxBullets>& bs, mjData* d) {
    for(auto& b:bs){b.active=b.scored=false;b.spawn_time=0;b.prev_pos=b.pos=b.vel={};
        if(b.mocap_id>=0)hide_bullet(d,b.mocap_id);}
}
Vec3 bullet_pos_at(const Bullet& b, double t) {
    double dt=t-b.spawn_time;
    return {b.pos.x+b.vel.x*dt,b.pos.y+b.vel.y*dt,b.pos.z+b.vel.z*dt-0.5*kGravity*dt*dt};
}
void spawn_bullet(std::array<Bullet,kMaxBullets>& bs, const mjData* d, int ms, int idx) {
    Bullet& b=bs[idx%kMaxBullets];
    auto*mz=d->site_xpos+3*ms,*mx=d->site_xmat+9*ms;
    b.active=true;b.scored=false;b.spawn_time=d->time;b.pos={mz[0],mz[1],mz[2]};
    b.prev_pos=b.pos;b.vel={mx[0],mx[3],mx[6]};b.vel=b.vel*kBulletSpeed;
}
void spawn_bullet_at(std::array<Bullet,kMaxBullets>& bs, const Vec3& p, const Vec3& dir,
                     double t, int idx) {
    Bullet& b=bs[idx%kMaxBullets];
    b.active=true;b.scored=false;b.spawn_time=t;b.pos=p;b.prev_pos=p;b.vel=dir*kBulletSpeed;
}
double ddot(const double* a, const Vec3& v){return a[0]*v.x+a[1]*v.y+a[2]*v.z;}
bool segment_hits_box(const mjModel* m, const mjData* d, const Vec3& a, const Vec3& b,
                      int body_id, int geom_id) {
    auto*bp=d->xpos+3*body_id,*bx=d->xmat+9*body_id,*half=m->geom_size+3*geom_id;
    Vec3 da{a.x-bp[0],a.y-bp[1],a.z-bp[2]},db{b.x-bp[0],b.y-bp[1],b.z-bp[2]};
    double av[3]={ddot(bx,da),ddot(bx+3,da),ddot(bx+6,da)};
    double bv[3]={ddot(bx,db),ddot(bx+3,db),ddot(bx+6,db)};
    double hv[3]={half[0]+kBulletRadius,half[1]+kBulletRadius,half[2]+kBulletRadius};
    double tmin=0,tmax=1;
    for(int i=0;i<3;++i){
        double dir=bv[i]-av[i];
        if(std::abs(dir)<1e-9){if(av[i]<-hv[i]||av[i]>hv[i])return false;continue;}
        double t1=(-hv[i]-av[i])/dir,t2=(hv[i]-av[i])/dir;
        if(t1>t2)std::swap(t1,t2); tmin=std::max(tmin,t1);tmax=std::min(tmax,t2);
        if(tmin>tmax)return false;
    }
    return true;
}
void update_bullets(std::array<Bullet,kMaxBullets>& bs, const mjModel* m, mjData* d,
                    int target_body, const int* armor_ids, int& score, bool& last_hit) {
    for(auto& b:bs){
        if(!b.active)continue;
        Vec3 cur=bullet_pos_at(b,d->time);
        if(!b.scored){
            for(int i=0;i<kNumArmorPlates;++i)
                if(segment_hits_box(m,d,b.prev_pos,cur,target_body,armor_ids[i])){
                    b.scored=true;b.active=false;++score;last_hit=true;
                    if(b.mocap_id>=0)hide_bullet(d,b.mocap_id);break;
                }
            if(b.scored)continue;
        }
        b.prev_pos=cur;
        if(b.mocap_id>=0){d->mocap_pos[3*b.mocap_id+0]=cur.x;
            d->mocap_pos[3*b.mocap_id+1]=cur.y;d->mocap_pos[3*b.mocap_id+2]=cur.z;}
        if(cur.z<-1||d->time-b.spawn_time>3){b.active=false;
            if(b.mocap_id>=0)hide_bullet(d,b.mocap_id);}
    }
}
void write_target_mocap(mjData* d, int id, const Vec3& pos, double yaw) {
    d->mocap_pos[3*id+0]=pos.x;d->mocap_pos[3*id+1]=pos.y;d->mocap_pos[3*id+2]=pos.z;
    double c=std::cos(yaw), s=std::sin(yaw);
    double rot[9]={c,s,0,-s,c,0,0,0,1};
    mju_mat2Quat(d->mocap_quat+4*id,rot);
}
void cool_heat(double& h, double now, double& last) {
    h=std::max(0.0,h-kHeatCooling*std::max(0.0,now-last));last=now;
}
void reset_cmd(hw::Command& cmd, const mjData* d, int yqp, int pqp) {
    cmd.control=true;cmd.found=true;cmd.shoot=false;
    cmd.yaw=d->qpos[yqp];cmd.pitch=d->qpos[pqp];
    cmd.yaw_vel=cmd.pitch_vel=cmd.yaw_accel=cmd.pitch_accel=0;
}
hw::Command make_p_cmd(const AimPrediction& pred, const SimGimbalStatus& st) {
    hw::Command cmd;cmd.control=true;cmd.found=true;cmd.shoot=false;
    cmd.yaw=pred.target_yaw;cmd.pitch=std::clamp(pred.target_pitch,kPitchMin,kPitchMax);
    cmd.yaw_vel=std::clamp(kAutoYawKp*pred.yaw_error,-kMaxYawVel,kMaxYawVel);
    cmd.pitch_vel=std::clamp(kAutoPitchKp*pred.pitch_error,-kMaxPitchVel,kMaxPitchVel);
    cmd.yaw_accel=(cmd.yaw_vel-st.yaw_vel)/kRenderDt;
    cmd.pitch_accel=(cmd.pitch_vel-st.pitch_vel)/kRenderDt;
    return cmd;
}
} // namespace

int main(int argc, char** argv) {
    std::string diff="hard", xml="gimbal.xml";
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if((a=="--difficulty"||a=="-d")&&i+1<argc)diff=argv[++i];
        else if(a[0]!='-')xml=a;}
    int di=(diff=="easy")?0:(diff=="medium")?1:2;

    auto make_target=[&]()->std::unique_ptr<ITarget>{
        if(di==0)return std::make_unique<TargetEasy>();
        if(di==1)return std::make_unique<TargetMedium>();
        return std::make_unique<TargetHard>();
    };
    auto target=make_target();

    char err[1024]={};
    mjModel* m=mj_loadXML(xml.c_str(),nullptr,err,sizeof(err));
    if(!m){std::fprintf(stderr,"%s\n",err);return 1;}
    mjData* d=mj_makeData(m);

    int yj=mj_name2id(m,mjOBJ_JOINT,"yaw"),pj=mj_name2id(m,mjOBJ_JOINT,"pitch");
    int ym=mj_name2id(m,mjOBJ_ACTUATOR,"yaw_motor"),pm=mj_name2id(m,mjOBJ_ACTUATOR,"pitch_motor");
    int tb=mj_name2id(m,mjOBJ_BODY,"target");
    int armor_ids[kNumArmorPlates];
    for(int i=0;i<kNumArmorPlates;++i){char nm[16];std::snprintf(nm,sizeof(nm),"armor_%d",i);
        armor_ids[i]=mj_name2id(m,mjOBJ_GEOM,nm);
        if(armor_ids[i]<0){std::fprintf(stderr,"Missing armor_%d\n",i);return 1;}}
    int tm=(tb>=0)?m->body_mocapid[tb]:-1;
    int ms=mj_name2id(m,mjOBJ_SITE,"muzzle_site"),gc=mj_name2id(m,mjOBJ_CAMERA,"gimbal_pov");
    if(yj<0||pj<0||tm<0||ms<0||gc<0){std::fprintf(stderr,"Missing MJCF names\n");return 1;}
    std::array<int,kMaxBullets> bm{};
    for(int i=0;i<kMaxBullets;++i){char nm[32];std::snprintf(nm,sizeof(nm),"bullet_%d",i);
        int bi=mj_name2id(m,mjOBJ_BODY,nm);
        if(bi<0||m->body_mocapid[bi]<0){std::fprintf(stderr,"Missing %s\n",nm);return 1;}
        bm[i]=m->body_mocapid[bi];}

    int yqp=m->jnt_qposadr[yj],pqp=m->jnt_qposadr[pj],yqv=m->jnt_dofadr[yj],pqv=m->jnt_dofadr[pj];

    if(!glfwInit())return 1;
    GLFWwindow* w=glfwCreateWindow(1200,900,"mujocoaim",nullptr,nullptr);
    if(!w){glfwTerminate();return 1;}
    glfwMakeContextCurrent(w);glfwSwapInterval(1);

    mjvCamera cam{};mjvOption opt{};mjvScene scn{};mjrContext con{};
    mjv_defaultCamera(&cam);mjv_defaultOption(&opt);
    mjv_makeScene(m,&scn,1000);mjr_makeContext(m,&con,mjFONTSCALE_150);
    cam.type=mjCAMERA_FIXED;cam.fixedcamid=gc;

    double mx=0,my=0;bool mi=false;
    bool rp=false,cp=false,ap=false,ab=false,sp=false;
    int sf=0,sc=0;bool lh=false;
    double ft=-1,lt=-1,lat=-1,bh=0,lhu=0;
    std::array<Bullet,kMaxBullets> bullets{};
    for(int i=0;i<kMaxBullets;++i)bullets[i].mocap_id=bm[i];
    reset_bullets(bullets,d);
    hw::Command cmd{};reset_cmd(cmd,d,yqp,pqp);
    glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);

    DetectLagBuffer dbuf(300);
    InterceptPredictor ipred;
    PpoPredictor ppred;ppred.load("src/aim_predictor.onnx");
    AimApproach aim_ap=AIM_VEL_EXTRAP;
    std::vector<PendingShot> pshots;

    target->reset();
    bool gp=false; // G key previous state

    while(!glfwWindowShouldClose(w)){
        bool rn=glfwGetKey(w,GLFW_KEY_R)==GLFW_PRESS;
        if(rn&&!rp){
            mj_resetData(m,d);reset_cmd(cmd,d,yqp,pqp);
            cam.type=mjCAMERA_FIXED;cam.fixedcamid=gc;
            glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);
            mi=false;ab=false;ap=false;sp=false;sf=0;sc=0;lh=false;
            ft=lt=lat=-1;bh=0;lhu=0;reset_bullets(bullets,d);
            dbuf.clear();ipred.clear();pshots.clear();target->reset();
        }rp=rn;

        bool cn=glfwGetKey(w,GLFW_KEY_F)==GLFW_PRESS;
        if(cn&&!cp){
            if(cam.type==mjCAMERA_FIXED){cam.type=mjCAMERA_FREE;cam.fixedcamid=-1;
                cam.distance=4.4;cam.lookat[0]=1.2;cam.lookat[1]=0;cam.lookat[2]=0.4;
                cam.elevation=-18;cam.azimuth=135;
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_NORMAL);}
            else{cam.type=mjCAMERA_FIXED;cam.fixedcamid=gc;
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);mi=false;}
        }cp=cn;

        // Difficulty toggle (G key)
        bool gn=glfwGetKey(w,GLFW_KEY_G)==GLFW_PRESS;
        if(gn&&!gp){
            di=(di+1)%3;target=make_target();target->reset();
            sf=0;sc=0;lh=false;ft=lt=lat=-1;sp=false;
            reset_bullets(bullets,d);dbuf.clear();ipred.clear();pshots.clear();
        }gp=gn;

        bool an=glfwGetKey(w,GLFW_KEY_T)==GLFW_PRESS;
        if(an&&!ap){
            ab=!ab;sf=0;sc=0;lh=false;ft=lt=lat=-1;sp=false;
            reset_bullets(bullets,d);dbuf.clear();ipred.clear();pshots.clear();
            if(ab){cam.type=mjCAMERA_FIXED;cam.fixedcamid=gc;
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);mi=false;}
        }ap=an;
        if(glfwGetKey(w,GLFW_KEY_ESCAPE)==GLFW_PRESS)glfwSetWindowShouldClose(w,true);

        auto ts=target->update(d->time);
        write_target_mocap(d,tm,ts.pos,ts.yaw);
        mj_forward(m,d);

        dbuf.push(d->time,target->position(),target->velocity());
        cool_heat(bh,d->time,lhu);

        SimGimbalStatus gs{};
        gs.yaw=d->qpos[yqp];gs.yaw_vel=d->qvel[yqv];
        gs.pitch=d->qpos[pqp];gs.pitch_vel=d->qvel[pqv];

        Vec3 obs_pos=target->position(),obs_vel=target->velocity();
        dbuf.lookup(d->time-kDetectionLagS,obs_pos,obs_vel);
        auto*mz=d->site_xpos+3*ms;Vec3 mpos{mz[0],mz[1],mz[2]};

        AimPrediction pred{};
        if(aim_ap==AIM_INTERCEPT){ipred.observe(d->time-kDetectionLagS,obs_pos);
            pred=ipred.predict(mpos,gs.yaw,gs.pitch);}
        else if(aim_ap==AIM_PPO){pred.target_yaw=std::atan2(obs_pos.y-mpos.y,obs_pos.x-mpos.x);
            pred.target_pitch=std::atan2(obs_pos.z-mpos.z,norm_xy(obs_pos-mpos));}
        else{Vec3 cp=obs_pos+obs_vel*kDetectionLagS;
            static Vec3 pv=obs_vel;static double pt=0;
            double da=d->time-pt;Vec3 acc{};
            if(da>1e-6){acc=(obs_vel-pv)*(1.0/da);double a=norm(acc);if(a>5)acc=acc*(5.0/a);}
            pv=obs_vel;pt=d->time;
            pred=predict_aim(mpos,cp,obs_vel,acc,kDetectionLagS+kShootDelayS,gs.yaw,gs.pitch);}

        double nx,ny;glfwGetCursorPos(w,&nx,&ny);
        if(!mi){mx=nx;my=ny;mi=true;}
        double dxm=nx-mx,dym=ny-my;mx=nx;my=ny;

        {static bool yp=false;bool yn=glfwGetKey(w,GLFW_KEY_Y)==GLFW_PRESS;
         if(yn&&!yp){aim_ap=static_cast<AimApproach>((aim_ap+1)%AIM_COUNT);
             dbuf.clear();ipred.clear();}yp=yn;}

        double dyv=0,dpv=0;
        if(ab){
        	if(aim_ap==AIM_PPO&&ppred.loaded()){
                double ts=(lat<0)?0.0:d->time-lat;
                auto pa=ppred.predict(mpos,obs_pos,gs.yaw,gs.pitch,gs.yaw_vel,gs.pitch_vel,bh,ts);
                cmd.control=true;cmd.found=true;cmd.yaw=gs.yaw;cmd.pitch=gs.pitch;
                cmd.yaw_vel=pa.yaw_vel;cmd.pitch_vel=pa.pitch_vel;
                cmd.yaw_accel=(pa.yaw_vel-gs.yaw_vel)/kRenderDt;
                cmd.pitch_accel=(pa.pitch_vel-gs.pitch_vel)/kRenderDt;
                cmd.shoot=pa.fire;dyv=pa.yaw_vel;dpv=pa.pitch_vel;}
        	else if(aim_ap==AIM_INTERCEPT&&ipred.last_intercept_valid()){
                auto mc=make_mpc_command(mpos,ipred.last_model(),pred,gs.yaw,gs.pitch,
                                         gs.yaw_vel,gs.pitch_vel,ipred.last_intercept_t());
                cmd.control=true;cmd.found=true;cmd.yaw=mc.yaw;cmd.yaw_vel=mc.yaw_vel;
                cmd.yaw_accel=mc.yaw_accel;cmd.pitch=mc.pitch;cmd.pitch_vel=mc.pitch_vel;
                cmd.pitch_accel=mc.pitch_accel;dyv=mc.yaw_vel;dpv=mc.pitch_vel;}
        	else{Vec3 delta=obs_pos-mpos;double h2=delta.x*delta.x+delta.y*delta.y;
                double ffy=0,ffp=0;
                if(h2>1e-6){ffy=(delta.x*obs_vel.y-delta.y*obs_vel.x)/h2;
                    double h=std::sqrt(h2),r2=h2+delta.z*delta.z;
                    if(r2>1e-6){double hd=(delta.x*obs_vel.x+delta.y*obs_vel.y)/h;
                        ffp=(h*obs_vel.z-delta.z*hd)/r2;}}
                cmd=make_p_cmd(pred,gs);
                cmd.yaw_vel=std::clamp(ffy+cmd.yaw_vel,-kMaxYawVel,kMaxYawVel);
                cmd.pitch_vel=std::clamp(ffp+cmd.pitch_vel,-kMaxPitchVel,kMaxPitchVel);
                dyv=cmd.yaw_vel;dpv=cmd.pitch_vel;}}
        else if(cam.type==mjCAMERA_FIXED){
            double yd=-dxm*kMouseYawSens,pd=-dym*kMousePitchSens;
            dyv=std::clamp(yd/kRenderDt,-kMaxYawVel,kMaxYawVel);
            dpv=std::clamp(pd/kRenderDt,-kMaxPitchVel,kMaxPitchVel);
            cmd.control=true;cmd.found=true;cmd.yaw=wrap_pi(d->qpos[yqp]+yd);
            cmd.pitch=std::clamp(d->qpos[pqp]+pd,kPitchMin,kPitchMax);
            cmd.yaw_vel=dyv;cmd.pitch_vel=dpv;}

        double fs=d->time;
        while(d->time-fs<kRenderDt){d->ctrl[ym]=dyv;d->ctrl[pm]=dpv;
            d->qpos[yqp]=std::clamp(wrap_pi(d->qpos[yqp]+d->ctrl[ym]*m->opt.timestep),-2.8,2.8);
            d->qpos[pqp]=std::clamp(d->qpos[pqp]+d->ctrl[pm]*m->opt.timestep,kPitchMin,kPitchMax);
            d->qvel[yqv]=d->ctrl[ym];d->qvel[pqv]=d->ctrl[pm];
            d->time+=m->opt.timestep;mj_forward(m,d);}
        mj_forward(m,d);

        for(size_t i=0;i<pshots.size();){auto&ps=pshots[i];
            if(d->time>=ps.trigger_time+kShootDelayS){spawn_bullet_at(bullets,ps.muzzle_pos,
                ps.muzzle_dir,d->time,ps.shot_index);lh=false;pshots.erase(pshots.begin()+i);}
            else ++i;}

        bool mn=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
        bool me=cam.type==mjCAMERA_FIXED&&mn&&!sp;
        bool aa=ab&&std::abs(wrap_pi(cmd.yaw-d->qpos[yqp]))<0.015
                  &&std::abs(cmd.pitch-d->qpos[pqp])<0.015;
        double tr=norm_xy(obs_pos-mpos),tspeed=norm(target->velocity());
        bool ir=tr>=kMinRange&&tr<=kMaxRange&&tspeed<=kMaxTargetSpeed;
        bool ar=aa&&ir&&sf+pshots.size()<(size_t)kMaxShots&&bh+kHeatPerShot<=kHeatLimit
                &&(lat<0||d->time-lat>=kAutoCooldown);
        bool fn=sf+pshots.size()<(size_t)kMaxShots&&(me||ar);
        if(aim_ap==AIM_PPO){
            // PPO learned WHEN to fire — use its decision, safety-limit only.
            cmd.shoot = cmd.shoot && sf+pshots.size()<(size_t)kMaxShots
                        && bh+kHeatPerShot<=kHeatLimit
                        && (lat<0||d->time-lat>=kAutoCooldown);
        } else {
            cmd.shoot=fn;
        }
        if(cmd.shoot){auto*fmz=d->site_xpos+3*ms,*fmx=d->site_xmat+9*ms;
            Vec3 fpos{fmz[0],fmz[1],fmz[2]},fdir{fmx[0],fmx[3],fmx[6]};
            if(kShootDelayS>0)pshots.push_back({d->time,fpos,fdir,sf});
            else{spawn_bullet(bullets,d,ms,sf);lh=false;}
            if(sf==0)ft=d->time;++sf;if(sf==kMaxShots)lt=d->time;
            bh+=kHeatPerShot;if(ar)lat=d->time;}
        sp=mn;
        update_bullets(bullets,m,d,tb,armor_ids,sc,lh);
        if(cam.type==mjCAMERA_FIXED&&!ab){cmd.yaw=d->qpos[yqp];cmd.pitch=d->qpos[pqp];}

        if(cam.type==mjCAMERA_FREE&&glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS){
            if(glfwGetKey(w,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS||glfwGetKey(w,GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS)
                mjv_moveCamera(m,mjMOUSE_ZOOM,0,dym*0.01,&cam);
            else mjv_moveCamera(m,mjMOUSE_ROTATE_H,dxm*0.005,dym*0.005,&cam);}

        mjrRect vp{};glfwGetFramebufferSize(w,&vp.width,&vp.height);
        mjv_updateScene(m,d,&opt,nullptr,&cam,mjCAT_ALL,&scn);mjr_render(vp,&scn,&con);

        double se=(ft<0)?0:((sf>=kMaxShots?lt:d->time)-ft);
        char left[1500];
        std::snprintf(left,sizeof(left),
            "mujocoaim -d %s | G:difficulty T:aimbot %s Y:approach[%s] F:POV R:reset Esc:quit\n"
            "shots %d/100 score %d last=%s heat %.0f/%.0f time %.2fs lag=%.0fms delay=%.0fms\n"
            "target=(%.2f,%.2f,%.2f) yaw=%+.3f pitch=%+.3f err yaw=%+.3f pitch=%+.3f\n"
            "cmd{shoot=%d yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f}\n"
            "status yaw=%+.3f yaw_vel=%+.3f pitch=%+.3f pitch_vel=%+.3f ctrl=(%+.2f,%+.2f)",
            kDiffNames[di],ab?"ON":"OFF",kAimNames[aim_ap],sf,sc,
            lh?"HIT":pshots.empty()?"MISS":"...",bh,kHeatLimit,se,
            kDetectionLagS*1000,kShootDelayS*1000,
            obs_pos.x,obs_pos.y,obs_pos.z,pred.target_yaw,pred.target_pitch,
            pred.yaw_error,pred.pitch_error,cmd.shoot?1:0,
            cmd.yaw,cmd.yaw_vel,cmd.pitch,cmd.pitch_vel,
            d->qpos[yqp],d->qvel[yqv],d->qpos[pqp],d->qvel[pqv],
            d->ctrl[ym],d->ctrl[pm]);
        mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,left,nullptr,&con);
        glfwSwapBuffers(w);glfwPollEvents();}

    mjv_freeScene(&scn);mjr_freeContext(&con);
    mj_deleteData(d);mj_deleteModel(m);glfwDestroyWindow(w);glfwTerminate();
    return 0;
}
