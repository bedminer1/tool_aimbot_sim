/**
 * @file    src/aim_models/aim_predictor_intercept.cpp
 * @brief   EKF + circular motion model + intercept solver + MPC — full implementation
 *
 * @details
 * ~350 lines implementing the four-stage pipeline described in the header.
 * Key implementation details:
 *
 * EKF predict step (constant-velocity + constant-ω model):
 *   p += v·dt, θ += ω·dt, v and ω unchanged.
 *   Jacobian F is mostly identity with dt terms in position-velocity coupling.
 *
 * EKF update step (nonlinear observation):
 *   h(x) = [p_x + r·cos(θ), p_y + r·sin(θ), p_z]
 *   Jacobian H: ∂h/∂p = I, ∂h/∂θ = [-r·sin(θ), r·cos(θ), 0], rest zeros.
 *
 * Intercept solver:
 *   Finds smallest t > 0 solving |p_c(t) + r·(cos(θ(t)), sin(θ(t)))| = v_b·t.
 *   Uses a quadratic seed (ignoring the orbit term) then damped Newton.
 *   If Newton diverges (rare — small-radius orbit at typical ranges),
 *   falls back to bisection in [0, t_max].
 *
 * MPC controller:
 *   Feedforward angular velocity = model.omega (target's angular velocity).
 *   Feedback = kp × error (position-level P-controller).
 *   Total = feedforward + feedback, clamped to actuator limits.
 *
 * @see aim_predictor_intercept.hpp
 * @author  bedminer1
 * @date    2026-08-03
 */

#include "aim_models/aim_predictor_intercept.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

// ══════════════════════════════════════════════════════════════════════════════
// Circular model helpers
// ══════════════════════════════════════════════════════════════════════════════

Vec3 model_position_at(const CircularModel& m, double t)
{
    double a = m.theta0 + m.omega * t;
    return {m.p_c0.x + m.v_c.x * t + m.r * std::cos(a),
            m.p_c0.y + m.v_c.y * t + m.r * std::sin(a),
            m.p_c0.z + m.v_c.z * t};
}

Vec3 model_velocity_at(const CircularModel& m, double t)
{
    double a = m.theta0 + m.omega * t;
    return {m.v_c.x - m.r * m.omega * std::sin(a),
            m.v_c.y + m.r * m.omega * std::cos(a),
            m.v_c.z};
}

// ══════════════════════════════════════════════════════════════════════════════
// EKF
// ══════════════════════════════════════════════════════════════════════════════

CircularEkf::CircularEkf() { reset(); }

void CircularEkf::reset()
{
    std::memset(x_, 0, sizeof(x_));
    std::memset(P_, 0, sizeof(P_));
    for (int i = 0; i < N; ++i) P_[i * N + i] = 1.0;
    t_ = 0; init_ = false;
}

void CircularEkf::init(double t, const Vec3& z)
{
    x_[0]=z.x; x_[1]=z.y; x_[2]=z.z;
    x_[3]=0; x_[4]=0; x_[5]=0;
    x_[6]=0; x_[7]=10.47;
    for (int i=0;i<N*N;++i) P_[i]=0;
    P_[0*N+0]=P_[1*N+1]=P_[2*N+2]=1.0;
    P_[3*N+3]=P_[4*N+4]=P_[5*N+5]=P_[7*N+7]=100.0;
    P_[6*N+6]=1.0;
    t_=t; init_=true;
}

void CircularEkf::update(double t, const Vec3& z)
{
    if (!init_) { init(t, z); return; }
    double dt = t - t_;
    if (dt <= 0) return;
    predict(dt);
    correct(z);
    t_ = t;
}

void CircularEkf::predict(double dt)
{
    x_[0]+=x_[3]*dt; x_[1]+=x_[4]*dt; x_[2]+=x_[5]*dt;
    x_[6]+=x_[7]*dt;
    while(x_[6]>kPi) x_[6]-=2*kPi; while(x_[6]<-kPi) x_[6]+=2*kPi;

    double F[N*N]{};
    for(int i=0;i<N;++i) F[i*N+i]=1.0;
    F[0*N+3]=F[1*N+4]=F[2*N+5]=F[6*N+7]=dt;

    double FP[N*N],FT[N*N],Pn[N*N];
    mat_mult(N,N,N,F,P_,FP);
    mat_transpose(N,N,F,FT);
    mat_mult(N,N,N,FP,FT,Pn);

    const double qd[8]={q_p_,q_p_,q_p_,q_v_,q_v_,q_v_,q_th_,q_w_};
    for(int i=0;i<N;++i) Pn[i*N+i]+=qd[i]*dt;
    std::memcpy(P_,Pn,sizeof(P_));
}

void CircularEkf::correct(const Vec3& z)
{
    double c=std::cos(x_[6]), s=std::sin(x_[6]), r=kArmorR;

    double H[M*N]{};
    H[0*N+0]=1; H[0*N+6]=-r*s;
    H[1*N+1]=1; H[1*N+6]= r*c;
    H[2*N+2]=1;

    double HT[N*M]; mat_transpose(M,N,H,HT);
    double PHt[N*M]; mat_mult(N,N,M,P_,HT,PHt);
    double S[M*M];   mat_mult(M,N,M,H,PHt,S);
    for(int i=0;i<M;++i) S[i*M+i]+=r_xyz_;

    double Si[M*M];
    if(!mat_inv_3x3(S,Si)) return;

    double K[N*M]; mat_mult(N,M,M,PHt,Si,K);

    double hx[M]={x_[0]+r*c, x_[1]+r*s, x_[2]};
    double y[M]={z.x-hx[0], z.y-hx[1], z.z-hx[2]};
    for(int i=0;i<N;++i) for(int j=0;j<M;++j) x_[i]+=K[i*M+j]*y[j];
    while(x_[6]>kPi) x_[6]-=2*kPi; while(x_[6]<-kPi) x_[6]+=2*kPi;

    double KH[N*N]{}; mat_mult(N,M,N,K,H,KH);
    double IKH[N*N]{};
    for(int i=0;i<N;++i){ IKH[i*N+i]=1; for(int j=0;j<N;++j) IKH[i*N+j]-=KH[i*N+j]; }
    double Pn[N*N]; mat_mult(N,N,N,IKH,P_,Pn);
    std::memcpy(P_,Pn,sizeof(P_));
}

void CircularEkf::mat_mult(int rows, int inner, int cols, const double* A, const double* B, double* C)
{
    std::memset(C,0,rows*cols*sizeof(double));
    for(int i=0;i<rows;++i) for(int k=0;k<inner;++k){
        double aik=A[i*inner+k];
        for(int j=0;j<cols;++j) C[i*cols+j]+=aik*B[k*cols+j];
    }
}

void CircularEkf::mat_transpose(int rows, int cols, const double* A, double* AT)
{
    for(int i=0;i<rows;++i) for(int j=0;j<cols;++j) AT[j*rows+i]=A[i*cols+j];
}

bool CircularEkf::mat_inv_3x3(const double* A, double* Ai)
{
    double det=A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*A[7]-A[4]*A[6]);
    if(std::abs(det)<1e-15) return false;
    double id=1.0/det;
    Ai[0]=(A[4]*A[8]-A[5]*A[7])*id; Ai[1]=(A[2]*A[7]-A[1]*A[8])*id; Ai[2]=(A[1]*A[5]-A[2]*A[4])*id;
    Ai[3]=(A[5]*A[6]-A[3]*A[8])*id; Ai[4]=(A[0]*A[8]-A[2]*A[6])*id; Ai[5]=(A[2]*A[3]-A[0]*A[5])*id;
    Ai[6]=(A[3]*A[7]-A[4]*A[6])*id; Ai[7]=(A[1]*A[6]-A[0]*A[7])*id; Ai[8]=(A[0]*A[4]-A[1]*A[3])*id;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Analytical intercept time solver
// ══════════════════════════════════════════════════════════════════════════════

namespace {

double f_of(double t, const Vec3& pc, const Vec3& vc, double r, double th0, double om, double vb2)
{
    double a=th0+om*t;
    double px=pc.x+vc.x*t+r*std::cos(a), py=pc.y+vc.y*t+r*std::sin(a), pz=pc.z+vc.z*t;
    return px*px+py*py+pz*pz-vb2*t*t;
}

double bisect(double tl, double th, double fl, const Vec3& pc, const Vec3& vc, double r, double th0, double om, double vb2)
{
    for(int i=0;i<60;++i){
        double tm=0.5*(tl+th), fm=f_of(tm,pc,vc,r,th0,om,vb2);
        if(fm==0||(th-tl)<1e-9) return tm;
        if(fl*fm<0) th=tm; else { tl=tm; fl=fm; }
    }
    return 0.5*(tl+th);
}

} // namespace

InterceptResult solve_intercept_time(
    const Vec3& p_c0, const Vec3& v_c, double r, double theta0, double omega, double v_b, double t_max)
{
    InterceptResult out{false,0};
    constexpr double TM=1e-4;
    double vb2=v_b*v_b, dc=norm(p_c0), vcm=norm(v_c);

    double thi=t_max;
    { double dn=v_b-vcm; if(dn>1e-3) thi=std::min(t_max,1.5*(dc+r)/dn); }
    double tlo=std::max(TM,(dc-r)/(v_b+vcm)*0.5);

    double cth=std::cos(theta0), sth=std::sin(theta0);
    double p0x=p_c0.x+r*cth, p0y=p_c0.y+r*sth, p0z=p_c0.z;
    double v0x=v_c.x-r*omega*sth, v0y=v_c.y+r*omega*cth, v0z=v_c.z;
    double A=v0x*v0x+v0y*v0y+v0z*v0z-vb2;
    double B=2*(p0x*v0x+p0y*v0y+p0z*v0z);
    double C=p0x*p0x+p0y*p0y+p0z*p0z;

    double t=dc/v_b, fb=std::abs(f_of(t,p_c0,v_c,r,theta0,omega,vb2));
    double disc=B*B-4*A*C;
    if(disc>=0&&std::abs(A)>1e-12){
        double sq=std::sqrt(disc);
        for(double cand:{-B-sq,-B+sq}){ cand/=2*A;
            if(cand>TM&&cand<=thi){ double fc=std::abs(f_of(cand,p_c0,v_c,r,theta0,omega,vb2)); if(fc<fb){t=cand;fb=fc;} }
        }
    }else if(std::abs(A)<1e-12&&std::abs(B)>1e-12){
        double cand=-C/B;
        if(cand>TM&&cand<=thi){ double fc=std::abs(f_of(cand,p_c0,v_c,r,theta0,omega,vb2)); if(fc<fb){t=cand;fb=fc;} }
    }

    for(int iter=0;iter<6;++iter){
        double a=theta0+omega*t;
        double px=p_c0.x+v_c.x*t+r*std::cos(a), py=p_c0.y+v_c.y*t+r*std::sin(a), pz=p_c0.z+v_c.z*t;
        double vx=v_c.x-r*omega*std::sin(a), vy=v_c.y+r*omega*std::cos(a), vz=v_c.z;
        double f=px*px+py*py+pz*pz-vb2*t*t, fp=2*(px*vx+py*vy+pz*vz-vb2*t);
        if(std::abs(f)<1e-10*vb2) break;
        if(std::abs(fp)<1e-12) break;
        double step=f/fp, ms=0.5*std::max(t,TM);
        if(std::abs(step)>ms) step=std::copysign(ms,step);
        double tn=t-step; tn=std::clamp(tn,TM,thi);
        double fn=f_of(tn,p_c0,v_c,r,theta0,omega,vb2);
        for(int bk=0;std::abs(fn)>=std::abs(f)&&bk<5;++bk){ step*=.5; tn=t-step; tn=std::clamp(tn,TM,thi); fn=f_of(tn,p_c0,v_c,r,theta0,omega,vb2); }
        if(std::abs(tn-t)<1e-7){t=tn;break;}
        t=tn;
    }

    double ff=f_of(t,p_c0,v_c,r,theta0,omega,vb2);
    if(std::abs(ff)>1e-4*vb2){
        double tp=TM, fp=f_of(TM,p_c0,v_c,r,theta0,omega,vb2);
        for(int i=1;i<=128;++i){
            double ts=TM+(thi-TM)*i/128, fs=f_of(ts,p_c0,v_c,r,theta0,omega,vb2);
            if(fp*fs<=0){ t=bisect(tp,ts,fp,p_c0,v_c,r,theta0,omega,vb2); break; }
            tp=ts; fp=fs;
        }
        if(std::abs(f_of(t,p_c0,v_c,r,theta0,omega,vb2))>1e-4*vb2) return out;
    }

    if(t<=TM||t>t_max) return out;
    out.valid=true; out.t=t; return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// InterceptPredictor
// ══════════════════════════════════════════════════════════════════════════════

AimPrediction InterceptPredictor::predict(const Vec3& muzzle_pos, double cy, double cp)
{
    AimPrediction out{};
    last_intercept_valid_ = false;

    if(!ekf_.initialized()) {
        // Zero-th observation: aim at origin (gimbal forward), no lead.
        Vec3 delta{5,0,0};  // assume target ~5m ahead
        out.target_yaw = 0;
        double h = std::hypot(delta.x, delta.y);
        out.target_pitch = std::atan2(delta.z, h);
        out.yaw_error = -cy;
        out.pitch_error = out.target_pitch - cp;
        return out;
    }

    CircularModel m{};
    m.p_c0 = ekf_.center_pos();
    m.v_c  = ekf_.center_vel();
    m.r    = ekf_.radius();
    m.theta0 = ekf_.phase();
    m.omega  = ekf_.omega();

    CircularModel shifted = m;
    shifted.p_c0 = m.p_c0 - muzzle_pos;

    InterceptResult ir = solve_intercept_time(
        shifted.p_c0, shifted.v_c, m.r, m.theta0, m.omega, kBulletSpeed, 1.5);

    if(!ir.valid) {
        return predict_aim(muzzle_pos,
            m.p_c0 + Vec3{m.r*std::cos(m.theta0), m.r*std::sin(m.theta0), 0},
            m.v_c, Vec3{0,0,0}, 0.0, cy, cp);
    }

    // ── Iterative ballistic TOF refinement ────────────────────────────
    // The flat-trajectory intercept time ignores the gravity arc, so the
    // bullet arrives later than predicted.  Refine: compute ballistic pitch,
    // recompute TOF from arc length, predict target at corrected time.
    double t_int = ir.t;
    Vec3 target = model_position_at(m, t_int);
    Vec3 delta = target - muzzle_pos;
    double yaw = std::atan2(delta.y, delta.x);
    double pitch;
    if(!ballistic_pitch(norm_xy(delta), delta.z, pitch))
        pitch = std::atan2(delta.z, norm_xy(delta));

    for (int iter = 0; iter < 4; ++iter) {
        // Time-of-flight with ballistic arc: horizontal / (v·cos(θ)).
        double tof = norm_xy(delta) / std::max(0.1, kBulletSpeed * std::cos(pitch));
        // Sanity: don't let TOF shrink below the flat-trajectory estimate.
        tof = std::max(tof, ir.t);
        t_int = tof;
        target = model_position_at(m, t_int);
        delta = target - muzzle_pos;
        yaw = std::atan2(delta.y, delta.x);
        if(!ballistic_pitch(norm_xy(delta), delta.z, pitch))
            pitch = std::atan2(delta.z, norm_xy(delta));
    }

    last_model_ = m;
    last_intercept_t_ = t_int;
    last_intercept_valid_ = true;

    out.target_yaw = yaw;
    out.target_pitch = pitch;
    out.yaw_error = yaw - cy;
    while(out.yaw_error > kPi) out.yaw_error -= 2*kPi;
    while(out.yaw_error <-kPi) out.yaw_error += 2*kPi;
    out.pitch_error = pitch - cp;
    out.predicted_target_pos = target;
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// MPC controller
// ══════════════════════════════════════════════════════════════════════════════

MpcCommand make_mpc_command(
    const Vec3& muzzle_pos, const CircularModel& model, const AimPrediction& pred,
    double current_yaw, double current_pitch,
    double current_yaw_vel, double current_pitch_vel,
    double intercept_t, const MpcConfig& cfg)
{
    MpcCommand out{};

    Vec3 target = model_position_at(model, intercept_t);
    Vec3 delta = target - muzzle_pos;
    double target_yaw = std::atan2(delta.y, delta.x);
    double target_pitch = pred.target_pitch;  // ballistic-corrected

    // Feedforward: angular velocity of target at intercept time.
    Vec3 tvel = model_velocity_at(model, intercept_t);
    double h = norm_xy(delta), h2 = h*h;

    double ff_yaw = 0;
    if(h2 > 1e-6) ff_yaw = (delta.x*tvel.y - delta.y*tvel.x) / h2;

    double ff_pitch = 0;
    double r2 = h2 + delta.z*delta.z;
    if(r2 > 1e-6) {
        double hdot = (delta.x*tvel.x + delta.y*tvel.y) / std::max(1e-6, h);
        ff_pitch = (h*tvel.z - delta.z*hdot) / r2;
    }

    double ye = target_yaw - current_yaw;
    while(ye > kPi) ye -= 2*kPi; while(ye < -kPi) ye += 2*kPi;
    double pe = std::clamp(target_pitch, kPitchMin, kPitchMax) - current_pitch;

    out.yaw_vel   = std::clamp(ff_yaw + cfg.kp_yaw*ye, -cfg.max_yaw_vel, cfg.max_yaw_vel);
    out.pitch_vel = std::clamp(ff_pitch + cfg.kp_pitch*pe, -cfg.max_pitch_vel, cfg.max_pitch_vel);
    out.yaw    = target_yaw;
    out.pitch  = std::clamp(target_pitch, kPitchMin, kPitchMax);
    out.yaw_accel   = (out.yaw_vel - current_yaw_vel) / cfg.dt;
    out.pitch_accel = (out.pitch_vel - current_pitch_vel) / cfg.dt;
    return out;
}
