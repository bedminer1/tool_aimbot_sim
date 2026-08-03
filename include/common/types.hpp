#pragma once

#include <algorithm>
#include <cmath>

// ── Shared types ───────────────────────────────────────────────────────────

struct Vec3 { double x = 0.0, y = 0.0, z = 0.0; };

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator*(const Vec3& v, double s) { return {v.x*s, v.y*s, v.z*s}; }
inline double norm_xy(const Vec3& v) { return std::hypot(v.x, v.y); }
inline double norm(const Vec3& v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }

// ── Math constants ─────────────────────────────────────────────────────────

constexpr double kPi = 3.14159265358979323846;

inline double wrap_pi(double x) {
    while (x > kPi) x -= 2.0*kPi;
    while (x < -kPi) x += 2.0*kPi;
    return x;
}

// ── Utility ────────────────────────────────────────────────────────────────

inline double smoothstep(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}
