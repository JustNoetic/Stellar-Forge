#pragma once

#include "core/constants.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <utility>

namespace sf::math {

struct StateVector {
    glm::dvec3 pos{0.0};
    glm::dvec3 vel{0.0};
};

inline std::pair<double, double> stumpff_cs(double z) noexcept {
    if (std::abs(z) < 0.1) {
        double c = 0.5;
        double s = 1.0 / 6.0;
        double tc = c;
        double ts = s;
        for (int k = 1; k < 9; ++k) {
            tc *= -z / ((2 * k + 1) * (2 * k + 2));
            ts *= -z / ((2 * k + 2) * (2 * k + 3));
            c += tc;
            s += ts;
        }
        return {c, s};
    }
    if (z > 0.0) {
        double u = std::sqrt(z);
        double half_sin = std::sin(0.5 * u);
        return {2.0 * half_sin * half_sin / z, (u - std::sin(u)) / (u * u * u)};
    }
    double u = std::sqrt(-z);
    double half_sinh = std::sinh(0.5 * u);
    return {2.0 * half_sinh * half_sinh / (-z), (std::sinh(u) - u) / (u * u * u)};
}

inline std::pair<double, double> universal_kepler_value(double x, double alpha, double beta) noexcept {
    double z = alpha * x * x;
    if (z < -490000.0) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        return {inf, inf};
    }
    auto [c, s] = stumpff_cs(z);
    double value = beta * x * x * c + (1.0 - alpha) * x * x * x * s + x;
    double derivative = beta * x * (1.0 - z * s) + (1.0 - alpha) * x * x * c + 1.0;
    return {value, derivative};
}

inline StateVector propagate_conic_state(const glm::dvec3& rel_pos, const glm::dvec3& rel_vel, double mu, double dt) noexcept {
    double r0 = glm::length(rel_pos);
    if (dt == 0.0 || r0 <= 0.0) {
        return {rel_pos, rel_vel};
    }
    if (mu <= 0.0) {
        return {rel_pos + dt * rel_vel, rel_vel};
    }
    double speed = std::sqrt(mu / r0);
    double scale = r0 / speed;
    double alpha = 2.0 - glm::dot(rel_vel, rel_vel) / (speed * speed);
    double tau = dt / scale;
    if (alpha > 0.0) {
        double period = TWO_PI / (alpha * std::sqrt(alpha));
        if (std::abs(tau) > 0.5 * period) {
            tau = (tau > 0.0 ? 1.0 : -1.0) * std::fmod(std::abs(tau), period);
            if (tau > 0.5 * period) tau -= period;
            else if (tau < -0.5 * period) tau += period;
        }
    }
    if (tau == 0.0) {
        return {rel_pos, rel_vel};
    }
    double direction = tau > 0.0 ? 1.0 : -1.0;
    double target = std::abs(tau);
    double beta = direction * glm::dot(rel_pos, rel_vel) / (r0 * speed);
    double lo = 0.0;
    double hi = target;
    if (alpha < 0.0) {
        hi = std::min(hi, std::max(1.0, 2.0 * std::log1p(target) / std::sqrt(-alpha)));
    }
    for (int i = 0; i < 128; ++i) {
        auto [val, _] = universal_kepler_value(hi, alpha, beta);
        if (val >= target) break;
        hi *= 2.0;
    }
    double x = 0.5 * hi;
    for (int i = 0; i < 128; ++i) {
        auto [val, deriv] = universal_kepler_value(x, alpha, beta);
        double err = val - target;
        if (std::abs(err) <= 4e-14 * target) break;
        if (err > 0.0) hi = x;
        else lo = x;
        double candidate = deriv > 0.0 ? (x - err / deriv) : std::numeric_limits<double>::infinity();
        if (!std::isfinite(candidate) || candidate <= lo || candidate >= hi) {
            candidate = 0.5 * (lo + hi);
        }
        x = candidate;
    }
    x *= direction;
    double z = alpha * x * x;
    auto [c, s] = stumpff_cs(z);
    double f = 1.0 - x * x * c;
    double g = scale * (tau - x * x * x * s);
    glm::dvec3 out_pos = f * rel_pos + g * rel_vel;
    double radius = glm::length(out_pos);
    if (radius <= 0.0) {
        return {out_pos, rel_vel};
    }
    double fdot = speed * x * (z * s - 1.0) / radius;
    double gdot = 1.0 - r0 * x * x * c / radius;
    return {out_pos, fdot * rel_pos + gdot * rel_vel};
}

inline double conic_time_at_angle(double sl_p, double eccentricity, double theta, double mu) noexcept {
    double e = eccentricity;
    double chi = 0.0;
    if (e < 1.0) {
        double k = std::sqrt((1.0 - e) * (1.0 + e));
        double anomaly = std::atan2(k * std::sin(theta), e + std::cos(theta));
        chi = std::sqrt(sl_p) * anomaly / k;
    } else if (e > 1.0) {
        double k = std::sqrt((e - 1.0) * (e + 1.0));
        double anomaly = std::asinh(k * std::sin(theta) / (1.0 + e * std::cos(theta)));
        chi = std::sqrt(sl_p) * anomaly / k;
    } else {
        chi = std::sqrt(sl_p) * std::tan(0.5 * theta);
    }
    double alpha = (1.0 - e) * (1.0 + e) / sl_p;
    auto [_, s] = stumpff_cs(alpha * chi * chi);
    double periapsis = sl_p / (1.0 + e);
    return (periapsis * chi + e * chi * chi * chi * s) / std::sqrt(mu);
}

inline glm::dvec3 pole_to_ecliptic(double pole_ra_deg, double pole_dec_deg) noexcept {
    double ra = pole_ra_deg * DEG_TO_RAD;
    double dec = pole_dec_deg * DEG_TO_RAD;
    double x = std::cos(dec) * std::cos(ra);
    double y = std::cos(dec) * std::sin(ra);
    double z = std::sin(dec);
    double cos_e = std::cos(OBLIQUITY);
    double sin_e = std::sin(OBLIQUITY);
    glm::dvec3 pole{x, y * cos_e + z * sin_e, -y * sin_e + z * cos_e};
    double n = glm::length(pole);
    return n > 0.0 ? (pole / n) : glm::dvec3{0.0, 0.0, 1.0};
}

inline std::pair<double, double> ecliptic_to_pole(const glm::dvec3& pole_ecl) noexcept {
    double cos_e = std::cos(OBLIQUITY);
    double sin_e = std::sin(OBLIQUITY);
    double x = pole_ecl.x;
    double y = pole_ecl.y * cos_e - pole_ecl.z * sin_e;
    double z = pole_ecl.y * sin_e + pole_ecl.z * cos_e;
    double norm = std::sqrt(x * x + y * y + z * z);
    if (norm > 0.0) {
        x /= norm;
        y /= norm;
        z /= norm;
    }
    double dec = std::asin(std::clamp(z, -1.0, 1.0)) * RAD_TO_DEG;
    double ra = std::fmod(std::atan2(y, x) * RAD_TO_DEG + 360.0, 360.0);
    return {ra, dec};
}

inline glm::dmat3 build_equatorial_frame(const glm::dvec3& pole_ecl) noexcept {
    glm::dvec3 z = glm::normalize(pole_ecl);
    glm::dvec3 ref{0.0, 0.0, 1.0};
    glm::dvec3 x = glm::cross(ref, z);
    double x_len = glm::length(x);
    x = x_len < 1e-10 ? glm::dvec3{1.0, 0.0, 0.0} : (x / x_len);
    glm::dvec3 y = glm::cross(z, x);
    return glm::dmat3(x, y, z);
}

inline double kepler_solve(double M, double e, double tol = 1e-12) noexcept {
    double E = M;
    for (int i = 0; i < 100; ++i) {
        double dE = (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
        E -= dE;
        if (std::abs(dE) < tol) break;
    }
    return E;
}

inline StateVector orbital_to_cartesian(double a, double e, double inc, double Omega, double omega, double M, double mu) noexcept {
    double E = kepler_solve(M, e);
    double cos_E = std::cos(E);
    double sin_E = std::sin(E);
    double denom = 1.0 - e * cos_E;
    double r = a * denom;
    double root = std::sqrt(std::max(0.0, 1.0 - e * e));
    double cos_f = (cos_E - e) / denom;
    double sin_f = root * sin_E / denom;
    double x_orb = r * cos_f;
    double y_orb = r * sin_f;
    double fac = std::sqrt(mu / std::max(a * a * a, 1e-30));
    double vx_orb = -fac * a * sin_E / denom;
    double vy_orb = fac * a * root * cos_E / denom;

    double cos_O = std::cos(Omega), sin_O = std::sin(Omega);
    double cos_i = std::cos(inc), sin_i = std::sin(inc);
    double cos_w = std::cos(omega), sin_w = std::sin(omega);

    glm::dvec3 P{
        cos_O * cos_w - sin_O * sin_w * cos_i,
        sin_O * cos_w + cos_O * sin_w * cos_i,
        sin_w * sin_i
    };
    glm::dvec3 Q{
        -cos_O * sin_w - sin_O * cos_w * cos_i,
        -sin_O * sin_w + cos_O * cos_w * cos_i,
        cos_w * sin_i
    };
    return {x_orb * P + y_orb * Q, vx_orb * P + vy_orb * Q};
}

inline StateVector get_cartesian_from_keplerian(double parent_m, double body_m, double a, double e,
                                                double inc_deg, double Omega_deg, double omega_deg, double M_deg) noexcept {
    double mu = G_CONST * (parent_m + body_m);
    return orbital_to_cartesian(a, e, inc_deg * DEG_TO_RAD, Omega_deg * DEG_TO_RAD, omega_deg * DEG_TO_RAD, M_deg * DEG_TO_RAD, mu);
}

inline StateVector vector_orbital_to_cartesian(double a, double e, const glm::dvec3& n_vec, const glm::dvec3& e_vec, double M, double mu) noexcept {
    double E = kepler_solve(M, e);
    double cos_E = std::cos(E);
    double sin_E = std::sin(E);
    double denom = 1.0 - e * cos_E;
    double r = a * denom;
    double root = std::sqrt(std::max(0.0, 1.0 - e * e));
    double x_orb = r * (cos_E - e) / denom;
    double y_orb = r * root * sin_E / denom;
    double fac = std::sqrt(mu / std::max(a * a * a, 1e-30));
    double vx_orb = -fac * a * sin_E / denom;
    double vy_orb = fac * a * root * cos_E / denom;
    glm::dvec3 q_vec = glm::cross(n_vec, e_vec);
    return {x_orb * e_vec + y_orb * q_vec, vx_orb * e_vec + vy_orb * q_vec};
}

struct KeplerVectors {
    double a = 0.0;
    double e = 0.0;
    double M0 = 0.0;
    glm::dvec3 n{0.0, 0.0, 1.0};
    glm::dvec3 u_e{1.0, 0.0, 0.0};
};

inline KeplerVectors cartesian_to_orbital_vectors(const glm::dvec3& r, const glm::dvec3& v, double mu) noexcept {
    double r_len = glm::length(r);
    double v2 = glm::dot(v, v);
    if (r_len < 1e-30 || mu < 1e-30) {
        return {0.0, 0.0, 0.0, glm::dvec3{0.0, 0.0, 1.0}, glm::dvec3{1.0, 0.0, 0.0}};
    }
    glm::dvec3 h = glm::cross(r, v);
    double h_len = glm::length(h);
    if (h_len < 1e-30) {
        return {r_len, 0.0, 0.0, glm::dvec3{0.0, 0.0, 1.0}, glm::dvec3{1.0, 0.0, 0.0}};
    }
    glm::dvec3 n = h / h_len;
    glm::dvec3 e_vec = glm::cross(v, h) / mu - r / r_len;
    double e = glm::length(e_vec);
    glm::dvec3 u_e{1.0, 0.0, 0.0};
    if (e > 1e-10) {
        u_e = e_vec / e;
    } else {
        if (std::abs(n.z) < 0.999999) {
            glm::dvec3 node{-n.y, n.x, 0.0};
            double node_len = glm::length(node);
            u_e = node_len > 1e-15 ? (node / node_len) : glm::dvec3{1.0, 0.0, 0.0};
        } else {
            u_e = glm::dvec3{1.0, 0.0, 0.0};
        }
    }
    u_e = glm::normalize(u_e - n * glm::dot(u_e, n));

    double energy = 0.5 * v2 - mu / r_len;
    double a = std::abs(energy) < 1e-30 ? 1e30 : (-mu / (2.0 * energy));

    double nu = 0.0;
    if (e > 1e-10) {
        double r_dot_e = glm::dot(r, e_vec) / (r_len * e);
        nu = std::acos(std::clamp(r_dot_e, -1.0, 1.0));
        if (glm::dot(r, v) < 0.0) nu = TWO_PI - nu;
    } else {
        double r_dot_ue = glm::dot(r, u_e) / r_len;
        nu = std::acos(std::clamp(r_dot_ue, -1.0, 1.0));
        if (glm::dot(glm::cross(u_e, r), n) < 0.0) nu = TWO_PI - nu;
    }

    double M = 0.0;
    if (e < 1.0) {
        double cos_nu = std::cos(nu);
        double sin_nu = std::sin(nu);
        double denom = 1.0 + e * cos_nu;
        if (std::abs(denom) < 1e-12) denom = 1e-12;
        double sin_E = sin_nu * std::sqrt(std::max(0.0, 1.0 - e * e)) / denom;
        double cos_E = (e + cos_nu) / denom;
        double E = std::atan2(sin_E, cos_E);
        M = E - e * std::sin(E);
        M = std::fmod(M, TWO_PI);
        if (M < 0.0) M += TWO_PI;
    }
    return {a, e, M, n, u_e};
}

inline glm::dvec3 axis_angle_rotation(const glm::dvec3& v, const glm::dvec3& axis, double theta) noexcept {
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);
    return v * cos_t + glm::cross(axis, v) * sin_t + axis * glm::dot(v, axis) * (1.0 - cos_t);
}

inline glm::mat4 reverse_z_perspective(float fov_rad, float aspect, float z_near) noexcept {
    float f = 1.0f / std::tan(0.5f * fov_rad);
    glm::mat4 m(0.0f);
    m[0][0] = f / aspect;
    m[1][1] = -f;
    m[2][2] = 0.0f;
    m[2][3] = -1.0f;
    m[3][2] = z_near;
    return m;
}

struct KeplerianElements {
    double a = 0.0;
    double e = 0.0;
    double inc_deg = 0.0;
    double omega_node_deg = 0.0;
    double arg_peri_deg = 0.0;
    double true_anomaly_deg = 0.0;
    double mean_anomaly_deg = 0.0;
    double period_days = 0.0;
};

inline KeplerianElements compute_keplerian_elements(const glm::dvec3& rel_pos, const glm::dvec3& rel_vel, double mu) noexcept {
    double rx = rel_pos.x, ry = rel_pos.y, rz = rel_pos.z;
    double vx = rel_vel.x, vy = rel_vel.y, vz = rel_vel.z;
    double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    double v2 = vx * vx + vy * vy + vz * vz;
    if (r < 1e-30 || mu < 1e-30) {
        return {};
    }
    double hx = ry * vz - rz * vy;
    double hy = rz * vx - rx * vz;
    double hz = rx * vy - ry * vx;
    double h = std::sqrt(hx * hx + hy * hy + hz * hz);
    if (h < 1e-30) {
        return {};
    }
    double vhx = vy * hz - vz * hy;
    double vhy = vz * hx - vx * hz;
    double vhz = vx * hy - vy * hx;
    double ex = vhx / mu - rx / r;
    double ey = vhy / mu - ry / r;
    double ez = vhz / mu - rz / r;
    double e = std::sqrt(ex * ex + ey * ey + ez * ez);

    double energy = v2 / 2.0 - mu / r;
    double a = std::abs(energy) < 1e-30 ? 1e30 : (-mu / (2.0 * energy));
    double inc = std::acos(std::clamp(hz / h, -1.0, 1.0));

    double nx = -hy;
    double ny = hx;
    double n_mag = std::sqrt(nx * nx + ny * ny);

    double Omega = 0.0;
    if (n_mag > 1e-15) {
        Omega = std::atan2(ny, nx);
        if (Omega < 0.0) Omega += TWO_PI;
    }

    double omega = 0.0;
    if (n_mag > 1e-15 && e > 1e-10) {
        double n_dot_e = (nx * ex + ny * ey) / (n_mag * e);
        omega = std::acos(std::clamp(n_dot_e, -1.0, 1.0));
        if (ez < 0.0) omega = TWO_PI - omega;
    } else if (e > 1e-10) {
        omega = std::atan2(ey, ex);
        if (omega < 0.0) omega += TWO_PI;
    }

    double nu = 0.0;
    if (e > 1e-10) {
        double r_dot_e = (rx * ex + ry * ey + rz * ez) / (r * e);
        nu = std::acos(std::clamp(r_dot_e, -1.0, 1.0));
        double rdotv = rx * vx + ry * vy + rz * vz;
        if (rdotv < 0.0) nu = TWO_PI - nu;
    } else {
        if (n_mag > 1e-15) {
            double r_dot_n = (rx * nx + ry * ny) / (r * n_mag);
            nu = std::acos(std::clamp(r_dot_n, -1.0, 1.0));
            if (rz < 0.0) nu = TWO_PI - nu;
        } else {
            nu = std::atan2(ry, rx);
            if (nu < 0.0) nu += TWO_PI;
        }
    }

    double M = 0.0;
    if (e < 1.0) {
        double cos_nu = std::cos(nu);
        double sin_nu = std::sin(nu);
        double denom = 1.0 + e * cos_nu;
        double sin_E = sin_nu * std::sqrt(std::max(0.0, 1.0 - e * e)) / (denom > 1e-12 ? denom : 1e-12);
        double cos_E = (e + cos_nu) / (denom > 1e-12 ? denom : 1e-12);
        double E = std::atan2(sin_E, cos_E);
        M = E - e * sin_E;
        if (M < 0.0) M += TWO_PI;
    }

    double period_days = 0.0;
    if (a > 0.0 && e < 1.0) {
        double period_yr = TWO_PI * std::sqrt((a * a * a) / mu);
        period_days = period_yr * 365.25;
    }

    return {a, e, inc * RAD_TO_DEG, Omega * RAD_TO_DEG, omega * RAD_TO_DEG, nu * RAD_TO_DEG, M * RAD_TO_DEG, period_days};
}

inline std::pair<glm::dvec3, glm::dvec3> rotate_equatorial_to_ecliptic(const glm::dvec3& pos, const glm::dvec3& vel, const glm::dvec3& pole_ecl) noexcept {
    double px = pole_ecl.x, py = pole_ecl.y, pz = pole_ecl.z;
    if (std::abs(px) < 1e-8 && std::abs(py) < 1e-8 && pz > 0.0) {
        return {pos, vel};
    }
    glm::dvec3 N{-py, px, 0.0};
    double n_mag = glm::length(N);
    glm::dvec3 X, Y, Z;
    if (n_mag < 1e-8) {
        X = {1.0, 0.0, 0.0};
        Y = {0.0, -1.0, 0.0};
        Z = {0.0, 0.0, -1.0};
    } else {
        X = N / n_mag;
        Z = glm::normalize(pole_ecl);
        Y = glm::cross(Z, X);
    }
    glm::dmat3 R(X, Y, Z); // Column-major: X is col 0, Y is col 1, Z is col 2
    return {R * pos, R * vel};
}

inline std::pair<glm::dvec3, glm::dvec3> rotate_ecliptic_to_equatorial(const glm::dvec3& pos, const glm::dvec3& vel, const glm::dvec3& pole_ecl) noexcept {
    double px = pole_ecl.x, py = pole_ecl.y, pz = pole_ecl.z;
    if (std::abs(px) < 1e-8 && std::abs(py) < 1e-8 && pz > 0.0) {
        return {pos, vel};
    }
    glm::dvec3 N{-py, px, 0.0};
    double n_mag = glm::length(N);
    glm::dvec3 X, Y, Z;
    if (n_mag < 1e-8) {
        X = {1.0, 0.0, 0.0};
        Y = {0.0, -1.0, 0.0};
        Z = {0.0, 0.0, -1.0};
    } else {
        X = N / n_mag;
        Z = glm::normalize(pole_ecl);
        Y = glm::cross(Z, X);
    }
    glm::dmat3 R(X, Y, Z);
    glm::dmat3 R_inv = glm::transpose(R);
    return {R_inv * pos, R_inv * vel};
}

struct HydrostaticShape {
    double oblateness = 0.0;
    double j2 = 0.0;
    double j4 = 0.0;
};

inline HydrostaticShape compute_hydrostatic_shape(double mass_msun, double radius_km, double rotation_period_hr, std::string_view body_type) noexcept {
    if (mass_msun <= 0.0 || radius_km <= 0.0 || rotation_period_hr <= 0.0) {
        return {};
    }
    double omega = TWO_PI / (rotation_period_hr * 3600.0);
    double r_eq_m = radius_km * 1000.0;
    double mass_kg = mass_msun * 1.98847e30;
    constexpr double G_SI = 6.6743e-11;
    double m_param = (omega * omega * r_eq_m * r_eq_m * r_eq_m) / (G_SI * mass_kg);
    double chi = 0.65;
    if (body_type == "Moon" || body_type == "Dwarf Planet") {
        chi = 1.25;
    } else if (body_type == "Terrestrial") {
        chi = 0.95;
    }
    double f = std::clamp(chi * m_param, 0.0, 0.5);
    double j2 = m_param * (chi - 0.5);
    double j4 = -0.15 * (f * f);
    return {f, j2, j4};
}

} // namespace sf::math
