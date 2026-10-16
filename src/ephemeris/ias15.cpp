#include "ephemeris/ias15.hpp"
#include <cmath>
#include <algorithm>

namespace sf::ephemeris {

namespace {

constexpr std::array<double, 8> IAS15_H = {
    0.0,
    0.0562625605369221464656521910318,
    0.180240691736892364987579942780,
    0.352624717113169637373907769648,
    0.547153626330555383001448554766,
    0.734210177215410531523210605558,
    0.885320946839095768090359771030,
    0.977520613561287501891174488626
};

constexpr std::array<double, 28> IAS15_RR = {
    0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278,
    0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278,
    0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686,
    0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584,
    0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621,
    0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798,
    0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492,
    0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945,
    0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639,
    0.0921996667221917338008147
};

constexpr std::array<double, 21> IAS15_C = {
    -0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321,
    -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399,
    0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219,
    -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973,
    -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991,
    0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398,
    -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588
};

constexpr std::array<double, 21> IAS15_D = {
    0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321,
    0.0001780977692217433881125, 0.0457929855060279188954539, 0.5891279693869841488271399,
    0.0000100202365223291272096, 0.0084318571535257015445000, 0.2535340690545692665214616,
    1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490,
    0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991,
    0.0000000317188154017613665, 0.0002762930909826476593130, 0.0360285539837364596003871,
    0.5767330002770787313544596, 2.2485887607691597933926895, 2.7558127197720458314421588
};

inline double ias15_sqrt7(double a) noexcept {
    if (a <= 0.0) return 0.0;
    double scale = 1.0;
    while (a < 1e-7 && a > 0.0) {
        scale *= 0.1;
        a *= 1e7;
    }
    while (a > 1e2) {
        scale *= 10.0;
        a *= 1e-7;
    }
    double x = 1.0;
    for (int iter = 0; iter < 20; ++iter) {
        double x2 = x * x;
        double x6 = x2 * x2 * x2;
        x += (a / x6 - x) / 7.0;
    }
    return x * scale;
}

inline void add_cs(double& val, double& cs, double inp) noexcept {
    double y = inp - cs;
    double t = val + y;
    cs = (t - val) - y;
    val = t;
}

void predict_next_step(double ratio, size_t n3, const double* _e, const double* _b, double* e, double* b) {
    if (ratio > 20.0) {
        for (size_t k = 0; k < n3; ++k) {
            for (size_t i = 0; i < 7; ++i) {
                e[i * n3 + k] = 0.0;
                b[i * n3 + k] = 0.0;
            }
        }
    } else {
        double q1 = ratio;
        double q2 = q1 * q1;
        double q3 = q1 * q2;
        double q4 = q2 * q2;
        double q5 = q2 * q3;
        double q6 = q3 * q3;
        double q7 = q3 * q4;

        for (size_t k = 0; k < n3; ++k) {
            double b0 = _b[0 * n3 + k], b1 = _b[1 * n3 + k], b2 = _b[2 * n3 + k];
            double b3 = _b[3 * n3 + k], b4 = _b[4 * n3 + k], b5 = _b[5 * n3 + k], b6 = _b[6 * n3 + k];
            double be0 = b0 - _e[0 * n3 + k];
            double be1 = b1 - _e[1 * n3 + k];
            double be2 = b2 - _e[2 * n3 + k];
            double be3 = b3 - _e[3 * n3 + k];
            double be4 = b4 - _e[4 * n3 + k];
            double be5 = b5 - _e[5 * n3 + k];
            double be6 = b6 - _e[6 * n3 + k];

            e[0 * n3 + k] = q1 * (b6 * 7.0 + b5 * 6.0 + b4 * 5.0 + b3 * 4.0 + b2 * 3.0 + b1 * 2.0 + b0);
            e[1 * n3 + k] = q2 * (b6 * 21.0 + b5 * 15.0 + b4 * 10.0 + b3 * 6.0 + b2 * 3.0 + b1);
            e[2 * n3 + k] = q3 * (b6 * 35.0 + b5 * 20.0 + b4 * 10.0 + b3 * 4.0 + b2);
            e[3 * n3 + k] = q4 * (b6 * 35.0 + b5 * 15.0 + b4 * 5.0 + b3);
            e[4 * n3 + k] = q5 * (b6 * 21.0 + b5 * 6.0 + b4);
            e[5 * n3 + k] = q6 * (b6 * 7.0 + b5);
            e[6 * n3 + k] = q7 * b6;

            b[0 * n3 + k] = e[0 * n3 + k] + be0;
            b[1 * n3 + k] = e[1 * n3 + k] + be1;
            b[2 * n3 + k] = e[2 * n3 + k] + be2;
            b[3 * n3 + k] = e[3 * n3 + k] + be3;
            b[4 * n3 + k] = e[4 * n3 + k] + be4;
            b[5 * n3 + k] = e[5 * n3 + k] + be5;
            b[6 * n3 + k] = e[6 * n3 + k] + be6;
        }
    }
}

} // namespace

void Ias15Integrator::resize_buffers(size_t n) {
    num_bodies_ = n;
    size_t n3 = n * 3;
    bodies_.resize(n);
    g_.assign(7 * n3, 0.0);
    e_.assign(7 * n3, 0.0);
    b_.assign(7 * n3, 0.0);
    csb_.assign(7 * n3, 0.0);
    er_.assign(7 * n3, 0.0);
    br_.assign(7 * n3, 0.0);
    at_.assign(n3, 0.0);
    x0_.assign(n3, 0.0);
    v0_.assign(n3, 0.0);
    a0_.assign(n3, 0.0);
    csx_.assign(n3, 0.0);
    csv_.assign(n3, 0.0);

    gr_M_.resize(n);
    gr_jpos_.resize(n);
    gr_jvel_.resize(n);
    gr_jacc_.resize(n);
    gr_jgr_a_.resize(n);
    gr_igr_a_.resize(n);
}

void Ias15Integrator::update_parameters(const SystemDefinition& def) {
    has_gr_ = (def.star_idx >= 0 && def.star_idx < static_cast<int>(def.bodies.size()) && def.bodies[def.star_idx].is_star);
    phys_star_idx_ = has_gr_ ? def.star_idx : -1;

    oblate_bodies_.clear();
    for (size_t i = 0; i < def.bodies.size(); ++i) {
        const auto& b = def.bodies[i];
        if (b.j2 > 0.0) {
            double req = (b.oblateness > 0.0 && b.oblateness < 1.0) ? (b.radius_au() / std::cbrt(1.0 - b.oblateness)) : b.radius_au();
            glm::dvec3 p = (glm::length(b.pole_ecl) > 1e-12) ? glm::normalize(b.pole_ecl) : glm::dvec3{0.0, 0.0, 1.0};
            oblate_bodies_.push_back({i, b.j2, b.j4, req, p, b.mass_msun});
        }
    }
    has_j2_ = !oblate_bodies_.empty();
}

void Ias15Integrator::reset(const SystemDefinition& def, std::span<const BodyMotion> motions, units::Years t) {
    t_ = t.value;
    dt_ = 1e-4;
    dt_last_done_ = 0.0;
    resize_buffers(motions.size());
    for (size_t i = 0; i < motions.size(); ++i) {
        bodies_[i].pos = motions[i].position.value;
        bodies_[i].vel = motions[i].velocity.value;
        bodies_[i].acc = glm::dvec3{0.0};
        bodies_[i].mass = def.bodies[i].mass_msun;
    }
    update_parameters(def);
    compute_all_accelerations();
}

void Ias15Integrator::shift_com(glm::dvec3 pos_shift, glm::dvec3 vel_shift) {
    for (size_t i = 0; i < num_bodies_; ++i) {
        bodies_[i].pos -= pos_shift;
        bodies_[i].vel -= vel_shift;
        x0_[3 * i + 0] -= pos_shift.x;
        x0_[3 * i + 1] -= pos_shift.y;
        x0_[3 * i + 2] -= pos_shift.z;
        v0_[3 * i + 0] -= vel_shift.x;
        v0_[3 * i + 1] -= vel_shift.y;
        v0_[3 * i + 2] -= vel_shift.z;
    }
}

void Ias15Integrator::compute_all_accelerations() {
    for (size_t i = 0; i < num_bodies_; ++i) {
        bodies_[i].acc = glm::dvec3{0.0};
    }

    for (size_t i = 0; i < num_bodies_; ++i) {
        const auto& pi = bodies_[i].pos;
        double mi = bodies_[i].mass;
        for (size_t j = i + 1; j < num_bodies_; ++j) {
            glm::dvec3 d = bodies_[j].pos - pi;
            double r2 = glm::dot(d, d);
            if (r2 <= 0.0 || !std::isfinite(r2)) continue;
            double r_inv = 1.0 / std::sqrt(r2);
            double r3_inv = r_inv * r_inv * r_inv;
            double f = G_ * r3_inv;

            double mj = bodies_[j].mass;
            if (mj > 0.0) bodies_[i].acc += f * mj * d;
            if (mi > 0.0) bodies_[j].acc -= f * mi * d;
        }
    }

    if (has_gr_ && num_bodies_ > 1) {
        gr_M_[0] = bodies_[0].mass;
        for (size_t i = 1; i < num_bodies_; ++i) {
            gr_M_[i] = gr_M_[i - 1] + bodies_[i].mass;
        }

        glm::dvec3 rx_com = bodies_[0].pos;
        glm::dvec3 vx_com = bodies_[0].vel;
        glm::dvec3 ax_com = bodies_[0].acc;

        for (size_t i = 1; i < num_bodies_; ++i) {
            gr_jpos_[i] = bodies_[i].pos - rx_com;
            gr_jvel_[i] = bodies_[i].vel - vx_com;
            gr_jacc_[i] = bodies_[i].acc - ax_com;

            double mi = bodies_[i].mass;
            if (gr_M_[i] > 0.0) {
                double frac = mi / gr_M_[i];
                rx_com += frac * gr_jpos_[i];
                vx_com += frac * gr_jvel_[i];
                ax_com += frac * gr_jacc_[i];
            }
        }

        double mu = G_ * bodies_[0].mass;
        double c2 = C_AU_YR * C_AU_YR;

        for (size_t i = 1; i < num_bodies_; ++i) {
            glm::dvec3 x = gr_jpos_[i];
            glm::dvec3 v = gr_jvel_[i];
            glm::dvec3 a = gr_jacc_[i];

            double r2 = glm::dot(x, x);
            if (r2 <= 0.0 || !std::isfinite(r2)) continue;
            double ri = std::sqrt(r2);

            glm::dvec3 vi = v;
            double vi2 = glm::dot(vi, vi);
            double A = (0.5 * vi2 + 3.0 * mu / ri) / c2;

            for (int q = 0; q < 10; ++q) {
                glm::dvec3 old_vi = vi;
                vi = v / (1.0 - A);
                vi2 = glm::dot(vi, vi);
                A = (0.5 * vi2 + 3.0 * mu / ri) / c2;

                glm::dvec3 dvi = vi - old_vi;
                if (vi2 > 0.0 && glm::dot(dvi, dvi) / vi2 < 4.93038e-32) {
                    break;
                }
            }

            double B = (mu / ri - 1.5 * vi2) * mu / (r2 * ri * c2);
            double rdotrdot = glm::dot(x, v);
            glm::dvec3 vidot = a + B * x;
            double vdotvdot = glm::dot(vi, vidot);
            double D = (vdotvdot - 3.0 * mu / (r2 * ri) * rdotrdot) / c2;

            gr_jgr_a_[i] = B * (1.0 - A) * x - A * a - D * vi;
        }

        glm::dvec3 a_running{0.0};
        for (size_t i = num_bodies_ - 1; i > 0; --i) {
            double mi = bodies_[i].mass;
            if (gr_M_[i] > 0.0) {
                double frac = mi / gr_M_[i];
                a_running -= frac * gr_jgr_a_[i];
            }
            gr_igr_a_[i] = gr_jgr_a_[i] + a_running;
        }
        gr_igr_a_[0] = a_running;

        for (size_t i = 0; i < num_bodies_; ++i) {
            bodies_[i].acc += gr_igr_a_[i];
        }
    }

    if (has_j2_ && !oblate_bodies_.empty()) {
        for (const auto& obl : oblate_bodies_) {
            size_t body_idx = obl.index;
            if (body_idx >= num_bodies_) continue;
            double j2 = obl.j2;
            double j4 = obl.j4;
            if (j2 == 0.0 && j4 == 0.0) continue;
            double req = obl.req_au;
            double M = bodies_[body_idx].mass;
            if (M <= 0.0) continue;

            glm::dvec3 p = obl.pole_ecl;
            double base_j2 = -1.5 * j2 * G_ * M * (req * req);
            double base_j4 = (15.0 / 8.0) * j4 * G_ * M * (req * req * req * req);

            const glm::dvec3& bpos = bodies_[body_idx].pos;

            for (size_t i = 0; i < num_bodies_; ++i) {
                if (i == body_idx) continue;
                glm::dvec3 d = bodies_[i].pos - bpos;
                double r2 = glm::dot(d, d);
                if (r2 <= 0.0 || !std::isfinite(r2)) continue;
                double r = std::sqrt(r2);
                double z_dot = glm::dot(d, p);
                double z_r = z_dot / r;
                double z_r_sq = z_r * z_r;

                glm::dvec3 a{0.0};
                if (j2 != 0.0) {
                    double prefac_j2 = base_j2 / (r2 * r2 * r);
                    double term1 = 1.0 - 5.0 * z_r_sq;
                    double term2 = 2.0 * z_dot;
                    a += prefac_j2 * (term1 * d + term2 * p);
                }
                if (j4 != 0.0) {
                    double prefac_j4 = base_j4 / (r2 * r2 * r2 * r);
                    double z_r_4 = z_r_sq * z_r_sq;
                    double term1 = 1.0 - 14.0 * z_r_sq + 21.0 * z_r_4;
                    double term2 = 4.0 * z_r - (28.0 / 3.0) * z_r_sq * z_r;
                    a += prefac_j4 * (term1 * d + term2 * r * p);
                }

                bodies_[i].acc += a;
                double mi = bodies_[i].mass;
                if (mi > 0.0) {
                    bodies_[body_idx].acc -= a * (mi / M);
                }
            }
        }
    }
}

Ias15Integrator::StepResult Ias15Integrator::step(double dt) {
    size_t n3 = 3 * num_bodies_;
    constexpr double safety_factor = 0.25;

    compute_all_accelerations();

    for (size_t k = 0; k < num_bodies_; ++k) {
        x0_[3 * k + 0] = bodies_[k].pos.x;
        x0_[3 * k + 1] = bodies_[k].pos.y;
        x0_[3 * k + 2] = bodies_[k].pos.z;
        v0_[3 * k + 0] = bodies_[k].vel.x;
        v0_[3 * k + 1] = bodies_[k].vel.y;
        v0_[3 * k + 2] = bodies_[k].vel.z;
        a0_[3 * k + 0] = bodies_[k].acc.x;
        a0_[3 * k + 1] = bodies_[k].acc.y;
        a0_[3 * k + 2] = bodies_[k].acc.z;
    }

    std::fill(csb_.begin(), csb_.end(), 0.0);

    for (size_t k = 0; k < n3; ++k) {
        g_[0 * n3 + k] = b_[6 * n3 + k] * IAS15_D[15] + b_[5 * n3 + k] * IAS15_D[10] + b_[4 * n3 + k] * IAS15_D[6] +
                         b_[3 * n3 + k] * IAS15_D[3] + b_[2 * n3 + k] * IAS15_D[1] + b_[1 * n3 + k] * IAS15_D[0] + b_[0 * n3 + k];
        g_[1 * n3 + k] = b_[6 * n3 + k] * IAS15_D[16] + b_[5 * n3 + k] * IAS15_D[11] + b_[4 * n3 + k] * IAS15_D[7] +
                         b_[3 * n3 + k] * IAS15_D[4] + b_[2 * n3 + k] * IAS15_D[2] + b_[1 * n3 + k];
        g_[2 * n3 + k] = b_[6 * n3 + k] * IAS15_D[17] + b_[5 * n3 + k] * IAS15_D[12] + b_[4 * n3 + k] * IAS15_D[8] +
                         b_[3 * n3 + k] * IAS15_D[5] + b_[2 * n3 + k];
        g_[3 * n3 + k] = b_[6 * n3 + k] * IAS15_D[18] + b_[5 * n3 + k] * IAS15_D[13] + b_[4 * n3 + k] * IAS15_D[9] + b_[3 * n3 + k];
        g_[4 * n3 + k] = b_[6 * n3 + k] * IAS15_D[19] + b_[5 * n3 + k] * IAS15_D[14] + b_[4 * n3 + k];
        g_[5 * n3 + k] = b_[6 * n3 + k] * IAS15_D[20] + b_[5 * n3 + k];
        g_[6 * n3 + k] = b_[6 * n3 + k];
    }

    double predictor_corrector_error = 1e300;
    double predictor_corrector_error_last = 2.0;
    int iterations = 0;
    double dt_new = dt;

    while (true) {
        if (predictor_corrector_error < 1e-16) break;
        if (iterations > 2 && predictor_corrector_error_last <= predictor_corrector_error) break;
        if (iterations >= 12) break;

        predictor_corrector_error_last = predictor_corrector_error;
        predictor_corrector_error = 0.0;
        ++iterations;

        for (int n = 1; n <= 7; ++n) {
            double hn = IAS15_H[n];
            double dt_h = dt * hn;

            for (size_t i = 0; i < num_bodies_; ++i) {
                for (int c = 0; c < 3; ++c) {
                    size_t k = 3 * i + c;
                    double b0 = b_[0 * n3 + k], b1 = b_[1 * n3 + k], b2 = b_[2 * n3 + k], b3 = b_[3 * n3 + k];
                    double b4 = b_[4 * n3 + k], b5 = b_[5 * n3 + k], b6 = b_[6 * n3 + k];
                    double a0k = a0_[k], v0k = v0_[k];

                    double poly_x = ((((((((b6 * 7.0 * hn / 9.0 + b5) * 3.0 * hn / 4.0 + b4) * 5.0 * hn / 7.0 + b3) * 2.0 * hn / 3.0 + b2) * 3.0 * hn / 5.0 + b1) * hn / 2.0 + b0) * hn / 3.0 + a0k) * dt_h / 2.0 + v0k) * dt_h;
                    double xk = -csx_[k] + poly_x;
                    bodies_[i].pos[c] = xk + x0_[k];

                    double poly_v = (((((((b6 * 7.0 * hn / 8.0 + b5) * 6.0 * hn / 7.0 + b4) * 5.0 * hn / 6.0 + b3) * 4.0 * hn / 5.0 + b2) * 3.0 * hn / 4.0 + b1) * 2.0 * hn / 3.0 + b0) * hn / 2.0 + a0k) * dt_h;
                    double vk = -csv_[k] + poly_v;
                    bodies_[i].vel[c] = vk + v0_[k];
                }
            }

            compute_all_accelerations();

            for (size_t i = 0; i < num_bodies_; ++i) {
                at_[3 * i + 0] = bodies_[i].acc.x;
                at_[3 * i + 1] = bodies_[i].acc.y;
                at_[3 * i + 2] = bodies_[i].acc.z;
            }

            if (n == 1) {
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[0 * n3 + k];
                    g_[0 * n3 + k] = (at_[k] - a0_[k]) / IAS15_RR[0];
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], g_[0 * n3 + k] - tmp);
                }
            } else if (n == 2) {
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[1 * n3 + k];
                    g_[1 * n3 + k] = ((at_[k] - a0_[k]) / IAS15_RR[1] - g_[0 * n3 + k]) / IAS15_RR[2];
                    tmp = g_[1 * n3 + k] - tmp;
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], tmp * IAS15_C[0]);
                    add_cs(b_[1 * n3 + k], csb_[1 * n3 + k], tmp);
                }
            } else if (n == 3) {
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[2 * n3 + k];
                    g_[2 * n3 + k] = (((at_[k] - a0_[k]) / IAS15_RR[3] - g_[0 * n3 + k]) / IAS15_RR[4] - g_[1 * n3 + k]) / IAS15_RR[5];
                    tmp = g_[2 * n3 + k] - tmp;
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], tmp * IAS15_C[1]);
                    add_cs(b_[1 * n3 + k], csb_[1 * n3 + k], tmp * IAS15_C[2]);
                    add_cs(b_[2 * n3 + k], csb_[2 * n3 + k], tmp);
                }
            } else if (n == 4) {
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[3 * n3 + k];
                    g_[3 * n3 + k] = ((((at_[k] - a0_[k]) / IAS15_RR[6] - g_[0 * n3 + k]) / IAS15_RR[7] - g_[1 * n3 + k]) / IAS15_RR[8] - g_[2 * n3 + k]) / IAS15_RR[9];
                    tmp = g_[3 * n3 + k] - tmp;
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], tmp * IAS15_C[3]);
                    add_cs(b_[1 * n3 + k], csb_[1 * n3 + k], tmp * IAS15_C[4]);
                    add_cs(b_[2 * n3 + k], csb_[2 * n3 + k], tmp * IAS15_C[5]);
                    add_cs(b_[3 * n3 + k], csb_[3 * n3 + k], tmp);
                }
            } else if (n == 5) {
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[4 * n3 + k];
                    g_[4 * n3 + k] = (((((at_[k] - a0_[k]) / IAS15_RR[10] - g_[0 * n3 + k]) / IAS15_RR[11] - g_[1 * n3 + k]) / IAS15_RR[12] - g_[2 * n3 + k]) / IAS15_RR[13] - g_[3 * n3 + k]) / IAS15_RR[14];
                    tmp = g_[4 * n3 + k] - tmp;
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], tmp * IAS15_C[6]);
                    add_cs(b_[1 * n3 + k], csb_[1 * n3 + k], tmp * IAS15_C[7]);
                    add_cs(b_[2 * n3 + k], csb_[2 * n3 + k], tmp * IAS15_C[8]);
                    add_cs(b_[3 * n3 + k], csb_[3 * n3 + k], tmp * IAS15_C[9]);
                    add_cs(b_[4 * n3 + k], csb_[4 * n3 + k], tmp);
                }
            } else if (n == 6) {
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[5 * n3 + k];
                    g_[5 * n3 + k] = ((((((at_[k] - a0_[k]) / IAS15_RR[15] - g_[0 * n3 + k]) / IAS15_RR[16] - g_[1 * n3 + k]) / IAS15_RR[17] - g_[2 * n3 + k]) / IAS15_RR[18] - g_[3 * n3 + k]) / IAS15_RR[19] - g_[4 * n3 + k]) / IAS15_RR[20];
                    tmp = g_[5 * n3 + k] - tmp;
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], tmp * IAS15_C[10]);
                    add_cs(b_[1 * n3 + k], csb_[1 * n3 + k], tmp * IAS15_C[11]);
                    add_cs(b_[2 * n3 + k], csb_[2 * n3 + k], tmp * IAS15_C[12]);
                    add_cs(b_[3 * n3 + k], csb_[3 * n3 + k], tmp * IAS15_C[13]);
                    add_cs(b_[4 * n3 + k], csb_[4 * n3 + k], tmp * IAS15_C[14]);
                    add_cs(b_[5 * n3 + k], csb_[5 * n3 + k], tmp);
                }
            } else if (n == 7) {
                double maxak = 0.0;
                double maxb6ktmp = 0.0;
                for (size_t k = 0; k < n3; ++k) {
                    double tmp = g_[6 * n3 + k];
                    g_[6 * n3 + k] = (((((((at_[k] - a0_[k]) / IAS15_RR[21] - g_[0 * n3 + k]) / IAS15_RR[22] - g_[1 * n3 + k]) / IAS15_RR[23] - g_[2 * n3 + k]) / IAS15_RR[24] - g_[3 * n3 + k]) / IAS15_RR[25] - g_[4 * n3 + k]) / IAS15_RR[26] - g_[5 * n3 + k]) / IAS15_RR[27];
                    tmp = g_[6 * n3 + k] - tmp;
                    add_cs(b_[0 * n3 + k], csb_[0 * n3 + k], tmp * IAS15_C[15]);
                    add_cs(b_[1 * n3 + k], csb_[1 * n3 + k], tmp * IAS15_C[16]);
                    add_cs(b_[2 * n3 + k], csb_[2 * n3 + k], tmp * IAS15_C[17]);
                    add_cs(b_[3 * n3 + k], csb_[3 * n3 + k], tmp * IAS15_C[18]);
                    add_cs(b_[4 * n3 + k], csb_[4 * n3 + k], tmp * IAS15_C[19]);
                    add_cs(b_[5 * n3 + k], csb_[5 * n3 + k], tmp * IAS15_C[20]);
                    add_cs(b_[6 * n3 + k], csb_[6 * n3 + k], tmp);

                    double ak = std::abs(at_[k]);
                    if (ak > maxak) maxak = ak;
                    double b6ktmp = std::abs(tmp);
                    if (b6ktmp > maxb6ktmp) maxb6ktmp = b6ktmp;
                }
                predictor_corrector_error = (maxak > 0.0) ? (maxb6ktmp / maxak) : 0.0;
            }
        }
    }

    double min_timescale2 = 1e300;
    for (size_t i = 0; i < num_bodies_; ++i) {
        double a0i = 0.0, y2 = 0.0, y3 = 0.0, y4 = 0.0;
        for (int c = 0; c < 3; ++c) {
            size_t k = 3 * i + c;
            a0i += a0_[k] * a0_[k];
            double tmp2 = a0_[k] + b_[0 * n3 + k] + b_[1 * n3 + k] + b_[2 * n3 + k] + b_[3 * n3 + k] + b_[4 * n3 + k] + b_[5 * n3 + k] + b_[6 * n3 + k];
            y2 += tmp2 * tmp2;
            double tmp3 = b_[0 * n3 + k] + 2.0 * b_[1 * n3 + k] + 3.0 * b_[2 * n3 + k] + 4.0 * b_[3 * n3 + k] + 5.0 * b_[4 * n3 + k] + 6.0 * b_[5 * n3 + k] + 7.0 * b_[6 * n3 + k];
            y3 += tmp3 * tmp3;
            double tmp4 = 2.0 * b_[1 * n3 + k] + 6.0 * b_[2 * n3 + k] + 12.0 * b_[3 * n3 + k] + 20.0 * b_[4 * n3 + k] + 30.0 * b_[5 * n3 + k] + 42.0 * b_[6 * n3 + k];
            y4 += tmp4 * tmp4;
        }
        if (a0i == 0.0) continue;
        double denom = y3 + std::sqrt(y4 * y2);
        if (denom <= 0.0) continue;
        double timescale2 = 2.0 * y2 / denom;
        if (timescale2 < min_timescale2) min_timescale2 = timescale2;
    }

    if (min_timescale2 < 1e299) {
        dt_new = std::sqrt(min_timescale2) * dt * ias15_sqrt7(epsilon_ * 5040.0);
    } else {
        dt_new = dt / safety_factor;
    }

    if (std::abs(dt_new / dt) < safety_factor) {
        for (size_t i = 0; i < num_bodies_; ++i) {
            bodies_[i].pos = {x0_[3 * i + 0], x0_[3 * i + 1], x0_[3 * i + 2]};
            bodies_[i].vel = {v0_[3 * i + 0], v0_[3 * i + 1], v0_[3 * i + 2]};
            bodies_[i].acc = {a0_[3 * i + 0], a0_[3 * i + 1], a0_[3 * i + 2]};
        }
        if (dt_last_done_ != 0.0 && dt_new * dt_last_done_ > 0.0) {
            double ratio = dt_new / dt_last_done_;
            predict_next_step(ratio, n3, er_.data(), br_.data(), e_.data(), b_.data());
        }
        return {dt_new, 0.0, false};
    }

    if (std::abs(dt_new / dt) > 1.0) {
        if (dt_new / dt > 1.0 / safety_factor) {
            dt_new = dt / safety_factor;
        }
    }

    double dt2 = dt * dt;
    for (size_t k = 0; k < n3; ++k) {
        add_cs(x0_[k], csx_[k], b_[6 * n3 + k] / 72.0 * dt2);
        add_cs(x0_[k], csx_[k], b_[5 * n3 + k] / 56.0 * dt2);
        add_cs(x0_[k], csx_[k], b_[4 * n3 + k] / 42.0 * dt2);
        add_cs(x0_[k], csx_[k], b_[3 * n3 + k] / 30.0 * dt2);
        add_cs(x0_[k], csx_[k], b_[2 * n3 + k] / 20.0 * dt2);
        add_cs(x0_[k], csx_[k], b_[1 * n3 + k] / 12.0 * dt2);
        add_cs(x0_[k], csx_[k], b_[0 * n3 + k] / 6.0 * dt2);
        add_cs(x0_[k], csx_[k], a0_[k] / 2.0 * dt2);
        add_cs(x0_[k], csx_[k], v0_[k] * dt);

        add_cs(v0_[k], csv_[k], b_[6 * n3 + k] / 8.0 * dt);
        add_cs(v0_[k], csv_[k], b_[5 * n3 + k] / 7.0 * dt);
        add_cs(v0_[k], csv_[k], b_[4 * n3 + k] / 6.0 * dt);
        add_cs(v0_[k], csv_[k], b_[3 * n3 + k] / 5.0 * dt);
        add_cs(v0_[k], csv_[k], b_[2 * n3 + k] / 4.0 * dt);
        add_cs(v0_[k], csv_[k], b_[1 * n3 + k] / 3.0 * dt);
        add_cs(v0_[k], csv_[k], b_[0 * n3 + k] / 2.0 * dt);
        add_cs(v0_[k], csv_[k], a0_[k] * dt);
    }

    for (size_t i = 0; i < num_bodies_; ++i) {
        bodies_[i].pos = {x0_[3 * i + 0], x0_[3 * i + 1], x0_[3 * i + 2]};
        bodies_[i].vel = {v0_[3 * i + 0], v0_[3 * i + 1], v0_[3 * i + 2]};
    }

    er_ = e_;
    br_ = b_;

    double ratio = dt_new / dt;
    predict_next_step(ratio, n3, e_.data(), b_.data(), e_.data(), b_.data());

    return {dt_new, dt, true};
}

CommandResult Ias15Integrator::integrate(const SystemDefinition& def, std::vector<BodyMotion>& motions, units::Years target_time) {
    if (bodies_.size() != motions.size()) {
        reset(def, motions, units::Years{t_});
    }

    double t_target = target_time.value;
    int substeps = 0;
    constexpr int MAX_SUBSTEPS = 2000;

    while (substeps < MAX_SUBSTEPS) {
        double t_diff = t_target - t_;
        if (std::abs(t_diff) < 1e-12) {
            break;
        }

        double dt_step = dt_;
        if ((t_diff > 0.0 && dt_step < 0.0) || (t_diff < 0.0 && dt_step > 0.0)) {
            dt_step = -dt_step;
            dt_last_done_ = 0.0;
            b_.assign(7 * num_bodies_ * 3, 0.0);
            e_.assign(7 * num_bodies_ * 3, 0.0);
        }
        if (t_diff > 0.0 && dt_step > t_diff) {
            dt_step = t_diff;
        } else if (t_diff < 0.0 && dt_step < t_diff) {
            dt_step = t_diff;
        }

        auto [dt_new, dt_done, success] = step(dt_step);
        dt_ = dt_new;
        if (success) {
            t_ += dt_done;
            dt_last_done_ = dt_done;
        }
        ++substeps;
    }

    for (size_t i = 0; i < bodies_.size(); ++i) {
        motions[i].position = EclipticPosition{bodies_[i].pos};
        motions[i].velocity = EclipticVelocity{bodies_[i].vel};
    }

    t_ = t_target;
    return {true, true, {}};
}

} // namespace sf::ephemeris
