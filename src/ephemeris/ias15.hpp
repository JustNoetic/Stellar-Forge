#pragma once

#include "core/command_result.hpp"
#include "core/constants.hpp"
#include "core/coordinates.hpp"
#include "core/units.hpp"
#include "ephemeris/system_manager.hpp"

#include <array>
#include <atomic>
#include <span>
#include <vector>
#include <glm/glm.hpp>

namespace sf::ephemeris {

class Ias15Integrator {
public:
    void reset(const SystemDefinition& def, std::span<const BodyMotion> motions, units::Years t);
    void update_parameters(const SystemDefinition& def);
    CommandResult integrate(const SystemDefinition& def, std::vector<BodyMotion>& motions, units::Years target_time, int max_substeps = 2000, const std::atomic<bool>* cancel = nullptr);
    void shift_com(glm::dvec3 pos_shift, glm::dvec3 vel_shift);

    units::Years time() const noexcept { return units::Years{t_}; }
    double dt() const noexcept { return dt_; }
    void set_dt(double dt) noexcept { dt_ = dt; }
    double epsilon() const noexcept { return epsilon_; }
    void set_epsilon(double eps) noexcept { epsilon_ = eps; }

private:
    struct StepResult {
        double dt_new = 0.0;
        double dt_done = 0.0;
        bool success = false;
    };

    StepResult step(double dt_step);
    void compute_all_accelerations();
    void resize_buffers(size_t n);

    double G_ = G_CONST;
    double t_ = 0.0;
    double dt_ = 1e-4;
    double dt_last_done_ = 0.0;
    double epsilon_ = 1e-4;
    size_t num_bodies_ = 0;

    struct BodyState {
        glm::dvec3 pos{0.0};
        glm::dvec3 vel{0.0};
        glm::dvec3 acc{0.0};
        double mass = 0.0;
    };
    std::vector<BodyState> bodies_;

    bool has_gr_ = false;
    int phys_star_idx_ = -1;
    bool has_j2_ = false;

    struct OblateBody {
        size_t index = 0;
        double j2 = 0.0;
        double j4 = 0.0;
        double req_au = 0.0;
        glm::dvec3 pole_ecl{0.0, 0.0, 1.0};
        double mass = 0.0;
    };
    std::vector<OblateBody> oblate_bodies_;

    std::vector<double> g_;
    std::vector<double> e_;
    std::vector<double> b_;
    std::vector<double> csb_;
    std::vector<double> er_;
    std::vector<double> br_;
    std::vector<double> at_;
    std::vector<double> x0_;
    std::vector<double> v0_;
    std::vector<double> a0_;
    std::vector<double> csx_;
    std::vector<double> csv_;

    std::vector<double> gr_M_;
    std::vector<glm::dvec3> gr_jpos_;
    std::vector<glm::dvec3> gr_jvel_;
    std::vector<glm::dvec3> gr_jacc_;
    std::vector<glm::dvec3> gr_jgr_a_;
    std::vector<glm::dvec3> gr_igr_a_;
};

} // namespace sf::ephemeris
