#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <thread>
#include <vector>

namespace sf::physics {

inline constexpr size_t REFRACTION_IMPACT_SAMPLES = 512;
inline constexpr size_t REFRACTION_TRAVEL_SAMPLES = 256;

struct RefractionInput {
    double radius_km = 0, bottom_km = 0, ground_km = 0, top_km = 0, scale_km = 8;
    glm::dvec3 lorentz_lorenz{0};
    std::vector<double> log_density;
};

struct RefractionSolution {
    double apparent_zenith = 0, true_zenith = 0;
    bool visible = false;
    double rotation() const noexcept { return true_zenith-apparent_zenith; }
};

inline glm::dvec3 standard_air_dispersion() {
    auto n = [](double micrometres) {
        double s = 1/(micrometres*micrometres);
        return 5792105/(238.0185-s)+167917/(57.362-s);
    };
    double reference = n(.55);
    return {n(.68)/reference, 1.0, n(.44)/reference};
}

class PreparedRefraction {
public:
    double radius_km = 0, bottom_radius_km = 0, ground_radius_km = 0, top_radius_km = 0, scale_km = 8;
    glm::dvec3 lorentz_lorenz{0};
    std::vector<double> log_density;
    double impact_min_km = 0, impact_scale_km = 1, critical_radius_km = 0;
    std::array<double, REFRACTION_IMPACT_SAMPLES> impact{}, turn_radius{};
    std::array<glm::dvec3, REFRACTION_IMPACT_SAMPLES> deflection{};
    std::vector<double> travel;
    std::vector<glm::dvec3> node_invariant;
    std::vector<double> node_radii;
    std::vector<glm::dvec2> fine_profile;
    std::vector<glm::dvec3> fine_refractivity, fine_gradient;
    double fine_step_km = 1;
    glm::bvec3 monotone{false};

    double lorentz_lorenz_at(double r, int channel = 1) const noexcept {
        if (r >= top_radius_km || log_density.size() < 2) return 0;
        return lorentz_lorenz[channel]*std::exp(sample(r).x);
    }

    double index(double r, int channel = 1) const noexcept {
        if (r >= top_radius_km || log_density.size() < 2) return 1;
        if (fine_refractivity.size() > 1) return 1+fine_index(r, channel).x;
        double l = lorentz_lorenz_at(r, channel);
        return std::sqrt((1+2*l)/(1-l));
    }

    glm::dvec2 fine_index(double r, int channel) const noexcept {
        double x = (r-bottom_radius_km)/fine_step_km;
        if (x <= 0) return {fine_refractivity[0][channel]+fine_gradient[0][channel]*(r-bottom_radius_km), fine_gradient[0][channel]};
        size_t i = std::min(size_t(x), fine_refractivity.size()-2);
        double t = x-double(i), h = fine_step_km;
        double a = fine_refractivity[i][channel], b = fine_refractivity[i+1][channel];
        double da = fine_gradient[i][channel]*h, db = fine_gradient[i+1][channel]*h;
        double t2 = t*t, t3 = t2*t;
        return {(2*t3-3*t2+1)*a+(t3-2*t2+t)*da+(-2*t3+3*t2)*b+(t3-t2)*db,
                ((6*t2-6*t)*a+(3*t2-4*t+1)*da+(-6*t2+6*t)*b+(3*t2-2*t)*db)/h};
    }

    glm::dvec2 index_and_gradient(double r, int channel = 1) const noexcept {
        if (r >= top_radius_km || log_density.size() < 2) return {1, 0};
        if (fine_refractivity.size() > 1) {
            glm::dvec2 v = fine_index(r, channel);
            return {1+v.x, v.y};
        }
        glm::dvec2 s = sample(r);
        double l = lorentz_lorenz[channel]*std::exp(s.x);
        double n = std::sqrt((1+2*l)/(1-l));
        return {n, 3*l*s.y/(2*n*(1-l)*(1-l))};
    }

    double invariant(double r, int channel = 1) const noexcept { return index(r, channel)*r; }

    double turning_radius(double p, int channel = 1) const noexcept {
        if (p >= top_radius_km) return p;
        return downward_turn(p, top_radius_km, channel);
    }

    double travel_from_turn(double p, double r, double turn, int channel = 1, int panels = 0) const noexcept {
        if (r <= turn) return 0;
        double top = std::min(r, top_radius_km), inside = 0;
        if (top > turn) {
            double u_max = std::sqrt(top-turn);
            if (panels <= 0) panels = std::clamp(int(std::ceil(u_max/.05)), 16, 4096);
            for (int k = 0; k < panels; ++k) {
                double a = u_max*k/panels, b = u_max*(k+1)/panels;
                for (int q = 0; q < 8; ++q) {
                    double u = .5*(a+b)+.5*(b-a)*GL8[q].x;
                    double rr = turn+u*u;
                    double f = invariant(rr, channel);
                    double d = std::max((f-p)*(f+p), 1e-300);
                    inside += .5*(b-a)*GL8[q].y*2*u*p/(rr*std::sqrt(d));
                }
            }
        }
        if (r > top_radius_km) inside += vacuum_travel(p, r)-vacuum_travel(p, std::max(top_radius_km, turn));
        return inside;
    }

    double segment_travel(double p, double a, double b, int channel = 1) const noexcept {
        if (b < a) std::swap(a, b);
        if (b <= a) return 0;
        double turn = downward_turn(p, a, channel);
        if (!std::isnan(turn)) return travel_from_turn(p, b, turn, channel)-travel_from_turn(p, a, turn, channel);
        double integral = 0, upper = std::min(b, top_radius_km);
        const int panels = 256;
        for (int k = 0; k < panels && upper > a; ++k) {
            double lo = a+(upper-a)*k/panels, hi = a+(upper-a)*(k+1)/panels;
            for (int q = 0; q < 8; ++q) {
                double rr = .5*(lo+hi)+.5*(hi-lo)*GL8[q].x;
                double f = invariant(rr, channel);
                integral += .5*(hi-lo)*GL8[q].y*p/(rr*std::sqrt(std::max((f-p)*(f+p), 1e-300)));
            }
        }
        if (b > top_radius_km) integral += vacuum_travel(p, b)-vacuum_travel(p, std::max(a, top_radius_km));
        return integral;
    }

    double node_turn(double p) const noexcept {
        if (p >= top_radius_km) return p;
        size_t hi = node_invariant.size()-1;
        while (hi > 0 && node_radii[hi] >= top_radius_km) --hi;
        if (node_invariant[hi].y <= p) return node_radii[hi];
        size_t lo = 0;
        while (lo < hi && node_invariant[lo].y > p) ++lo;
        if (node_invariant[lo].y > p) return std::numeric_limits<double>::quiet_NaN();
        size_t a = lo, b = hi;
        for (size_t k = hi; k > lo; --k) if (node_invariant[k-1].y <= p) { a = k-1; b = k; break; }
        double fa = node_invariant[a].y, fb = node_invariant[b].y;
        double r = node_radii[a]+(node_radii[b]-node_radii[a])*(p-fa)/(fb-fa);
        for (int i = 0; i < 2; ++i) {
            glm::dvec2 ng = index_and_gradient(r);
            double f = ng.x*r, df = ng.x+ng.y*r;
            if (std::abs(df) > 1e-12) r = std::clamp(r-(f-p)/df, node_radii[a], node_radii[b]);
        }
        return r;
    }

    double table_travel(double p, double r) const noexcept {
        if (p >= top_radius_km || travel.empty()) return vacuum_travel(p, r);
        double row = std::clamp(impact_coordinate(p), 0.0, 1.0)*double(REFRACTION_IMPACT_SAMPLES-1);
        ptrdiff_t i = std::min(ptrdiff_t(row), ptrdiff_t(REFRACTION_IMPACT_SAMPLES)-2);
        double w = row-double(i);
        auto catmull = [&](auto value) {
            double y0 = value(i-1), y1 = value(i), y2 = value(i+1), y3 = value(i+2);
            return y1+.5*w*(y2-y0+w*(2*y0-5*y1+4*y2-y3+w*(3*(y1-y2)+y3-y0)));
        };
        auto clamp_row = [](ptrdiff_t k) { return size_t(std::clamp<ptrdiff_t>(k, 0, ptrdiff_t(REFRACTION_IMPACT_SAMPLES)-1)); };
        double turn = node_turn(p);
        if (std::isnan(turn)) turn = p-catmull([&](ptrdiff_t k) { size_t c = clamp_row(k); return impact[c]-turn_radius[c]; });
        double rr = std::min(r, top_radius_km);
        if (rr <= turn) return r > top_radius_km ? vacuum_travel(p, r)-vacuum_travel(p, top_radius_km) : 0.0;
        double span = std::sqrt(std::max(top_radius_km-turn, 0.0));
        double x = span > 0 ? std::sqrt(std::max(rr-turn, 0.0))/span : 0;
        double column = std::clamp(x, 0.0, 1.0)*double(REFRACTION_TRAVEL_SAMPLES-1);
        size_t j = std::min(size_t(column), REFRACTION_TRAVEL_SAMPLES-2);
        double f = column-double(j);
        double inside = span*catmull([&](ptrdiff_t k) {
            const double* t = &travel[clamp_row(k)*REFRACTION_TRAVEL_SAMPLES];
            return t[j]*(1-f)+t[j+1]*f;
        });
        if (r > top_radius_km) inside += vacuum_travel(p, r)-vacuum_travel(p, top_radius_km);
        return inside;
    }

    double impact_coordinate(double p) const noexcept {
        double span = std::asinh((top_radius_km-impact_min_km)/impact_scale_km);
        return std::sqrt(std::asinh(std::max(p-impact_min_km, 0.0)/impact_scale_km)/span);
    }

    double impact_at(double t) const noexcept {
        double span = std::asinh((top_radius_km-impact_min_km)/impact_scale_km);
        return impact_min_km+impact_scale_km*std::sinh(span*t*t);
    }

    glm::dvec3 total_deflection(double p) const noexcept {
        if (p >= top_radius_km) return glm::dvec3(0);
        double row = std::clamp(impact_coordinate(p), 0.0, 1.0)*double(REFRACTION_IMPACT_SAMPLES-1);
        size_t i = std::min(size_t(row), REFRACTION_IMPACT_SAMPLES-2);
        return glm::mix(deflection[i], deflection[i+1], row-double(i));
    }

    double horizon_impact(double camera_radius, int channel = 1) const noexcept {
        double upper = std::min(camera_radius, top_radius_km);
        if (upper <= ground_radius_km) return camera_radius;
        double best = camera_radius >= top_radius_km ? top_radius_km : invariant(camera_radius, channel);
        best = std::min(best, invariant(ground_radius_km, channel));
        for (size_t i = 0; i < node_invariant.size(); ++i) {
            double r = node_radius(i);
            if (r > ground_radius_km && r < upper) best = std::min(best, node_invariant[i][channel]);
        }
        return best;
    }

    double max_deflection() const noexcept {
        double best = 0;
        for (size_t i = 0; i < REFRACTION_IMPACT_SAMPLES; ++i)
            if (turn_radius[i] >= ground_radius_km) best = std::max(best, deflection[i].y);
        return best;
    }

    double lowest_visible_chord(double camera_radius) const noexcept {
        double p = horizon_impact(camera_radius);
        double bend = std::max(max_deflection(), total_deflection(p).y);
        double limb = std::asin(std::clamp(p/(index(camera_radius)*camera_radius), 0.0, 1.0));
        return camera_radius*std::sin(std::max(limb-bend, 0.0));
    }

    double direct_travel(double p, double a, double b, int channel = 1) const noexcept {
        if (b < a) std::swap(a, b);
        double top = std::min(b, top_radius_km), integral = 0;
        if (top > a) {
            for (int k = 0; k < 8; ++k) {
                for (int q = 0; q < 8; ++q) {
                    double x = (k+.5+.5*GL8[q].x)/8;
                    double rr = a+(top-a)*x*x;
                    double f = invariant(rr, channel);
                    integral += GL8[q].y/16*(top-a)*2*x*p/(rr*std::sqrt(std::max((f-p)*(f+p), 1e-300)));
                }
            }
        }
        if (b > top_radius_km) integral += vacuum_travel(p, b)-vacuum_travel(p, std::max(a, top_radius_km));
        return integral;
    }

    double reach_travel(double camera_radius, double zenith, double target_radius, bool& reached, int channel = 1,
                        bool tabulated = false, bool outbound = false) const noexcept {
        constexpr double unreachable = std::numeric_limits<double>::infinity();
        reached = false;
        double p = index(camera_radius, channel)*camera_radius*std::sin(zenith);
        double barrier = upward_barrier(p, camera_radius, channel);
        bool covered = tabulated && p >= impact_min_km;
        if (zenith <= .5*std::numbers::pi) {
            if (target_radius < camera_radius || barrier < std::min(target_radius, top_radius_km)) return unreachable;
            reached = true;
            if (!tabulated) return segment_travel(p, camera_radius, target_radius, channel);
            return covered ? table_travel(p, target_radius)-table_travel(p, camera_radius) : direct_travel(p, camera_radius, target_radius);
        }
        double turn = tabulated ? (covered ? node_turn(p) : -unreachable) : downward_turn(p, camera_radius, channel);
        bool grounded = std::isnan(turn) || turn < ground_radius_km;
        if (target_radius < camera_radius && !outbound) {
            if (target_radius < (grounded ? ground_radius_km : turn)) return unreachable;
            reached = true;
            if (!tabulated) return segment_travel(p, target_radius, camera_radius, channel);
            return covered ? table_travel(p, camera_radius)-table_travel(p, target_radius) : direct_travel(p, target_radius, camera_radius);
        }
        if (grounded || target_radius < turn || barrier < std::min(target_radius, top_radius_km)) return unreachable;
        reached = true;
        if (tabulated) return table_travel(p, camera_radius)+table_travel(p, target_radius);
        return travel_from_turn(p, camera_radius, turn, channel)+travel_from_turn(p, target_radius, turn, channel);
    }

    RefractionSolution solve(double camera_radius, double target_radius, double polar_angle, int channel = 1,
                             bool tabulated = false) const noexcept {
        RefractionSolution result;
        bool infinite = !std::isfinite(target_radius);
        double target = infinite ? 1e30 : target_radius;
        result.true_zenith = infinite ? polar_angle :
            std::atan2(target_radius*std::sin(polar_angle), target_radius*std::cos(polar_angle)-camera_radius);
        bool rising = target >= camera_radius;
        bool outbound = rising;
        double lo = 0, hi = std::numbers::pi;
        if (!rising) {
            double target_p = invariant(target, channel), camera_p = invariant(camera_radius, channel);
            if (target_p < camera_p && target_p >= impact_min_km && turning_radius(target_p, channel) <= target+1e-6) {
                double tangent = tabulated ? table_travel(target_p, camera_radius) : segment_travel(target_p, target, camera_radius, channel);
                outbound = polar_angle > tangent;
                lo = std::numbers::pi-std::asin(std::clamp(target_p/camera_p, 0.0, 1.0));
            }
        }
        bool ok = false;
        for (int i = 0; i < 64; ++i) {
            double z = .5*(lo+hi);
            double t = reach_travel(camera_radius, z, target, ok, channel, tabulated, outbound);
            bool too_far = !ok || t > polar_angle;
            if (outbound == too_far) hi = z;
            else lo = z;
        }
        result.apparent_zenith = .5*(lo+hi);
        double t = reach_travel(camera_radius, result.apparent_zenith, target, ok, channel, tabulated, outbound);
        result.visible = ok && std::abs(t-polar_angle) <= (tabulated ? 1e-4 : 1e-7)*std::max(1.0, polar_angle);
        return result;
    }

    static double vacuum_travel(double p, double r) noexcept {
        return r > p ? std::acos(std::clamp(p/r, -1.0, 1.0)) : 0.0;
    }

    double radius_at(double t) const noexcept {
        double s0 = std::asinh((bottom_radius_km-radius_km)/scale_km);
        double s1 = std::asinh((top_radius_km-radius_km)/scale_km);
        return radius_km+scale_km*std::sinh(s0+(s1-s0)*t);
    }

private:
    inline static constexpr std::array<glm::dvec2, 4> GL4{{
        {-.8611363115940526, .3478548451374538}, {-.3399810435848563, .6521451548625461},
        {.3399810435848563, .6521451548625461}, {.8611363115940526, .3478548451374538}
    }};

    inline static constexpr std::array<glm::dvec2, 8> GL8{{
        {-.9602898564975363, .1012285362903763}, {-.7966664774136267, .2223810344533745},
        {-.5255324099163290, .3137066458778873}, {-.1834346424956498, .3626837833783620},
        {.1834346424956498, .3626837833783620}, {.5255324099163290, .3137066458778873},
        {.7966664774136267, .2223810344533745}, {.9602898564975363, .1012285362903763}
    }};

    glm::dvec2 sample_profile(double r) const noexcept {
        double s0 = std::asinh((bottom_radius_km-radius_km)/scale_km);
        double s1 = std::asinh((top_radius_km-radius_km)/scale_km);
        double h = r-radius_km;
        double s = std::asinh(h/scale_km);
        double ds_dr = 1/std::sqrt(scale_km*scale_km+h*h);
        double steps = double(log_density.size()-1);
        double x = (s-s0)/(s1-s0)*steps;
        if (x <= 0) {
            double slope = (log_density[1]-log_density[0])/((s1-s0)/steps);
            return {log_density[0]+slope*(s-s0), slope*ds_dr};
        }
        size_t i = std::min(size_t(x), log_density.size()-2);
        double t = x-double(i);
        auto at = [&](ptrdiff_t k) {
            k = std::clamp<ptrdiff_t>(k, 0, ptrdiff_t(log_density.size())-1);
            return log_density[size_t(k)];
        };
        double y0 = at(ptrdiff_t(i)-1), y1 = at(ptrdiff_t(i)), y2 = at(ptrdiff_t(i)+1), y3 = at(ptrdiff_t(i)+2);
        if (i == 0) y0 = 2*y1-y2;
        if (i+2 >= log_density.size()) y3 = 2*y2-y1;
        double m1 = .5*(y2-y0), m2 = .5*(y3-y1);
        double t2 = t*t, t3 = t2*t;
        double value = (2*t3-3*t2+1)*y1+(t3-2*t2+t)*m1+(-2*t3+3*t2)*y2+(t3-t2)*m2;
        double slope = (6*t2-6*t)*y1+(3*t2-4*t+1)*m1+(-6*t2+6*t)*y2+(3*t2-2*t)*m2;
        return {value, slope*steps/(s1-s0)*ds_dr};
    }

    double node_radius(size_t i) const noexcept { return node_radii[i]; }

    size_t node_below(double r) const noexcept {
        double s0 = std::asinh((bottom_radius_km-radius_km)/scale_km), s1 = std::asinh((top_radius_km-radius_km)/scale_km);
        double x = (std::asinh((r-radius_km)/scale_km)-s0)/(s1-s0)*double(node_invariant.size()-1);
        size_t i = size_t(std::clamp(std::floor(x), 0.0, double(node_invariant.size()-1)));
        while (i > 0 && node_radius(i) > r) --i;
        return i;
    }

    double refine(double p, double lower, double upper, int channel, bool rising) const noexcept {
        for (int i = 0; i < 60; ++i) {
            double middle = .5*(lower+upper);
            bool below = invariant(middle, channel) <= p;
            if (below != rising) lower = middle;
            else upper = middle;
        }
        return rising ? lower : upper;
    }

    glm::dvec2 sample(double r) const noexcept {
        if (fine_profile.size() < 2) return sample_profile(r);
        double x = (r-bottom_radius_km)/fine_step_km;
        if (x <= 0) return {fine_profile[0].x+fine_profile[0].y*(r-bottom_radius_km), fine_profile[0].y};
        size_t i = std::min(size_t(x), fine_profile.size()-2);
        double t = x-double(i), h = fine_step_km;
        glm::dvec2 a = fine_profile[i], b = fine_profile[i+1];
        double t2 = t*t, t3 = t2*t;
        double value = (2*t3-3*t2+1)*a.x+(t3-2*t2+t)*h*a.y+(-2*t3+3*t2)*b.x+(t3-t2)*h*b.y;
        double slope = ((6*t2-6*t)*a.x+(3*t2-4*t+1)*h*a.y+(-6*t2+6*t)*b.x+(3*t2-2*t)*h*b.y)/h;
        return {value, slope};
    }

    double downward_turn(double p, double from, int channel) const noexcept {
        if (from >= top_radius_km) {
            if (p >= top_radius_km) return p;
            from = top_radius_km*(1-1e-15);
        }
        if (invariant(from, channel) <= p) return from;
        if (monotone[channel]) {
            if (node_invariant[0][channel] > p) return std::numeric_limits<double>::quiet_NaN();
            size_t lo = 0, hi = node_below(from);
            if (node_invariant[hi][channel] <= p) return refine(p, node_radii[hi], from, channel, false);
            while (hi-lo > 1) {
                size_t mid = (lo+hi)/2;
                (node_invariant[mid][channel] <= p ? lo : hi) = mid;
            }
            return refine(p, node_radii[lo], std::min(node_radii[hi], from), channel, false);
        }
        double upper = from;
        for (size_t i = node_below(from)+1; i-- > 0;) {
            double r = node_radius(i);
            if (r >= upper) continue;
            if (node_invariant[i][channel] <= p) return refine(p, r, upper, channel, false);
            upper = r;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    double upward_barrier(double p, double from, int channel) const noexcept {
        if (from >= top_radius_km || monotone[channel]) return std::numeric_limits<double>::infinity();
        double lower = from;
        for (size_t i = node_below(from)+1; i < node_invariant.size(); ++i) {
            double r = node_radius(i);
            if (r <= lower) continue;
            if (r >= top_radius_km) break;
            if (node_invariant[i][channel] < p*(1-1e-13)) return refine(p, lower, r, channel, true);
            lower = r;
        }
        return std::numeric_limits<double>::infinity();
    }

    friend PreparedRefraction prepare_refraction(RefractionInput input);
};

inline PreparedRefraction prepare_refraction(RefractionInput input) {
    PreparedRefraction result;
    result.radius_km = input.radius_km;
    result.bottom_radius_km = input.radius_km+input.bottom_km;
    result.ground_radius_km = input.radius_km+std::max(input.ground_km, input.bottom_km);
    result.top_radius_km = input.radius_km+input.top_km;
    result.scale_km = input.scale_km;
    result.lorentz_lorenz = input.lorentz_lorenz;
    result.log_density = std::move(input.log_density);
    const size_t fine = 8192;
    result.fine_step_km = (result.top_radius_km-result.bottom_radius_km)/double(fine-1);
    result.fine_profile.resize(fine);
    for (size_t i = 0; i < fine; ++i)
        result.fine_profile[i] = result.sample_profile(result.bottom_radius_km+result.fine_step_km*double(i));
    std::vector<glm::dvec3> refractivity(fine), gradient(fine);
    for (size_t i = 0; i < fine; ++i)
        for (int c = 0; c < 3; ++c) {
            glm::dvec2 ng = result.index_and_gradient(result.bottom_radius_km+result.fine_step_km*double(i), c);
            refractivity[i][c] = ng.x-1;
            gradient[i][c] = ng.y;
        }
    refractivity[fine-1] = gradient[fine-1] = glm::dvec3(0);
    result.fine_refractivity = std::move(refractivity);
    result.fine_gradient = std::move(gradient);
    const size_t nodes = 4*result.log_density.size();
    result.node_invariant.resize(nodes);
    result.node_radii.resize(nodes);
    double minimum = result.top_radius_km;
    for (size_t i = 0; i < nodes; ++i) {
        double r = result.radius_at(double(i)/double(nodes-1));
        result.node_radii[i] = r;
        for (int c = 0; c < 3; ++c) result.node_invariant[i][c] = r >= result.top_radius_km ? r : result.invariant(r, c);
        minimum = std::min(minimum, result.node_invariant[i].y);
    }
    result.impact_min_km = minimum;
    for (int c = 0; c < 3; ++c) {
        result.monotone[c] = true;
        for (size_t i = 1; i < nodes; ++i)
            if (result.node_invariant[i][c] <= result.node_invariant[i-1][c]) result.monotone[c] = false;
    }
    double ground_invariant = result.invariant(result.ground_radius_km), critical = ground_invariant;
    for (size_t i = 0; i < nodes; ++i)
        if (result.node_radii[i] > result.ground_radius_km && result.node_radii[i] < result.top_radius_km &&
            result.node_invariant[i].y < critical) {
            critical = result.node_invariant[i].y;
            result.critical_radius_km = result.node_radii[i];
        }
    result.impact_scale_km = std::max(result.scale_km, 1e-3);
    result.travel.assign(REFRACTION_IMPACT_SAMPLES*REFRACTION_TRAVEL_SAMPLES, 0.0);
    auto build_row = [&](size_t i) {
        double t = double(i)/double(REFRACTION_IMPACT_SAMPLES-1);
        double p = result.impact_at(t);
        if (i+1 == REFRACTION_IMPACT_SAMPLES) p = result.top_radius_km;
        result.impact[i] = p;
        double turn = result.turning_radius(p);
        if (std::isnan(turn)) turn = result.bottom_radius_km;
        result.turn_radius[i] = turn;
        double u_max = std::sqrt(std::max(result.top_radius_km-turn, 0.0));
        double* row = &result.travel[i*REFRACTION_TRAVEL_SAMPLES];
        double accumulated = 0;
        for (size_t j = 1; j < REFRACTION_TRAVEL_SAMPLES; ++j) {
            double a = u_max*double(j-1)/double(REFRACTION_TRAVEL_SAMPLES-1);
            double b = u_max*double(j)/double(REFRACTION_TRAVEL_SAMPLES-1);
            for (int q = 0; q < 4; ++q) {
                double u = .5*(a+b)+.5*(b-a)*PreparedRefraction::GL4[q].x;
                double rr = turn+u*u;
                double f = result.invariant(rr);
                accumulated += .5*(b-a)*PreparedRefraction::GL4[q].y*2*u*p/(rr*std::sqrt(std::max((f-p)*(f+p), 1e-300)));
            }
            row[j] = u_max > 0 ? accumulated/u_max : double(j)/double(REFRACTION_TRAVEL_SAMPLES-1)*std::sqrt(2/p);
        }
        double vacuum = PreparedRefraction::vacuum_travel(p, result.top_radius_km);
        result.deflection[i].y = 2*(accumulated-vacuum);
        for (int c : {0, 2}) {
            double channel_turn = result.turning_radius(p, c);
            result.deflection[i][c] = std::isnan(channel_turn) ? std::numeric_limits<double>::quiet_NaN() :
                2*(result.travel_from_turn(p, result.top_radius_km, channel_turn, c, 32)-vacuum);
        }
    };
    unsigned workers = std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
    std::vector<std::thread> threads;
    for (unsigned w = 1; w < workers; ++w)
        threads.emplace_back([&, w] { for (size_t i = w; i < REFRACTION_IMPACT_SAMPLES; i += workers) build_row(i); });
    for (size_t i = 0; i < REFRACTION_IMPACT_SAMPLES; i += workers) build_row(i);
    for (auto& thread : threads) thread.join();
    return result;
}

}
