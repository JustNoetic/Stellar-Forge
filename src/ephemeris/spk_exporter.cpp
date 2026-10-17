#include "ephemeris/spk_exporter.hpp"
#include "core/constants.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <set>

namespace sf::ephemeris {

namespace {

std::string sanitize_segid(std::string_view name) {
    std::string clean;
    clean.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            clean += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        } else if (clean.empty() || clean.back() != '_') {
            clean += '_';
        }
    }
    while (!clean.empty() && clean.back() == '_') clean.pop_back();
    if (clean.empty()) clean = "BODY";
    std::string segid = "SF_TIMELINE_" + clean;
    if (segid.size() > 40) segid.resize(40);
    return segid;
}

std::vector<int> assign_spice_ids(const SystemDefinition& def) {
    std::vector<int> ids;
    ids.reserve(def.bodies.size());
    std::set<int> used;

    for (const auto& body : def.bodies) {
        if (body.spice_id > 0) {
            used.insert(body.spice_id);
        }
    }

    int next_fictional = -100001;
    for (const auto& body : def.bodies) {
        int assigned = 0;
        if (body.spice_id > 0 && !used.contains(body.spice_id)) {
            assigned = body.spice_id;
        } else if (body.spice_id > 0) {
            assigned = body.spice_id;
        } else if (auto* kb = SpiceEphemeris::find_known_body_by_name(body.name)) {
            assigned = kb->id;
        }
        if (assigned == 0 || std::find(ids.begin(), ids.end(), assigned) != ids.end()) {
            while (used.contains(next_fictional)) {
                --next_fictional;
            }
            assigned = next_fictional--;
        }
        ids.push_back(assigned);
        used.insert(assigned);
    }
    return ids;
}

glm::dvec3 evaluate_hermite_pos(double t, const std::vector<double>& epochs,
                                const std::vector<glm::dvec3>& pos,
                                const std::vector<glm::dvec3>& vel) {
    size_t n = epochs.size();
    if (n == 0) return glm::dvec3(0.0);
    if (n == 1 || t <= epochs.front()) return pos.front();
    if (t >= epochs.back()) return pos.back();

    auto it = std::upper_bound(epochs.begin(), epochs.end(), t);
    size_t k = (it == epochs.end()) ? n - 2 : static_cast<size_t>(std::max<ptrdiff_t>(0, (it - epochs.begin()) - 1));
    if (k >= n - 1) k = n - 2;

    double t0 = epochs[k];
    double t1 = epochs[k + 1];
    double h = t1 - t0;
    if (h <= 0.0) return pos[k];

    double s = std::clamp((t - t0) / h, 0.0, 1.0);
    double s2 = s * s;
    double s3 = s2 * s;

    double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    double h10 = s3 - 2.0 * s2 + s;
    double h01 = -2.0 * s3 + 3.0 * s2;
    double h11 = s3 - s2;

    return h00 * pos[k] + (h10 * h) * vel[k] + h01 * pos[k + 1] + (h11 * h) * vel[k + 1];
}

double clenshaw_eval(int degree, const double* coeffs, double tau) {
    double b_k1 = 0.0, b_k2 = 0.0;
    for (int k = degree; k >= 1; --k) {
        double b_k = 2.0 * tau * b_k1 - b_k2 + coeffs[k];
        b_k2 = b_k1;
        b_k1 = b_k;
    }
    return tau * b_k1 - b_k2 + coeffs[0];
}

struct ChebyshevFitResult {
    std::vector<double> cdata;
    double intlen_s = 0.0;
    int n_intervals = 1;
    double max_resid_km = 0.0;
};

ChebyshevFitResult fit_chebyshev(const std::vector<double>& epochs,
                                 const std::vector<glm::dvec3>& pos_km,
                                 const std::vector<glm::dvec3>& vel_kms,
                                 int degree, double tol_km,
                                 const std::atomic<bool>* cancel_flag) {
    const double total_span = epochs.back() - epochs.front();
    const int nc = degree + 1;

    std::vector<double> roots(nc);
    for (int m = 0; m < nc; ++m) {
        roots[m] = std::cos((2.0 * m + 1.0) / (2.0 * nc) * std::numbers::pi);
    }

    std::vector<std::vector<double>> cos_weights(nc, std::vector<double>(nc));
    for (int j = 0; j < nc; ++j) {
        double scale = (j == 0) ? (1.0 / nc) : (2.0 / nc);
        for (int m = 0; m < nc; ++m) {
            cos_weights[j][m] = scale * std::cos(j * (2.0 * m + 1.0) / (2.0 * nc) * std::numbers::pi);
        }
    }

    int n_int = std::clamp(static_cast<int>(epochs.size() / 16), 1, 64);
    constexpr int MAX_INT = 16384;

    ChebyshevFitResult best{};

    while (n_int <= MAX_INT) {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) break;

        double intlen = total_span / n_int;
        std::vector<double> cdata(n_int * 3 * nc, 0.0);
        double max_resid = 0.0;

        for (int i = 0; i < n_int; ++i) {
            double a = epochs.front() + i * intlen;
            double b = a + intlen;
            double t_mid = 0.5 * (a + b);
            double r_rad = 0.5 * intlen;

            std::vector<glm::dvec3> sampled_pos(nc);
            for (int m = 0; m < nc; ++m) {
                double t_m = t_mid + r_rad * roots[m];
                sampled_pos[m] = evaluate_hermite_pos(t_m, epochs, pos_km, vel_kms);
            }

            double* rec = &cdata[i * 3 * nc];
            double* rec_x = rec;
            double* rec_y = rec + nc;
            double* rec_z = rec + 2 * nc;

            for (int j = 0; j < nc; ++j) {
                double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
                for (int m = 0; m < nc; ++m) {
                    double w = cos_weights[j][m];
                    sum_x += w * sampled_pos[m].x;
                    sum_y += w * sampled_pos[m].y;
                    sum_z += w * sampled_pos[m].z;
                }
                rec_x[j] = sum_x;
                rec_y[j] = sum_y;
                rec_z[j] = sum_z;
            }

            glm::dvec3 p_test = evaluate_hermite_pos(t_mid, epochs, pos_km, vel_kms);
            glm::dvec3 p_poly(
                clenshaw_eval(degree, rec_x, 0.0),
                clenshaw_eval(degree, rec_y, 0.0),
                clenshaw_eval(degree, rec_z, 0.0)
            );
            double resid = glm::length(p_test - p_poly);
            max_resid = std::max(max_resid, resid);
        }

        best = {std::move(cdata), intlen, n_int, max_resid};
        if (max_resid <= tol_km || n_int * 2 > MAX_INT) {
            break;
        }
        n_int *= 2;
    }

    return best;
}

} // namespace

SpkExportResult SpkExporter::export_timeline(
    const std::filesystem::path& destination_bsp,
    const TimelineRecording& recording,
    const SystemDefinition& system_def,
    const ReferenceHierarchy& hierarchy,
    std::shared_ptr<SpiceEphemeris> spice,
    const SpkExportOptions& options) {

    if (!spice || !spice->can_write_spk()) {
        return {false, "CSPICE library not loaded or does not support SPK writing."};
    }
    if (recording.empty() || recording.num_steps() < 2) {
        return {false, "Timeline recording must contain at least 2 steps."};
    }

    const size_t num_steps = recording.num_steps();
    const size_t num_bodies = recording.num_bodies();
    if (num_bodies == 0 || num_bodies != system_def.bodies.size()) {
        return {false, "Timeline body count does not match system definition."};
    }

    std::error_code ec;
    std::filesystem::create_directories(destination_bsp.parent_path(), ec);

    double epoch_et = 820540869.184;
    if (!system_def.epoch_utc.empty()) {
        double et = 0.0;
        if (spice->epoch(system_def.epoch_utc, et)) {
            epoch_et = et;
        }
    }

    std::vector<double> epochs(num_steps);
    for (size_t k = 0; k < num_steps; ++k) {
        epochs[k] = epoch_et + units::to_seconds(recording.times[k]).value;
    }

    std::vector<int> body_ids = assign_spice_ids(system_def);
    std::vector<int> center_ids(num_bodies, 0);

    if (options.use_hierarchical_center) {
        for (size_t i = 0; i < num_bodies; ++i) {
            int p = hierarchy.parent_index(static_cast<int>(i));
            if (p >= 0 && static_cast<size_t>(p) < num_bodies && p != static_cast<int>(i)) {
                center_ids[i] = body_ids[p];
            } else {
                center_ids[i] = 0;
            }
        }
    }

    std::filesystem::path temp_bsp = destination_bsp;
    temp_bsp += ".part";
    std::filesystem::remove(temp_bsp, ec);

    int handle = 0;
    auto open_res = spice->spk_open(temp_bsp, "STELLAR_FORGE_TIMELINE", handle);
    if (!open_res) {
        return {false, "Failed to open SPK file for writing: " + open_res.error};
    }

    nlohmann::json fit_stats = nlohmann::json::array();
    bool write_ok = true;
    std::string write_err;

    for (size_t i = 0; i < num_bodies; ++i) {
        if (options.cancel_flag && options.cancel_flag->load(std::memory_order_relaxed)) {
            write_ok = false;
            write_err = "Export cancelled by user.";
            break;
        }

        std::string segid = sanitize_segid(system_def.bodies[i].name);
        int center_id = center_ids[i];
        int parent_idx = hierarchy.parent_index(static_cast<int>(i));

        std::vector<glm::dvec3> pos_km(num_steps);
        std::vector<glm::dvec3> vel_kms(num_steps);

        for (size_t k = 0; k < num_steps; ++k) {
            const auto& m = recording.motions[k * num_bodies + i];
            glm::dvec3 p = m.position.value * AU_TO_KM;
            glm::dvec3 v = m.velocity.value * (AU_TO_KM / SECONDS_PER_YEAR);

            if (center_id != 0 && parent_idx >= 0 && static_cast<size_t>(parent_idx) < num_bodies) {
                const auto& mp = recording.motions[k * num_bodies + parent_idx];
                p -= mp.position.value * AU_TO_KM;
                v -= mp.velocity.value * (AU_TO_KM / SECONDS_PER_YEAR);
            }
            pos_km[k] = p;
            vel_kms[k] = v;
        }

        if (options.spk_type == 13) {
            std::vector<double> states(num_steps * 6);
            for (size_t k = 0; k < num_steps; ++k) {
                states[k * 6 + 0] = pos_km[k].x;
                states[k * 6 + 1] = pos_km[k].y;
                states[k * 6 + 2] = pos_km[k].z;
                states[k * 6 + 3] = vel_kms[k].x;
                states[k * 6 + 4] = vel_kms[k].y;
                states[k * 6 + 5] = vel_kms[k].z;
            }

            int deg = std::clamp(options.hermite_degree, 1, 7);
            if (deg % 2 == 0) --deg;
            auto wr = spice->spk_write_type13(handle, body_ids[i], center_id, "ECLIPJ2000",
                                             epochs.front(), epochs.back(), segid,
                                             deg, static_cast<int>(num_steps),
                                             states.data(), epochs.data());
            if (!wr) {
                write_ok = false;
                write_err = "Failed to write SPK Type 13 for " + system_def.bodies[i].name + ": " + wr.error;
                break;
            }
            fit_stats.push_back({
                {"body", system_def.bodies[i].name},
                {"spice_id", body_ids[i]},
                {"center_id", center_id},
                {"type", 13},
                {"hermite_degree", deg}
            });
        } else {
            int deg = std::clamp(options.cheb_degree, 3, 27);
            auto fit = fit_chebyshev(epochs, pos_km, vel_kms, deg, options.cheb_tol_km, options.cancel_flag);

            auto wr = spice->spk_write_type2(handle, body_ids[i], center_id, "ECLIPJ2000",
                                             epochs.front(), epochs.back(), segid,
                                             fit.intlen_s, fit.n_intervals, deg,
                                             fit.cdata.data(), epochs.front());
            if (!wr) {
                write_ok = false;
                write_err = "Failed to write SPK Type 2 for " + system_def.bodies[i].name + ": " + wr.error;
                break;
            }
            fit_stats.push_back({
                {"body", system_def.bodies[i].name},
                {"spice_id", body_ids[i]},
                {"center_id", center_id},
                {"type", 2},
                {"cheb_degree", deg},
                {"intervals", fit.n_intervals},
                {"interval_seconds", fit.intlen_s},
                {"max_fit_residual_km", fit.max_resid_km}
            });
        }

        if (options.progress_callback) {
            float frac = static_cast<float>(i + 1) / static_cast<float>(num_bodies);
            options.progress_callback(frac, "Encoded " + system_def.bodies[i].name);
        }
    }

    spice->spk_close(handle);

    if (!write_ok) {
        std::filesystem::remove(temp_bsp, ec);
        return {false, write_err};
    }

    std::filesystem::remove(destination_bsp, ec);
    std::filesystem::rename(temp_bsp, destination_bsp, ec);
    if (ec) {
        return {false, "Failed to move temporary SPK to destination: " + ec.message()};
    }

    nlohmann::json metadata = {
        {"generator", "Stellar Forge 2.0"},
        {"spk_type", options.spk_type},
        {"frame", "ECLIPJ2000"},
        {"epoch_et_seconds_past_j2000", epoch_et},
        {"display_epoch_utc", system_def.epoch_utc},
        {"time_span_years", recording.times.back().value - recording.times.front().value},
        {"time_span_et", epochs.back() - epochs.front()},
        {"fit_stats", fit_stats}
    };

    if (options.spk_type == 2) {
        metadata["chebyshev"] = {
            {"degree", options.cheb_degree},
            {"tolerance_km", options.cheb_tol_km}
        };
    } else {
        metadata["hermite_degree"] = options.hermite_degree;
    }

    nlohmann::json sidecar_bodies = nlohmann::json::array();
    for (size_t i = 0; i < num_bodies; ++i) {
        char color_buf[16]{};
        int r = static_cast<int>(std::clamp(system_def.bodies[i].color_rgb.r * 255.0f, 0.0f, 255.0f));
        int g = static_cast<int>(std::clamp(system_def.bodies[i].color_rgb.g * 255.0f, 0.0f, 255.0f));
        int b = static_cast<int>(std::clamp(system_def.bodies[i].color_rgb.b * 255.0f, 0.0f, 255.0f));
        std::snprintf(color_buf, sizeof(color_buf), "#%02x%02x%02x", r, g, b);

        sidecar_bodies.push_back({
            {"name", system_def.bodies[i].name},
            {"spice_id", body_ids[i]},
            {"center_id", center_ids[i]},
            {"color", color_buf}
        });
    }
    metadata["bodies"] = sidecar_bodies;

    std::filesystem::path sidecar_path = destination_bsp;
    sidecar_path += ".json";
    paths::write_json(sidecar_path, metadata);

    return {true, {}, destination_bsp, sidecar_path, std::move(metadata)};
}

} // namespace sf::ephemeris
