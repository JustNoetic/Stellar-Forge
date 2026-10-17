#include "ephemeris/spice_ephemeris.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/math_utils.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#else
#include <dlfcn.h>
#endif

namespace sf::ephemeris {
namespace {

template <typename Function>
Function symbol(void* library, const char* name) {
#ifdef _WIN32
    auto address = GetProcAddress(static_cast<HMODULE>(library), name);
#else
    auto address = dlsym(library, name);
#endif
    if (!address) throw std::runtime_error(std::string("CSPICE runtime is missing ") + name);
    return reinterpret_cast<Function>(address);
}

std::string to_upper(std::string_view sv) {
    std::string out(sv);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

std::string to_lower(std::string_view sv) {
    std::string out(sv);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string title_case_spice_name(std::string_view raw) {
    std::string out(raw);
    bool new_word = true;
    for (char& c : out) {
        if (c == '_') c = ' ';
        if ( std::isspace(static_cast<unsigned char>(c))) {
            new_word = true;
        } else if (new_word) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            new_word = false;
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return out;
}

static const std::vector<KnownSpiceBody>& known_spice_bodies() {
    static const std::vector<KnownSpiceBody> table = {
        {10, "Sun", "Star", "#fff5e6", 696340.0, 1.98847e30, "", false, "", ""},
        {199, "Mercury", "Terrestrial", "#a8a8a8", 2439.7, 3.3011e23, "Sun", false, "", ""},
        {299, "Venus", "Terrestrial", "#e0c8a0", 6051.8, 4.8675e24, "Sun", false, "", ""},
        {399, "Earth", "Terrestrial", "#3b5d9c", 6371.0, 5.972e24, "Sun", false, "", ""},
        {301, "Moon", "Moon", "#d0d0d0", 1737.4, 7.342e22, "Earth", false, "", ""},
        {-1024, "Artemis II (Orion)", "Spacecraft", "#00ffff", 0.010, 26000.0, "Earth", true, "2026-04-02 02:00:00", "2026-04-10 23:50:00"},
        {-82, "Cassini", "Spacecraft", "#e0a040", 0.010, 5712.0, "Saturn", true, "2004-07-01 14:00:00", "2017-09-15 09:58:00"},
        {-170, "JWST", "Spacecraft", "#ffd700", 0.010, 6500.0, "Earth", true, "", ""},
        {499, "Mars", "Terrestrial", "#c1440e", 3389.5, 6.4171e23, "Sun", false, "", ""},
        {4, "Mars", "Terrestrial", "#c1440e", 3389.5, 6.4171e23, "Sun", false, "", ""},
        {401, "Phobos", "Moon", "#a0a0a0", 11.26, 1.0659e16, "Mars", false, "", ""},
        {402, "Deimos", "Moon", "#a0a0a0", 6.2, 1.4762e15, "Mars", false, "", ""},
        {599, "Jupiter", "Gas Giant", "#c88b3a", 69911.0, 1.8982e27, "Sun", false, "", ""},
        {5, "Jupiter", "Gas Giant", "#c88b3a", 69911.0, 1.8982e27, "Sun", false, "", ""},
        {501, "Io", "Moon", "#c1b446", 1821.6, 8.9319e22, "Jupiter", false, "", ""},
        {502, "Europa", "Moon", "#9b7f59", 1560.8, 4.7998e22, "Jupiter", false, "", ""},
        {503, "Ganymede", "Moon", "#8b8173", 2631.2, 1.4819e23, "Jupiter", false, "", ""},
        {504, "Callisto", "Moon", "#5e5850", 2410.3, 1.0759e23, "Jupiter", false, "", ""},
        {505, "Amalthea", "Moon", "#e65f5c", 83.56, 2.068e18, "Jupiter", false, "", ""},
        {514, "Thebe", "Moon", "#5c544e", 48.74, 4.195e17, "Jupiter", false, "", ""},
        {515, "Adrastea", "Moon", "#78716c", 8.36, 1.948e15, "Jupiter", false, "", ""},
        {516, "Metis", "Moon", "#57534e", 21.59, 3.596e16, "Jupiter", false, "", ""},
        {699, "Saturn", "Gas Giant", "#e3d599", 58232.0, 5.6834e26, "Sun", false, "", ""},
        {6, "Saturn", "Gas Giant", "#e3d599", 58232.0, 5.6834e26, "Sun", false, "", ""},
        {601, "Mimas", "Moon", "#a0a0a0", 198.2, 3.7493e19, "Saturn", false, "", ""},
        {602, "Enceladus", "Moon", "#ffffff", 252.1, 1.0802e20, "Saturn", false, "", ""},
        {603, "Tethys", "Moon", "#a0a0a0", 531.1, 6.1744e20, "Saturn", false, "", ""},
        {604, "Dione", "Moon", "#a0a0a0", 561.4, 1.0954e21, "Saturn", false, "", ""},
        {605, "Rhea", "Moon", "#a0a0a0", 763.8, 2.3065e21, "Saturn", false, "", ""},
        {606, "Titan", "Moon", "#e0a040", 2574.7, 1.3452e23, "Saturn", false, "", ""},
        {607, "Hyperion", "Moon", "#a8a29e", 135.1, 5.584e18, "Saturn", false, "", ""},
        {608, "Iapetus", "Moon", "#505050", 734.5, 1.8056e21, "Saturn", false, "", ""},
        {609, "Phoebe", "Moon", "#374151", 106.5, 8.289e18, "Saturn", false, "", ""},
        {612, "Helene", "Moon", "#e2e8f0", 17.4, 2.547e16, "Saturn", false, "", ""},
        {613, "Telesto", "Moon", "#f1f5f9", 11.8, 7.192e15, "Saturn", false, "", ""},
        {614, "Calypso", "Moon", "#f8fafc", 10.4, 3.596e15, "Saturn", false, "", ""},
        {632, "Methone", "Moon", "#cbd5e1", 1.39, 1.498e13, "Saturn", false, "", ""},
        {634, "Polydeuces", "Moon", "#94a3b8", 0.70, 4.495e12, "Saturn", false, "", ""},
        {799, "Uranus", "Ice Giant", "#4b70dd", 25362.0, 8.6810e25, "Sun", false, "", ""},
        {7, "Uranus", "Ice Giant", "#4b70dd", 25362.0, 8.6810e25, "Sun", false, "", ""},
        {701, "Ariel", "Moon", "#a0a0a0", 578.9, 1.353e21, "Uranus", false, "", ""},
        {702, "Umbriel", "Moon", "#808080", 584.7, 1.275e21, "Uranus", false, "", ""},
        {703, "Titania", "Moon", "#a0a0a0", 788.4, 3.400e21, "Uranus", false, "", ""},
        {704, "Oberon", "Moon", "#808080", 761.4, 3.076e21, "Uranus", false, "", ""},
        {705, "Miranda", "Moon", "#a0a0a0", 235.8, 6.59e19, "Uranus", false, "", ""},
        {715, "Puck", "Moon", "#3a3a3a", 80.8, 2.892e18, "Uranus", false, "", ""},
        {899, "Neptune", "Ice Giant", "#274687", 24622.0, 1.0241e26, "Sun", false, "", ""},
        {8, "Neptune", "Ice Giant", "#274687", 24622.0, 1.0241e26, "Sun", false, "", ""},
        {801, "Triton", "Moon", "#a0a0a0", 1353.4, 2.14e22, "Neptune", false, "", ""},
        {802, "Nereid", "Moon", "#a0a0a0", 170.0, 3.1e19, "Neptune", false, "", ""},
        {803, "Naiad", "Moon", "#888888", 29.9, 1.888e17, "Neptune", false, "", ""},
        {804, "Thalassa", "Moon", "#808080", 39.7, 3.506e17, "Neptune", false, "", ""},
        {805, "Despina", "Moon", "#7d7d7d", 73.8, 2.128e18, "Neptune", false, "", ""},
        {806, "Galatea", "Moon", "#7a7a7a", 79.4, 2.113e18, "Neptune", false, "", ""},
        {807, "Larissa", "Moon", "#6e6e6e", 96.8, 4.900e18, "Neptune", false, "", ""},
        {808, "Proteus", "Moon", "#505050", 200.5, 4.390e19, "Neptune", false, "", ""},
        {814, "Hippocamp", "Moon", "#555555", 17.4, 4.944e16, "Neptune", false, "", ""},
        {999, "Pluto", "Dwarf Planet", "#ddc8b8", 1188.3, 1.303e22, "Sun", false, "", ""},
        {9, "Pluto", "Dwarf Planet", "#ddc8b8", 1188.3, 1.303e22, "Sun", false, "", ""},
        {901, "Charon", "Moon", "#a0a0a0", 606.0, 1.586e21, "Pluto", false, "", ""},
        {902, "Nix", "Moon", "#a0a0a0", 19.0, 4.5e16, "Pluto", false, "", ""},
        {903, "Hydra", "Moon", "#a0a0a0", 19.0, 4.8e16, "Pluto", false, "", ""},
        {904, "Kerberos", "Moon", "#a0a0a0", 6.0, 1.65e16, "Pluto", false, "", ""},
        {905, "Styx", "Moon", "#a0a0a0", 5.0, 7.5e15, "Pluto", false, "", ""}
    };
    return table;
}

static std::mutex g_spice_instance_mutex;
static std::weak_ptr<SpiceEphemeris> g_active_spice;

} // namespace

const std::vector<DefaultKernelInfo>& SpiceEphemeris::default_kernels() {
    static const std::vector<DefaultKernelInfo> catalog = {
        {"naif0012.tls", "Required Core", "Leapseconds (Required)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls", true},
        {"pck00010.tpc", "Required Core", "Planetary Constants (Required)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc", true},
        {"de440s.bsp", "Required Core", "Planetary Ephemeris (Required)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp", true},
        {"mar099s.bsp", "Mars Moons", "Mars Moons", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/mar099s.bsp", true},
        {"jup365.bsp", "Jupiter Moons", "Major Jupiter Moons (1.1GB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp", true},
        {"jup347.bsp", "Jupiter Moons", "Irregular Jupiter Moons (879MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup347.bsp", false},
        {"jup348.bsp", "Jupiter Moons", "Irregular Jupiter Moons (57MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup348.bsp", false},
        {"jup349.bsp", "Jupiter Moons", "Irregular Jupiter Moons (93MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup349.bsp", false},
        {"sat441.bsp", "Saturn Moons", "Major Saturn Moons (631MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp", true},
        {"sat455.bsp", "Saturn Moons", "Irregular Saturn Moons (278MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat455.bsp", false},
        {"sat456.bsp", "Saturn Moons", "Irregular Saturn Moons (115MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat456.bsp", false},
        {"sat457.bsp", "Saturn Moons", "Irregular Saturn Moons (190MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat457.bsp", false},
        {"sat459.bsp", "Saturn Moons", "Irregular Saturn Moons (80MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat459.bsp", false},
        {"ura111.bsp", "Uranus Moons", "Major Uranus Moons (Miranda, Ariel, etc.)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/ura111.bsp", false},
        {"nep095.bsp", "Neptune Moons", "Major & Inner Neptune Moons (Triton, Proteus, etc.)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep095.bsp", true},
        {"nep105.bsp", "Neptune Moons", "Nereid Ephemeris (201MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep105.bsp", false},
        {"nep104.bsp", "Neptune Moons", "Irregular Neptune Moons (318MB)", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep104.bsp", false},
        {"plu060.bsp", "Pluto System", "Pluto Moons", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/plu060.bsp", true}
    };
    return catalog;
}

const std::vector<std::string>& SpiceEphemeris::kernel_groups() {
    static const std::vector<std::string> groups = {
        "Required Core",
        "Mars Moons",
        "Jupiter Moons",
        "Saturn Moons",
        "Uranus Moons",
        "Neptune Moons",
        "Pluto System"
    };
    return groups;
}

const KnownSpiceBody* SpiceEphemeris::find_known_body(int spice_id) {
    for (const auto& entry : known_spice_bodies()) {
        if (entry.id == spice_id) return &entry;
    }
    return nullptr;
}

const KnownSpiceBody* SpiceEphemeris::find_known_body_by_name(std::string_view name) {
    std::string target = to_upper(name);
    for (const auto& entry : known_spice_bodies()) {
        if (to_upper(entry.name) == target) return &entry;
    }
    return nullptr;
}

std::shared_ptr<SpiceEphemeris> SpiceEphemeris::open(std::string& error) {
    std::lock_guard lock(g_spice_instance_mutex);
    if (auto source = g_active_spice.lock()) return source;
    auto source = std::shared_ptr<SpiceEphemeris>(new SpiceEphemeris);
    auto status = source->initialize();
    if (!status) {
        error = status.error;
        return {};
    }
    g_active_spice = source;
    return source;
}

std::shared_ptr<SpiceEphemeris> SpiceEphemeris::reload(std::string& error) {
    std::lock_guard lock(g_spice_instance_mutex);
    if (auto existing = g_active_spice.lock()) {
        auto status = existing->initialize();
        if (!status) {
            error = status.error;
            return {};
        }
        return existing;
    }
    auto source = std::shared_ptr<SpiceEphemeris>(new SpiceEphemeris);
    auto status = source->initialize();
    if (!status) {
        error = status.error;
        return {};
    }
    g_active_spice = source;
    return source;
}

CommandResult SpiceEphemeris::initialize() {
    std::lock_guard lock(mutex_);
    auto config = paths::read_json(paths::external_path("data", "ephemeris.json"));
    if (!config || !config->is_object()) {
        return {false, false, "Ephemeris requires data/ephemeris.json and a local CSPICE runtime/kernel set."};
    }
    try {
        if (!library_) {
            auto runtime = paths::external_path("data", "kernels", config->at("runtime").get<std::string>());
#ifdef _WIN32
            library_ = LoadLibraryExW(runtime.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
            library_ = dlopen(runtime.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            if (!library_) return {false, false, "Cannot load CSPICE runtime: " + runtime.string()};
            furnish_ = symbol<decltype(furnish_)>(library_, "furnsh_c");
            unload_ = symbol<decltype(unload_)>(library_, "unload_c");
            clear_ = symbol<decltype(clear_)>(library_, "kclear_c");
            ktotal_ = symbol<decltype(ktotal_)>(library_, "ktotal_c");
            kdata_ = symbol<decltype(kdata_)>(library_, "kdata_c");
            dafopr_ = symbol<decltype(dafopr_)>(library_, "dafopr_c");
            dafbfs_ = symbol<decltype(dafbfs_)>(library_, "dafbfs_c");
            daffna_ = symbol<decltype(daffna_)>(library_, "daffna_c");
            dafgs_ = symbol<decltype(dafgs_)>(library_, "dafgs_c");
            spkuds_ = symbol<decltype(spkuds_)>(library_, "spkuds_c");
            dafcls_ = symbol<decltype(dafcls_)>(library_, "dafcls_c");
            bodc2n_ = symbol<decltype(bodc2n_)>(library_, "bodc2n_c");
            bodn2c_ = symbol<decltype(bodn2c_)>(library_, "bodn2c_c");
            bodvrd_ = symbol<decltype(bodvrd_)>(library_, "bodvrd_c");
            et2utc_ = symbol<decltype(et2utc_)>(library_, "et2utc_c");
            error_action_ = symbol<decltype(error_action_)>(library_, "erract_c");
            error_print_ = symbol<decltype(error_print_)>(library_, "errprt_c");
            failed_ = symbol<decltype(failed_)>(library_, "failed_c");
            message_ = symbol<decltype(message_)>(library_, "getmsg_c");
            reset_ = symbol<decltype(reset_)>(library_, "reset_c");
            to_et_ = symbol<decltype(to_et_)>(library_, "str2et_c");
            state_ = symbol<decltype(state_)>(library_, "spkgeo_c");
        }
        reset_();
        clear_();
        loaded_kernels_.clear();
        char action[] = "RETURN", printing[] = "NONE";
        error_action_("SET", 0, action);
        error_print_("SET", 0, printing);
        auto status = result();
        if (!status) return status;
        for (const auto& entry : config->at("kernels")) {
            auto rel = entry.get<std::string>();
            auto path = paths::external_path("data", "kernels", rel);
            if (!std::filesystem::is_regular_file(path)) {
                return {false, false, "Missing ephemeris kernel: " + path.string()};
            }
            furnish_(path.string().c_str());
            status = result();
            if (!status) return status;
            loaded_kernels_.push_back(path);
        }
    } catch (const std::exception& error) {
        return {false, false, error.what()};
    }
    return {};
}

SpiceEphemeris::~SpiceEphemeris() {
    std::lock_guard lock(mutex_);
    if (clear_) clear_();
    if (!library_) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(library_));
#else
    dlclose(library_);
#endif
}

CommandResult SpiceEphemeris::result() const {
    if (!failed_()) return {};
    char message[2048]{};
    message_("LONG", sizeof(message), message);
    reset_();
    return {false, false, message};
}

CommandResult SpiceEphemeris::furnish_file(const std::filesystem::path& path) {
    std::lock_guard lock(mutex_);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return {false, false, "Kernel file not found: " + path.string()};
    }
    for (const auto& existing : loaded_kernels_) {
        if (std::filesystem::equivalent(existing, path, ec)) {
            return {};
        }
    }
    furnish_(path.string().c_str());
    auto status = result();
    if (status) loaded_kernels_.push_back(path);
    return status;
}

CommandResult SpiceEphemeris::epoch(std::string_view utc, double& et) const {
    std::lock_guard lock(mutex_);
    auto text = std::string(utc);
    to_et_(text.c_str(), &et);
    return result();
}

bool SpiceEphemeris::query_raw_state(int target, int observer, double et, double state_out[6]) const {
    double light_time = 0.0;
    state_(target, et, "ECLIPJ2000", observer, state_out, &light_time);
    if (!failed_()) return true;
    reset_();
    return false;
}

CommandResult SpiceEphemeris::state(int target, double et, BodyMotion& motion) const {
    return state_relative(target, 0, et, motion);
}

CommandResult SpiceEphemeris::state_relative(int target, int observer, double et, BodyMotion& motion) const {
    std::lock_guard lock(mutex_);
    if (!std::isfinite(et)) return {false, false, "Ephemeris time must be finite."};
    double state[6]{};
    double light_time = 0.0;
    state_(target, et, "ECLIPJ2000", observer, state, &light_time);
    if (failed_()) {
        char primary_msg[2048]{};
        message_("LONG", sizeof(primary_msg), primary_msg);
        reset_();
        bool recovered = false;
        if (target > 100 && target < 1000 && target % 100 == 99) {
            recovered = query_raw_state(target / 100, observer, et, state);
        }
        if (!recovered) {
            return {false, false, primary_msg};
        }
    }
    for (double value : state) {
        if (!std::isfinite(value)) return {false, false, "SPICE returned a nonfinite state."};
    }
    motion.position = EclipticPosition{glm::dvec3(state[0], state[1], state[2]) * KM_TO_AU};
    motion.velocity = EclipticVelocity{glm::dvec3(state[3], state[4], state[5]) * (KM_TO_AU * SECONDS_PER_YEAR)};
    return {};
}

std::string SpiceEphemeris::body_name(int spice_id) const {
    std::lock_guard lock(mutex_);
    if (!bodc2n_) return {};
    char buf[128]{};
    int found = 0;
    bodc2n_(spice_id, sizeof(buf), buf, &found);
    if (failed_()) {
        reset_();
        return {};
    }
    if (found && buf[0] != '\0') {
        return title_case_spice_name(buf);
    }
    return {};
}

std::string SpiceEphemeris::et_to_utc(double et, const char* format) const {
    std::lock_guard lock(mutex_);
    if (!et2utc_ || !std::isfinite(et)) return std::format("{:.0f}s ET", et);
    char buf[128]{};
    et2utc_(et, format, 0, sizeof(buf), buf);
    if (failed_()) {
        reset_();
        return std::format("{:.0f}s ET", et);
    }
    return std::string(buf);
}

bool SpiceEphemeris::body_gm_and_radius(int spice_id, double fallback_mass_kg, double fallback_radius_km,
                                        double& mass_msun, double& radius_rsun) const {
    std::lock_guard lock(mutex_);
    mass_msun = fallback_mass_kg / 1.98847e30;
    radius_rsun = fallback_radius_km / SOLAR_RADIUS_KM;
    if (!bodvrd_) return false;

    double gm_sun = 1.3271244004193938e11;
    int dim = 0;
    double vals[8]{};
    bodvrd_("10", "GM", 1, &dim, vals);
    if (failed_()) reset_();
    else if (dim >= 1 && vals[0] > 0.0) gm_sun = vals[0];

    auto query_gm = [&](int id, double& out_gm) -> bool {
        std::string id_str = std::to_string(id);
        int d = 0;
        double v[4]{};
        bodvrd_(id_str.c_str(), "GM", 1, &d, v);
        if (failed_()) {
            reset_();
            return false;
        }
        if (d >= 1 && v[0] >= 0.0) {
            out_gm = v[0];
            return true;
        }
        return false;
    };

    double gm_body = -1.0;
    if (!query_gm(spice_id, gm_body)) {
        if (spice_id > 100 && spice_id < 1000 && spice_id % 100 == 99) {
            query_gm(spice_id / 100, gm_body);
        } else if (spice_id > 0 && spice_id < 10) {
            query_gm(spice_id * 100 + 99, gm_body);
        }
    }
    if (gm_body >= 0.0 && gm_sun > 0.0) {
        mass_msun = gm_body / gm_sun;
    }

    auto query_rad = [&](int id, double& out_r_km) -> bool {
        std::string id_str = std::to_string(id);
        int d = 0;
        double v[4]{};
        bodvrd_(id_str.c_str(), "RADII", 3, &d, v);
        if (failed_()) {
            reset_();
            return false;
        }
        if (d >= 1 && v[0] > 0.0) {
            out_r_km = v[0];
            return true;
        }
        return false;
    };

    double r_km = -1.0;
    if (!query_rad(spice_id, r_km)) {
        if (spice_id > 100 && spice_id < 1000 && spice_id % 100 == 99) {
            query_rad(spice_id / 100, r_km);
        } else if (spice_id > 0 && spice_id < 10) {
            query_rad(spice_id * 100 + 99, r_km);
        }
    }
    if (r_km > 0.0) {
        radius_rsun = r_km / SOLAR_RADIUS_KM;
    }
    return (gm_body >= 0.0 || r_km > 0.0);
}

BspInspectResult SpiceEphemeris::inspect_bsp(const std::filesystem::path& bsp_path) {
    std::lock_guard lock(mutex_);
    BspInspectResult out{};
    std::error_code ec;
    if (bsp_path.empty() || !std::filesystem::is_regular_file(bsp_path, ec)) {
        out.error = "File not found: " + bsp_path.string();
        return out;
    }

    auto abs_path = std::filesystem::absolute(bsp_path, ec);
    out.filepath = abs_path.string();
    out.filename = abs_path.filename().string();

    auto tls_default = paths::external_path("data", "kernels", "default", "naif0012.tls");
    if (!std::filesystem::is_regular_file(tls_default, ec)) {
        tls_default = paths::external_path("data", "kernels", "naif0012.tls");
    }
    if (std::filesystem::is_regular_file(tls_default, ec)) {
        furnish_file(tls_default);
    }

    int handle = 0;
    bool close_handle = false;
    int spk_count = 0;
    if (ktotal_ && kdata_) {
        ktotal_("SPK", &spk_count);
        if (failed_()) reset_();
        for (int i = 0; i < spk_count; ++i) {
            char kfile[512]{}, ftype[64]{}, src[256]{};
            int khandle = 0, found = 0;
            kdata_(i, "SPK", sizeof(kfile), sizeof(ftype), sizeof(src), kfile, ftype, src, &khandle, &found);
            if (failed_()) {
                reset_();
                continue;
            }
            if (found && std::filesystem::equivalent(abs_path, std::filesystem::path(kfile), ec)) {
                handle = khandle;
                close_handle = false;
                break;
            }
        }
    }

    if (handle == 0) {
        dafopr_(abs_path.string().c_str(), &handle);
        auto st = result();
        if (!st) {
            out.error = st.error;
            return out;
        }
        close_handle = true;
    }

    std::vector<int> body_ids;
    std::unordered_map<int, int> center_map;
    std::unordered_map<int, std::pair<double, double>> cov_map;

    dafbfs_(handle);
    if (failed_()) {
        auto st = result();
        if (close_handle) dafcls_(handle);
        out.error = st.error;
        return out;
    }

    while (true) {
        int found = 0;
        daffna_(&found);
        if (failed_()) {
            reset_();
            break;
        }
        if (!found) break;

        double sum[128]{};
        dafgs_(sum);
        if (failed_()) {
            reset_();
            break;
        }
        int body = 0, center = 0, frame = 0, type_id = 0, baddr = 0, eaddr = 0;
        double b_epoch = 0.0, e_epoch = 0.0;
        spkuds_(sum, &body, &center, &frame, &type_id, &b_epoch, &e_epoch, &baddr, &eaddr);
        if (failed_()) {
            reset_();
            continue;
        }

        if (std::find(body_ids.begin(), body_ids.end(), body) == body_ids.end()) {
            body_ids.push_back(body);
        }
        if (!center_map.contains(body)) {
            center_map[body] = center;
        }
        auto it = cov_map.find(body);
        if (it == cov_map.end()) {
            cov_map[body] = {b_epoch, e_epoch};
        } else {
            it->second.first = std::min(it->second.first, b_epoch);
            it->second.second = std::max(it->second.second, e_epoch);
        }
    }

    if (close_handle) {
        dafcls_(handle);
        if (failed_()) reset_();
    }

    if (body_ids.empty()) {
        out.error = "No valid SPK segments found in " + out.filename;
        return out;
    }

    std::filesystem::path sidecar_path = abs_path;
    sidecar_path += ".json";
    if (std::filesystem::is_regular_file(sidecar_path, ec)) {
        if (auto j = paths::read_json(sidecar_path); j && j->is_object()) {
            out.sidecar = *j;
            int spk_t = out.sidecar.value("spk_type", 2);
            if (spk_t == 2) {
                int deg = 13;
                if (out.sidecar.contains("chebyshev") && out.sidecar["chebyshev"].is_object()) {
                    deg = out.sidecar["chebyshev"].value("degree", 13);
                }
                out.format_desc = std::format("Type 2 Chebyshev (Degree {}) [Stellar-Forge Export]", deg);
            } else if (spk_t == 13) {
                int deg = out.sidecar.value("hermite_degree", 5);
                out.format_desc = std::format("Type 13 Hermite (Degree {}) [Stellar-Forge Export]", deg);
            }
        }
    }

    if (out.sidecar.is_object() && out.sidecar.contains("bodies") && out.sidecar["bodies"].is_array()) {
        std::vector<int> ordered;
        for (const auto& sb : out.sidecar["bodies"]) {
            if (sb.is_object() && sb.contains("spice_id")) {
                int sbid = sb.value("spice_id", 0);
                if (std::find(body_ids.begin(), body_ids.end(), sbid) != body_ids.end() &&
                    std::find(ordered.begin(), ordered.end(), sbid) == ordered.end()) {
                    ordered.push_back(sbid);
                }
            }
        }
        for (int bid : body_ids) {
            if (std::find(ordered.begin(), ordered.end(), bid) == ordered.end()) {
                ordered.push_back(bid);
            }
        }
        body_ids = std::move(ordered);
    } else {
        std::sort(body_ids.begin(), body_ids.end(), [](int a, int b) {
            bool a_neg = (a < 0);
            bool b_neg = (b < 0);
            if (a_neg != b_neg) return !a_neg;
            return a < b;
        });
    }

    static constexpr const char* kPalette[] = {
        "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
        "#39c5cf", "#f0883e", "#56d364", "#79c0ff", "#e3b341",
        "#ff7b72", "#d2a8ff", "#2ea043", "#a5d6ff", "#ffab70"
    };
    constexpr size_t kPaletteCount = sizeof(kPalette) / sizeof(kPalette[0]);

    double min_et = std::numeric_limits<double>::infinity();
    double max_et = -std::numeric_limits<double>::infinity();

    for (size_t idx = 0; idx < body_ids.size(); ++idx) {
        int bid = body_ids[idx];
        auto cov = cov_map[bid];
        min_et = std::min(min_et, cov.first);
        max_et = std::max(max_et, cov.second);

        std::string b_name;
        std::string b_color;
        if (out.sidecar.is_object() && out.sidecar.contains("bodies") && out.sidecar["bodies"].is_array()) {
            for (const auto& sb : out.sidecar["bodies"]) {
                if (sb.is_object() && sb.value("spice_id", 0) == bid) {
                    b_name = sb.value("name", "");
                    b_color = sb.value("color", "");
                    break;
                }
            }
        }
        if (b_name.empty()) {
            if (const auto* known = find_known_body(bid)) {
                b_name = known->name;
                b_color = known->color_hex;
            }
        }
        if (b_name.empty()) {
            b_name = body_name(bid);
        }
        if (b_name.empty()) {
            b_name = (bid < 0) ? std::format("Spacecraft ({})", bid) : std::format("Object {}", bid);
        }
        if (b_color.empty()) {
            b_color = kPalette[idx % kPaletteCount];
        }

        int cid = center_map[bid];
        std::string c_name;
        if (cid == 0) {
            c_name = "Solar System Barycenter";
        } else if (const auto* known_c = find_known_body(cid)) {
            c_name = known_c->name;
        } else {
            c_name = body_name(cid);
            if (c_name.empty()) c_name = std::format("Object {}", cid);
        }

        BspBodyInfo binfo{};
        binfo.id = bid;
        binfo.name = std::move(b_name);
        binfo.center_id = cid;
        binfo.center_name = std::move(c_name);
        binfo.color_hex = std::move(b_color);
        binfo.start_et = cov.first;
        binfo.end_et = cov.second;
        binfo.utc_start = et_to_utc(cov.first, "C");
        binfo.utc_end = et_to_utc(cov.second, "C");
        out.bodies.push_back(std::move(binfo));
    }

    if (!std::isfinite(min_et) || !std::isfinite(max_et)) {
        min_et = 0.0;
        max_et = 1.0;
    }
    out.min_et = min_et;
    out.max_et = max_et;
    out.utc_min = et_to_utc(min_et, "C");
    out.utc_max = et_to_utc(max_et, "C");
    out.duration_days = (max_et - min_et) / 86400.0;
    out.valid = true;
    return out;
}

SpiceKernelManager::SpiceKernelManager() {
    ensure_directories();
    load_settings();
}

SpiceKernelManager::~SpiceKernelManager() {
    cancel_download();
    if (download_thread_.joinable()) {
        download_thread_.join();
    }
}

void SpiceKernelManager::ensure_directories() const {
    std::error_code ec;
    std::filesystem::create_directories(paths::external_path("data", "kernels"), ec);
    std::filesystem::create_directories(paths::external_path("data", "kernels", "default"), ec);
    std::filesystem::create_directories(paths::external_path("data", "kernels", "additional"), ec);
}

void SpiceKernelManager::cleanup_partial_downloads() const {
    std::error_code ec;
    std::filesystem::path dirs[] = {
        paths::external_path("data", "kernels", "default"),
        paths::external_path("data", "kernels", "additional"),
        paths::external_path("data", "kernels")
    };
    for (const auto& d : dirs) {
        if (!std::filesystem::is_directory(d, ec)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(d, ec)) {
            if (entry.is_regular_file(ec) && entry.path().extension() == ".part") {
                std::filesystem::remove(entry.path(), ec);
            }
        }
    }
}

void SpiceKernelManager::load_settings() {
    enabled_kernels_.clear();
    for (const auto& dk : SpiceEphemeris::default_kernels()) {
        enabled_kernels_[dk.filename] = dk.essential;
    }

    auto manifest_path = paths::external_path("data", "ephemeris.json");
    if (auto manifest = paths::read_json(manifest_path); manifest && manifest->is_object() && manifest->contains("kernels")) {
        for (const auto& k : manifest->at("kernels")) {
            if (k.is_string()) {
                std::filesystem::path kp(k.get<std::string>());
                enabled_kernels_[kp.filename().string()] = true;
            }
        }
    }

    auto settings_path = paths::external_path("data", "ephemeris_settings.json");
    std::error_code ec;
    if (std::filesystem::is_regular_file(settings_path, ec)) {
        if (auto saved = paths::read_json(settings_path); saved && saved->is_object()) {
            for (auto it = saved->begin(); it != saved->end(); ++it) {
                if (it.value().is_boolean()) {
                    enabled_kernels_[it.key()] = it.value().get<bool>();
                }
            }
            settings_initialized_ = true;
        }
    } else {
        settings_initialized_ = check_missing_kernels().empty();
    }
}

bool SpiceKernelManager::save_settings() {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [k, v] : enabled_kernels_) {
        j[k] = v;
    }
    auto settings_path = paths::external_path("data", "ephemeris_settings.json");
    bool ok = paths::write_json(settings_path, j, 4);
    if (ok) {
        settings_initialized_ = true;
        sync_ephemeris_manifest();
    }
    return ok;
}

void SpiceKernelManager::sync_ephemeris_manifest() const {
    auto manifest_path = paths::external_path("data", "ephemeris.json");
    std::string runtime = "cspice.dll";
#ifndef _WIN32
    runtime = "libcspice.so";
#endif
    if (auto existing = paths::read_json(manifest_path); existing && existing->is_object() && existing->contains("runtime")) {
        runtime = existing->value("runtime", runtime);
    }

    auto kernels_root = paths::external_path("data", "kernels");
    nlohmann::json kernels_arr = nlohmann::json::array();
    std::unordered_set<std::string> added;

    auto add_if_present = [&](const std::string& fname) {
        if (added.contains(fname)) return;
        auto found = find_kernel_path(fname);
        if (found.empty()) return;
        std::error_code ec;
        auto rel = std::filesystem::relative(found, kernels_root, ec);
        std::string rel_str = ec ? fname : rel.generic_string();
        kernels_arr.push_back(rel_str);
        added.insert(fname);
    };

    for (const auto& dk : SpiceEphemeris::default_kernels()) {
        if (is_kernel_enabled(dk.filename)) {
            add_if_present(dk.filename);
        }
    }
    for (const auto& ak : list_additional_kernels()) {
        if (ak.enabled) {
            add_if_present(ak.name);
        }
    }

    nlohmann::json manifest = {
        {"runtime", runtime},
        {"kernels", kernels_arr}
    };
    paths::write_json(manifest_path, manifest, 2);
}

bool SpiceKernelManager::is_kernel_enabled(std::string_view filename) const {
    auto it = enabled_kernels_.find(std::string(filename));
    return it != enabled_kernels_.end() ? it->second : false;
}

void SpiceKernelManager::set_kernel_enabled(std::string_view filename, bool enabled) {
    enabled_kernels_[std::string(filename)] = enabled;
}

std::filesystem::path SpiceKernelManager::find_kernel_path(std::string_view filename) const {
    std::error_code ec;
    std::filesystem::path candidates[] = {
        paths::external_path("data", "kernels", "default", filename),
        paths::external_path("data", "kernels", "additional", filename),
        paths::external_path("data", "kernels", filename)
    };
    for (const auto& c : candidates) {
        if (std::filesystem::is_regular_file(c, ec)) {
            return std::filesystem::absolute(c, ec);
        }
    }
    return {};
}

std::vector<AdditionalKernelInfo> SpiceKernelManager::list_additional_kernels() const {
    std::vector<AdditionalKernelInfo> results;
    std::unordered_set<std::string> default_names;
    for (const auto& dk : SpiceEphemeris::default_kernels()) {
        default_names.insert(dk.filename);
    }

    std::unordered_set<std::string> seen;
    std::filesystem::path search_dirs[] = {
        paths::external_path("data", "kernels", "additional"),
        paths::external_path("data", "kernels")
    };
    std::error_code ec;
    for (const auto& d : search_dirs) {
        if (!std::filesystem::is_directory(d, ec)) continue;
        std::vector<std::filesystem::path> entries;
        for (const auto& entry : std::filesystem::directory_iterator(d, ec)) {
            if (entry.is_regular_file(ec)) entries.push_back(entry.path());
        }
        std::sort(entries.begin(), entries.end());
        for (const auto& full_p : entries) {
            std::string item = full_p.filename().string();
            if (to_lower(full_p.extension().string()) != ".bsp") continue;
            if (default_names.contains(item) || seen.contains(item)) continue;
            seen.insert(item);
            double size_kb = static_cast<double>(std::filesystem::file_size(full_p, ec)) / 1024.0;
            results.push_back({
                item,
                std::filesystem::absolute(full_p, ec),
                size_kb,
                is_kernel_enabled(item)
            });
        }
    }
    return results;
}

bool SpiceKernelManager::add_additional_kernel(const std::filesystem::path& source_path, std::string& error, bool copy_file) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(source_path, ec)) {
        error = "Kernel file not found: " + source_path.string();
        return false;
    }
    ensure_directories();
    std::string fname = source_path.filename().string();
    auto dest_path = paths::external_path("data", "kernels", "additional", fname);
    if (copy_file && !std::filesystem::equivalent(source_path, dest_path, ec)) {
        std::filesystem::copy_file(source_path, dest_path, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            error = "Failed to copy kernel file: " + ec.message();
            return false;
        }
    }
    enabled_kernels_[fname] = true;
    save_settings();
    return true;
}

std::vector<AvailableBspFile> SpiceKernelManager::list_available_bsp_files() const {
    std::vector<AvailableBspFile> results;
    std::set<std::filesystem::path> seen;
    std::error_code ec;

    std::unordered_map<std::string, std::string> descriptions;
    for (const auto& dk : SpiceEphemeris::default_kernels()) {
        descriptions[dk.filename] = dk.description;
    }

    auto exp_dir = paths::external_path("exports", "ephemeris");
    if (std::filesystem::is_directory(exp_dir, ec)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(exp_dir, ec)) {
            if (!entry.is_regular_file(ec)) continue;
            if (to_lower(entry.path().extension().string()) != ".bsp") continue;
            auto abs_p = std::filesystem::weakly_canonical(entry.path(), ec);
            if (!seen.insert(abs_p).second) continue;
            auto rel = std::filesystem::relative(abs_p, exp_dir, ec);
            double size_kb = static_cast<double>(std::filesystem::file_size(abs_p, ec)) / 1024.0;
            results.push_back({abs_p, ec ? abs_p.filename().string() : rel.generic_string(), "Exported Ephemeris", size_kb});
        }
    }

    struct DirCat { std::filesystem::path dir; const char* category; };
    DirCat search_dirs[] = {
        {paths::external_path("data", "kernels", "additional"), "Additional Kernels"},
        {paths::external_path("data", "kernels", "default"), "Default Kernels"},
        {paths::external_path("data", "kernels"), "Kernel Library"}
    };

    for (const auto& dc : search_dirs) {
        if (!std::filesystem::is_directory(dc.dir, ec)) continue;
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(dc.dir, ec)) {
            if (entry.is_regular_file(ec) && to_lower(entry.path().extension().string()) == ".bsp") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& f : files) {
            auto abs_p = std::filesystem::weakly_canonical(f, ec);
            if (!seen.insert(abs_p).second) continue;
            std::string fname = abs_p.filename().string();
            double size_kb = static_cast<double>(std::filesystem::file_size(abs_p, ec)) / 1024.0;
            auto it = descriptions.find(fname);
            std::string disp = (it != descriptions.end() && it->second != fname)
                ? std::format("{} ({})", fname, it->second)
                : fname;
            results.push_back({abs_p, std::move(disp), dc.category, size_kb});
        }
    }
    return results;
}

std::vector<std::string> SpiceKernelManager::check_missing_kernels() const {
    std::vector<std::string> missing;
    for (const auto& dk : SpiceEphemeris::default_kernels()) {
        if (!is_kernel_enabled(dk.filename)) continue;
        if (find_kernel_path(dk.filename).empty()) {
            missing.push_back(dk.filename);
        }
    }
    return missing;
}

DownloadProgressSnapshot SpiceKernelManager::download_snapshot() const {
    std::lock_guard lock(download_mutex_);
    return {
        is_downloading_.load(),
        cancel_requested_.load(),
        completed_pending_.load(),
        download_progress_.load(),
        download_bytes_current_.load(),
        download_bytes_total_.load(),
        current_download_file_,
        download_status_,
        download_speed_str_,
        download_error_
    };
}

bool SpiceKernelManager::consume_download_completed() {
    return completed_pending_.exchange(false);
}

void SpiceKernelManager::clear_download_error() {
    std::lock_guard lock(download_mutex_);
    download_error_.clear();
}

void SpiceKernelManager::cancel_download() {
    cancel_requested_.store(true);
}

void SpiceKernelManager::start_download_missing() {
    if (is_downloading_.load()) return;
    if (download_thread_.joinable()) {
        download_thread_.join();
    }

    std::vector<DefaultKernelInfo> to_download;
    for (const auto& dk : SpiceEphemeris::default_kernels()) {
        if (is_kernel_enabled(dk.filename) && find_kernel_path(dk.filename).empty()) {
            to_download.push_back(dk);
        }
    }

    if (to_download.empty()) {
        sync_ephemeris_manifest();
        completed_pending_.store(true);
        return;
    }

    is_downloading_.store(true);
    cancel_requested_.store(false);
    completed_pending_.store(false);
    download_progress_.store(0.0f);
    download_bytes_current_.store(0);
    download_bytes_total_.store(0);
    {
        std::lock_guard lock(download_mutex_);
        current_download_file_.clear();
        download_status_ = "Starting download...";
        download_speed_str_ = "Calculating speed...";
        download_error_.clear();
    }

    download_thread_ = std::thread(&SpiceKernelManager::download_worker, this, std::move(to_download));
}

namespace {

#ifdef _WIN32
bool download_url_winhttp(const std::string& url, const std::filesystem::path& part_path,
                          std::atomic<bool>& cancel_requested,
                          std::function<void(uint64_t, uint64_t)> on_progress,
                          std::string& error_out) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, nullptr, 0);
    if (wlen <= 0) {
        error_out = "Invalid URL encoding.";
        return false;
    }
    std::wstring wurl(wlen - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, wurl.data(), wlen);

    URL_COMPONENTS url_comp{};
    url_comp.dwStructSize = sizeof(url_comp);
    wchar_t host[256]{};
    wchar_t path[1024]{};
    url_comp.lpszHostName = host;
    url_comp.dwHostNameLength = static_cast<DWORD>(std::size(host));
    url_comp.lpszUrlPath = path;
    url_comp.dwUrlPathLength = static_cast<DWORD>(std::size(path));

    if (!WinHttpCrackUrl(wurl.c_str(), static_cast<DWORD>(wurl.size()), 0, &url_comp)) {
        error_out = "Failed to parse URL: " + url;
        return false;
    }

    HINTERNET session = WinHttpOpen(L"StellarForge/2.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) {
        error_out = "Failed to open WinHTTP session.";
        return false;
    }

    HINTERNET connect = WinHttpConnect(session, host, url_comp.nPort, 0);
    if (!connect) {
        WinHttpCloseHandle(session);
        error_out = "Failed to connect to host.";
        return false;
    }

    DWORD flags = (url_comp.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET request = WinHttpOpenRequest(connect, L"GET", path, nullptr, WINHTTP_NO_REFERER,
                                           WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!request) {
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        error_out = "Failed to open HTTP request.";
        return false;
    }

    bool ok = false;
    if (WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
        WinHttpReceiveResponse(request, nullptr)) {
        DWORD status_code = 0;
        DWORD status_size = sizeof(status_code);
        WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);
        if (status_code == 200) {
            wchar_t cl_buf[64]{};
            DWORD cl_size = sizeof(cl_buf);
            uint64_t total_bytes = 0;
            if (WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX,
                                    cl_buf, &cl_size, WINHTTP_NO_HEADER_INDEX)) {
                total_bytes = _wcstoui64(cl_buf, nullptr, 10);
            }

            std::ofstream out(part_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                error_out = "Cannot open file for writing: " + part_path.string();
            } else {
                std::vector<char> buffer(64 * 1024);
                uint64_t downloaded = 0;
                ok = true;
                while (!cancel_requested.load()) {
                    DWORD avail = 0;
                    if (!WinHttpQueryDataAvailable(request, &avail)) {
                        error_out = "Network read error while downloading.";
                        ok = false;
                        break;
                    }
                    if (avail == 0) break;
                    DWORD to_read = std::min<DWORD>(avail, static_cast<DWORD>(buffer.size()));
                    DWORD read_bytes = 0;
                    if (!WinHttpReadData(request, buffer.data(), to_read, &read_bytes)) {
                        error_out = "Failed to read HTTP response stream.";
                        ok = false;
                        break;
                    }
                    if (read_bytes == 0) break;
                    out.write(buffer.data(), read_bytes);
                    if (!out) {
                        error_out = "Disk write error while saving kernel.";
                        ok = false;
                        break;
                    }
                    downloaded += read_bytes;
                    on_progress(downloaded, total_bytes);
                }
                out.close();
                if (cancel_requested.load()) ok = false;
            }
        } else {
            error_out = std::format("HTTP status {} from {}", status_code, url);
        }
    } else {
        error_out = "Failed to connect to " + url;
    }

    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    return ok;
}
#endif

} // namespace

void SpiceKernelManager::download_worker(std::vector<DefaultKernelInfo> to_download) {
    ensure_directories();
    cleanup_partial_downloads();
    size_t total_files = to_download.size();

    for (size_t i = 0; i < total_files; ++i) {
        if (cancel_requested_.load()) break;
        const auto& dk = to_download[i];
        auto dest_path = paths::external_path("data", "kernels", "default", dk.filename);
        auto part_path = dest_path;
        part_path += ".part";

        {
            std::lock_guard lock(download_mutex_);
            current_download_file_ = dk.filename;
            download_status_ = std::format("Downloading {} ({}/{})...", dk.filename, i + 1, total_files);
            download_speed_str_ = "Calculating speed...";
        }
        download_bytes_current_.store(0);
        download_bytes_total_.store(0);

        auto last_sample_time = std::chrono::steady_clock::now();
        uint64_t last_sample_bytes = 0;

        auto progress_cb = [&](uint64_t cur_bytes, uint64_t total_bytes) {
            download_bytes_current_.store(cur_bytes);
            download_bytes_total_.store(total_bytes);

            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_sample_time).count();
            if (dt >= 0.4) {
                double speed_bps = static_cast<double>(cur_bytes - last_sample_bytes) / dt;
                std::string spd;
                if (speed_bps >= 1024.0 * 1024.0) {
                    spd = std::format("{:.2f} MB/s", speed_bps / (1024.0 * 1024.0));
                } else {
                    spd = std::format("{:.1f} KB/s", std::max(0.0, speed_bps / 1024.0));
                }
                {
                    std::lock_guard lock(download_mutex_);
                    download_speed_str_ = std::move(spd);
                }
                last_sample_time = now;
                last_sample_bytes = cur_bytes;
            }

            if (total_bytes > 0) {
                float file_prog = std::min(1.0f, static_cast<float>(cur_bytes) / static_cast<float>(total_bytes));
                download_progress_.store((static_cast<float>(i) + file_prog) / static_cast<float>(total_files));
            }
        };

        std::string err;
        bool ok = false;
#ifdef _WIN32
        ok = download_url_winhttp(dk.url, part_path, cancel_requested_, progress_cb, err);
#else
        err = "Automatic kernel downloading requires Windows WinHTTP.";
#endif
        std::error_code ec;
        if (cancel_requested_.load()) {
            std::filesystem::remove(part_path, ec);
            break;
        }
        if (!ok) {
            std::filesystem::remove(part_path, ec);
            cleanup_partial_downloads();
            {
                std::lock_guard lock(download_mutex_);
                download_status_ = "Error downloading " + dk.filename;
                download_error_ = err;
                download_speed_str_.clear();
            }
            is_downloading_.store(false);
            return;
        }

        std::filesystem::remove(dest_path, ec);
        std::filesystem::rename(part_path, dest_path, ec);
        if (ec) {
            std::lock_guard lock(download_mutex_);
            download_status_ = "Error saving " + dk.filename;
            download_error_ = ec.message();
            is_downloading_.store(false);
            return;
        }
    }

    cleanup_partial_downloads();
    if (cancel_requested_.load()) {
        std::lock_guard lock(download_mutex_);
        download_status_ = "Download cancelled.";
        download_speed_str_.clear();
        is_downloading_.store(false);
        return;
    }

    sync_ephemeris_manifest();
    download_progress_.store(1.0f);
    {
        std::lock_guard lock(download_mutex_);
        current_download_file_.clear();
        download_status_ = "Ready";
        download_speed_str_.clear();
    }
    is_downloading_.store(false);
    completed_pending_.store(true);
}

LoadedSystem SpiceKernelManager::build_ephemeris_system(const SystemDefinition* template_def,
                                                        std::string_view epoch_utc,
                                                        std::string& error) {
    sync_ephemeris_manifest();
    auto spice = SpiceEphemeris::reload(error);
    if (!spice) return {};

    std::string target_epoch = epoch_utc.empty() ? "2026-06-03" : std::string(epoch_utc);
    double et = 0.0;
    auto ep_status = spice->epoch(target_epoch, et);
    if (!ep_status) {
        error = ep_status.error;
        return {};
    }

    LoadedSystem result{};
    result.definition.name = "Ephemeris Mode";
    result.definition.motion_model = MotionModel::Spice;
    result.definition.epoch_utc = target_epoch;

    std::unordered_set<int> included_ids;
    std::unordered_set<std::string> included_names;

    auto mark_included = [&](int spid, const std::string& name) {
        included_ids.insert(spid);
        if (spid > 100 && spid < 1000 && spid % 100 == 99) included_ids.insert(spid / 100);
        if (spid > 0 && spid < 10) included_ids.insert(spid * 100 + 99);
        included_names.insert(to_upper(name));
    };

    if (template_def) {
        for (const auto& b : template_def->bodies) {
            BodyDefinition copy = b;
            if (copy.spice_id == 0) {
                if (const auto* known = SpiceEphemeris::find_known_body_by_name(copy.name)) {
                    copy.spice_id = known->id;
                }
            }
            if (copy.spice_id == 0) continue;

            BodyMotion st{};
            if (!spice->state(copy.spice_id, et, st)) {
                continue;
            }
            mark_included(copy.spice_id, copy.name);
            result.definition.bodies.push_back(std::move(copy));
            result.motions.push_back(st);
        }
    }

    for (const auto& kpath : spice->loaded_kernels()) {
        if (to_lower(kpath.extension().string()) != ".bsp") continue;
        auto info = spice->inspect_bsp(kpath);
        if (!info.valid) continue;

        for (const auto& binfo : info.bodies) {
            int spid = binfo.id;
            if (spid == 1 || spid == 2 || spid == 3 || spid == 4) continue;
            if (included_ids.contains(spid) || included_names.contains(to_upper(binfo.name))) continue;

            BodyMotion st{};
            if (!spice->state(spid, et, st)) continue;

            const auto* known = SpiceEphemeris::find_known_body(spid);
            BodyDefinition bdef{};
            bdef.name = binfo.name;
            bdef.spice_id = spid;
            bdef.is_spacecraft = (spid < 0) || (known && known->is_spacecraft);
            bdef.type = known ? known->type : (bdef.is_spacecraft ? "Spacecraft" : "Moon");
            bdef.is_star = (bdef.type == "Star");
            std::string col_hex = known ? known->color_hex : (bdef.is_spacecraft ? "#79c0ff" : "#a0a0a0");
            bdef.color_rgb = hex_to_rgb(col_hex);
            bdef.color_pole_rgb = bdef.color_rgb;

            double fallback_m_kg = known ? known->mass_kg : (bdef.is_spacecraft ? 26000.0 : 1e15);
            double fallback_r_km = known ? known->radius_km : (bdef.is_spacecraft ? 0.010 : 10.0);
            spice->body_gm_and_radius(spid, fallback_m_kg, fallback_r_km, bdef.mass_msun, bdef.radius_rsun);

            std::string parent_name = "Sun";
            if (known && !known->parent_name.empty()) {
                parent_name = known->parent_name;
            } else if (spid < 0) {
                BodyMotion st_start = st;
                spice->state(spid, binfo.start_et, st_start);
                struct CandidateParent { int id; const char* name; };
                static constexpr CandidateParent kCandidates[] = {
                    {399, "Earth"}, {699, "Saturn"}, {599, "Jupiter"},
                    {499, "Mars"}, {799, "Uranus"}, {899, "Neptune"}
                };
                double min_d = std::numeric_limits<double>::infinity();
                for (const auto& cp : kCandidates) {
                    BodyMotion pst{};
                    if (spice->state(cp.id, binfo.start_et, pst)) {
                        double d_km = glm::length(st_start.position.value - pst.position.value) * AU_TO_KM;
                        if (d_km < 5e7 && d_km < min_d) {
                            min_d = d_km;
                            parent_name = cp.name;
                        }
                    }
                }
            } else if (spid > 100 && spid < 1000 && spid % 100 != 99) {
                int p_planet = (spid / 100) * 100 + 99;
                if (const auto* kp = SpiceEphemeris::find_known_body(p_planet)) {
                    parent_name = kp->name;
                }
            }

            bdef.parent_id = parent_name;
            bdef.pole_ecl = {0.0, 0.0, 1.0};
            bdef.source_json = {
                {"name", bdef.name},
                {"type", bdef.type},
                {"spice_id", bdef.spice_id},
                {"parentId", bdef.parent_id},
                {"m", bdef.mass_msun},
                {"r", bdef.radius_rsun},
                {"color", col_hex},
                {"is_spacecraft", bdef.is_spacecraft}
            };

            mark_included(spid, bdef.name);
            result.definition.bodies.push_back(std::move(bdef));
            result.motions.push_back(st);
        }
    }

    std::unordered_map<std::string, size_t> raw_idx;
    for (size_t i = 0; i < result.definition.bodies.size(); ++i) {
        raw_idx[result.definition.bodies[i].name] = i;
    }
    std::string star_name = "Sun";
    for (const auto& b : result.definition.bodies) {
        if (b.is_star) {
            star_name = b.name;
            break;
        }
    }
    for (auto& b : result.definition.bodies) {
        if (b.is_star) {
            b.parent_id.clear();
        } else if (b.parent_id.empty() || !raw_idx.contains(b.parent_id) || b.parent_id == b.name) {
            b.parent_id = star_name;
        }
    }

    std::vector<size_t> order;
    std::unordered_set<size_t> visited;
    std::unordered_set<size_t> visiting;
    auto visit = [&](auto& self, size_t idx) -> void {
        if (visited.contains(idx)) return;
        if (visiting.contains(idx)) {
            result.definition.bodies[idx].parent_id.clear();
            return;
        }
        visiting.insert(idx);
        const auto& pid = result.definition.bodies[idx].parent_id;
        if (!pid.empty() && raw_idx.contains(pid)) {
            self(self, raw_idx[pid]);
        }
        visiting.erase(idx);
        visited.insert(idx);
        order.push_back(idx);
    };
    for (size_t i = 0; i < result.definition.bodies.size(); ++i) {
        visit(visit, i);
    }

    std::vector<BodyDefinition> sorted_bodies;
    std::vector<BodyMotion> sorted_motions;
    sorted_bodies.reserve(order.size());
    sorted_motions.reserve(order.size());
    result.definition.name_to_idx.clear();
    result.definition.star_idx = 0;

    for (size_t idx : order) {
        auto b = std::move(result.definition.bodies[idx]);
        auto m = result.motions[idx];
        int new_i = static_cast<int>(sorted_bodies.size());
        result.definition.name_to_idx[b.name] = new_i;
        if (b.is_star || b.parent_id.empty()) {
            b.parent_id.clear();
            b.parent_idx = -1;
            if (b.is_star) result.definition.star_idx = new_i;
        } else {
            auto it = result.definition.name_to_idx.find(b.parent_id);
            b.parent_idx = (it != result.definition.name_to_idx.end()) ? it->second : result.definition.star_idx;
        }
        sorted_bodies.push_back(std::move(b));
        sorted_motions.push_back(m);
    }
    result.definition.bodies = std::move(sorted_bodies);
    result.motions = std::move(sorted_motions);
    return result;
}

LoadedSystem SpiceKernelManager::build_generic_bsp_system(const std::filesystem::path& bsp_path, std::string& error) {
    auto spice = SpiceEphemeris::open(error);
    if (!spice) return {};

    auto f_status = spice->furnish_file(bsp_path);
    if (!f_status) {
        error = f_status.error;
        return {};
    }

    auto info = spice->inspect_bsp(bsp_path);
    if (!info.valid || info.bodies.empty()) {
        error = info.error.empty() ? "No bodies found in BSP file." : info.error;
        return {};
    }

    double epoch_et = info.min_et;
    if (info.sidecar.is_object() && info.sidecar.contains("epoch_et_seconds_past_j2000")) {
        epoch_et = info.sidecar.value("epoch_et_seconds_past_j2000", info.min_et);
    }
    std::string epoch_utc = spice->et_to_utc(epoch_et, "ISOC");

    std::vector<BspBodyInfo> all_bodies = info.bodies;
    std::unordered_set<int> present_ids;
    for (const auto& b : all_bodies) {
        present_ids.insert(b.id);
        if (b.id > 100 && b.id < 1000 && b.id % 100 == 99) present_ids.insert(b.id / 100);
        if (b.id > 0 && b.id < 10) present_ids.insert(b.id * 100 + 99);
    }

    auto ensure_context_body = [&](int cid) {
        if (cid == 0 || present_ids.contains(cid)) return;
        int actual_id = (cid > 0 && cid < 10) ? (cid * 100 + 99) : cid;
        if (present_ids.contains(actual_id)) return;
        BodyMotion test_st{};
        if (!spice->state(actual_id, epoch_et, test_st)) return;
        const auto* known = SpiceEphemeris::find_known_body(actual_id);
        BspBodyInfo cbody{};
        cbody.id = actual_id;
        cbody.name = known ? known->name : spice->body_name(actual_id);
        if (cbody.name.empty()) cbody.name = std::format("Object {}", actual_id);
        cbody.center_id = (actual_id == 10) ? 0 : 10;
        cbody.color_hex = known ? known->color_hex : "#a0a0a0";
        cbody.start_et = epoch_et;
        cbody.end_et = info.max_et;
        present_ids.insert(actual_id);
        present_ids.insert(cid);
        all_bodies.push_back(std::move(cbody));
    };

    ensure_context_body(10);
    for (size_t i = 0; i < info.bodies.size(); ++i) {
        ensure_context_body(info.bodies[i].center_id);
    }

    std::unordered_map<int, std::string> id_to_name;
    for (const auto& b : all_bodies) {
        id_to_name[b.id] = b.name;
        if (b.id > 100 && b.id < 1000 && b.id % 100 == 99) id_to_name[b.id / 100] = b.name;
        if (b.id > 0 && b.id < 10) id_to_name[b.id * 100 + 99] = b.name;
    }

    struct RawEntry {
        BodyDefinition def;
        BodyMotion motion;
    };
    std::vector<RawEntry> entries;
    std::unordered_set<std::string> used_names;

    for (const auto& b : all_bodies) {
        BodyMotion st{};
        if (!spice->state(b.id, epoch_et, st)) {
            if (!spice->state(b.id, b.start_et, st)) continue;
        }

        const auto* known = SpiceEphemeris::find_known_body(b.id);
        BodyDefinition bdef{};
        bdef.name = b.name;
        if (used_names.contains(bdef.name)) {
            bdef.name = std::format("{} ({})", b.name, b.id);
        }
        used_names.insert(bdef.name);
        bdef.spice_id = b.id;
        bdef.is_spacecraft = (b.id < 0) || (known && known->is_spacecraft);
        bdef.is_star = (b.id == 10) || (known && known->type == "Star");
        bdef.type = bdef.is_star ? "Star" : (known ? known->type : (bdef.is_spacecraft ? "Spacecraft" : "Terrestrial"));
        bdef.color_rgb = hex_to_rgb(b.color_hex);
        bdef.color_pole_rgb = bdef.color_rgb;
        bdef.pole_ecl = {0.0, 0.0, 1.0};

        double fallback_m_kg = known ? known->mass_kg : (bdef.is_star ? 1.98847e30 : (bdef.is_spacecraft ? 26000.0 : 1e20));
        double fallback_r_km = known ? known->radius_km : (bdef.is_star ? 696340.0 : (bdef.is_spacecraft ? 0.010 : 1000.0));
        spice->body_gm_and_radius(b.id, fallback_m_kg, fallback_r_km, bdef.mass_msun, bdef.radius_rsun);
        if (bdef.radius_rsun <= 0.0) bdef.radius_rsun = 0.001;

        if (!bdef.is_star) {
            if (b.center_id != 0 && id_to_name.contains(b.center_id) && id_to_name[b.center_id] != bdef.name) {
                bdef.parent_id = id_to_name[b.center_id];
            } else if (known && !known->parent_name.empty()) {
                for (const auto& [k_id, k_name] : id_to_name) {
                    if (to_upper(k_name) == to_upper(known->parent_name) && k_name != bdef.name) {
                        bdef.parent_id = k_name;
                        break;
                    }
                }
            }
            if (bdef.parent_id.empty() && id_to_name.contains(10) && id_to_name[10] != bdef.name) {
                bdef.parent_id = id_to_name[10];
            }
        }

        bdef.source_json = {
            {"name", bdef.name},
            {"type", bdef.type},
            {"spice_id", bdef.spice_id},
            {"parentId", bdef.parent_id},
            {"m", bdef.mass_msun},
            {"r", bdef.radius_rsun},
            {"color", b.color_hex},
            {"is_spacecraft", bdef.is_spacecraft}
        };
        entries.push_back({std::move(bdef), st});
    }

    if (entries.empty()) {
        error = "No bodies in the BSP kernel have valid states at the start epoch.";
        return {};
    }

    bool has_root = false;
    for (const auto& e : entries) {
        if (e.def.parent_id.empty()) {
            has_root = true;
            break;
        }
    }
    if (!has_root) {
        entries.front().def.parent_id.clear();
    }

    std::unordered_map<std::string, size_t> raw_idx;
    for (size_t i = 0; i < entries.size(); ++i) {
        raw_idx[entries[i].def.name] = i;
    }
    for (auto& e : entries) {
        if (!e.def.parent_id.empty() && !raw_idx.contains(e.def.parent_id)) {
            e.def.parent_id.clear();
        }
    }

    std::vector<size_t> order;
    std::unordered_set<size_t> visited;
    std::unordered_set<size_t> visiting;
    auto visit = [&](auto& self, size_t idx) -> void {
        if (visited.contains(idx)) return;
        if (visiting.contains(idx)) {
            entries[idx].def.parent_id.clear();
            return;
        }
        visiting.insert(idx);
        const auto& pid = entries[idx].def.parent_id;
        if (!pid.empty() && raw_idx.contains(pid)) {
            self(self, raw_idx[pid]);
        }
        visiting.erase(idx);
        visited.insert(idx);
        order.push_back(idx);
    };
    for (size_t i = 0; i < entries.size(); ++i) {
        visit(visit, i);
    }

    LoadedSystem loaded{};
    loaded.definition.name = "Ephemeris: " + bsp_path.stem().string();
    loaded.definition.motion_model = MotionModel::Spice;
    loaded.definition.epoch_utc = epoch_utc;
    loaded.definition.custom_bsp_path = std::filesystem::absolute(bsp_path).string();

    for (size_t idx : order) {
        auto& e = entries[idx];
        int new_i = static_cast<int>(loaded.definition.bodies.size());
        loaded.definition.name_to_idx[e.def.name] = new_i;
        if (!e.def.parent_id.empty()) {
            auto it = loaded.definition.name_to_idx.find(e.def.parent_id);
            e.def.parent_idx = (it != loaded.definition.name_to_idx.end()) ? it->second : -1;
        } else {
            e.def.parent_idx = -1;
        }
        if (e.def.is_star && loaded.definition.star_idx == 0) {
            loaded.definition.star_idx = new_i;
        }
        loaded.definition.bodies.push_back(std::move(e.def));
        loaded.motions.push_back(e.motion);
    }
    return loaded;
}

} // namespace sf::ephemeris
