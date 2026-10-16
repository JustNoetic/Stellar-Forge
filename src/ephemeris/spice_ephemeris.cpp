#include "ephemeris/spice_ephemeris.hpp"
#include "core/path_utils.hpp"
#include <cmath>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
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

}

std::shared_ptr<SpiceEphemeris> SpiceEphemeris::open(std::string& error) {
    static std::weak_ptr<SpiceEphemeris> active;
    if (auto source = active.lock()) return source;
    auto source = std::shared_ptr<SpiceEphemeris>(new SpiceEphemeris);
    auto status = source->initialize();
    if (!status) { error = status.error; return {}; }
    active = source;
    return source;
}

CommandResult SpiceEphemeris::initialize() {
    auto config = paths::read_json(paths::external_path("data", "ephemeris.json"));
    if (!config || !config->is_object()) return {false, false, "Ephemeris requires data/ephemeris.json and a local CSPICE runtime/kernel set."};
    try {
        auto runtime = paths::external_path("data", "kernels", config->at("runtime").get<std::string>());
#ifdef _WIN32
        library_ = LoadLibraryExW(runtime.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        library_ = dlopen(runtime.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library_) return {false, false, "Cannot load CSPICE runtime: " + runtime.string()};
        furnish_ = symbol<decltype(furnish_)>(library_, "furnsh_c");
        error_action_ = symbol<decltype(error_action_)>(library_, "erract_c");
        error_print_ = symbol<decltype(error_print_)>(library_, "errprt_c");
        failed_ = symbol<decltype(failed_)>(library_, "failed_c");
        message_ = symbol<decltype(message_)>(library_, "getmsg_c");
        reset_ = symbol<decltype(reset_)>(library_, "reset_c");
        to_et_ = symbol<decltype(to_et_)>(library_, "str2et_c");
        state_ = symbol<decltype(state_)>(library_, "spkgeo_c");
        clear_ = symbol<decltype(clear_)>(library_, "kclear_c");
        char action[] = "RETURN", printing[] = "NONE";
        error_action_("SET", 0, action);
        error_print_("SET", 0, printing);
        auto status = result();
        if (!status) return status;
        for (const auto& entry : config->at("kernels")) {
            auto path = paths::external_path("data", "kernels", entry.get<std::string>());
            if (!std::filesystem::is_regular_file(path)) return {false, false, "Missing ephemeris kernel: " + path.string()};
            furnish_(path.string().c_str());
            status = result();
            if (!status) return status;
        }
    } catch (const std::exception& error) {
        return {false, false, error.what()};
    }
    return {};
}

SpiceEphemeris::~SpiceEphemeris() {
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

CommandResult SpiceEphemeris::epoch(std::string_view utc, double& et) const {
    auto text = std::string(utc);
    to_et_(text.c_str(), &et);
    return result();
}

CommandResult SpiceEphemeris::state(int target, double et, BodyMotion& motion) const {
    if (!std::isfinite(et)) return {false, false, "Ephemeris time must be finite."};
    double state[6]{}, light_time = 0.0;
    state_(target, et, "ECLIPJ2000", 0, state, &light_time);
    auto status = result();
    if (!status) return status;
    for (double value : state) if (!std::isfinite(value)) return {false, false, "SPICE returned a nonfinite state."};
    motion.position = EclipticPosition{glm::dvec3(state[0], state[1], state[2]) * KM_TO_AU};
    motion.velocity = EclipticVelocity{glm::dvec3(state[3], state[4], state[5]) * (KM_TO_AU * SECONDS_PER_YEAR)};
    return {};
}

}
