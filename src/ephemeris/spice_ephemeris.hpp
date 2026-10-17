#pragma once

#include "core/command_result.hpp"
#include "ephemeris/system_manager.hpp"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sf::ephemeris {

struct DefaultKernelInfo {
    std::string filename;
    std::string group;
    std::string description;
    std::string url;
    bool essential = false;
};

struct AdditionalKernelInfo {
    std::string name;
    std::filesystem::path path;
    double size_kb = 0.0;
    bool enabled = false;
};

struct AvailableBspFile {
    std::filesystem::path path;
    std::string name;
    std::string category;
    double size_kb = 0.0;
};

struct BspBodyInfo {
    int id = 0;
    std::string name;
    int center_id = 0;
    std::string center_name;
    std::string color_hex = "#58a6ff";
    double start_et = 0.0;
    double end_et = 0.0;
    std::string utc_start;
    std::string utc_end;
};

struct BspInspectResult {
    bool valid = false;
    std::string error;
    std::string filepath;
    std::string filename;
    std::string format_desc = "Standard NASA SPICE SPK";
    std::vector<BspBodyInfo> bodies;
    double min_et = 0.0;
    double max_et = 0.0;
    std::string utc_min;
    std::string utc_max;
    double duration_days = 0.0;
    nlohmann::json sidecar;
};

struct DownloadProgressSnapshot {
    bool is_downloading = false;
    bool cancel_requested = false;
    bool completed_pending = false;
    float progress = 0.0f;
    uint64_t bytes_current = 0;
    uint64_t bytes_total = 0;
    std::string current_file;
    std::string status_text;
    std::string speed_str;
    std::string error;
};

struct KnownSpiceBody {
    int id = 0;
    std::string name;
    std::string type = "Moon";
    std::string color_hex = "#a0a0a0";
    double radius_km = 10.0;
    double mass_kg = 1e15;
    std::string parent_name;
    bool is_spacecraft = false;
    std::string mission_start_utc;
    std::string mission_end_utc;
};

class SpiceEphemeris {
public:
    static std::shared_ptr<SpiceEphemeris> open(std::string& error);
    static std::shared_ptr<SpiceEphemeris> reload(std::string& error);
    ~SpiceEphemeris();
    SpiceEphemeris(const SpiceEphemeris&) = delete;
    SpiceEphemeris& operator=(const SpiceEphemeris&) = delete;

    CommandResult epoch(std::string_view utc, double& et) const;
    CommandResult state(int target, double et, BodyMotion& motion) const;
    CommandResult state_relative(int target, int observer, double et, BodyMotion& motion) const;
    CommandResult furnish_file(const std::filesystem::path& path);
    BspInspectResult inspect_bsp(const std::filesystem::path& bsp_path);

    bool can_write_spk() const noexcept { return spkopn_ && (spkw02_ || spkw13_) && spkcls_; }
    CommandResult spk_open(const std::filesystem::path& path, std::string_view internal_name, int& handle);
    CommandResult spk_write_type2(int handle, int body, int center, std::string_view frame,
                                  double first_et, double last_et, std::string_view segid,
                                  double intlen_sec, int n_intervals, int degree,
                                  const double* cdata, double btime_et);
    CommandResult spk_write_type13(int handle, int body, int center, std::string_view frame,
                                   double first_et, double last_et, std::string_view segid,
                                   int degree, int num_states, const double* states_6d,
                                   const double* epochs_et);
    CommandResult spk_close(int handle);

    std::string body_name(int spice_id) const;
    std::string et_to_utc(double et, const char* format = "C") const;
    bool body_gm_and_radius(int spice_id, double fallback_mass_kg, double fallback_radius_km,
                            double& mass_msun, double& radius_rsun) const;
    const std::vector<std::filesystem::path>& loaded_kernels() const noexcept { return loaded_kernels_; }

    static const std::vector<DefaultKernelInfo>& default_kernels();
    static const std::vector<std::string>& kernel_groups();
    static const KnownSpiceBody* find_known_body(int spice_id);
    static const KnownSpiceBody* find_known_body_by_name(std::string_view name);

private:
    SpiceEphemeris() = default;
    CommandResult initialize();
    CommandResult result() const;
    bool query_raw_state(int target, int observer, double et, double state_out[6]) const;

    mutable std::recursive_mutex mutex_;
    void* library_ = nullptr;
    std::vector<std::filesystem::path> loaded_kernels_;

    void (*furnish_)(const char*) = nullptr;
    void (*unload_)(const char*) = nullptr;
    void (*clear_)() = nullptr;
    void (*ktotal_)(const char*, int*) = nullptr;
    void (*kdata_)(int, const char*, int, int, int, char*, char*, char*, int*, int*) = nullptr;
    void (*dafopr_)(const char*, int*) = nullptr;
    void (*dafbfs_)(int) = nullptr;
    void (*daffna_)(int*) = nullptr;
    void (*dafgs_)(double*) = nullptr;
    void (*spkuds_)(const double*, int*, int*, int*, int*, double*, double*, int*, int*) = nullptr;
    void (*dafcls_)(int) = nullptr;
    void (*bodc2n_)(int, int, char*, int*) = nullptr;
    void (*bodn2c_)(const char*, int*, int*) = nullptr;
    void (*bodvrd_)(const char*, const char*, int, int*, double*) = nullptr;
    void (*et2utc_)(double, const char*, int, int, char*) = nullptr;
    void (*error_action_)(const char*, int, char*) = nullptr;
    void (*error_print_)(const char*, int, char*) = nullptr;
    int (*failed_)() = nullptr;
    void (*message_)(const char*, int, char*) = nullptr;
    void (*reset_)() = nullptr;
    void (*to_et_)(const char*, double*) = nullptr;
    void (*state_)(int, double, const char*, int, double*, double*) = nullptr;
    void (*spkopn_)(const char*, const char*, int, int*) = nullptr;
    void (*spkw02_)(int, int, int, const char*, double, double, const char*, double, int, int, const double*, double) = nullptr;
    void (*spkw03_)(int, int, int, const char*, double, double, const char*, double, int, int, const double*, double) = nullptr;
    void (*spkw13_)(int, int, int, const char*, double, double, const char*, int, int, const void*, const double*) = nullptr;
    void (*spkcls_)(int) = nullptr;
};

class SpiceKernelManager {
public:
    SpiceKernelManager();
    ~SpiceKernelManager();
    SpiceKernelManager(const SpiceKernelManager&) = delete;
    SpiceKernelManager& operator=(const SpiceKernelManager&) = delete;

    void load_settings();
    bool save_settings();
    bool settings_initialized() const noexcept { return settings_initialized_; }

    bool is_kernel_enabled(std::string_view filename) const;
    void set_kernel_enabled(std::string_view filename, bool enabled);
    const std::unordered_map<std::string, bool>& enabled_kernels() const noexcept { return enabled_kernels_; }

    std::filesystem::path find_kernel_path(std::string_view filename) const;
    std::vector<AdditionalKernelInfo> list_additional_kernels() const;
    bool add_additional_kernel(const std::filesystem::path& source_path, std::string& error, bool copy_file = true);
    std::vector<AvailableBspFile> list_available_bsp_files() const;
    std::vector<std::string> check_missing_kernels() const;

    void start_download_missing();
    void cancel_download();
    DownloadProgressSnapshot download_snapshot() const;
    bool consume_download_completed();
    void clear_download_error();

    LoadedSystem build_ephemeris_system(const SystemDefinition* template_def, std::string_view epoch_utc, std::string& error);
    LoadedSystem build_generic_bsp_system(const std::filesystem::path& bsp_path, std::string& error);

private:
    void ensure_directories() const;
    void cleanup_partial_downloads() const;
    void sync_ephemeris_manifest() const;
    void download_worker(std::vector<DefaultKernelInfo> to_download);

    std::unordered_map<std::string, bool> enabled_kernels_;
    bool settings_initialized_ = false;

    mutable std::mutex download_mutex_;
    std::thread download_thread_;
    std::atomic<bool> is_downloading_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> completed_pending_{false};
    std::atomic<float> download_progress_{0.0f};
    std::atomic<uint64_t> download_bytes_current_{0};
    std::atomic<uint64_t> download_bytes_total_{0};
    std::string current_download_file_;
    std::string download_status_;
    std::string download_speed_str_;
    std::string download_error_;
};

} // namespace sf::ephemeris
