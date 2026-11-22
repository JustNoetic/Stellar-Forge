#include "rendering/hdri_exporter.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace sf::rendering {

namespace {

constexpr uint32_t FACE_GUARD_PX = 2;

constexpr HdriPresetInfo PRESETS[] = {
    {"720p (1440 x 720)",   1440,  720,   512},
    {"1080p (2160 x 1080)", 2160,  1080,  768},
    {"2K (2048 x 1024)",    2048,  1024,  1024},
    {"4K (4096 x 2048)",    4096,  2048,  1024},
    {"8K (8192 x 4096)",    8192,  4096,  2048},
    {"16K (16384 x 8192)",  16384, 8192,  4096},
    {"32K (32768 x 16384)", 32768, 16384, 8192},
};

const char* face_label(uint32_t face) noexcept {
    static constexpr const char* LABELS[6] = {"+X (Front)", "-X (Back)", "+Y (Left)", "-Y (Right)", "+Z (Zenith)", "-Z (Nadir)"};
    return face < 6 ? LABELS[face] : "";
}

float sanitize_radiance(float v) noexcept {
    return (std::isfinite(v) && v > 0.0f) ? v : 0.0f;
}

template <typename T>
void write_le(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_attr_header(std::ostream& out, const char* name, const char* type, int32_t size) {
    out.write(name, static_cast<std::streamsize>(std::strlen(name) + 1));
    out.write(type, static_cast<std::streamsize>(std::strlen(type) + 1));
    write_le(out, size);
}

void write_exr_header(std::ostream& out, uint32_t width, uint32_t height) {
    const uint32_t magic = 0x01312f76u;
    const uint32_t version = 2u;
    write_le(out, magic);
    write_le(out, version);

    // Channels must be sorted alphabetically for OpenEXR readers.
    const char* channels[] = {"B", "G", "R"};
    int32_t chlist_size = 1;
    for (const char* ch : channels) {
        chlist_size += static_cast<int32_t>(std::strlen(ch) + 1) + 16;
    }
    write_attr_header(out, "channels", "chlist", chlist_size);
    for (const char* ch : channels) {
        out.write(ch, static_cast<std::streamsize>(std::strlen(ch) + 1));
        write_le<int32_t>(out, 2); // FLOAT
        const uint8_t linear_pad[4] = {0, 0, 0, 0};
        out.write(reinterpret_cast<const char*>(linear_pad), 4);
        write_le<int32_t>(out, 1);
        write_le<int32_t>(out, 1);
    }
    write_le<uint8_t>(out, 0);

    write_attr_header(out, "compression", "compression", 1);
    write_le<uint8_t>(out, 0); // NO_COMPRESSION

    const int32_t win[4] = {0, 0, static_cast<int32_t>(width) - 1, static_cast<int32_t>(height) - 1};
    write_attr_header(out, "dataWindow", "box2i", 16);
    out.write(reinterpret_cast<const char*>(win), 16);
    write_attr_header(out, "displayWindow", "box2i", 16);
    out.write(reinterpret_cast<const char*>(win), 16);

    write_attr_header(out, "lineOrder", "lineOrder", 1);
    write_le<uint8_t>(out, 0); // INCREASING_Y

    write_attr_header(out, "pixelAspectRatio", "float", 4);
    write_le<float>(out, 1.0f);

    const float center[2] = {0.0f, 0.0f};
    write_attr_header(out, "screenWindowCenter", "v2f", 8);
    out.write(reinterpret_cast<const char*>(center), 8);

    write_attr_header(out, "screenWindowWidth", "float", 4);
    write_le<float>(out, 1.0f);

    write_le<uint8_t>(out, 0);
}

void face_basis(uint32_t face, const glm::dvec3& ex, const glm::dvec3& ey, const glm::dvec3& ez,
                glm::dvec3& out_fwd, glm::dvec3& out_up) noexcept {
    switch (face) {
        case 0: out_fwd =  ex; out_up =  ez; break;
        case 1: out_fwd = -ex; out_up =  ez; break;
        case 2: out_fwd =  ey; out_up =  ez; break;
        case 3: out_fwd = -ey; out_up =  ez; break;
        case 4: out_fwd =  ez; out_up = -ex; break;
        default: out_fwd = -ez; out_up = ex; break;
    }
}

void map_direction_to_face(double bx, double by, double bz, uint32_t& out_face, double& out_s, double& out_t) noexcept {
    const double ax = std::abs(bx);
    const double ay = std::abs(by);
    const double az = std::abs(bz);
    if (ax >= ay && ax >= az) {
        if (bx >= 0.0) { out_face = 0; out_s = -by / bx; out_t = bz / bx; }
        else           { out_face = 1; out_s =  by / -bx; out_t = bz / -bx; }
    } else if (ay >= ax && ay >= az) {
        if (by >= 0.0) { out_face = 2; out_s =  bx / by; out_t = bz / by; }
        else           { out_face = 3; out_s = -bx / -by; out_t = bz / -by; }
    } else {
        if (bz >= 0.0) { out_face = 4; out_s = -by / bz; out_t = -bx / bz; }
        else           { out_face = 5; out_s = -by / -bz; out_t =  bx / -bz; }
    }
}

} // namespace

HdriPresetInfo HdriExporter::preset_info(HdriQuality quality, uint32_t max_image_dim) noexcept {
    size_t idx = std::min(static_cast<size_t>(quality), std::size(PRESETS) - 1);
    HdriPresetInfo info = PRESETS[idx];
    uint32_t max_face = (max_image_dim > FACE_GUARD_PX * 2) ? (max_image_dim - FACE_GUARD_PX * 2) : 512u;
    info.face_size = std::min(info.face_size, max_face);
    return info;
}

std::optional<std::filesystem::path> HdriExporter::save_file_dialog(const std::filesystem::path& initial_path, void* owner_hwnd) {
#ifdef _WIN32
    wchar_t file_buf[4096] = {};
    if (!initial_path.empty()) {
        std::wstring init_w = initial_path.wstring();
        size_t n = std::min(init_w.size(), std::size(file_buf) - 1);
        std::memcpy(file_buf, init_w.data(), n * sizeof(wchar_t));
        file_buf[n] = L'\0';
    }
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = static_cast<HWND>(owner_hwnd);
    ofn.lpstrFilter = L"OpenEXR 32-bit HDRI (*.exr)\0*.exr\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = file_buf;
    ofn.nMaxFile = static_cast<DWORD>(std::size(file_buf));
    ofn.lpstrDefExt = L"exr";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (GetSaveFileNameW(&ofn)) {
        return std::filesystem::path(file_buf);
    }
#else
    (void)initial_path;
    (void)owner_hwnd;
#endif
    return std::nullopt;
}

void HdriExporter::begin(const HdriExportOptions& options, const Camera& camera, const RenderSettings& settings,
                         uint32_t max_image_dim, bool fp32_supported, bool was_paused) {
    release_face_storage();
    options_ = options;
    if (options_.output_path.extension() != ".exr") {
        options_.output_path.replace_extension(".exr");
    }
    preset_ = preset_info(options_.quality, max_image_dim);
    padded_face_ = preset_.face_size + FACE_GUARD_PX * 2;
    target_format_ = fp32_supported ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    was_paused_ = was_paused;
    face_index_ = 0;
    error_.clear();

    base_camera_ = camera;
    export_settings_ = settings;
    export_settings_.hdr_enabled = true;
    export_settings_.exposure = (options_.exposure_mode == HdriExposureMode::NormalizedCurrent)
        ? settings.exposure : 1.0f;
    export_settings_.show_orbits = false;
    export_settings_.bloom_mode = 0;
    export_settings_.atmosphere_shadow_temporal = false;
    export_settings_.atmosphere_shadow_rotation = false;
    export_settings_.atmosphere_shadow_animate_jitter = false;

    if (options_.orientation == HdriOrientation::CameraHorizon) {
        basis_x_ = glm::normalize(camera.forward_dir());
        basis_z_ = glm::normalize(camera.up_dir());
        basis_y_ = glm::normalize(glm::cross(basis_z_, basis_x_));
        basis_z_ = glm::normalize(glm::cross(basis_x_, basis_y_));
    } else {
        basis_x_ = {1.0, 0.0, 0.0};
        basis_y_ = {0.0, 0.0, -1.0};
        basis_z_ = {0.0, 1.0, 0.0};
    }

#ifdef _WIN32
    use_file_backing_ = (preset_.width > 16384u);
#else
    use_file_backing_ = false;
#endif
    phase_ = Phase::RenderFace;
    status_text_ = std::format("Rendering cube face 1 / 6: {}", face_label(0));
}

bool HdriExporter::step(rhi::VkContext& ctx, CelestialRenderer& celestial, StarCatalog& stars,
                        const ephemeris::SystemSnapshot& system) {
    if (phase_ == Phase::Idle || phase_ == Phase::Done || phase_ == Phase::Failed) {
        return false;
    }
    try {
        if (phase_ == Phase::RenderFace) {
            render_single_face(ctx, celestial, stars, system, face_index_);
            ++face_index_;
            if (face_index_ < 6) {
                status_text_ = std::format("Rendering cube face {} / 6: {}", face_index_ + 1, face_label(face_index_));
            } else {
                release_gpu_targets(ctx);
                phase_ = Phase::WriteExr;
                status_text_ = std::format("Projecting & writing 32-bit EXR ({} x {})...", preset_.width, preset_.height);
            }
            return true;
        }
        if (phase_ == Phase::WriteExr) {
            write_equirectangular_exr();
            release_face_storage();
            phase_ = Phase::Done;
            status_text_ = std::format("Exported {}", options_.output_path.string());
            log::info("HDRI skybox exported: {} ({}x{}, 32-bit EXR)", options_.output_path.string(), preset_.width, preset_.height);
            return false;
        }
    } catch (const std::exception& e) {
        log::error("HDRI export failed: {}", e.what());
        error_ = e.what();
        release_gpu_targets(ctx);
        release_face_storage();
        phase_ = Phase::Failed;
        return false;
    }
    return false;
}

void HdriExporter::cancel(rhi::VkContext& ctx) {
    release_gpu_targets(ctx);
    release_face_storage();
    phase_ = Phase::Idle;
    error_.clear();
    status_text_.clear();
}

void HdriExporter::shutdown(rhi::VkContext& ctx) {
    release_gpu_targets(ctx);
    release_face_storage();
}

float HdriExporter::progress() const noexcept {
    if (phase_ == Phase::Done) return 1.0f;
    if (phase_ == Phase::WriteExr) return 6.0f / 7.0f;
    if (phase_ == Phase::RenderFace) return static_cast<float>(face_index_) / 7.0f;
    return 0.0f;
}

void HdriExporter::release_gpu_targets(rhi::VkContext& ctx) {
    ctx.wait_idle();
    ctx.destroy_buffer(readback_);
    ctx.destroy_image(export_color_);
    ctx.destroy_image(export_depth_);
}

void HdriExporter::release_face_storage() {
    for (auto& vec : ram_faces_) {
        std::vector<float>().swap(vec);
    }
#ifdef _WIN32
    for (auto& mf : mapped_faces_) {
        if (mf.data) {
            UnmapViewOfFile(mf.data);
            mf.data = nullptr;
        }
        if (mf.mapping_handle) {
            CloseHandle(static_cast<HANDLE>(mf.mapping_handle));
            mf.mapping_handle = nullptr;
        }
        if (mf.file_handle) {
            CloseHandle(static_cast<HANDLE>(mf.file_handle));
            mf.file_handle = nullptr;
        }
        if (!mf.path.empty()) {
            std::error_code ec;
            std::filesystem::remove(mf.path, ec);
            mf.path.clear();
        }
    }
#endif
}

void HdriExporter::render_single_face(rhi::VkContext& ctx, CelestialRenderer& celestial, StarCatalog& stars,
                                      const ephemeris::SystemSnapshot& system, uint32_t face) {
    const VkExtent2D ext{padded_face_, padded_face_};
    const VkDeviceSize pixel_bytes = (target_format_ == VK_FORMAT_R32G32B32A32_SFLOAT) ? 16 : 8;
    const VkDeviceSize readback_bytes = VkDeviceSize(padded_face_) * padded_face_ * pixel_bytes;

    if (!export_color_ || export_color_.extent.width != padded_face_) {
        release_gpu_targets(ctx);
        export_color_ = ctx.create_image_2d(
            padded_face_, padded_face_, target_format_,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        export_depth_ = ctx.create_image_2d(
            padded_face_, padded_face_, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        readback_ = ctx.create_host_buffer(readback_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    Camera face_cam = base_camera_;
    face_cam.custom_basis = true;
    face_cam.cam_look_aim = false;
    face_cam.horizon_align = false;
    face_basis(face, basis_x_, basis_y_, basis_z_, face_cam.custom_forward, face_cam.up);
    face_cam.fov_deg = 2.0 * std::atan(double(padded_face_) / double(preset_.face_size)) * RAD_TO_DEG;

    // Wait briefly for terrain/surface texture streaming workers so side and rear faces load full-detail tiles.
    const auto settle_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (int iter = 0; iter < 60; ++iter) {
        ctx.immediate_submit([&](VkCommandBuffer cmd) {
            celestial.prepare(cmd, ctx, face_cam, system, export_settings_, ext, false, nullptr);
        });
        bool pending = celestial.terrain_stats().stream.pending > 0 ||
                       celestial.texture_stats().pending > 0 ||
                       celestial.normal_stats().pending > 0;
        if (!pending || std::chrono::steady_clock::now() >= settle_deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        ctx.transition_image(cmd, export_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        ctx.transition_image(cmd, export_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        celestial.prepare(cmd, ctx, face_cam, system, export_settings_, ext, false, nullptr);
        auto refraction_writes = celestial.refraction_writes(ctx);
        stars.cull(cmd, ctx, face_cam, system.time(), export_settings_, ext, refraction_writes);

        auto color_att = rhi::RenderingAttachment::clear_color(export_color_, 0.0f, 0.0f, 0.0f, 1.0f);
        auto depth_att = rhi::RenderingAttachment::clear_depth_reverse_z(export_depth_, 0.0f);
        ctx.begin_rendering(cmd, ext, std::span{&color_att, 1}, &depth_att);
        celestial.render(cmd, ctx, face_cam, system, export_settings_, ext, nullptr, false, target_format_);
        stars.render(cmd, ctx, face_cam, system.time(), export_settings_, ext, refraction_writes, target_format_);
        ctx.end_rendering(cmd);

        celestial.composite_atmospheres(cmd, ctx, export_color_, export_depth_, nullptr);

        ctx.transition_image(cmd, export_color_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {padded_face_, padded_face_, 1};
        vkCmdCopyImageToBuffer(cmd, export_color_.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.handle, 1, &copy);
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });

    VK_CHECK(vmaInvalidateAllocation(ctx.allocator(), readback_.allocation, 0, VK_WHOLE_SIZE));

    const size_t texels = size_t(padded_face_) * padded_face_;
    const size_t rgb_floats = texels * 3;
    std::vector<float> temp_rgb;
    std::vector<float>& dst = use_file_backing_ ? temp_rgb : ram_faces_[face];
    dst.resize(rgb_floats);

    if (target_format_ == VK_FORMAT_R32G32B32A32_SFLOAT) {
        const auto* src = static_cast<const float*>(readback_.mapped);
        for (size_t i = 0; i < texels; ++i) {
            dst[i * 3 + 0] = sanitize_radiance(src[i * 4 + 0]);
            dst[i * 3 + 1] = sanitize_radiance(src[i * 4 + 1]);
            dst[i * 3 + 2] = sanitize_radiance(src[i * 4 + 2]);
        }
    } else {
        const auto* src = static_cast<const uint16_t*>(readback_.mapped);
        for (size_t i = 0; i < texels; ++i) {
            dst[i * 3 + 0] = sanitize_radiance(glm::unpackHalf1x16(src[i * 4 + 0]));
            dst[i * 3 + 1] = sanitize_radiance(glm::unpackHalf1x16(src[i * 4 + 1]));
            dst[i * 3 + 2] = sanitize_radiance(glm::unpackHalf1x16(src[i * 4 + 2]));
        }
    }

#ifdef _WIN32
    if (use_file_backing_) {
        if (options_.output_path.has_parent_path()) {
            std::filesystem::create_directories(options_.output_path.parent_path());
        }
        auto tmp_path = options_.output_path;
        tmp_path += std::format(".face{}.tmp", face);
        HANDLE hf = CreateFileW(tmp_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (hf == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to create temporary face buffer: " + tmp_path.string());
        }
        const size_t total_bytes = rgb_floats * sizeof(float);
        size_t written_total = 0;
        const auto* byte_ptr = reinterpret_cast<const char*>(dst.data());
        while (written_total < total_bytes) {
            DWORD chunk = static_cast<DWORD>(std::min<size_t>(total_bytes - written_total, 64ull << 20));
            DWORD written = 0;
            if (!WriteFile(hf, byte_ptr + written_total, chunk, &written, nullptr) || written == 0) {
                CloseHandle(hf);
                std::error_code ec;
                std::filesystem::remove(tmp_path, ec);
                throw std::runtime_error("Failed writing temporary face buffer: " + tmp_path.string());
            }
            written_total += written;
        }
        HANDLE hm = CreateFileMappingW(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) {
            CloseHandle(hf);
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            throw std::runtime_error("Failed to map temporary face buffer: " + tmp_path.string());
        }
        const void* view = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            CloseHandle(hm);
            CloseHandle(hf);
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            throw std::runtime_error("Failed to map view of temporary face buffer: " + tmp_path.string());
        }
        mapped_faces_[face] = {tmp_path, hf, hm, static_cast<const float*>(view)};
    }
#endif
}

void HdriExporter::write_equirectangular_exr() {
    if (options_.output_path.has_parent_path()) {
        std::filesystem::create_directories(options_.output_path.parent_path());
    }
    std::ofstream file(options_.output_path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Could not open output EXR file: " + options_.output_path.string());
    }

    const uint32_t W = preset_.width;
    const uint32_t H = preset_.height;
    const uint32_t N = preset_.face_size;
    const uint32_t P = padded_face_;

    write_exr_header(file, W, H);
    const uint64_t table_pos = static_cast<uint64_t>(file.tellp());
    const uint64_t data_start = table_pos + uint64_t(H) * sizeof(uint64_t);
    const int32_t row_data_bytes = static_cast<int32_t>(W) * 3 * static_cast<int32_t>(sizeof(float));
    const uint64_t block_stride = 8ull + static_cast<uint64_t>(row_data_bytes);

    std::vector<uint64_t> offsets(H);
    for (uint32_t y = 0; y < H; ++y) {
        offsets[y] = data_start + uint64_t(y) * block_stride;
    }
    file.write(reinterpret_cast<const char*>(offsets.data()), static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));

    std::array<const float*, 6> face_ptrs{};
    for (uint32_t f = 0; f < 6; ++f) {
        face_ptrs[f] = use_file_backing_ ? mapped_faces_[f].data : ram_faces_[f].data();
        if (!face_ptrs[f]) {
            throw std::runtime_error("Missing rendered cube face data during EXR assembly.");
        }
    }

    std::vector<float> row_planar(size_t(W) * 3);
    float* plane_b = row_planar.data();
    float* plane_g = plane_b + W;
    float* plane_r = plane_g + W;

    std::vector<double> cos_phi(W), sin_phi(W);
    for (uint32_t x = 0; x < W; ++x) {
        const double u = (double(x) + 0.5) / double(W);
        const double phi = PI * (1.0 - 2.0 * u);
        cos_phi[x] = std::cos(phi);
        sin_phi[x] = std::sin(phi);
    }

    const double half_n = 0.5 * double(N);
    const double pad_d = double(FACE_GUARD_PX);
    const int max_idx = static_cast<int>(P) - 1;

    for (uint32_t y = 0; y < H; ++y) {
        const double theta = PI * (double(y) + 0.5) / double(H);
        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);

        for (uint32_t x = 0; x < W; ++x) {
            const double bx = sin_theta * cos_phi[x];
            const double by = sin_theta * sin_phi[x];
            const double bz = cos_theta;

            uint32_t face = 0;
            double s = 0.0, t = 0.0;
            map_direction_to_face(bx, by, bz, face, s, t);

            const double fx = pad_d + half_n * (s + 1.0) - 0.5;
            const double fy = pad_d + half_n * (1.0 - t) - 0.5;

            const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, max_idx);
            const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, max_idx);
            const int x1 = std::min(x0 + 1, max_idx);
            const int y1 = std::min(y0 + 1, max_idx);
            const float wx = static_cast<float>(std::clamp(fx - double(x0), 0.0, 1.0));
            const float wy = static_cast<float>(std::clamp(fy - double(y0), 0.0, 1.0));

            const float* fp = face_ptrs[face];
            const float* p00 = fp + (size_t(y0) * P + size_t(x0)) * 3;
            const float* p10 = fp + (size_t(y0) * P + size_t(x1)) * 3;
            const float* p01 = fp + (size_t(y1) * P + size_t(x0)) * 3;
            const float* p11 = fp + (size_t(y1) * P + size_t(x1)) * 3;

            const float w00 = (1.0f - wx) * (1.0f - wy);
            const float w10 = wx * (1.0f - wy);
            const float w01 = (1.0f - wx) * wy;
            const float w11 = wx * wy;

            plane_r[x] = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;
            plane_g[x] = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
            plane_b[x] = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
        }

        const int32_t y_i = static_cast<int32_t>(y);
        write_le(file, y_i);
        write_le(file, row_data_bytes);
        file.write(reinterpret_cast<const char*>(row_planar.data()), static_cast<std::streamsize>(row_data_bytes));
        if (!file) {
            throw std::runtime_error("Failed writing EXR scanline data: " + options_.output_path.string());
        }
    }
}

} // namespace sf::rendering
