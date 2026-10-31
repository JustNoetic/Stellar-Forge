#include "terrain/surface_baker.hpp"
#include "core/constants.hpp"
#include "core/path_utils.hpp"
#include "ephemeris/system_manager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <objbase.h>
#include <wincodec.h>
#endif

namespace sf::terrain {

namespace {

int calc_recommended_lod(uint32_t width) noexcept {
    double ratio = std::max(1.0, static_cast<double>(width) / static_cast<double>(COLOR_CORE * 4));
    return std::clamp(static_cast<int>(std::ceil(std::log2(ratio))), 0, 5);
}

#ifdef _WIN32
std::wstring quote_arg(std::wstring_view arg) {
    std::wstring out = L"\"";
    for (wchar_t c : arg) {
        if (c == L'"') out += L"\\\"";
        else out += c;
    }
    out += L"\"";
    return out;
}

bool decode_with_wic(const std::filesystem::path& path, StagedImageInfo& out) {
    HRESULT hr_co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    bool should_uninit = SUCCEEDED(hr_co);

    IWICImagingFactory* factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        if (should_uninit) CoUninitialize();
        return false;
    }

    IWICBitmapDecoder* decoder = nullptr;
    std::wstring wpath = path.wstring();
    hr = factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnDemand, &decoder);
    if (FAILED(hr) || !decoder) {
        factory->Release();
        if (should_uninit) CoUninitialize();
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr) || !frame) {
        decoder->Release();
        factory->Release();
        if (should_uninit) CoUninitialize();
        return false;
    }

    UINT src_w = 0, src_h = 0;
    hr = frame->GetSize(&src_w, &src_h);
    if (FAILED(hr) || src_w == 0 || src_h == 0) {
        frame->Release();
        decoder->Release();
        factory->Release();
        if (should_uninit) CoUninitialize();
        return false;
    }

    out.width = src_w;
    out.height = src_h;
    out.recommended_lod = calc_recommended_lod(src_w);

    IWICBitmapScaler* scaler = nullptr;
    hr = factory->CreateBitmapScaler(&scaler);
    if (FAILED(hr) || !scaler) {
        frame->Release();
        decoder->Release();
        factory->Release();
        if (should_uninit) CoUninitialize();
        return false;
    }

    hr = scaler->Initialize(frame, SURFACE_PREVIEW_W, SURFACE_PREVIEW_H, WICBitmapInterpolationModeFant);
    if (FAILED(hr)) {
        hr = scaler->Initialize(frame, SURFACE_PREVIEW_W, SURFACE_PREVIEW_H, WICBitmapInterpolationModeLinear);
    }

    bool decoded = false;
    if (SUCCEEDED(hr)) {
        WICPixelFormatGUID pf{};
        if (SUCCEEDED(scaler->GetPixelFormat(&pf)) && pf == GUID_WICPixelFormat16bppGray) {
            std::vector<uint16_t> gray16(SURFACE_PREVIEW_W * SURFACE_PREVIEW_H);
            if (SUCCEEDED(scaler->CopyPixels(nullptr, SURFACE_PREVIEW_W * 2,
                                             static_cast<UINT>(gray16.size() * 2),
                                             reinterpret_cast<BYTE*>(gray16.data())))) {
                uint16_t gmin = 65535, gmax = 0;
                for (uint16_t v : gray16) {
                    gmin = std::min(gmin, v);
                    gmax = std::max(gmax, v);
                }
                double span = (gmax > gmin) ? static_cast<double>(gmax - gmin) : 1.0;
                out.preview_rgba.resize(SURFACE_PREVIEW_W * SURFACE_PREVIEW_H * 4);
                for (size_t i = 0; i < gray16.size(); ++i) {
                    uint8_t g = static_cast<uint8_t>(std::clamp(std::round((gray16[i] - gmin) / span * 255.0), 0.0, 255.0));
                    out.preview_rgba[i * 4 + 0] = g;
                    out.preview_rgba[i * 4 + 1] = g;
                    out.preview_rgba[i * 4 + 2] = g;
                    out.preview_rgba[i * 4 + 3] = 255;
                }
                decoded = true;
            }
        }

        if (!decoded) {
            IWICFormatConverter* converter = nullptr;
            if (SUCCEEDED(factory->CreateFormatConverter(&converter)) && converter) {
                if (SUCCEEDED(converter->Initialize(scaler, GUID_WICPixelFormat32bppRGBA,
                                                    WICBitmapDitherTypeNone, nullptr, 0.0,
                                                    WICBitmapPaletteTypeCustom))) {
                    out.preview_rgba.resize(SURFACE_PREVIEW_W * SURFACE_PREVIEW_H * 4);
                    if (SUCCEEDED(converter->CopyPixels(nullptr, SURFACE_PREVIEW_W * 4,
                                                        static_cast<UINT>(out.preview_rgba.size()),
                                                        out.preview_rgba.data()))) {
                        for (size_t i = 0; i < SURFACE_PREVIEW_W * SURFACE_PREVIEW_H; ++i) {
                            out.preview_rgba[i * 4 + 3] = 255;
                        }
                        decoded = true;
                    }
                }
                converter->Release();
            }
        }
    }

    scaler->Release();
    frame->Release();
    decoder->Release();
    factory->Release();
    if (should_uninit) CoUninitialize();
    return decoded;
}
#endif

} // namespace

SurfaceBaker::~SurfaceBaker() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
}

std::filesystem::path SurfaceBaker::resolve_python_executable() {
    std::error_code ec;
    auto p1 = paths::external_path("venv", "Scripts", "python.exe");
    if (std::filesystem::is_regular_file(p1, ec)) return p1;

    auto p2 = paths::external_path("..", "Stellar Forge", "venv", "Scripts", "python.exe");
    if (std::filesystem::is_regular_file(p2, ec)) return p2;

    return "python.exe";
}

std::optional<std::filesystem::path> SurfaceBaker::open_image_file_dialog(void* owner_hwnd) {
#ifdef _WIN32
    wchar_t file_buf[4096] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = static_cast<HWND>(owner_hwnd);
    ofn.lpstrFilter = L"Equirectangular Images (*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp)\0*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = file_buf;
    ofn.nMaxFile = static_cast<DWORD>(std::size(file_buf));
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (GetOpenFileNameW(&ofn)) {
        return std::filesystem::path(file_buf);
    }
#else
    (void)owner_hwnd;
#endif
    return std::nullopt;
}

std::optional<std::filesystem::path> SurfaceBaker::find_existing_source_image(
    std::string_view system_name, std::string_view body_name, bool is_height) {
    if (body_name.empty()) return std::nullopt;
    std::string sys_folder = (system_name == "Ephemeris Mode" || system_name.empty())
        ? std::string(ephemeris::SystemManager::SOLAR_SYSTEM_NAME)
        : std::string(system_name);

    std::string target_stem = is_height ? std::format("{}_heightmap", body_name) : std::string(body_name);
    for (char& c : target_stem) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    static constexpr std::string_view exts[] = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".webp", ".bmp"};
    const std::filesystem::path folders[] = {
        paths::external_path("textures", sys_folder, body_name),
        paths::external_path("textures", "Solar System", body_name),
    };

    std::error_code ec;
    for (const auto& folder : folders) {
        if (!std::filesystem::is_directory(folder, ec)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(folder, ec)) {
            if (!entry.is_regular_file(ec)) continue;
            std::string stem = entry.path().stem().string();
            std::string ext = entry.path().extension().string();
            for (char& c : stem) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (stem != target_stem) continue;
            for (auto allowed : exts) {
                if (ext == allowed) return entry.path();
            }
        }
    }
    return std::nullopt;
}

std::optional<StagedImageInfo> SurfaceBaker::inspect_and_load_image(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return std::nullopt;

    StagedImageInfo info{};
    info.path = path;

#ifdef _WIN32
    if (decode_with_wic(path, info)) {
        return info;
    }

    auto py = resolve_python_executable();
    auto script = paths::external_path("scripts", "bake_surface_maps.py");
    auto temp_raw = paths::external_path("cache", "staged_preview.rgba");
    std::filesystem::create_directories(temp_raw.parent_path(), ec);

    std::wstring cmd = std::format(L"{} -u {} --inspect-image {} --preview-out {}",
                                   quote_arg(py.wstring()),
                                   quote_arg(script.wstring()),
                                   quote_arg(path.wstring()),
                                   quote_arg(temp_raw.wstring()));

    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE r_pipe = nullptr, w_pipe = nullptr;
    if (!CreatePipe(&r_pipe, &w_pipe, &sa, 0)) return std::nullopt;
    SetHandleInformation(r_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = w_pipe;
    si.hStdError = w_pipe;

    PROCESS_INFORMATION pi{};
    std::wstring work_dir = paths::external_path(".").wstring();
    BOOL created = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                  CREATE_NO_WINDOW, nullptr, work_dir.c_str(), &si, &pi);
    CloseHandle(w_pipe);
    if (!created) {
        CloseHandle(r_pipe);
        return std::nullopt;
    }

    std::string output;
    char buf[512];
    DWORD read_bytes = 0;
    while (ReadFile(r_pipe, buf, sizeof(buf), &read_bytes, nullptr) && read_bytes > 0) {
        output.append(buf, read_bytes);
    }
    WaitForSingleObject(pi.hProcess, 15000);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(r_pipe);

    auto pos = output.find("INSPECT|");
    if (pos != std::string::npos) {
        unsigned int w = 0, h = 0;
        int rec = 3;
        if (std::sscanf(output.c_str() + pos, "INSPECT|%u|%u|%d", &w, &h, &rec) == 3 && w > 0 && h > 0) {
            info.width = w;
            info.height = h;
            info.recommended_lod = std::clamp(rec, 0, 5);
            if (auto bytes = paths::read_binary_file(temp_raw);
                bytes && bytes->size() == SURFACE_PREVIEW_W * SURFACE_PREVIEW_H * 4) {
                info.preview_rgba = std::move(*bytes);
                return info;
            }
        }
    }
#endif
    return std::nullopt;
}

std::vector<uint8_t> SurfaceBaker::build_solid_preview(float r, float g, float b) {
    uint8_t cr = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
    uint8_t cg = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
    uint8_t cb = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
    std::vector<uint8_t> rgba(SURFACE_PREVIEW_W * SURFACE_PREVIEW_H * 4);
    for (size_t i = 0; i < SURFACE_PREVIEW_W * SURFACE_PREVIEW_H; ++i) {
        rgba[i * 4 + 0] = cr;
        rgba[i * 4 + 1] = cg;
        rgba[i * 4 + 2] = cb;
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

std::vector<uint8_t> SurfaceBaker::build_color_pack_preview(const SurfaceSource& source) {
    std::array<std::vector<std::byte>, 6> roots;
    try {
        for (uint32_t face = 0; face < 6; ++face) {
            roots[face] = source.load({face, 0, 0, 0});
            if (roots[face].size() < COLOR_BYTES) return build_solid_preview(0.2f, 0.2f, 0.25f);
        }
    } catch (...) {
        return build_solid_preview(0.2f, 0.2f, 0.25f);
    }

    std::vector<uint8_t> out(SURFACE_PREVIEW_W * SURFACE_PREVIEW_H * 4, 255);
    double polar = std::clamp(source.polar_ratio, 0.05, 1.0);
    for (uint32_t py = 0; py < SURFACE_PREVIEW_H; ++py) {
        double lat = (0.5 - (static_cast<double>(py) + 0.5) / SURFACE_PREVIEW_H) * PI;
        double cos_lat = std::cos(lat);
        double dy = polar * std::sin(lat);
        for (uint32_t px = 0; px < SURFACE_PREVIEW_W; ++px) {
            double lon = ((static_cast<double>(px) + 0.5) / SURFACE_PREVIEW_W - 0.5) * TWO_PI;
            double dx = cos_lat * std::sin(lon);
            double dz = cos_lat * std::cos(lon);
            glm::dvec3 dir = glm::normalize(glm::dvec3(dx, dy, dz));
            TileId at = tile_at(dir, 0);
            glm::dvec3 axis = cube_vector(at.face, 0.0, 0.0);
            double div = glm::dot(dir, axis);
            double fu = glm::dot(dir, cube_vector(at.face, 1.0, 0.0) - axis) / div;
            double fv = glm::dot(dir, cube_vector(at.face, 0.0, 1.0) - axis) / div;
            int tx = std::clamp(static_cast<int>((fu + 1.0) * 0.5 * COLOR_CORE) + static_cast<int>(COLOR_GUTTER),
                                0, static_cast<int>(COLOR_SIDE - 1));
            int ty = std::clamp(static_cast<int>((fv + 1.0) * 0.5 * COLOR_CORE) + static_cast<int>(COLOR_GUTTER),
                                0, static_cast<int>(COLOR_SIDE - 1));
            const auto* src_px = reinterpret_cast<const uint8_t*>(roots[at.face].data()) + (ty * COLOR_SIDE + tx) * 4;
            size_t dst_idx = (static_cast<size_t>(py) * SURFACE_PREVIEW_W + px) * 4;
            out[dst_idx + 0] = src_px[0];
            out[dst_idx + 1] = src_px[1];
            out[dst_idx + 2] = src_px[2];
            out[dst_idx + 3] = 255;
        }
    }
    return out;
}

std::vector<uint8_t> SurfaceBaker::build_height_pack_preview(const HeightSource& source) {
    std::array<HeightTile, 6> roots;
    try {
        for (uint32_t face = 0; face < 6; ++face) {
            roots[face] = source.read({face, 0, 0, 0});
        }
    } catch (...) {
        return build_solid_preview(0.15f, 0.15f, 0.18f);
    }

    std::vector<uint8_t> out(SURFACE_PREVIEW_W * SURFACE_PREVIEW_H * 4, 255);
    double polar = std::clamp(source.info().polar_ratio, 0.05, 1.0);
    double h_min = source.info().minimum_m;
    double h_max = source.info().maximum_m;
    double span = (h_max > h_min + 1e-6) ? (h_max - h_min) : 1.0;

    for (uint32_t py = 0; py < SURFACE_PREVIEW_H; ++py) {
        double lat = (0.5 - (static_cast<double>(py) + 0.5) / SURFACE_PREVIEW_H) * PI;
        double cos_lat = std::cos(lat);
        double dy = polar * std::sin(lat);
        for (uint32_t px = 0; px < SURFACE_PREVIEW_W; ++px) {
            double lon = ((static_cast<double>(px) + 0.5) / SURFACE_PREVIEW_W - 0.5) * TWO_PI;
            double dx = cos_lat * std::sin(lon);
            double dz = cos_lat * std::cos(lon);
            glm::dvec3 dir = glm::normalize(glm::dvec3(dx, dy, dz));
            TileId at = tile_at(dir, 0);
            glm::dvec3 axis = cube_vector(at.face, 0.0, 0.0);
            double div = glm::dot(dir, axis);
            double fu = glm::dot(dir, cube_vector(at.face, 1.0, 0.0) - axis) / div;
            double fv = glm::dot(dir, cube_vector(at.face, 0.0, 1.0) - axis) / div;
            const auto& tile = roots[at.face];
            int tx = std::clamp(static_cast<int>( std::round((fu + 1.0) * 0.5 * tile.cells) ) + static_cast<int>(tile.gutter),
                                0, static_cast<int>(tile.side - 1));
            int ty = std::clamp(static_cast<int>( std::round((fv + 1.0) * 0.5 * tile.cells) ) + static_cast<int>(tile.gutter),
                                0, static_cast<int>(tile.side - 1));
            double h = tile.heights_m[static_cast<size_t>(ty) * tile.side + tx];
            uint8_t g = static_cast<uint8_t>(std::clamp(std::round((h - h_min) / span * 255.0), 0.0, 255.0));
            size_t dst_idx = (static_cast<size_t>(py) * SURFACE_PREVIEW_W + px) * 4;
            out[dst_idx + 0] = g;
            out[dst_idx + 1] = g;
            out[dst_idx + 2] = g;
            out[dst_idx + 3] = 255;
        }
    }
    return out;
}

bool SurfaceBaker::remove_from_catalog(std::string_view body_name, bool is_height) {
    auto catalog_path = paths::external_path("data", "surfaces.json");
    auto j_opt = paths::read_json(catalog_path);
    if (!j_opt || !j_opt->is_object() || !j_opt->contains("bodies") || !(*j_opt)["bodies"].is_array()) {
        return false;
    }
    auto& bodies = (*j_opt)["bodies"];
    std::string key = is_height ? "height" : "albedo";
    bool modified = false;
    for (auto it = bodies.begin(); it != bodies.end();) {
        if (it->is_object() && it->value("body", "") == body_name) {
            if (it->contains(key)) {
                it->erase(key);
                modified = true;
            }
            if (!it->contains("albedo") && !it->contains("height")) {
                it = bodies.erase(it);
                continue;
            }
        }
        ++it;
    }
    if (!modified) return false;
    return paths::write_json(catalog_path, *j_opt, 2);
}

bool SurfaceBaker::start_bake(SurfaceBakeRequest request) {
    if (active_.load(std::memory_order_relaxed)) return false;
    if (worker_.joinable()) worker_.join();

    {
        std::lock_guard lock(mutex_);
        status_text_ = "Starting baker...";
        backend_name_.clear();
        result_ = {};
    }
    progress_.store(0.0f, std::memory_order_relaxed);
    completed_.store(false, std::memory_order_relaxed);
    start_time_ = std::chrono::steady_clock::now();
    active_.store(true, std::memory_order_release);

    worker_ = std::jthread([this, req = std::move(request)](std::stop_token) {
        run_worker(std::move(req));
    });
    return true;
}

std::optional<SurfaceBakeResult> SurfaceBaker::poll() {
    if (!completed_.load(std::memory_order_acquire)) return std::nullopt;
    if (worker_.joinable()) worker_.join();
    completed_.store(false, std::memory_order_relaxed);
    std::lock_guard lock(mutex_);
    return result_;
}

double SurfaceBaker::elapsed_seconds() const noexcept {
    if (!active_.load(std::memory_order_relaxed)) return 0.0;
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

std::string SurfaceBaker::status_text() const {
    std::lock_guard lock(mutex_);
    return status_text_;
}

std::string SurfaceBaker::backend_name() const {
    std::lock_guard lock(mutex_);
    return backend_name_;
}

void SurfaceBaker::run_worker(SurfaceBakeRequest req) {
    SurfaceBakeResult res{};
    res.body_name = req.body_name;
    res.is_height = req.is_height;
    res.level = req.level;
    res.height_min_km = req.height_min_km;
    res.height_max_km = req.height_max_km;

#ifdef _WIN32
    auto py = resolve_python_executable();
    auto script = paths::external_path("scripts", "bake_surface_maps.py");
    std::string sys_folder = (req.system_name == "Ephemeris Mode" || req.system_name.empty())
        ? std::string(ephemeris::SystemManager::SOLAR_SYSTEM_NAME)
        : req.system_name;
    auto copy_dir = paths::external_path("textures", sys_folder, req.body_name);

    std::wstring cmd = std::format(
        L"{} -u {} --single-body {} --map-type {} --source-file {} --level {} --polar-ratio {:.12g} --radius-m {:.12g} --height-min-km {:.8g} --height-max-km {:.8g} --backend {}",
        quote_arg(py.wstring()),
        quote_arg(script.wstring()),
        quote_arg(std::filesystem::path(req.body_name).wstring()),
        req.is_height ? L"height" : L"albedo",
        quote_arg(req.source_file.wstring()),
        req.level,
        req.polar_ratio,
        req.radius_m,
        req.height_min_km,
        req.height_max_km,
        std::filesystem::path(req.backend).wstring()
    );
    if (req.copy_to_system_textures) {
        cmd += std::format(L" --copy-to-dir {}", quote_arg(copy_dir.wstring()));
    }

    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE r_pipe = nullptr, w_pipe = nullptr;
    if (!CreatePipe(&r_pipe, &w_pipe, &sa, 0)) {
        res.error = "Failed to create process output pipe.";
        std::lock_guard lock(mutex_);
        result_ = std::move(res);
        active_.store(false, std::memory_order_release);
        completed_.store(true, std::memory_order_release);
        return;
    }
    SetHandleInformation(r_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = w_pipe;
    si.hStdError = w_pipe;

    PROCESS_INFORMATION pi{};
    std::wstring work_dir = paths::external_path(".").wstring();
    BOOL created = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                  CREATE_NO_WINDOW, nullptr, work_dir.c_str(), &si, &pi);
    CloseHandle(w_pipe);

    if (!created) {
        CloseHandle(r_pipe);
        res.error = "Failed to launch Python texture baker process.";
        std::lock_guard lock(mutex_);
        result_ = std::move(res);
        active_.store(false, std::memory_order_release);
        completed_.store(true, std::memory_order_release);
        return;
    }

    std::string pending;
    std::string last_non_empty_line;
    char buf[512];
    DWORD bytes_read = 0;

    auto process_line = [&](std::string_view line) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.remove_suffix(1);
        }
        if (line.empty()) return;
        if (line.starts_with("BACKEND|")) {
            std::string bname(line.substr(8));
            std::lock_guard lock(mutex_);
            backend_name_ = bname;
            res.backend_used = bname;
        } else if (line.starts_with("PROGRESS|")) {
            auto rest = line.substr(9);
            auto p1 = rest.find('|');
            if (p1 != std::string_view::npos) {
                auto p2 = rest.find('|', p1 + 1);
                if (p2 != std::string_view::npos) {
                    int done = std::atoi(std::string(rest.substr(0, p1)).c_str());
                    int total = std::max(1, std::atoi(std::string(rest.substr(p1 + 1, p2 - p1 - 1)).c_str()));
                    std::string msg(rest.substr(p2 + 1));
                    progress_.store(std::clamp(static_cast<float>(done) / static_cast<float>(total), 0.0f, 1.0f),
                                    std::memory_order_relaxed);
                    std::lock_guard lock(mutex_);
                    status_text_ = std::move(msg);
                }
            }
        } else if (line.starts_with("RESULT|")) {
            auto rest = line.substr(7);
            auto p1 = rest.find('|');
            if (p1 != std::string_view::npos) {
                std::string rel_path(rest.substr(p1 + 1));
                res.pack_path = paths::external_path(rel_path);
            }
        } else {
            last_non_empty_line = std::string(line);
        }
    };

    while (ReadFile(r_pipe, buf, sizeof(buf), &bytes_read, nullptr) && bytes_read > 0) {
        pending.append(buf, bytes_read);
        size_t nl = 0;
        while ((nl = pending.find('\n')) != std::string::npos) {
            process_line(std::string_view(pending).substr(0, nl));
            pending.erase(0, nl + 1);
        }
    }
    if (!pending.empty()) {
        process_line(pending);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(r_pipe);

    auto end_time = std::chrono::steady_clock::now();
    res.elapsed_sec = std::chrono::duration<double>(end_time - start_time_).count();

    if (exit_code == 0 && !res.pack_path.empty()) {
        res.success = true;
        progress_.store(1.0f, std::memory_order_relaxed);
    } else {
        res.success = false;
        res.error = last_non_empty_line.empty()
            ? std::format("Baker exited with code {}.", exit_code)
            : last_non_empty_line;
    }
#else
    res.error = "In-sim baking is only supported on Windows.";
#endif

    {
        std::lock_guard lock(mutex_);
        result_ = std::move(res);
    }
    active_.store(false, std::memory_order_release);
    completed_.store(true, std::memory_order_release);
}

} // namespace sf::terrain
