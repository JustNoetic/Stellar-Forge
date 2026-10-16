#include "core/path_utils.hpp"

#include <fstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace sf::paths {

namespace {

std::filesystem::path detect_project_root() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH]{};
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len > 0) {
        std::filesystem::path exe_dir = std::filesystem::path(buf).parent_path();
        for (std::filesystem::path cur = exe_dir; !cur.empty(); cur = cur.parent_path()) {
            if (std::filesystem::exists(cur / "CMakeLists.txt") || std::filesystem::exists(cur / "data")) {
                return cur;
            }
            if (cur == cur.root_path()) {
                break;
            }
        }
        return exe_dir;
    }
#endif
    return std::filesystem::current_path();
}

} // namespace

const std::filesystem::path& project_root() {
    static const std::filesystem::path root = detect_project_root();
    return root;
}

std::optional<std::string> read_text_file(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        return std::nullopt;
    }
    ifs.seekg(0, std::ios::end);
    const auto sz = ifs.tellg();
    if (sz < 0) {
        return std::nullopt;
    }
    std::string out(static_cast<size_t>(sz), '\0');
    ifs.seekg(0, std::ios::beg);
    ifs.read(out.data(), sz);
    return out;
}

std::optional<std::vector<uint8_t>> read_binary_file(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        return std::nullopt;
    }
    ifs.seekg(0, std::ios::end);
    const auto sz = ifs.tellg();
    if (sz < 0) {
        return std::nullopt;
    }
    std::vector<uint8_t> out(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(out.data()), sz);
    return out;
}

bool write_binary_file(const std::filesystem::path& path, const void* data, size_t size) {
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    return ofs.good();
}

std::optional<nlohmann::json> read_json(const std::filesystem::path& path) {
    auto text = read_text_file(path);
    if (!text) {
        return std::nullopt;
    }
    auto parsed = nlohmann::json::parse(*text, nullptr, false);
    if (parsed.is_discarded()) {
        return std::nullopt;
    }
    return parsed;
}

bool write_json(const std::filesystem::path& path, const nlohmann::json& j, int indent) {
    std::string dumped = j.dump(indent);
    return write_binary_file(path, dumped.data(), dumped.size());
}

} // namespace sf::paths
