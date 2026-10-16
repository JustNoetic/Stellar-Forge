#pragma once

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace sf::paths {

const std::filesystem::path& project_root();

template <typename... Parts>
std::filesystem::path bundled_path(Parts&&... parts) {
    std::filesystem::path p = project_root();
    ((p /= std::forward<Parts>(parts)), ...);
    return p;
}

template <typename... Parts>
std::filesystem::path external_path(Parts&&... parts) {
    std::filesystem::path p = project_root();
    ((p /= std::forward<Parts>(parts)), ...);
    return p;
}

std::optional<std::string> read_text_file(const std::filesystem::path& path);
std::optional<std::vector<uint8_t>> read_binary_file(const std::filesystem::path& path);
bool write_binary_file(const std::filesystem::path& path, const void* data, size_t size);
std::optional<nlohmann::json> read_json(const std::filesystem::path& path);
bool write_json(const std::filesystem::path& path, const nlohmann::json& j, int indent = 2);

} // namespace sf::paths
