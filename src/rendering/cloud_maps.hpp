#pragma once

#include "rhi/vk_context.hpp"
#include <string>
#include <unordered_map>

namespace sf::rendering {

class CloudMaps {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    const rhi::Image& image() const noexcept { return atlas_; }
    uint32_t layer(const std::string& name) const noexcept;

private:
    rhi::Image atlas_;
    std::unordered_map<std::string, uint32_t> layers_;
};

}
