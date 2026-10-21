#include "rendering/refraction_renderer.hpp"
#include "core/constants.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sf::rendering {

namespace {

constexpr VkDeviceSize STAGING_BYTES = 4ull << 20;

void write_ubo(rhi::VkContext& ctx, rhi::Buffer& buffer, const RefractionGPU& value) {
    std::memcpy(buffer.mapped, &value, sizeof(RefractionGPU));
    ctx.flush_buffer(buffer, 0, sizeof(RefractionGPU));
}

bool same_observer(const RefractionGPU& a, const RefractionGPU& b) {
    return a.frame == b.frame && a.mapping == b.mapping && a.model == b.model && a.profile == b.profile && a.escape == b.escape;
}

}

void RefractionFallback::init(rhi::VkContext& ctx) {
    RefractionGPU disabled{};
    ubo_ = ctx.create_host_buffer(sizeof(RefractionGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &disabled);
    observer_ = ctx.create_image_2d(2, 2, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    const float zeros[8]{};
    ctx.upload_image_2d(observer_, zeros, sizeof(zeros));
}

void RefractionFallback::shutdown(rhi::VkContext& ctx) {
    ctx.destroy_image(observer_);
    ctx.destroy_buffer(ubo_);
}

std::array<rhi::PushDescriptorWrite, 2> RefractionFallback::writes(rhi::VkContext& ctx) const {
    return {rhi::PushDescriptorWrite::uniform_buffer(20, ubo_, 0, sizeof(RefractionGPU)),
            rhi::PushDescriptorWrite::sampled_image(21, observer_, ctx.sampler_nearest_clamp())};
}

void RefractionRenderer::init(rhi::VkContext& ctx) {
    staging_.init(ctx, STAGING_BYTES);
    RefractionGPU disabled{};
    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        ubos_[i] = ctx.create_host_buffer(sizeof(RefractionGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &disabled);
        observers_[i] = ctx.create_image_2d(REFRACTION_OBSERVER_WIDTH, REFRACTION_OBSERVER_HEIGHT, VK_FORMAT_R32G32_SFLOAT,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }
    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        for (auto& image : observers_)
            ctx.transition_image(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    });
    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
    };
    pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "refraction/observer_table.comp", bindings);
}

void RefractionRenderer::shutdown(rhi::VkContext& ctx) {
    rhi::ShaderCache::destroy_pipeline(ctx, pipeline_);
    for (auto& t : tables_) {
        ctx.destroy_image(t.travel);
        ctx.destroy_image(t.rows);
        ctx.destroy_image(t.profile);
    }
    tables_.clear();
    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        ctx.destroy_image(observers_[i]);
        ctx.destroy_buffer(ubos_[i]);
    }
    staging_.shutdown(ctx);
}

RefractionRenderer::Tables* RefractionRenderer::tables_for(VkCommandBuffer cmd, rhi::VkContext& ctx,
                                                           const std::shared_ptr<const physics::PreparedAtmosphere>& atmosphere) {
    for (auto it = tables_.begin(); it != tables_.end(); ++it) {
        if (it->atmosphere != atmosphere) continue;
        if (it->ready) {
            it->last_serial = ctx.recording_serial();
            return &*it;
        }
        if (it->last_serial > ctx.completed_serial()) return nullptr;
        ctx.destroy_image(it->travel);
        ctx.destroy_image(it->rows);
        ctx.destroy_image(it->profile);
        tables_.erase(it);
        break;
    }
    auto packed = pack_refraction_tables(*atmosphere->refraction);
    Tables t;
    t.atmosphere = atmosphere;
    t.last_serial = ctx.recording_serial();
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    t.travel = ctx.create_image_2d(physics::REFRACTION_TRAVEL_SAMPLES, physics::REFRACTION_IMPACT_SAMPLES, VK_FORMAT_R32_SFLOAT, usage);
    t.rows = ctx.create_image_2d(physics::REFRACTION_IMPACT_SAMPLES, 1, VK_FORMAT_R32G32_SFLOAT, usage);
    t.profile = ctx.create_image_2d(REFRACTION_PROFILE_TEXELS_X, packed.profile_rows, VK_FORMAT_R32G32_SFLOAT, usage);
    auto upload = [&](rhi::Image& image, std::span<const std::byte> bytes) {
        ctx.transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = image.extent;
        bool staged = staging_.image(cmd, ctx, image, region, bytes);
        ctx.transition_image(cmd, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        return staged;
    };
    t.ready = upload(t.travel, std::as_bytes(std::span(packed.travel))) &&
              upload(t.rows, std::as_bytes(std::span(packed.rows))) &&
              upload(t.profile, std::as_bytes(std::span(packed.profile)));
    tables_.push_back(std::move(t));
    return tables_.back().ready ? &tables_.back() : nullptr;
}

void RefractionRenderer::retire_tables(rhi::VkContext& ctx, const physics::PreparedAtmosphere* keep) {
    constexpr size_t RETAINED = 4;
    while (tables_.size() > RETAINED) {
        auto oldest = tables_.end();
        for (auto it = tables_.begin(); it != tables_.end(); ++it)
            if (it->atmosphere.get() != keep && it->last_serial <= ctx.completed_serial() &&
                (oldest == tables_.end() || it->last_serial < oldest->last_serial)) oldest = it;
        if (oldest == tables_.end()) return;
        ctx.destroy_image(oldest->travel);
        ctx.destroy_image(oldest->rows);
        ctx.destroy_image(oldest->profile);
        tables_.erase(oldest);
    }
    for (auto it = tables_.begin(); it != tables_.end();) {
        if (!it->ready && it->atmosphere.get() != keep && it->last_serial <= ctx.completed_serial()) {
            ctx.destroy_image(it->travel);
            ctx.destroy_image(it->rows);
            ctx.destroy_image(it->profile);
            it = tables_.erase(it);
        } else ++it;
    }
}

void RefractionRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings) {
    uint32_t f = ctx.frame_index();
    RefractionGPU params{};
    int previous = host_.body;
    host_ = {};
    if (settings.refraction_enabled && settings.show_atmospheres) {
        auto eye = camera.eye_pos_render();
        double best = std::numeric_limits<double>::infinity(), kept = best;
        for (size_t i = 0; i < system.size(); ++i) {
            const auto& body = system.body(i);
            if (!body.render.atmosphere || !body.render.atmosphere->refraction || !body.definition.atmosphere.enabled) continue;
            const auto& r = *body.render.atmosphere->refraction;
            double polar_ratio = std::clamp(1.0-body.definition.oblateness, 0.05, 1.0);
            glm::dvec3 pole = glm::normalize(body.render.pole);
            glm::dvec3 relative = (eye.value-body.render.position.value)*AU_TO_KM;
            relative += pole*glm::dot(relative, pole)*(1.0/polar_ratio-1.0);
            double distance = std::max(0.0, glm::length(relative)-r.top_radius_km);
            if (int(i) == previous) kept = distance;
            if (distance < best) {
                best = distance;
                host_.body = int(i);
            }
        }
        if (previous >= 0 && std::isfinite(kept) && kept <= 1.1*best+1e-9) host_.body = previous;
    }
    const physics::PreparedRefraction* source = nullptr;
    if (host_.body >= 0) {
        const auto& body = system.body(size_t(host_.body));
        const auto& r = *body.render.atmosphere->refraction;
        double polar_ratio = std::clamp(1.0-body.definition.oblateness, 0.05, 1.0);
        glm::dvec3 pole = glm::normalize(body.render.pole);
        glm::dvec3 center = camera_relative(body.render.position, camera.eye_pos_render());
        glm::dvec3 relative = -center*AU_TO_KM;
        relative += pole*glm::dot(relative, pole)*(1.0/polar_ratio-1.0);
        host_.camera_radius_km = glm::length(relative);
        host_.ground_radius_km = r.ground_radius_km;
        host_.max_displacement = r.max_deflection()/polar_ratio;
        host_.lowest_visible_chord_km = r.lowest_visible_chord(host_.camera_radius_km);
        if (host_.camera_radius_km > r.bottom_radius_km) {
            Tables* tables = tables_for(cmd, ctx, body.render.atmosphere);
            if (tables) {
                params = refraction_parameters(r, host_.camera_radius_km);
                params.host = glm::vec4(glm::vec3(center), 1.0f);
                params.pole = glm::vec4(glm::vec3(pole), float(1.0/polar_ratio-1.0));
                source = &r;
                if (built_source_[f] != source || !same_observer(built_[f], params)) {
                    write_ubo(ctx, ubos_[f], params);
                    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_SHADER_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.handle);
                    rhi::PushDescriptorWrite writes[] = {
                        rhi::PushDescriptorWrite::uniform_buffer(0, ubos_[f], 0, sizeof(RefractionGPU)),
                        rhi::PushDescriptorWrite::sampled_image(1, tables->travel, ctx.sampler_nearest_clamp()),
                        rhi::PushDescriptorWrite::sampled_image(2, tables->rows, ctx.sampler_nearest_clamp()),
                        rhi::PushDescriptorWrite::sampled_image(3, tables->profile, ctx.sampler_nearest_clamp()),
                        rhi::PushDescriptorWrite::storage_image(4, observers_[f]),
                    };
                    ctx.push_descriptors(cmd, pipeline_, writes);
                    uint32_t timer = ctx.begin_gpu_timer(cmd, "refraction_observer");
                    vkCmdDispatch(cmd, (REFRACTION_OBSERVER_WIDTH+7)/8, (REFRACTION_OBSERVER_HEIGHT+7)/8, 1);
                    ctx.end_gpu_timer(cmd, timer);
                    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_SHADER_READ_BIT);
                    built_[f] = params;
                    built_source_[f] = source;
                    ++observer_builds_;
                }
            }
        }
    }
    if (!source) {
        params = {};
        host_.body = -1;
    }
    write_ubo(ctx, ubos_[f], params);
    retire_tables(ctx, source && host_.body >= 0 ? system.body(size_t(host_.body)).render.atmosphere.get() : nullptr);
}

std::array<rhi::PushDescriptorWrite, 2> RefractionRenderer::writes(rhi::VkContext& ctx) const {
    uint32_t f = ctx.frame_index();
    return {rhi::PushDescriptorWrite::uniform_buffer(20, ubos_[f], 0, sizeof(RefractionGPU)),
            rhi::PushDescriptorWrite::sampled_image(21, observers_[f], ctx.sampler_nearest_clamp(), VK_IMAGE_LAYOUT_GENERAL)};
}

std::array<rhi::PushDescriptorWrite, 3> RefractionRenderer::ray_writes(rhi::VkContext& ctx) const {
    uint32_t f = ctx.frame_index();
    for (const auto& tables : tables_) {
        if (!tables.ready || tables.atmosphere->refraction.get() != built_source_[f] || host_.body < 0) continue;
        return {rhi::PushDescriptorWrite::sampled_image(22, tables.travel, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(23, tables.rows, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(24, tables.profile, ctx.sampler_nearest_clamp())};
    }
    return {rhi::PushDescriptorWrite::sampled_image(22, observers_[f], ctx.sampler_nearest_clamp(), VK_IMAGE_LAYOUT_GENERAL),
            rhi::PushDescriptorWrite::sampled_image(23, observers_[f], ctx.sampler_nearest_clamp(), VK_IMAGE_LAYOUT_GENERAL),
            rhi::PushDescriptorWrite::sampled_image(24, observers_[f], ctx.sampler_nearest_clamp(), VK_IMAGE_LAYOUT_GENERAL)};
}

}
