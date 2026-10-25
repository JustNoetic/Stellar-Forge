#include "rendering/eclipse_refraction_renderer.hpp"
#include "core/constants.hpp"
#include "physics/eclipse_shadow.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "rhi/staging_uploads.hpp"
#include <chrono>
#include <future>

namespace sf::rendering {

struct EclipseRefractionRenderer::Impl {
    using Clock = std::chrono::steady_clock;
    struct Slot {
        std::shared_ptr<const physics::PreparedAtmosphere> atmosphere;
        int bucket = 0;
        uint64_t touched = 0, last_serial = 0;
        bool ready = false;
        float support = 1, alpha = 0, max_bend = 0;
        Clock::time_point visible_at{};
    };
    struct Result { uint32_t slot = 0; physics::EclipseShadowTable table; double milliseconds = 0; };
    rhi::Image atlas;
    rhi::StagingUploads staging;
    std::array<Slot, 16> slots;
    std::future<Result> worker;
    uint32_t worker_slot = 0;
    std::optional<Result> completed;
    uint64_t frame = 0;
    uint32_t build_count = 0;
    double bake_ms = 0;
    std::vector<glm::vec4> records;
    std::vector<glm::vec2> bounds;
};

EclipseRefractionRenderer::EclipseRefractionRenderer() : impl_(std::make_unique<Impl>()) {}
EclipseRefractionRenderer::~EclipseRefractionRenderer() = default;

void EclipseRefractionRenderer::init(rhi::VkContext& ctx) {
    impl_->atlas = ctx.create_image_2d(physics::ECLIPSE_SHADOW_WIDTH, physics::ECLIPSE_SHADOW_HEIGHT, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, 16);
    std::vector<glm::vec4> pixels(physics::ECLIPSE_SHADOW_WIDTH*physics::ECLIPSE_SHADOW_HEIGHT*16, glm::vec4(0));
    std::fill_n(pixels.begin(), physics::ECLIPSE_SHADOW_WIDTH*physics::ECLIPSE_SHADOW_HEIGHT,
        glm::vec4(glm::log2(glm::vec3(.0002f, .00001f, .000001f)), 0));
    ctx.upload_image_2d(impl_->atlas, pixels.data(), pixels.size()*sizeof(glm::vec4));
    impl_->staging.init(ctx, 2ull << 20);
}

void EclipseRefractionRenderer::shutdown(rhi::VkContext& ctx) {
    if (impl_->worker.valid()) impl_->worker.wait();
    impl_->worker = {};
    impl_->staging.shutdown(ctx);
    ctx.destroy_image(impl_->atlas);
    impl_->slots = {};
    impl_->completed.reset();
    impl_->records.clear(); impl_->bounds.clear();
}

void EclipseRefractionRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& system,
                                       const RenderSettings& settings) {
    auto& state = *impl_;
    ++state.frame;
    state.records.clear();
    state.bounds.assign(std::min(system.size(), size_t(1024)), glm::vec2(0));
    if (state.worker.valid() && state.worker.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try { state.completed = state.worker.get(); }
        catch (const std::exception& error) {
            state.slots[state.worker_slot] = {};
            log::warn("Eclipse shadow bake failed: {}", error.what());
        }
    }
    if (!settings.eclipse_refraction_enabled || !settings.show_atmospheres) return;
    if (state.completed) {
        auto& result = *state.completed;
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, result.slot, 1};
        ctx.transition_image_range(cmd, state.atlas, range, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, result.slot, 1};
        region.imageExtent = state.atlas.extent;
        bool uploaded = state.staging.image(cmd, ctx, state.atlas, region, std::as_bytes(std::span(result.table.pixels)));
        ctx.transition_image_range(cmd, state.atlas, range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        if (uploaded) {
            auto& slot = state.slots[result.slot];
            bool previously_visible = std::any_of(state.slots.begin()+1, state.slots.end(), [&](const auto& other) {
                return other.ready && other.atmosphere == slot.atmosphere;
            });
            slot.ready = true; slot.support = result.table.support;
            slot.max_bend = result.table.max_bend;
            slot.visible_at = Impl::Clock::now()-(previously_visible ? std::chrono::seconds(1) : std::chrono::seconds(0));
            ++state.build_count; state.bake_ms = result.milliseconds;
            state.completed.reset();
        }
    }
    std::vector<glm::dvec4> stars;
    for (size_t i = 0; i < state.bounds.size() && stars.size() < 16; ++i)
        if (system.body(i).definition.is_star)
            stars.emplace_back(system.body(i).render.position.value, system.body(i).opaque_radius_au());
    if (stars.empty()) stars.emplace_back(0, 0, 0, SOLAR_RADII_TO_AU);
    state.records.assign(state.bounds.size()*stars.size(), glm::vec4(0));
    bool any = false;
    for (size_t i = 0; i < state.bounds.size(); ++i) {
        const auto& body = system.body(i);
        if (!body.render.atmosphere || !body.render.atmosphere->refraction || !body.definition.atmosphere.enabled || body.definition.is_star) continue;
        auto atmosphere = body.render.atmosphere;
        double top = atmosphere->refraction->top_radius_km*KM_TO_AU;
        for (size_t s = 0; s < stars.size(); ++s) {
            double alpha = stars[s].w/glm::length(glm::dvec3(stars[s])-body.render.position.value);
            if (!(alpha >= 1e-6 && alpha <= .25)) continue;
            if (settings.eclipse_refraction_probe) {
                state.records[i*stars.size()+s] = {1, float(alpha), 4, 1};
                state.bounds[i] = {float(top), .12f}; any = true;
                continue;
            }
            bool eligible = false;
            for (size_t j = 0; j < state.bounds.size() && !eligible; ++j) {
                if (j == i || system.body(j).definition.is_star) continue;
                auto receiver = system.body(j);
                double radius = receiver.opaque_radius_au();
                if (settings.terrain_enabled)
                    radius += std::max({0.0, double(settings.terrain_height_km), receiver.definition.height_range_km.y})*KM_TO_AU;
                if (settings.show_rings) for (const auto& ring : receiver.definition.rings)
                    radius = std::max(radius, receiver.definition.radius_au()*ring.outer);
                glm::dvec3 delta = receiver.render.position.value-body.render.position.value;
                glm::dvec3 sun = glm::dvec3(stars[s])-body.render.position.value;
                eligible = eclipse_could_shadow({glm::vec3(delta), float(radius)}, {glm::vec3(sun), float(stars[s].w)},
                    {glm::vec3(0), float(top+.12*(glm::length(delta)+radius))});
            }
            if (!eligible) continue;
            int bucket = int(std::round(std::log(alpha)/std::log(1.025)));
            int exact = -1, closest = -1;
            for (int k = 1; k < int(state.slots.size()); ++k) {
                auto& slot = state.slots[size_t(k)];
                if (slot.atmosphere != atmosphere) continue;
                if (slot.bucket == bucket) { exact = k; slot.touched = state.frame; }
                if (slot.ready && std::abs(slot.bucket-bucket) <= 2 &&
                    (closest < 0 || std::abs(slot.bucket-bucket) < std::abs(state.slots[size_t(closest)].bucket-bucket))) closest = k;
            }
            if (closest >= 0) state.slots[size_t(closest)].touched = state.frame;
            if (exact < 0 && !state.worker.valid() && !state.completed) {
                int replace = -1;
                for (int k = 1; k < int(state.slots.size()); ++k) {
                    const auto& slot = state.slots[size_t(k)];
                    if (slot.last_serial > ctx.completed_serial() || slot.touched == state.frame) continue;
                    if (replace < 0 || !slot.atmosphere || slot.touched < state.slots[size_t(replace)].touched) replace = k;
                    if (!slot.atmosphere) break;
                }
                if (replace >= 0) {
                    auto& slot = state.slots[size_t(replace)];
                    slot = {}; slot.atmosphere = atmosphere; slot.bucket = bucket; slot.touched = state.frame;
                    slot.alpha = float(std::pow(1.025, bucket));
                    state.worker_slot = uint32_t(replace);
                    state.worker = std::async(std::launch::async, [atmosphere, angle = double(slot.alpha), replace] {
                        auto start = Impl::Clock::now();
                        auto table = physics::bake_eclipse_shadow(physics::prepare_eclipse_profile(*atmosphere), angle);
                        double elapsed = std::chrono::duration<double, std::milli>(Impl::Clock::now()-start).count();
                        return Impl::Result{uint32_t(replace), std::move(table), elapsed};
                    });
                }
            }
            if (closest >= 0) {
                auto& slot = state.slots[size_t(closest)];
                slot.touched = state.frame; slot.last_serial = ctx.recording_serial();
                float fade = std::clamp(std::chrono::duration<float>(Impl::Clock::now()-slot.visible_at).count()/.15f, 0.0f, 1.0f);
                state.records[i*stars.size()+s] = {float(closest+1), slot.alpha, slot.support, fade};
                state.bounds[i] = {float(top), std::max(state.bounds[i].y, slot.max_bend*1.01f)}; any = true;
            }
        }
    }
    if (!any) state.records.clear();
}

rhi::PushDescriptorWrite EclipseRefractionRenderer::write(rhi::VkContext& ctx) const {
    return rhi::PushDescriptorWrite::sampled_image(26, impl_->atlas, ctx.sampler_linear_clamp());
}
std::span<const glm::vec4> EclipseRefractionRenderer::records() const { return impl_->records; }
std::span<const glm::vec2> EclipseRefractionRenderer::bounds() const { return impl_->bounds; }
uint32_t EclipseRefractionRenderer::builds() const { return impl_->build_count; }
double EclipseRefractionRenderer::last_bake_ms() const { return impl_->bake_ms; }

}
