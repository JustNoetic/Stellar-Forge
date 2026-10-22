#include "core/path_utils.hpp"
#include "rendering/atmosphere_renderer.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "rendering/ring_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace sf::rendering {

static_assert(sizeof(AtmosphereParametersGPU) == 13328);
static_assert(sizeof(AtmosphereParametersGPU) <= 16384);
static_assert(sizeof(AtmosphereShadowHeaderGPU) == 560);
static_assert(sizeof(AtmosphereShadowCasterGPU) == 48);

static bool prepare_shadows(rhi::VkContext& ctx, rhi::Buffer& buffer, const ephemeris::SystemSnapshot& system,
                            uint32_t receiver, glm::dvec3 eye, float outer, float focal,
                            const RenderSettings& settings, std::span<const RingHostGPU> rings,
                            AtmosphereShadowHeaderGPU& previous, std::vector<AtmosphereShadowCasterGPU>& previous_casters,
                            uint32_t sequence, uint32_t& changes) {
    AtmosphereShadowHeaderGPU header;
    const auto origin = system.body(receiver).render.position.value;
    header.eye_focal = {glm::vec3((eye-origin)*AU_TO_KM), focal};
    header.flags = {settings.atmosphere_shadows_enabled ? 1u : 0u,
                    (settings.ring_shadow_single_box ? 1u : 0u) | (settings.ring_shadow_reference_linear ? 2u : 0u),
                    settings.eclipse_reference_smoothstep ? 1u : 0u, settings.atmosphere_shadow_reference ? 1u : 0u};
    uint32_t effective_sequence = !settings.atmosphere_shadow_reference &&
        (settings.atmosphere_shadow_temporal || settings.atmosphere_shadow_animate_jitter) ? sequence : 0u;
    header.temporal = {effective_sequence, settings.atmosphere_shadow_noise_type,
                       settings.atmosphere_shadow_samples, settings.atmosphere_shadow_adaptive ? 1u : 0u};
    std::vector<AtmosphereShadowCasterGPU> casters;
    uint32_t star_index = 0;
    for (uint32_t s = 0; s < system.size() && star_index < header.stars.size(); ++s) {
        const auto star = system.body(s);
        if (!star.definition.is_star || glm::length(star.render.position.value-origin) < 1e-12) continue;
        auto& light = header.stars[star_index];
        light = {glm::vec3((star.render.position.value-origin)*AU_TO_KM), float(star.opaque_radius_au()*AU_TO_KM)};
        auto& range = header.ranges[star_index++];
        if (!settings.atmosphere_shadows_enabled) continue;
        range.x = uint32_t(casters.size());
        for (uint32_t j = 0; j < system.size(); ++j) {
            const auto caster = system.body(j);
            if (j == receiver || caster.definition.is_star || caster.opaque_radius_au() <= 0.0) continue;
            glm::vec4 position(glm::vec3((caster.render.position.value-origin)*AU_TO_KM),
                               float(caster.opaque_radius_au()*AU_TO_KM));
            if (!eclipse_could_shadow({0, 0, 0, outer}, light, position)) continue;
            casters.push_back({position, {glm::vec3(caster.render.pole),
                position.w*float(std::clamp(1.0-caster.definition.oblateness, 0.05, 1.0))}, {}});
        }
        range.y = uint32_t(casters.size())-range.x;
        range.z = uint32_t(casters.size());
        if (settings.show_rings && settings.ring_shadows_enabled) {
            for (const auto& ring : rings) {
                uint32_t host = ring.counts_indices.x;
                if (host >= system.size() || ring.pole_max_r.w <= ring.center_min_r.w) continue;
                glm::vec3 center((system.body(host).render.position.value-origin)*AU_TO_KM);
                glm::vec3 pole(ring.pole_max_r);
                float inner = float(ring.center_min_r.w*AU_TO_KM), radius = float(ring.pole_max_r.w*AU_TO_KM);
                glm::vec4 receiver_eye(glm::vec3(origin-eye), outer*float(KM_TO_AU));
                glm::vec4 light_eye(glm::vec3(star.render.position.value-eye), light.w*float(KM_TO_AU));
                if (!ring_could_shadow_surface(receiver_eye, light_eye,
                    {glm::vec3(ring.center_min_r), ring.pole_max_r.w}, {pole, ring.center_min_r.w}, focal, settings.ring_shadow_softness)) continue;
                double light_distance = glm::length(glm::dvec3(light));
                double denominator = std::abs(glm::dot(glm::dvec3(light), glm::dvec3(pole)))-outer-light.w;
                float support = -1.0f;
                if (light_distance > outer && denominator > 1e-6*(light_distance+outer)) {
                    double minimum_mu = denominator/(light_distance+outer+light.w);
                    double hit_distance = (std::abs(glm::dot(glm::dvec3(center), glm::dvec3(pole)))+outer)/minimum_mu;
                    double stretch = 1.0/minimum_mu;
                    double stellar_blur = 2.0*std::max(settings.ring_shadow_softness, 1.0f)*light.w/(light_distance-outer)*hit_distance*stretch;
                    double pixel_blur = 0.75*(glm::length(glm::dvec3(header.eye_focal))+outer)/focal*stretch;
                    double profile_blur = 0.75*(radius-inner)/4096.0;
                    support = float(radius+std::max({stellar_blur, pixel_blur, profile_blur})+0.01);
                }
                casters.push_back({{center, radius}, {pole, inner}, {support, float(ring.counts_indices.y), settings.ring_shadow_softness, 0}});
            }
        }
        range.w = uint32_t(casters.size())-range.z;
    }
    VkDeviceSize bytes = sizeof(header)+casters.size()*sizeof(AtmosphereShadowCasterGPU);
    if (buffer.size < bytes) {
        VkDeviceSize capacity = std::max(bytes, buffer.size*2);
        ctx.destroy_buffer(buffer);
        buffer = ctx.create_host_buffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    std::memcpy(buffer.mapped, &header, sizeof(header));
    if (!casters.empty()) std::memcpy(static_cast<std::byte*>(buffer.mapped)+sizeof(header), casters.data(), bytes-sizeof(header));
    ctx.flush_buffer(buffer, 0, bytes);
    changes = 0;
    if (glm::vec3(previous.eye_focal) != glm::vec3(header.eye_focal)) changes |= ShadowHistoryCamera;
    if (previous.eye_focal.w != header.eye_focal.w) changes |= ShadowHistoryProjection;
    if (previous.flags != header.flags || glm::uvec3(previous.temporal.y, previous.temporal.z, previous.temporal.w) !=
        glm::uvec3(header.temporal.y, header.temporal.z, header.temporal.w)) changes |= ShadowHistorySampling;
    if (previous.stars != header.stars) changes |= ShadowHistoryLighting;
    if (previous.ranges != header.ranges || previous_casters.size() != casters.size() || (!casters.empty() &&
        std::memcmp(previous_casters.data(), casters.data(), casters.size()*sizeof(AtmosphereShadowCasterGPU)) != 0))
        changes |= ShadowHistoryCasters;
    auto previous_key = previous;
    previous_key.temporal.x = header.temporal.x;
    if (changes == 0 && std::memcmp(&previous_key, &header, sizeof(header)) != 0) changes |= ShadowHistorySampling;
    previous = header;
    previous_casters = std::move(casters);
    return !previous_casters.empty();
}

void AtmosphereRenderer::init(rhi::VkContext& ctx) {
    constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
    const rhi::DescriptorLayoutBinding trans[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding multi[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding sky[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding composite[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}};
    transmittance_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/transmittance.comp", trans);
    multiscatter_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/multiscatter.comp", multi);
    sky_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/sky_view.comp", sky);
    std::vector<rhi::DescriptorLayoutBinding> aerial(std::begin(sky), std::end(sky));
    aerial.push_back({5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    aerial_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/aerial_perspective.comp", aerial);
    composite_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/composite.comp", composite);
    const rhi::DescriptorLayoutBinding shadow[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}, {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}};
    shadow_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow.comp", shadow);
    std::vector<rhi::DescriptorLayoutBinding> shadow_transparency(std::begin(shadow), std::end(shadow));
    shadow_transparency.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    shadow_transparency.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_transparency.comp", shadow_transparency);
    auto accumulate = std::vector<rhi::DescriptorLayoutBinding>(std::begin(shadow), std::end(shadow));
    accumulate.push_back({23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    accumulate.push_back({24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    accumulate.push_back({25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_accumulate_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_accumulate.comp", accumulate, sizeof(glm::uvec2));
    auto rotate = accumulate;
    rotate.push_back({26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    rotate.push_back({27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    shadow_rotate_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_rotate.comp", rotate, sizeof(glm::uvec2));
    rotate.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    rotate.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_rotate_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_rotate_transparency.comp", rotate, sizeof(glm::uvec2));
    const rhi::DescriptorLayoutBinding reproject[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}, {23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    shadow_reproject_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_reproject.comp", reproject, 2*sizeof(glm::mat4));
    accumulate.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    accumulate.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_accumulate_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_accumulate_transparency.comp", accumulate, sizeof(glm::uvec2));
    shadow_coarse_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_coarse.comp", shadow);
    std::vector<rhi::DescriptorLayoutBinding> shadow_vrs(std::begin(shadow), std::end(shadow));
    shadow_vrs.push_back({21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage});
    shadow_vrs.push_back({22, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage});
    shadow_vrs_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_vrs.comp", shadow_vrs);
    shadow_vrs.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    shadow_vrs.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_vrs_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_vrs_transparency.comp", shadow_vrs);
    std::vector<rhi::DescriptorLayoutBinding> transparent_bindings(std::begin(composite), std::end(composite));
    transparent_bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    transparent_bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    composite_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/composite_transparency.comp", transparent_bindings);
    std::erase_if(transparent_bindings, [](auto binding) { return binding.binding == 4; });
    transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", transparent_bindings);
    const rhi::DescriptorLayoutBinding surface[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    surface_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/surface_irradiance.comp", surface, sizeof(uint32_t));
    surface_lut_ = ctx.create_image_2d(128, 32, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 1, 64);
    ctx.immediate_submit([&](VkCommandBuffer cmd) { ctx.transition_image(cmd, surface_lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); });
    auto stbn_data = paths::read_binary_file(paths::external_path("textures", "noise", "stbn_scalar.bin"));
    if (stbn_data && stbn_data->size() == 128 * 128 * 64) {
        stbn_tex_ = ctx.create_image_3d(128, 128, 64, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ctx.upload_image_2d(stbn_tex_, stbn_data->data(), stbn_data->size(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (!stbn_tex_) {
        uint8_t dummy_val = 128;
        dummy_stbn_ = ctx.create_image_3d(1, 1, 1, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ctx.upload_image_2d(dummy_stbn_, &dummy_val, 1,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    for (auto& buffer : surface_records_) buffer = ctx.create_host_buffer(1024*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (auto& frame : shadow_diagnostics_) frame.counters = ctx.create_host_buffer(8*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (ctx.dual_source_blend_enabled()) {
        rhi::GraphicsPipelineDesc desc;
        desc.vert_path = "post/fullscreen.vert";
        desc.frag_path = "atmosphere/composite.frag";
        for (auto binding : composite) {
            if (binding.binding == 4) continue;
            binding.stages = VK_SHADER_STAGE_FRAGMENT_BIT;
            desc.bindings.push_back(binding);
        }
        desc.depth_format = VK_FORMAT_UNDEFINED;
        desc.depth_test = desc.depth_write = false;
        desc.cull_mode = VK_CULL_MODE_NONE;
        desc.dual_source_blend = true;
        dual_source_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    }
}

void AtmosphereRenderer::shutdown(rhi::VkContext& ctx) {
    for (auto& t : tables_) {
        ctx.destroy_image(t->transmittance);
        ctx.destroy_image(t->multiscatter);
        ctx.destroy_image(t->sky);
        ctx.destroy_image(t->throughput);
        ctx.destroy_image(t->aerial_radiance);
        ctx.destroy_image(t->aerial_optical_depth);
        ctx.destroy_image(t->aerial_range);
        for (auto& u : t->uniforms) ctx.destroy_buffer(u);
        for (auto& shadow : t->shadows) ctx.destroy_buffer(shadow);
        ctx.destroy_image(t->shadow_fraction);
        ctx.destroy_image(t->shadow_depth);
        ctx.destroy_image(t->shadow_energy);
        ctx.destroy_image(t->shadow_loss);
    }
    tables_.clear();
    active_.clear();
    surface_layers_.fill(nullptr);
    rhi::ShaderCache::destroy_pipeline(ctx, transmittance_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, multiscatter_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, sky_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, aerial_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, composite_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, transparent_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, composite_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, dual_source_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, surface_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_accumulate_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_accumulate_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_reproject_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_rotate_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_rotate_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_coarse_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_vrs_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_vrs_transparency_pipeline_);
    ctx.destroy_image(shadow_coarse_fraction_);
    ctx.destroy_image(shadow_coarse_guide_);
    ctx.destroy_image(shadow_reproject_energy_);
    ctx.destroy_image(shadow_reproject_loss_);
    for (auto& retired : retired_shadows_) ctx.destroy_image(retired.image);
    retired_shadows_.clear();
    ctx.destroy_image(surface_lut_);
    if (stbn_tex_) ctx.destroy_image(stbn_tex_);
    if (dummy_stbn_) ctx.destroy_image(dummy_stbn_);
    for (auto& buffer : surface_records_) ctx.destroy_buffer(buffer);
    for (auto& frame : shadow_diagnostics_) ctx.destroy_buffer(frame.counters);
}

ShadowHistoryDiagnostics AtmosphereRenderer::shadow_history_diagnostics(rhi::VkContext& ctx) const {
    if (!shadow_diagnostics_enabled_ || !shadow_temporal_enabled_ || stats_.shadow_updates == 0) return {};
    auto result = shadow_last_diagnostics_;
    const ShadowDiagnosticFrame* latest = nullptr;
    for (const auto& frame : shadow_diagnostics_)
        if (frame.serial > result.serial && frame.serial <= ctx.completed_serial() && (!latest || frame.serial > latest->serial))
            latest = &frame;
    if (!latest) return result;
    VK_CHECK(vmaInvalidateAllocation(ctx.allocator(), latest->counters.allocation, 0, VK_WHOLE_SIZE));
    std::array<uint32_t, 8> values;
    std::memcpy(values.data(), latest->counters.mapped, sizeof(values));
    result.columns = values[0];
    result.serial = latest->serial;
    result.depth_resets = values[1];
    result.refining = values[2];
    result.complete = values[3];
    result.minimum_frames = values[0] ? values[4] : 0;
    result.maximum_frames = values[5];
    result.reprojected = values[6];
    result.reprojection_rejected = values[7];
    result.reset_reasons = latest->reset_reasons;
    result.valid = true;
    return result;
}

void AtmosphereRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings, VkExtent2D extent,
                                 std::span<const float> minimum_heights_km, std::span<const RingHostGPU> rings,
                                 const rhi::Image* ring_shadow_lut, uint64_t ring_shadow_revision, bool shadow_diagnostics) {
    active_.clear();
    ++prepare_serial_;
    std::erase_if(retired_shadows_, [&](auto& entry) {
        if (entry.serial > ctx.completed_serial()) return false;
        ctx.destroy_image(entry.image);
        return true;
    });
    ring_shadow_lut_ = ring_shadow_lut ? ring_shadow_lut : &surface_lut_;
    shadow_last_diagnostics_ = shadow_diagnostics ? shadow_history_diagnostics(ctx) : ShadowHistoryDiagnostics{};
    stats_ = {};
    shadow_diagnostics_enabled_ = shadow_diagnostics;
    auto& diagnostics = shadow_diagnostics_[ctx.frame_index()];
    diagnostics.serial = 0;
    diagnostics.reset_reasons = 0;
    if (shadow_diagnostics) {
        const std::array<uint32_t, 8> values{0, 0, 0, 0, 64, 0, 0, 0};
        std::memcpy(diagnostics.counters.mapped, values.data(), sizeof(values));
        ctx.flush_buffer(diagnostics.counters);
    }
    shadow_vrs_enabled_ = settings.atmosphere_shadow_vrs;
    shadow_temporal_enabled_ = settings.atmosphere_shadow_temporal && !settings.atmosphere_shadow_reference;
    bool rotation_extent_changed = shadow_reproject_energy_ &&
        (shadow_reproject_energy_.extent.width != extent.width || shadow_reproject_energy_.extent.height != extent.height);
    if (!shadow_temporal_enabled_ || !settings.atmosphere_shadow_rotation || !settings.show_atmospheres || system.empty() || rotation_extent_changed) {
        for (auto* image : {&shadow_reproject_energy_, &shadow_reproject_loss_}) {
            if (*image) retired_shadows_.push_back({*image, ctx.recording_serial()});
            *image = {};
        }
    }
    dual_source_ = settings.atmosphere_dual_source && bool(dual_source_pipeline_);
    auto& records = surface_records_[ctx.frame_index()];
    std::memset(records.mapped, 0, 1024*sizeof(glm::vec4));
    ctx.flush_buffer(records);
    if (!settings.show_atmospheres || system.empty()) {
        for (auto& table : tables_) {
            for (auto* image : {&table->shadow_fraction, &table->shadow_depth, &table->shadow_energy, &table->shadow_loss}) {
                if (*image) retired_shadows_.push_back({*image, ctx.recording_serial()});
                *image = {};
            }
            table->shadow_history_valid = false;
        }
        return;
    }
    struct Candidate { uint32_t index; double distance; };
    std::vector<Candidate> candidates;
    double tan_y = std::tan(camera.fov_deg*DEG_TO_RAD*0.5);
    double tan_x = tan_y*double(extent.width)/std::max(1u, extent.height);
    double focal = double(extent.height)/(2.0*tan_y);
    glm::dvec3 eye = camera.eye_pos_render().value;
    for (uint32_t i = 0; i < system.size(); ++i) {
        const auto b = system.body(i);
        const auto& a = b.definition.atmosphere;
        if (!a.enabled || b.definition.is_star || b.definition.is_black_hole || b.definition.is_spacecraft) continue;
        double radius = b.definition.radius_au()*AU_TO_KM;
        glm::dvec3 to_body = (b.render.position.value-eye)*AU_TO_KM;
        double distance = glm::length(to_body);
        double height = b.render.atmosphere ? b.render.atmosphere->height_km : a.height_km;
        double outer = radius+height;
        if (distance > outer) {
            double depth = glm::dot(to_body, camera.forward_dir());
            if (depth+outer <= 0.0 ||
                std::abs(glm::dot(to_body, camera.right_dir())) > depth*tan_x+outer*std::sqrt(1.0+tan_x*tan_x) ||
                std::abs(glm::dot(to_body, camera.up_dir())) > depth*tan_y+outer*std::sqrt(1.0+tan_y*tan_y)) continue;
        }
        if (outer/std::max(distance, 1e-12)*focal < 0.5) continue;
        candidates.push_back({i, distance});
    }
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.distance > b.distance; });
    constexpr size_t retained_budget = 256 * 1024 * 1024;
    constexpr size_t table_bytes = 256*64*16 + 32*32*8 + 2*192*108*8 +
        2*192*108*64*8 + 192*108*4 + sizeof(AtmosphereParametersGPU)*rhi::FRAMES_IN_FLIGHT;
    const size_t capacity = std::max(candidates.size(), retained_budget/table_bytes);
    const auto definition = system.share_definition();
    std::vector<size_t> slots(candidates.size(), SIZE_MAX);
    std::vector<bool> reserved(tables_.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = 0; j < tables_.size(); ++j) {
            const auto& t = *tables_[j];
            if (t.valid && t.definition == definition && t.generation == system.generation() &&
                t.body_index == candidates[i].index) {
                slots[i] = j;
                reserved[j] = true;
                break;
            }
        }
    }
    for (auto& slot : slots) {
        if (slot != SIZE_MAX) continue;
        size_t available = SIZE_MAX;
        for (size_t j = 0; j < tables_.size(); ++j) {
            if (reserved[j]) continue;
            const auto& t = *tables_[j];
            bool stale = t.definition != definition || t.generation != system.generation();
            if (stale) { available = j; break; }
            if (available == SIZE_MAX || t.last_used < tables_[available]->last_used) available = j;
        }
        bool stale = available != SIZE_MAX && (tables_[available]->definition != definition ||
            tables_[available]->generation != system.generation());
        if (!stale && reserved.size() < capacity) {
            slot = reserved.size();
            reserved.push_back(true);
        } else {
            slot = available;
            reserved[slot] = true;
        }
    }
    std::vector<size_t> surface_slots(candidates.size(), SIZE_MAX);
    std::array<bool, 32> surface_reserved{};
    for (size_t i = 0; i < candidates.size() && i < surface_layers_.size(); ++i) {
        if (slots[i] >= tables_.size()) continue;
        for (size_t j = 0; j < surface_layers_.size(); ++j) {
            if (surface_layers_[j] == tables_[slots[i]].get()) {
                surface_slots[i] = j;
                surface_reserved[j] = true;
                break;
            }
        }
    }
    for (size_t i = 0; i < candidates.size() && i < surface_layers_.size(); ++i) {
        if (surface_slots[i] != SIZE_MAX) continue;
        auto layer = std::find(surface_reserved.begin(), surface_reserved.end(), false);
        surface_slots[i] = size_t(layer-surface_reserved.begin());
        *layer = true;
    }
    auto image = [&](uint32_t w, uint32_t h, VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT) {
        return ctx.create_image_2d(w, h, format,
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    };
    auto bake = [&](const rhi::Pipeline& pipeline, std::span<const rhi::PushDescriptorWrite> writes, uint32_t x, uint32_t y, uint32_t z = 1) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        ctx.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, x, y, z);
    };
    auto writable = [&](rhi::Image& img) { ctx.transition_image(cmd, img, VK_IMAGE_LAYOUT_GENERAL); };
    auto readable = [&](rhi::Image& img) {
        ctx.transition_image(cmd, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    };
    uint32_t timer = candidates.empty() ? UINT32_MAX : ctx.begin_gpu_timer(cmd, "atmosphere_luts");
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto candidate = candidates[i];
        size_t slot = slots[i];
        if (slot == tables_.size()) {
            auto t = std::make_unique<Tables>();
            t->transmittance = image(256, 64, VK_FORMAT_R32G32B32A32_SFLOAT);
            t->multiscatter = image(32, 32);
            t->sky = image(192, 108);
            t->throughput = image(192, 108);
            t->aerial_radiance = ctx.create_image_3d(192, 108, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
            t->aerial_optical_depth = ctx.create_image_3d(192, 108, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
            t->aerial_range = image(192, 108, VK_FORMAT_R32_SFLOAT);
            readable(t->aerial_radiance);
            readable(t->aerial_optical_depth);
            readable(t->aerial_range);
            for (auto& u : t->uniforms) u = ctx.create_host_buffer(sizeof(AtmosphereParametersGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            for (auto& shadow : t->shadows) shadow = ctx.create_host_buffer(sizeof(AtmosphereShadowHeaderGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            tables_.push_back(std::move(t));
        }
        auto& t = *tables_[slot];
        const auto b = system.body(candidate.index);
        const auto& a = b.definition.atmosphere;
        float height = b.render.atmosphere ? b.render.atmosphere->height_km : a.height_km;
        double polar_ratio = std::clamp(1.0-b.definition.oblateness, 0.05, 1.0);
        glm::dvec3 pole = glm::normalize(b.render.pole);
        auto atmosphere_space = [&](glm::dvec3 v) { return v+pole*glm::dot(v, pole)*(1.0/polar_ratio-1.0); };
        glm::dvec3 relative = atmosphere_space((eye-b.render.position.value)*AU_TO_KM);
        double distance = glm::length(relative);
        glm::dvec3 z = distance > 1e-12 ? relative/distance : glm::dvec3(0.0, 1.0, 0.0);
        glm::dvec3 x = glm::normalize(glm::cross(std::abs(z.y) < 0.99 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), z));
        glm::dvec3 y = glm::cross(z, x);
        float radius = static_cast<float>(b.definition.radius_au()*AU_TO_KM);
        AtmosphereParametersGPU p{};
        double minimum = std::min(0.0, b.definition.height_range_km.x);
        if (candidate.index < minimum_heights_km.size()) minimum = std::min(minimum, double(minimum_heights_km[candidate.index]));
        float bottom = std::max(radius*0.01f, radius+static_cast<float>(minimum/polar_ratio));
        if (a.pressure_reference) bottom = float(b.opaque_radius_au()*AU_TO_KM);
        p.radii_heights = {bottom, radius+height, a.rayleigh_scale_height_km, a.mie_scale_height_km};
        p.reference_shape = {radius, static_cast<float>(polar_ratio), a.pressure_reference ? bottom : radius, 0.0f};
        p.world_pole = {glm::vec3(pole), static_cast<float>(1.0/polar_ratio-1.0)};
        p.rayleigh = {a.rayleigh_scattering, 0.0f};
        p.mie_scattering = {a.mie_scattering, a.mie_asymmetry};
        p.mie_extinction = {a.mie_extinction, 0.0f};
        p.ground_albedo = {a.ground_albedo, 0.0f};
        if (const auto& physical = b.render.atmosphere) {
            p.radii_heights.z = float(physical->surface.scale_height_km);
            p.radii_heights.w = physical->mie_scale_height_km;
            p.mie_scattering = {physical->mie_scattering, physical->mie_asymmetry};
            p.mie_extinction = {physical->mie_extinction, 0.0f};
            p.rayleigh = {physical->rayleigh, 0.0f};
            p.mixed_absorption = {physical->mixed_absorption, 0.0f};
            p.ozone_absorption = {physical->ozone_absorption, 0.0f};
            p.ozone_params = {float(a.ozone_peak_km), float(a.ozone_width_km), 0.0f, 0.0f};
            p.profile_domain = physical->profile_domain;
            p.log_density = physical->log_density;
            p.aerosol_count.x = float(physical->aerosols.size());
            for (size_t i = 0; i < physical->aerosols.size(); ++i) {
                const auto& layer = physical->aerosols[i];
                p.aerosol_shape[i] = layer.shape;
                p.aerosol_scattering[i] = layer.scattering;
                p.aerosol_extinction[i] = layer.extinction;
                std::copy(layer.phase.begin(), layer.phase.end(), p.aerosol_phase.begin()+i*physics::AEROSOL_PHASE_SAMPLES);
            }
            if (a.pressure_pa == 0) p.mie_scattering = p.mie_extinction = glm::vec4(0.0f);
        }
        p.camera = {0.0f, 0.0f, std::max(static_cast<float>(distance), bottom+0.001f), 0.0f};
        p.axis_x = {glm::vec3(x), 0.0f};
        p.axis_y = {glm::vec3(y), 0.0f};
        p.axis_z = {glm::vec3(z), 0.0f};
        for (uint32_t s = 0; s < system.size() && p.camera.w < 16.0f; ++s) {
            const auto star = system.body(s);
            if (!star.definition.is_star) continue;
            glm::dvec3 delta = star.render.position.value-b.render.position.value;
            double d = glm::length(delta);
            if (d < 1e-12) continue;
            glm::dvec3 direction = glm::normalize(atmosphere_space(delta/d));
            uint32_t index = static_cast<uint32_t>(p.camera.w++);
            p.sun_direction[index] = {glm::vec3(glm::dot(direction, x), glm::dot(direction, y), glm::dot(direction, z)),
                float(std::asin(std::clamp(star.opaque_radius_au()/d, 1e-7, 1.0)))};
            float irradiance = settings.hdr_enabled ? static_cast<float>(std::max(star.definition.lum_eq, 0.0)/(d*d)) : 1.0f;
            p.sun_irradiance[index] = {star.definition.color_rgb*(irradiance*static_cast<float>(PI)), 0.0f};
        }
        auto projection = camera.proj_matrix(float(extent.width)/std::max(1u, extent.height), camera.compute_near_plane_au());
        auto view = camera.view_matrix_eye_relative();
        p.inverse_view_projection = glm::inverse(projection*view);
        p.depth_params.x = settings.hdr_enabled ? settings.exposure : 1.0f;
        p.depth_params.y = settings.aerial_perspective_enabled ? 1.0f : 0.0f;
        uint32_t shadow_changes = 0;
        t.shadow_active = prepare_shadows(ctx, t.shadows[ctx.frame_index()], system, candidate.index, eye, radius+height,
                        float(focal), settings, ring_shadow_lut ? rings : std::span<const RingHostGPU>{},
                        t.shadow_header, t.shadow_casters, uint32_t(prepare_serial_), shadow_changes);
        auto& uniform = t.uniforms[ctx.frame_index()];
        std::memcpy(uniform.mapped, &p, sizeof(p));
        ctx.flush_buffer(uniform);
        bool identity_dirty = t.definition != system.share_definition() || t.generation != system.generation() ||
            t.body_index != candidate.index || t.geometry_revision != b.revisions.geometry || t.lighting_revision != b.revisions.lighting;
        t.shadow_reset_reasons = shadow_changes;
        if (!t.shadow_history_valid) t.shadow_reset_reasons |= ShadowHistoryInitial;
        if (identity_dirty) t.shadow_reset_reasons |= ShadowHistoryDefinition;
        if (t.shadow_recorded_prepare+1 != prepare_serial_) t.shadow_reset_reasons |= ShadowHistoryInterrupted;
        if (view != t.shadow_view || glm::vec3(p.camera) != glm::vec3(t.cached.camera) ||
            p.axis_x != t.cached.axis_x || p.axis_y != t.cached.axis_y || p.axis_z != t.cached.axis_z)
            t.shadow_reset_reasons |= ShadowHistoryCamera;
        if (projection != t.shadow_projection) t.shadow_reset_reasons |= ShadowHistoryProjection;
        if (p.camera.w != t.cached.camera.w || p.sun_direction != t.cached.sun_direction || p.sun_irradiance != t.cached.sun_irradiance)
            t.shadow_reset_reasons |= ShadowHistoryLighting;
        if (p.depth_params != t.cached.depth_params) t.shadow_reset_reasons |= ShadowHistoryTransport;
        auto atmosphere_key = t.cached;
        atmosphere_key.camera = p.camera;
        atmosphere_key.axis_x = p.axis_x; atmosphere_key.axis_y = p.axis_y; atmosphere_key.axis_z = p.axis_z;
        atmosphere_key.sun_direction = p.sun_direction;
        atmosphere_key.sun_irradiance = p.sun_irradiance;
        atmosphere_key.inverse_view_projection = p.inverse_view_projection;
        atmosphere_key.depth_params = p.depth_params;
        if (std::memcmp(&p, &atmosphere_key, sizeof(p)) != 0) t.shadow_reset_reasons |= ShadowHistoryAtmosphere;
        if (t.shadow_ring_revision != ring_shadow_revision || t.shadow_ring_image != ring_shadow_lut_->handle)
            t.shadow_reset_reasons |= ShadowHistoryRings;
        if (t.shadow_reset_reasons == 0 && std::memcmp(&p, &t.cached, sizeof(p)) != 0)
            t.shadow_reset_reasons |= ShadowHistoryTransport;
        bool rotation_mode_unchanged = t.shadow_rotation_enabled == settings.atmosphere_shadow_rotation;
        if (!rotation_mode_unchanged) t.shadow_reset_reasons |= ShadowHistorySampling;
        auto relative_eye = eye-b.render.position.value;
        if (relative_eye != t.shadow_eye_relative) t.shadow_reset_reasons |= ShadowHistoryCamera;
        t.shadow_history_reusable = shadow_temporal_enabled_ && t.shadow_history_valid && !identity_dirty &&
            t.shadow_recorded_prepare+1 == prepare_serial_ && shadow_changes == 0 &&
            std::memcmp(&p, &t.cached, sizeof(p)) == 0 &&
            t.shadow_ring_revision == ring_shadow_revision && t.shadow_ring_image == ring_shadow_lut_->handle && rotation_mode_unchanged &&
            relative_eye == t.shadow_eye_relative;
        auto rotation_key = t.cached;
        rotation_key.inverse_view_projection = p.inverse_view_projection;
        t.shadow_history_reproject = shadow_temporal_enabled_ && settings.atmosphere_shadow_rotation &&
            p.camera.w == 1.0f && !t.shadow_history_reusable && t.shadow_history_valid && !identity_dirty && rotation_mode_unchanged &&
            t.shadow_recorded_prepare+1 == prepare_serial_ && shadow_changes == 0 &&
            relative_eye == t.shadow_eye_relative && view != t.shadow_view && projection == t.shadow_projection &&
            std::memcmp(&p, &rotation_key, sizeof(p)) == 0 &&
            t.shadow_ring_revision == ring_shadow_revision && t.shadow_ring_image == ring_shadow_lut_->handle;
        t.shadow_reprojection = t.shadow_projection*t.shadow_view*p.inverse_view_projection;
        t.shadow_previous_inverse = t.cached.inverse_view_projection;
        t.shadow_eye_relative = relative_eye;
        t.shadow_rotation_enabled = settings.atmosphere_shadow_rotation;
        t.shadow_ring_revision = ring_shadow_revision;
        t.shadow_ring_image = ring_shadow_lut_->handle;
        t.shadow_view = view;
        t.shadow_projection = projection;
        bool optics_dirty = !t.valid || identity_dirty || p.reference_shape != t.cached.reference_shape || p.radii_heights != t.cached.radii_heights || p.rayleigh != t.cached.rayleigh ||
            p.mie_scattering != t.cached.mie_scattering || p.mie_extinction != t.cached.mie_extinction || p.ground_albedo != t.cached.ground_albedo ||
            p.mixed_absorption != t.cached.mixed_absorption || p.ozone_absorption != t.cached.ozone_absorption ||
            p.ozone_params != t.cached.ozone_params || p.profile_domain != t.cached.profile_domain || p.log_density != t.cached.log_density ||
            p.aerosol_count != t.cached.aerosol_count || p.aerosol_shape != t.cached.aerosol_shape ||
            p.aerosol_scattering != t.cached.aerosol_scattering || p.aerosol_extinction != t.cached.aerosol_extinction ||
            p.aerosol_phase != t.cached.aerosol_phase;
        bool sky_dirty = optics_dirty || p.camera != t.cached.camera || p.sun_direction != t.cached.sun_direction || p.sun_irradiance != t.cached.sun_irradiance;
        bool aerial_dirty = !t.aerial_valid || sky_dirty;
        if (aerial_dirty) t.aerial_valid = false;
        auto ubo = rhi::PushDescriptorWrite::uniform_buffer(0, uniform);
        auto sampler = ctx.sampler_linear_clamp();
        if (optics_dirty) {
            for (auto& layer : surface_layers_) if (layer == &t) layer = nullptr;
            writable(t.transmittance);
            rhi::PushDescriptorWrite writes[] = {ubo, rhi::PushDescriptorWrite::storage_image(1, t.transmittance)};
            bake(transmittance_pipeline_, writes, 32, 8);
            readable(t.transmittance);
            writable(t.multiscatter);
            rhi::PushDescriptorWrite multi_writes[] = {ubo, rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler), rhi::PushDescriptorWrite::storage_image(2, t.multiscatter)};
            bake(multiscatter_pipeline_, multi_writes, 32, 32);
            readable(t.multiscatter);
            ++stats_.optical_lut_updates;
        }
        size_t surface_slot = surface_slots[i];
        if (surface_slot != SIZE_MAX && candidate.index < 1024 && b.render.atmosphere) {
            if (optics_dirty || surface_layers_[surface_slot] != &t) {
                uint32_t surface_timer = ctx.begin_gpu_timer(cmd, "atmosphere_surface_lut");
                writable(surface_lut_);
                rhi::PushDescriptorWrite writes[] = {ubo,
                    rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                    rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                    rhi::PushDescriptorWrite::storage_image(3, surface_lut_)};
                ctx.push_constants(cmd, surface_pipeline_, uint32_t(surface_slot));
                bake(surface_pipeline_, writes, 16, 4);
                readable(surface_lut_);
                ctx.end_gpu_timer(cmd, surface_timer);
                ++stats_.surface_lut_updates;
                surface_layers_[surface_slot] = &t;
            }
            if (settings.atmosphere_surface_lighting)
                static_cast<glm::vec4*>(records.mapped)[candidate.index] = {bottom, radius+height,
                    float(b.opaque_radius_au()*AU_TO_KM), float(surface_slot+1)};
        }
        if (sky_dirty) {
            uint32_t sky_timer = ctx.begin_gpu_timer(cmd, "atmosphere_sky");
            writable(t.sky);
            writable(t.throughput);
            rhi::PushDescriptorWrite writes[] = {ubo,
                rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                rhi::PushDescriptorWrite::storage_image(3, t.sky),
                rhi::PushDescriptorWrite::storage_image(4, t.throughput)};
            bake(sky_pipeline_, writes, 24, 14);
            readable(t.sky);
            readable(t.throughput);
            ctx.end_gpu_timer(cmd, sky_timer);
            ++stats_.sky_lut_updates;
        }
        if (settings.aerial_perspective_enabled && aerial_dirty) {
            uint32_t aerial_timer = ctx.begin_gpu_timer(cmd, "atmosphere_aerial");
            writable(t.aerial_radiance);
            writable(t.aerial_optical_depth);
            writable(t.aerial_range);
            rhi::PushDescriptorWrite writes[] = {ubo,
                rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                rhi::PushDescriptorWrite::storage_image(3, t.aerial_radiance),
                rhi::PushDescriptorWrite::storage_image(4, t.aerial_optical_depth),
                rhi::PushDescriptorWrite::storage_image(5, t.aerial_range)};
            bake(aerial_pipeline_, writes, 24, 27);
            readable(t.aerial_radiance);
            readable(t.aerial_optical_depth);
            readable(t.aerial_range);
            ctx.end_gpu_timer(cmd, aerial_timer);
            ++stats_.aerial_lut_updates;
            t.aerial_valid = true;
        }
        t.cached = p;
        t.definition = system.share_definition();
        t.generation = system.generation();
        t.body_index = candidate.index;
        t.geometry_revision = b.revisions.geometry;
        t.lighting_revision = b.revisions.lighting;
        t.valid = true;
        t.last_used = prepare_serial_;
        active_.push_back(&t);
    }
    ctx.end_gpu_timer(cmd, timer);
    for (auto& table : tables_) {
        if (table->last_used == prepare_serial_ && table->shadow_active) continue;
        for (auto* image : {&table->shadow_fraction, &table->shadow_depth, &table->shadow_energy, &table->shadow_loss}) {
            if (*image) retired_shadows_.push_back({*image, ctx.recording_serial()});
            *image = {};
        }
        table->shadow_history_valid = false;
    }
    ctx.flush_buffer(records);
    stats_.visible = static_cast<uint32_t>(active_.size());
}

void AtmosphereRenderer::composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                                   std::span<const rhi::PushDescriptorWrite> transparency_writes) {
    if (active_.empty()) return;
    ctx.transition_image(cmd, color, dual_source_ ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL);
    ctx.transition_image(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint32_t timer = ctx.begin_gpu_timer(cmd, "atmosphere_composite");
    if (!transparency_writes.empty()) ctx.memory_barrier(cmd);
    for (auto* t : active_) {
        if (!t->shadow_active) continue;
        if (t->shadow_fraction.extent.width != color.extent.width || t->shadow_fraction.extent.height != color.extent.height) {
            for (auto* image : {&t->shadow_fraction, &t->shadow_depth}) {
                if (*image) retired_shadows_.push_back({*image, ctx.recording_serial()});
                *image = ctx.create_image_2d(color.extent.width, color.extent.height,
                    image == &t->shadow_fraction ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 2);
                ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            t->shadow_history_reusable = false;
            t->shadow_history_reproject = false;
            t->shadow_reset_reasons |= ShadowHistoryViewport;
        }
        const auto& header = t->shadow_header;
        bool temporal = shadow_temporal_enabled_;
        for (auto* image : {&t->shadow_energy, &t->shadow_loss}) {
            bool resize = image->extent.width != color.extent.width || image->extent.height != color.extent.height;
            if (*image && (!temporal || resize)) {
                retired_shadows_.push_back({*image, ctx.recording_serial()});
                *image = {};
            }
            if (temporal && !*image) {
                *image = ctx.create_image_2d(color.extent.width, color.extent.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT, 1, 2);
                t->shadow_history_reusable = false;
                t->shadow_history_reproject = false;
                t->shadow_reset_reasons |= ShadowHistoryInitial;
                ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_GENERAL);
            }
        }
        bool vrs = !temporal && shadow_vrs_enabled_ && header.flags.w == 0u && t->cached.camera.w == 1.0f &&
                   color.extent.width >= 64 && color.extent.height >= 64;
        ctx.transition_image(cmd, t->shadow_fraction, VK_IMAGE_LAYOUT_GENERAL);
        ctx.transition_image(cmd, t->shadow_depth, VK_IMAGE_LAYOUT_GENERAL);
        bool reproject = temporal && t->shadow_history_reproject;
        auto& pipeline = reproject ? (transparency_writes.empty() ? shadow_rotate_pipeline_ : shadow_rotate_transparency_pipeline_)
                             : temporal ? (transparency_writes.empty() ? shadow_accumulate_pipeline_ : shadow_accumulate_transparency_pipeline_)
                             : vrs ? (transparency_writes.empty() ? shadow_vrs_pipeline_ : shadow_vrs_transparency_pipeline_)
                             : (transparency_writes.empty() ? shadow_pipeline_ : shadow_transparency_pipeline_);
        auto& stbn = stbn_tex_ ? stbn_tex_ : dummy_stbn_;
        std::vector<rhi::PushDescriptorWrite> writes{
            rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(7, t->transmittance, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(8, t->multiscatter, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(12, *ring_shadow_lut_, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_image(18, t->shadow_fraction),
            rhi::PushDescriptorWrite::storage_image(19, t->shadow_depth),
            rhi::PushDescriptorWrite::sampled_image(20, stbn, ctx.sampler_nearest_clamp())};
        uint32_t shadow_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadows");
        if (temporal) {
            ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            writes.push_back(rhi::PushDescriptorWrite::storage_image(23, t->shadow_energy));
            writes.push_back(rhi::PushDescriptorWrite::storage_image(24, t->shadow_loss));
            writes.push_back(rhi::PushDescriptorWrite::storage_buffer(25, shadow_diagnostics_[ctx.frame_index()].counters));
            if (reproject) {
                for (auto* image : {&shadow_reproject_energy_, &shadow_reproject_loss_}) {
                    if (image->extent.width != color.extent.width || image->extent.height != color.extent.height) {
                        if (*image) retired_shadows_.push_back({*image, ctx.recording_serial()});
                        *image = ctx.create_image_2d(color.extent.width, color.extent.height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
                    }
                    ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_GENERAL);
                }
                rhi::PushDescriptorWrite reprojection_writes[] = {
                    rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
                    rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
                    rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
                    rhi::PushDescriptorWrite::storage_image(23, t->shadow_energy),
                    rhi::PushDescriptorWrite::storage_image(24, t->shadow_loss),
                    rhi::PushDescriptorWrite::storage_image(26, shadow_reproject_energy_),
                    rhi::PushDescriptorWrite::storage_image(27, shadow_reproject_loss_)};
                const std::array<glm::mat4, 2> transforms{t->shadow_reprojection, t->shadow_previous_inverse};
                auto reproject_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadow_reproject");
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadow_reproject_pipeline_.handle);
                ctx.push_descriptors(cmd, shadow_reproject_pipeline_, reprojection_writes);
                ctx.push_constants(cmd, shadow_reproject_pipeline_, transforms);
                vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
                ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                ctx.end_gpu_timer(cmd, reproject_timer);
                writes.push_back(rhi::PushDescriptorWrite::storage_image(26, shadow_reproject_energy_));
                writes.push_back(rhi::PushDescriptorWrite::storage_image(27, shadow_reproject_loss_));
                ++stats_.shadow_rotation_updates;
            }
            ctx.push_constants(cmd, pipeline, glm::uvec2(t->shadow_history_reusable, shadow_diagnostics_enabled_));
            if (!t->shadow_history_reusable && !reproject) {
                ++stats_.shadow_history_resets;
                stats_.shadow_history_reset_reasons |= t->shadow_reset_reasons;
            }
        }
        if (vrs) {
            uint32_t width = (color.extent.width+1)/2, height = (color.extent.height+1)/2;
            if (shadow_coarse_fraction_.extent.width != width || shadow_coarse_fraction_.extent.height != height) {
                for (auto* image : {&shadow_coarse_fraction_, &shadow_coarse_guide_}) {
                    if (*image) retired_shadows_.push_back({*image, ctx.recording_serial()});
                    *image = ctx.create_image_2d(width, height,
                        image == &shadow_coarse_fraction_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
                }
            }
            ctx.transition_image(cmd, shadow_coarse_fraction_, VK_IMAGE_LAYOUT_GENERAL);
            ctx.transition_image(cmd, shadow_coarse_guide_, VK_IMAGE_LAYOUT_GENERAL);
            auto coarse_writes = writes;
            std::erase_if(coarse_writes, [](auto write) { return write.binding == 18 || write.binding == 19; });
            coarse_writes.push_back(rhi::PushDescriptorWrite::storage_image(18, shadow_coarse_fraction_));
            coarse_writes.push_back(rhi::PushDescriptorWrite::storage_image(19, shadow_coarse_guide_));
            uint32_t coarse_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadow_coarse");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadow_coarse_pipeline_.handle);
            ctx.push_descriptors(cmd, shadow_coarse_pipeline_, coarse_writes);
            vkCmdDispatch(cmd, (width+7)/8, (height+7)/8, 1);
            ctx.end_gpu_timer(cmd, coarse_timer);
            for (auto* image : {&shadow_coarse_fraction_, &shadow_coarse_guide_})
                ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            writes.push_back(rhi::PushDescriptorWrite::sampled_image(21, shadow_coarse_fraction_, ctx.sampler_nearest_clamp()));
            writes.push_back(rhi::PushDescriptorWrite::sampled_image(22, shadow_coarse_guide_, ctx.sampler_nearest_clamp()));
            ++stats_.shadow_vrs_updates;
        }
        writes.insert(writes.end(), transparency_writes.begin(), transparency_writes.end());
        uint32_t resolve_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadow_resolve");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        ctx.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
        ctx.end_gpu_timer(cmd, resolve_timer);
        for (auto* image : {&t->shadow_fraction, &t->shadow_depth})
            ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        ctx.end_gpu_timer(cmd, shadow_timer);
        t->shadow_history_valid = temporal;
        t->shadow_recorded_prepare = prepare_serial_;
        ++stats_.shadow_updates;
    }
    if (shadow_diagnostics_enabled_ && shadow_temporal_enabled_ && stats_.shadow_updates != 0) {
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        auto& diagnostics = shadow_diagnostics_[ctx.frame_index()];
        diagnostics.serial = ctx.recording_serial();
        diagnostics.reset_reasons = stats_.shadow_history_reset_reasons;
    }
    if (dual_source_ && !transparency_writes.empty()) {
        for (auto* t : active_) {
            auto& stbn = stbn_tex_ ? stbn_tex_ : dummy_stbn_;
            std::vector<rhi::PushDescriptorWrite> writes{
                rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
                rhi::PushDescriptorWrite::sampled_image(1, t->sky, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(2, t->throughput, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(5, t->aerial_radiance, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(6, t->aerial_optical_depth, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(7, t->transmittance, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(8, t->multiscatter, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(9, t->aerial_range, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
                rhi::PushDescriptorWrite::sampled_image(12, *ring_shadow_lut_, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(16, t->shadow_active ? t->shadow_fraction : surface_lut_, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(17, t->shadow_active ? t->shadow_depth : surface_lut_, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(20, stbn, ctx.sampler_nearest_clamp())};
            writes.insert(writes.end(), transparency_writes.begin(), transparency_writes.end());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, transparent_pipeline_.handle);
            ctx.push_descriptors(cmd, transparent_pipeline_, writes);
            vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
            ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
    }
    if (dual_source_) {
        auto attachment = rhi::RenderingAttachment::load_color(color);
        ctx.begin_rendering(cmd, {color.extent.width, color.extent.height}, {&attachment, 1});
        ctx.set_viewport_and_scissor(cmd, {color.extent.width, color.extent.height});
    }
    for (auto* t : active_) {
        auto& stbn = stbn_tex_ ? stbn_tex_ : dummy_stbn_;
        rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(1, t->sky, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(2, t->throughput, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::storage_image(4, color),
            rhi::PushDescriptorWrite::sampled_image(5, t->aerial_radiance, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(6, t->aerial_optical_depth, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(7, t->transmittance, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(8, t->multiscatter, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(9, t->aerial_range, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(12, *ring_shadow_lut_, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(16, t->shadow_active ? t->shadow_fraction : surface_lut_, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(17, t->shadow_active ? t->shadow_depth : surface_lut_, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(20, stbn, ctx.sampler_nearest_clamp())};
        if (dual_source_) {
            std::vector<rhi::PushDescriptorWrite> graphics;
            for (const auto& write : writes) if (write.binding != 4) graphics.push_back(write);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dual_source_pipeline_.handle);
            ctx.push_descriptors(cmd, dual_source_pipeline_, graphics);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        } else {
            auto& pipeline = transparency_writes.empty() ? composite_pipeline_ : composite_transparency_pipeline_;
            std::vector<rhi::PushDescriptorWrite> compute(std::begin(writes), std::end(writes));
            compute.insert(compute.end(), transparency_writes.begin(), transparency_writes.end());
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
            ctx.push_descriptors(cmd, pipeline, compute);
            vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
            ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
    }
    if (dual_source_) ctx.end_rendering(cmd);
    ctx.end_gpu_timer(cmd, timer);
}

}
