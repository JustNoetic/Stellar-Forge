#include "rendering/celestial_renderer.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "core/constants.hpp"
#include "core/path_utils.hpp"
#include "core/timing.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace sf::rendering {

static_assert(sizeof(BodyInstanceGPU) == 288);
static_assert(sizeof(SurfaceFluidGPU) == 128);
static_assert(sizeof(SurfaceFluidHeaderGPU) == 32+OCEAN_WAVE_COUNT*32);
static_assert(terrain::MAX_PATCHES >= MAX_CELESTIAL_BODIES*6);

namespace {

glm::dmat3 fluid_rotation(glm::dvec3 pole, double spin) {
    glm::dvec3 reference = std::abs(pole.x) < 0.9 ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0);
    glm::dvec3 x = glm::normalize(reference - pole * glm::dot(reference, pole));
    glm::dvec3 z = glm::cross(x, pole);
    double c = std::cos(spin), s = std::sin(spin);
    return glm::dmat3(x * c - z * s, pole, x * s + z * c);
}

double cox_munk_variance(double wind_ms) {
    return 0.003 + 0.00512 * std::max(wind_ms, 0.0);
}

double surface_gravity(const ephemeris::BodyDefinition& d) {
    double radius_m = d.radius_au() * AU_TO_KM * 1000.0;
    double g = 6.6743e-11 * d.mass_msun * 1.98847e30 / std::max(radius_m * radius_m, 1.0);
    return std::isfinite(g) && g > 0.05 ? g : 9.81;
}

void build_ocean_waves(SurfaceFluidHeaderGPU& header, glm::dvec3 camera_fixed_m, double wind_ms, double wind_direction_deg,
                       double gravity, double seconds) {
    glm::dvec3 up = glm::length(camera_fixed_m) > 1.0 ? glm::normalize(camera_fixed_m) : glm::dvec3(1.0, 0.0, 0.0);
    glm::dvec3 east = glm::cross(glm::dvec3(0.0, 1.0, 0.0), up);
    east = glm::length(east) > 1e-6 ? glm::normalize(east) : glm::dvec3(0.0, 0.0, 1.0);
    glm::dvec3 north = glm::cross(up, east);
    double wind = std::max(wind_ms, 0.5);
    double peak_omega = 0.877 * gravity / wind;
    double peak_wavelength = TWO_PI * gravity / (peak_omega * peak_omega);
    double wavelength = std::clamp(1.5 * peak_wavelength, 1.0, 2000.0);
    double variance = cox_munk_variance(wind_ms);
    double steepness = std::sqrt(2.0 * variance / OCEAN_WAVE_COUNT);
    double wind_angle = wind_direction_deg * DEG_TO_RAD;
    for (uint32_t i = 0; i < OCEAN_WAVE_COUNT; ++i) {
        double spread = (std::fmod(i * 0.6180339887498949, 1.0) * 2.0 - 1.0) * PI * (0.25 + 0.5 * i / double(OCEAN_WAVE_COUNT - 1));
        double angle = wind_angle + spread;
        glm::dvec3 direction = east * std::cos(angle) + north * std::sin(angle);
        double k = TWO_PI / wavelength;
        double omega = std::sqrt(gravity * k);
        double offset = std::fmod(i * 2.399963229728653, TWO_PI);
        double time_phase = std::fmod(offset - omega * seconds, TWO_PI);
        double focus_phase = std::fmod(k * glm::dot(direction, camera_fixed_m) + time_phase, TWO_PI);
        header.waves[i].direction_k = glm::vec4(glm::vec3(direction), static_cast<float>(k));
        header.waves[i].phases = glm::vec4(static_cast<float>(focus_phase), static_cast<float>(time_phase),
                                           static_cast<float>(steepness), static_cast<float>(wavelength));
        wavelength *= 0.68;
    }
    header.header.x = static_cast<float>(variance);
    header.header.y = static_cast<float>(std::fmod(seconds, 1000.0));
}

struct SphereVertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

} // namespace

void CelestialRenderer::init(rhi::VkContext& ctx, bool load_surface_assets) {
    build_sphere_mesh(ctx, 64, 32);
    cloud_maps_.init(ctx);
    textures_.init(ctx);
    normals_.init(ctx, true);
    rings_.init(ctx);
    atmospheres_.init(ctx);
    transparency_.init(ctx);
    refraction_.init(ctx);
    eclipse_refraction_.init(ctx);

    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        scene_ubos_[i] = ctx.create_host_buffer(sizeof(SceneCelestialUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        instance_ssbos_[i] = ctx.create_host_buffer(
            MAX_CELESTIAL_BODIES * sizeof(BodyInstanceGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        eclipse_ssbos_[i] = ctx.create_host_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        fluid_ssbos_[i] = ctx.create_host_buffer(
            sizeof(SurfaceFluidHeaderGPU)+MAX_CELESTIAL_BODIES*sizeof(SurfaceFluidGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {29, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };

    rhi::GraphicsPipelineDesc sphere_desc{};
    sphere_desc.vert_path = "celestial/sphere.vert";
    sphere_desc.frag_path = "celestial/sphere.frag";
    sphere_desc.bindings.assign(std::begin(bindings), std::end(bindings));
    sphere_desc.vertex_bindings = {
        {0, sizeof(SphereVertex), VK_VERTEX_INPUT_RATE_VERTEX},
    };
    sphere_desc.vertex_attributes = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, normal)},
    };
    sphere_desc.color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
    sphere_desc.depth_format = VK_FORMAT_D32_SFLOAT;
    sphere_desc.depth_test = true;
    sphere_desc.depth_write = true;
    sphere_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    sphere_desc.cull_mode = VK_CULL_MODE_BACK_BIT;
    sphere_desc.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    sphere_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, sphere_desc);

    for (uint32_t mode = 0; mode < 2; ++mode) {
        rhi::GraphicsPipelineDesc cloud_desc = sphere_desc;
        cloud_desc.vert_path = "celestial/cloud.vert";
        cloud_desc.frag_path = mode == 0 ? "celestial/cloud_sheet.frag" : "celestial/cloud.frag";
        cloud_desc.capture_statistics = std::getenv("STELLAR_SHADER_STATS") != nullptr;
        cloud_desc.bindings = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
            {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
            {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
            {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
            {29, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        };
        cloud_desc.depth_write = false;
        cloud_desc.premultiplied_blend = true;
        cloud_desc.cull_mode = VK_CULL_MODE_FRONT_BIT;
        cloud_pipeline_[mode] = rhi::ShaderCache::create_graphics_pipeline(ctx, cloud_desc);
        if (ctx.fragment_stores_enabled()) {
            auto capture = cloud_desc;
            capture.macros = {{"TRANSPARENCY_CAPTURE", "1"}};
            capture.bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
            capture.bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
            cloud_transparency_pipeline_[mode] = rhi::ShaderCache::create_graphics_pipeline(ctx, capture);
        }
        cloud_desc.macros = {{"CLOUD_DEPTH_ONLY", "1"}};
        cloud_desc.depth_write = true;
        cloud_desc.premultiplied_blend = false;
        cloud_desc.additive_blend = true;
        cloud_depth_pipeline_[mode] = rhi::ShaderCache::create_graphics_pipeline(ctx, cloud_desc);
    }

    rhi::GraphicsPipelineDesc point_desc{};
    point_desc.vert_path = "celestial/point_celestial.vert";
    point_desc.frag_path = "celestial/point_celestial.frag";
    point_desc.bindings.assign(std::begin(bindings), std::end(bindings));
    point_desc.bindings.push_back({27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1});
    point_desc.push_constant_size = sizeof(uint32_t);
    point_desc.color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
    point_desc.depth_format = VK_FORMAT_D32_SFLOAT;
    point_desc.depth_test = false;
    point_desc.depth_write = false;
    point_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    point_desc.additive_blend = true;
    point_desc.cull_mode = VK_CULL_MODE_NONE;

    point_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, point_desc);
    point_desc.depth_format = VK_FORMAT_UNDEFINED;
    atmosphere_point_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, point_desc);
    terrain_.init(ctx, load_surface_assets);
    for (const auto& [body, source] : terrain_.height_sources()) normals_.set_height_source(body, source);
    if (load_surface_assets) {
        for (const auto& [body, path] : textures_.load_catalog(paths::external_path("data", "surfaces.json")))
            set_height_source(body, terrain::HeightSource::open(path));
    }
}

void CelestialRenderer::shutdown(rhi::VkContext& ctx) {
    cloud_maps_.shutdown(ctx);
    eclipse_refraction_.shutdown(ctx);
    refraction_.shutdown(ctx);
    transparency_.shutdown(ctx);
    atmospheres_.shutdown(ctx);
    rings_.shutdown(ctx);
    terrain_.shutdown(ctx);
    textures_.shutdown(ctx);
    normals_.shutdown(ctx);
    rhi::ShaderCache::destroy_pipeline(ctx, point_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, atmosphere_point_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, sphere_pipeline_);
    for (uint32_t mode = 0; mode < 2; ++mode) {
        rhi::ShaderCache::destroy_pipeline(ctx, cloud_pipeline_[mode]);
        rhi::ShaderCache::destroy_pipeline(ctx, cloud_depth_pipeline_[mode]);
        rhi::ShaderCache::destroy_pipeline(ctx, cloud_transparency_pipeline_[mode]);
    }
    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        ctx.destroy_buffer(fluid_ssbos_[i]);
        ctx.destroy_buffer(instance_ssbos_[i]);
        ctx.destroy_buffer(material_phases_[i]);
        ctx.destroy_buffer(eclipse_ssbos_[i]);
        ctx.destroy_buffer(scene_ubos_[i]);
    }
    ctx.destroy_buffer(sphere_ibo_);
    ctx.destroy_buffer(sphere_vbo_);
}

void CelestialRenderer::build_sphere_mesh(rhi::VkContext& ctx, uint32_t sectors, uint32_t stacks) {
    std::vector<SphereVertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve((stacks + 1) * (sectors + 1));
    indices.reserve(stacks * sectors * 6);

    for (uint32_t i = 0; i <= stacks; ++i) {
        double stack_angle = PI * 0.5 - static_cast<double>(i) * (PI / static_cast<double>(stacks));
        double xy = std::cos(stack_angle);
        double y = std::sin(stack_angle);

        for (uint32_t j = 0; j <= sectors; ++j) {
            double sector_angle = static_cast<double>(j) * (TWO_PI / static_cast<double>(sectors));
            double x = xy * std::cos(sector_angle);
            double z = xy * std::sin(sector_angle);
            glm::vec3 n{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
            vertices.push_back({n, n});
        }
    }

    for (uint32_t i = 0; i < stacks; ++i) {
        uint32_t k1 = i * (sectors + 1);
        uint32_t k2 = k1 + sectors + 1;
        for (uint32_t j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices.push_back(k1);
                indices.push_back(k1 + 1);
                indices.push_back(k2);
            }
            if (i != (stacks - 1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2 + 1);
                indices.push_back(k2);
            }
        }
    }

    sphere_index_count_ = static_cast<uint32_t>(indices.size());
    sphere_vbo_ = ctx.create_device_buffer(vertices.size() * sizeof(SphereVertex),
                                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data());
    sphere_ibo_ = ctx.create_device_buffer(indices.size() * sizeof(uint32_t),
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data());
}

void CelestialRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                                const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent,
                                bool shadow_diagnostics, std::map<std::string, double>* timings) {
    atmosphere_points_ready_ = false;
    auto measure = [&](const char* name, auto&& function) {
        ScopedTimer timer(timings ? &(*timings)[name] : nullptr);
        function();
    };
    uint32_t count = std::min(static_cast<uint32_t>(sys.size()), MAX_CELESTIAL_BODIES);
    measure("cpu_refraction_prepare_ms", [&] { refraction_.prepare(cmd, ctx, cam, sys, settings); });
    measure("cpu_textures_prepare_ms", [&] { textures_.prepare(cmd, ctx, cam, sys, extent, settings.surface_textures_enabled); });
    measure("cpu_normals_prepare_ms", [&] { normals_.prepare(cmd, ctx, cam, sys, extent, settings.terrain_imported_enabled && settings.terrain_normal_maps_enabled); });
    measure("cpu_rings_prepare_ms", [&] { rings_.prepare(cmd, ctx, cam, sys, count, refraction_.host().body); });
    measure("cpu_ringshine_prepare_ms", [&] {
    uint32_t ringshine_timer = ctx.begin_gpu_timer(cmd, "ringshine_bake");
    rings_.update_ringshine(cmd, ctx, cam, sys, settings, count, extent);
    ctx.end_gpu_timer(cmd, ringshine_timer);
    });
    measure("cpu_terrain_prepare_ms", [&] {
    terrain_.prepare(ctx, cam, sys, settings, extent, count, &refraction_.host());
    terrain_.bake(cmd, ctx);
    atmosphere_minima_.resize(count);
    for (uint32_t i = 0; i < count; ++i)
        atmosphere_minima_[i] = static_cast<float>(terrain_.minimum_height_km(sys.body(i).definition, settings));
    });
    measure("cpu_planetshine_prepare_ms", [&] {
    materials_.update(sys);
    planetshine_ = prepare_planetshine(sys, settings.planetshine_enabled && !settings.direct_sunlight_only,
        settings.hdr_enabled, MAX_CELESTIAL_BODIES, &materials_);
    });
    measure("cpu_atmosphere_prepare_ms", [&] { atmospheres_.prepare(cmd, ctx, cam, sys, settings, extent, atmosphere_minima_, rings_.hosts(), &rings_.shadow_lut(), rings_.shadow_revision(), shadow_diagnostics, planetshine_, &rings_); });
    measure("cpu_transparency_prepare_ms", [&] {
    bool visible_clouds = false;
    if (settings.clouds_enabled) {
        double pixels = extent.height/(2.0*std::tan(cam.fov_deg*DEG_TO_RAD*0.5));
        for (uint32_t i = 0; i < count && !visible_clouds; ++i) {
            const auto& body = sys.body(i);
            const auto& cloud = body.definition.clouds;
            double distance = glm::length(camera_relative(body.render.position, cam.eye_pos_render()));
            visible_clouds = cloud.enabled && cloud.coverage > 0 && cloud.density > 0 && cloud.extinction_per_km > 0 &&
                body.definition.radius_au()/std::max(distance, 1e-15)*pixels > 2;
        }
    }
    transparency_.prepare(cmd, ctx, extent, !settings.transparency_reference_legacy && !sys.empty() &&
        (visible_clouds || (settings.show_rings && rings_.host_count() > 0) || (settings.show_orbits && sys.size() > 1)));
    });
    measure("cpu_eclipse_refraction_prepare_ms", [&] { eclipse_refraction_.prepare(cmd, ctx, sys, settings); });
}

void CelestialRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                               const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent,
                               std::map<std::string, double>* timings, bool gpu_diagnostics) {
    bool profile_gpu = timings || gpu_diagnostics;
    auto setup_start = timings ? SteadyClock::now() : SteadyClock::time_point{};
    ring_eclipse_candidate_counts_.clear();
    surface_ring_candidate_masks_.clear();
    uint32_t count = std::min(static_cast<uint32_t>(sys.size()), MAX_CELESTIAL_BODIES);
    if (count == 0) {
        return;
    }

    uint32_t f_idx = ctx.frame_index();
    float fb_w = static_cast<float>(std::max(1u, extent.width));
    float fb_h = static_cast<float>(std::max(1u, extent.height));
    float aspect = fb_w / fb_h;
    float fov_rad = static_cast<float>(cam.fov_deg * DEG_TO_RAD);
    float fov_factor = 1.0f / (2.0f * std::tan(0.5f * fov_rad));
    float z_near = cam.compute_near_plane_au();

    glm::dvec3 eye_pos = cam.eye_pos_render().value;
    glm::mat4 view = cam.view_matrix_eye_relative();
    glm::mat4 proj = cam.proj_matrix(aspect, z_near);
    glm::dmat3 view_rotation = glm::dmat3(glm::mat3(view));

    SceneCelestialUBO ubo{};
    ubo.view = view;
    ubo.view_proj = proj * view;
    ubo.projection = proj;
    ubo.screen_params = {fb_w, fb_h, fov_factor, settings.exposure};
    ubo.point_params.x = std::clamp(settings.star_point_size, 1.5f, 6.0f);
    ubo.num_bodies = count;
    ubo.hdr_enabled = settings.hdr_enabled ? 1u : 0u;
    ubo.num_ring_hosts = (settings.show_rings && settings.ring_shadows_enabled) ? rings_.host_count() : 0u;
    ubo.ringshine_params = {settings.ringshine_enabled && settings.show_rings && !settings.direct_sunlight_only ? 1u : 0u,
        rings_.host_count(), settings.planetshine_enabled && !settings.direct_sunlight_only ? 1u : 0u, settings.ring_planetshine_mode};
    body_instances_.resize(count);
    auto* inst_ptr = body_instances_.data();
    auto eclipse_records = eclipse_refraction_.records();
    auto eclipse_bounds = eclipse_refraction_.bounds();
    std::array<uint32_t, MAX_SCENE_STARS> star_bodies{};

    for (uint32_t i = 0; i < count; ++i) {
        const auto& b = sys.body(i);
        glm::dvec3 rel_eye_d = camera_relative(b.render.position, cam.eye_pos_render());
        glm::vec3 rel_eye = glm::vec3(rel_eye_d);
        float r_au = static_cast<float>(b.opaque_radius_au());

        BodyInstanceGPU inst{};
        inst.pos_radius = glm::vec4(rel_eye, r_au);
        inst.view_center = glm::vec4(glm::vec3(view_rotation * rel_eye_d), 0.0f);
        inst.color_is_star = glm::vec4(b.definition.color_rgb, b.definition.is_star ? 1.0f : 0.0f);
        inst.pole_oblateness = glm::vec4(glm::vec3(b.render.pole), static_cast<float>(b.definition.oblateness));
        inst.pole_color_lum = glm::vec4(b.definition.color_pole_rgb, static_cast<float>(b.definition.lum_eq));
        inst.spin_flags = glm::vec4(static_cast<float>(b.render.spin_angle_rad),
                                    static_cast<float>(b.definition.lum_pole),
                                    b.definition.is_black_hole ? 1.0f : 0.0f,
                                    static_cast<float>(i));
        inst.surface_flags.x = terrain_.covers(i) ? 1u : 0u;
        if (i < planetshine_.size()) inst.planetshine = planetshine_[i];
        inst.material = materials_.material(i);
        inst_ptr[i] = inst;

        if (b.definition.is_star && ubo.num_stars < MAX_SCENE_STARS) {
            uint32_t s = ubo.num_stars++;
            star_bodies[s] = i;
            ubo.stars_pos_radius[s] = glm::vec4(rel_eye, r_au);
            ubo.stars_color_lum[s] = glm::vec4(b.definition.color_rgb, static_cast<float>(std::max(b.definition.lum_eq, 1e-4)));
        }
    }

    if (ubo.num_stars == 0) {
        ubo.num_stars = 1;
        ubo.stars_pos_radius[0] = glm::vec4(glm::vec3(-eye_pos), static_cast<float>(SOLAR_RADII_TO_AU));
        ubo.stars_color_lum[0] = glm::vec4(1.0f, 0.95f, 0.90f, 1.0f);
    } else {
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t s = 0; s < ubo.num_stars; ++s)
                if (!sys.illuminates(star_bodies[s], i)) inst_ptr[i].surface_flags.y |= 1u << s;
    }

    uint32_t ring_count = settings.show_rings ? rings_.host_count() : 0u;
    static_assert(MAX_RING_HOSTS < 32);
    surface_ring_candidate_masks_.assign(ubo.num_ring_hosts > 0u && !settings.ring_shadow_reference_all_hosts
                                       ? count * ubo.num_stars : 0u, 0u);
    ring_eclipse_candidate_counts_.assign(ring_count * ubo.num_stars, 0u);
    eclipse_words_.assign((count + ring_count) * ubo.num_stars * 2, 0u);
    for (uint32_t i = 0; i < count + ring_count; ++i) {
        glm::vec4 receiver_bounds;
        if (i < count) {
            const auto& receiver = inst_ptr[i];
            if (receiver.color_is_star.w > 0.5f || receiver.spin_flags.z > 0.5f) continue;
            float distance = std::max(glm::length(glm::vec3(receiver.pos_radius)), 1e-12f);
            float radius_px = receiver.pos_radius.w / distance * fb_h * fov_factor;
            if (radius_px <= 1.49f && receiver.surface_flags.x == 0u && eclipse_records.empty()) continue;
            receiver_bounds = receiver.pos_radius;
            if (receiver.surface_flags.x != 0u)
                receiver_bounds.w += terrain_.relief_extent_au(i);
            for (uint32_t s = 0; !surface_ring_candidate_masks_.empty() && s < ubo.num_stars; ++s) {
                uint32_t mask = 0u;
                for (uint32_t h = 0; h < ubo.num_ring_hosts; ++h) {
                    const auto& host = rings_.hosts()[h];
                    if (ring_could_shadow_surface(
                            receiver_bounds, ubo.stars_pos_radius[s],
                            {glm::vec3(host.center_min_r), host.pole_max_r.w},
                            {glm::vec3(host.pole_max_r), host.center_min_r.w}, fb_h * fov_factor, settings.ring_shadow_softness))
                        mask |= 1u << h;
                }
                surface_ring_candidate_masks_[i * ubo.num_stars + s] = mask;
            }
        } else {
            const auto& host = rings_.hosts()[i - count];
            receiver_bounds = glm::vec4(glm::vec3(host.center_min_r), host.pole_max_r.w);
        }
        for (uint32_t s = 0; s < ubo.num_stars; ++s) {
            uint32_t range = (i * ubo.num_stars + s) * 2;
            uint32_t start = static_cast<uint32_t>(eclipse_words_.size());
            for (uint32_t j = 0; j < count; ++j) {
                if (i == j || inst_ptr[j].color_is_star.w > 0.5f) continue;
                if (i >= count && inst_ptr[j].pos_radius.w <= 0.0f) continue;
                auto caster = inst_ptr[j].pos_radius;
                if (j < eclipse_bounds.size() && eclipse_bounds[j].x > 0) {
                    double distance = glm::length(glm::dvec3(caster)-glm::dvec3(receiver_bounds))+receiver_bounds.w;
                    caster.w = float(eclipse_bounds[j].x+distance*eclipse_bounds[j].y);
                }
                if (eclipse_could_shadow(receiver_bounds, ubo.stars_pos_radius[s], caster))
                    eclipse_words_.push_back(j);
            }
            eclipse_words_[range] = start;
            eclipse_words_[range + 1] = static_cast<uint32_t>(eclipse_words_.size()) - start;
            if (i >= count)
                ring_eclipse_candidate_counts_[(i - count) * ubo.num_stars + s] = eclipse_words_[range + 1];
        }
    }
    ubo.ring_shadow_params = {static_cast<uint32_t>(eclipse_words_.size()),
                              settings.ring_shadow_reference_all_hosts ? 1u : 0u,
                              (settings.ring_shadow_single_box ? 1u : 0u) | (settings.ring_shadow_reference_linear ? 2u : 0u),
                              std::bit_cast<uint32_t>(settings.ring_shadow_softness)};
    ubo.eclipse_params.x = settings.eclipse_reference_smoothstep ? 1u : 0u;
    eclipse_words_.insert(eclipse_words_.end(), surface_ring_candidate_masks_.begin(), surface_ring_candidate_masks_.end());
    if (!eclipse_records.empty()) {
        ubo.eclipse_params.y = uint32_t(eclipse_words_.size());
        eclipse_words_.resize(eclipse_words_.size()+eclipse_records.size()*4);
        std::memcpy(eclipse_words_.data()+ubo.eclipse_params.y, eclipse_records.data(), eclipse_records.size_bytes());
    }
    auto& eclipse_buffer = eclipse_ssbos_[f_idx];
    VkDeviceSize eclipse_bytes = eclipse_words_.size() * sizeof(uint32_t);
    if (eclipse_bytes > eclipse_buffer.size) {
        VkDeviceSize capacity = std::max(eclipse_bytes, eclipse_buffer.size * 2);
        ctx.destroy_buffer(eclipse_buffer);
        eclipse_buffer = ctx.create_host_buffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    std::memcpy(eclipse_buffer.mapped, eclipse_words_.data(), eclipse_bytes);
    std::memcpy(instance_ssbos_[f_idx].mapped, body_instances_.data(), count * sizeof(BodyInstanceGPU));
    std::memcpy(scene_ubos_[f_idx].mapped, &ubo, sizeof(SceneCelestialUBO));
    ctx.flush_buffer(eclipse_buffer, 0, eclipse_bytes);
    ctx.flush_buffer(instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU));
    ctx.flush_buffer(scene_ubos_[f_idx], 0, sizeof(SceneCelestialUBO));

    auto& phases = material_phases_[f_idx];
    size_t phase_bytes = materials_.atlas().size()*sizeof(float);
    if (phases.size < phase_bytes) {
        ctx.destroy_buffer(phases);
        phases = ctx.create_host_buffer(phase_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        material_uploads_[f_idx] = 0;
    }
    if (material_uploads_[f_idx] != materials_.revision()) {
        std::memcpy(phases.mapped, materials_.atlas().data(), phase_bytes);
        ctx.flush_buffer(phases, 0, phase_bytes);
        material_uploads_[f_idx] = materials_.revision();
    }
    prepare_fluids(f_idx, cam, sys, settings, count, fb_h * fov_factor);
    VkDeviceSize fluid_bytes = sizeof(SurfaceFluidHeaderGPU)+count*sizeof(SurfaceFluidGPU);
    ctx.flush_buffer(fluid_ssbos_[f_idx], 0, fluid_bytes);
    auto refraction = refraction_.writes(ctx);
    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
        rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
        rhi::PushDescriptorWrite::storage_buffer(2, eclipse_buffer, 0, eclipse_bytes),
        rhi::PushDescriptorWrite::sampled_image(5, textures_.image(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(6, textures_.pages(f_idx)),
        rhi::PushDescriptorWrite::storage_buffer(7, rings_.host_buffer(f_idx), 0, MAX_RING_HOSTS * sizeof(RingHostGPU)),
        rhi::PushDescriptorWrite::sampled_image(9, rings_.shadow_lut(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(10, normals_.image(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(11, normals_.pages(f_idx)),
        rhi::PushDescriptorWrite::sampled_image(12, atmospheres_.surface_lut(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(13, atmospheres_.surface_records(f_idx)),
        rhi::PushDescriptorWrite::sampled_image(16, rings_.ringshine_atlas(f_idx), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(25, phases, 0, phase_bytes),
        rhi::PushDescriptorWrite::storage_buffer(28, fluid_ssbos_[f_idx], 0, fluid_bytes),
        rhi::PushDescriptorWrite::sampled_image(29, cloud_maps_.image(), ctx.sampler_linear_repeat()),
        eclipse_refraction_.write(ctx),
        refraction[0],
        refraction[1],
    };

    if (timings) (*timings)["cpu_surface_setup_ms"] = elapsed_ms(setup_start);
    auto draw_start = timings ? SteadyClock::now() : SteadyClock::time_point{};
    auto sphere_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_spheres") : UINT32_MAX;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sphere_pipeline_.handle);
    ctx.push_descriptors(cmd, sphere_pipeline_, writes);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vbo_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, sphere_ibo_.handle, 0, VK_INDEX_TYPE_UINT32);
    for (uint32_t first = 0; first < count;) {
        if (terrain_.covers(first)) { ++first; continue; }
        uint32_t end = first+1;
        while (end < count && !terrain_.covers(end)) ++end;
        vkCmdDrawIndexed(cmd, sphere_index_count_, end-first, 0, 0, first);
        first = end;
    }

    ctx.end_gpu_timer(cmd, sphere_timer);
    auto terrain_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_terrain_draw") : UINT32_MAX;
    terrain_.render(cmd, ctx, writes);
    ctx.end_gpu_timer(cmd, terrain_timer);

    if (!cloud_bodies_.empty()) {
        auto cloud_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_clouds") : UINT32_MAX;
        std::vector<rhi::PushDescriptorWrite> cloud_writes = {
            rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
            rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
            rhi::PushDescriptorWrite::sampled_image(12, atmospheres_.surface_lut(), ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_buffer(13, atmospheres_.surface_records(f_idx)),
            rhi::PushDescriptorWrite::storage_buffer(28, fluid_ssbos_[f_idx], 0, fluid_bytes),
            rhi::PushDescriptorWrite::sampled_image(29, cloud_maps_.image(), ctx.sampler_linear_repeat()),
        };
        uint32_t mode = std::min(settings.cloud_mode, 1u);
        bool capture = transparency_.active() && cloud_transparency_pipeline_[mode].handle;
        const auto& pipeline = capture ? cloud_transparency_pipeline_[mode] : cloud_pipeline_[mode];
        if (capture) {
            auto capture_writes = transparency_.writes();
            cloud_writes.insert(cloud_writes.end(), capture_writes.begin(), capture_writes.end());
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
        ctx.push_descriptors(cmd, pipeline, cloud_writes);
        vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vbo_.handle, &offset);
        vkCmdBindIndexBuffer(cmd, sphere_ibo_.handle, 0, VK_INDEX_TYPE_UINT32);
        for (uint32_t body : cloud_bodies_) vkCmdDrawIndexed(cmd, sphere_index_count_, 1, 0, 0, body);
        if (!capture) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cloud_depth_pipeline_[mode].handle);
            ctx.push_descriptors(cmd, cloud_depth_pipeline_[mode], cloud_writes);
            for (uint32_t body : cloud_bodies_) vkCmdDrawIndexed(cmd, sphere_index_count_, 1, 0, 0, body);
        }
        ctx.end_gpu_timer(cmd, cloud_timer);
    }

    auto points_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_points") : UINT32_MAX;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipeline_.handle);
    std::vector<rhi::PushDescriptorWrite> point_writes(std::begin(writes), std::end(writes));
    point_writes.push_back(rhi::PushDescriptorWrite::storage_buffer(27, atmospheres_.point_responses(f_idx)));
    ctx.push_descriptors(cmd, point_pipeline_, point_writes);
    ctx.push_constants(cmd, point_pipeline_, uint32_t(0));
    atmosphere_points_ready_ = true;
    for (uint32_t first = 0; first < count;) {
        if (terrain_.covers(first)) { ++first; continue; }
        uint32_t end = first+1;
        while (end < count && !terrain_.covers(end)) ++end;
        vkCmdDraw(cmd, 6, end-first, 0, first);
        first = end;
    }

    ctx.end_gpu_timer(cmd, points_timer);
    auto refraction_rays = refraction_.ray_writes(ctx);
    rhi::PushDescriptorWrite ring_base_writes[] = {
        rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
        rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
        rhi::PushDescriptorWrite::storage_buffer(2, eclipse_buffer, 0, eclipse_bytes),
        refraction[0],
        refraction[1],
        refraction_rays[0], refraction_rays[1], refraction_rays[2],
        eclipse_refraction_.write(ctx),
        rhi::PushDescriptorWrite::storage_buffer(25, phases, 0, phase_bytes),
    };
    auto transparency_writes = transparency_.writes();
    rings_.render(cmd, ctx, settings, ring_base_writes, {ubo.stars_pos_radius, ubo.num_stars},
        transparency_.active() ? std::span{transparency_writes} : std::span<const rhi::PushDescriptorWrite>{});
    if (timings) (*timings)["cpu_surface_draw_ms"] = elapsed_ms(draw_start);
}

void CelestialRenderer::prepare_fluids(uint32_t frame, const Camera& cam, const ephemeris::SystemSnapshot& sys,
                                       const RenderSettings& settings, uint32_t count, float pixel_scale) {
    auto* bytes = static_cast<std::byte*>(fluid_ssbos_[frame].mapped);
    auto* fluids = reinterpret_cast<SurfaceFluidGPU*>(bytes + sizeof(SurfaceFluidHeaderGPU));
    SurfaceFluidHeaderGPU header{};
    header.flags = {settings.oceans_enabled ? 1u : 0u, settings.clouds_enabled ? 1u : 0u, settings.cloud_mode, UINT32_MAX};
    double seconds = sys.time().value * SECONDS_PER_YEAR;
    double best_altitude = std::numeric_limits<double>::infinity();
    int focus = -1;
    glm::dvec3 focus_camera_fixed{0.0};
    std::vector<std::pair<double, uint32_t>> clouds;

    for (uint32_t i = 0; i < count; ++i) {
        const auto& b = sys.body(i);
        const auto& d = b.definition;
        SurfaceFluidGPU f{};
        if (!d.is_star && !d.is_black_hole && (d.ocean.enabled || d.clouds.enabled)) {
            double radius_km = d.radius_au() * AU_TO_KM;
            glm::dvec3 camera_offset = -camera_relative(b.render.position, cam.eye_pos_render());
            double distance = glm::length(camera_offset);
            if (d.ocean.enabled) {
                float mode = d.ocean.mask == "global" ? 3.0f : d.ocean.mask == "height" ? 2.0f : 1.0f;
                f.ocean_deep = glm::vec4(d.ocean.deep_color, mode);
                f.ocean_shallow = glm::vec4(d.ocean.shallow_color, static_cast<float>(d.ocean.foam));
                f.ocean_params = glm::vec4(static_cast<float>(d.ocean.wind_speed_ms),
                    static_cast<float>(cox_munk_variance(d.ocean.wind_speed_ms)),
                    static_cast<float>(radius_km + d.ocean.sea_level_km), static_cast<float>(radius_km));
                double altitude = distance / std::max(d.radius_au(), 1e-15) - 1.0;
                if (settings.oceans_enabled && altitude < best_altitude) {
                    best_altitude = altitude;
                    focus = static_cast<int>(i);
                    glm::dvec3 pole = glm::length(b.render.pole) > 1e-4 ? glm::normalize(b.render.pole) : glm::dvec3(0.0, 1.0, 0.0);
                    focus_camera_fixed = glm::transpose(fluid_rotation(pole, b.render.spin_angle_rad))
                                       * camera_offset * (AU_TO_KM * 1000.0);
                }
            }
            if (d.clouds.enabled) {
                f.cloud_color = glm::vec4(d.clouds.color, 1.0f);
                f.cloud_shape = glm::vec4(static_cast<float>(d.clouds.coverage), static_cast<float>(b.render.cloud_base_altitude_km),
                    static_cast<float>(d.clouds.thickness_km), static_cast<float>(d.clouds.density * d.clouds.extinction_per_km));
                f.cloud_style = glm::vec4(2.0f, static_cast<float>(d.clouds.detail_size_km),
                    static_cast<float>(d.clouds.erosion), static_cast<float>(cloud_maps_.layer(d.clouds.coverage_map)));
                f.cloud_noise.x = static_cast<float>(d.clouds.shape_size_km);
                double drift = std::fmod(d.clouds.wind_speed_ms * seconds / (radius_km * 1000.0), TWO_PI);
                double evolution = std::fmod(d.clouds.evolution * seconds / 3600.0, 1000.0);
                f.cloud_motion = glm::vec4(static_cast<float>(d.clouds.feature_size_km), static_cast<float>(drift),
                    static_cast<float>(evolution), static_cast<float>(radius_km));
                double apparent_px = d.radius_au() / std::max(distance, 1e-15) * pixel_scale;
                if (settings.clouds_enabled && d.clouds.coverage > 0 && d.clouds.density > 0 &&
                    d.clouds.extinction_per_km > 0 && apparent_px > 2.0) clouds.emplace_back(distance, i);
            }
        }
        fluids[i] = f;
    }

    double wind = 7.0, direction = 30.0, gravity = 9.81;
    if (focus >= 0) {
        const auto& d = sys.body(static_cast<uint32_t>(focus)).definition;
        wind = d.ocean.wind_speed_ms;
        direction = d.ocean.wind_direction_deg;
        gravity = surface_gravity(d);
        header.flags.w = static_cast<uint32_t>(focus);
    }
    build_ocean_waves(header, focus_camera_fixed, wind, direction, gravity, seconds);
    std::memcpy(bytes, &header, sizeof(header));

    std::sort(clouds.begin(), clouds.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    cloud_bodies_.clear();
    for (const auto& entry : clouds) cloud_bodies_.push_back(entry.second);
}

void CelestialRenderer::composite_atmospheres(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                                              rhi::Image* physical_color) {
    auto transparency_writes = transparency_.writes();
    auto refraction = refraction_.writes(ctx);
    uint32_t frame = ctx.frame_index();
    auto draw = [&](uint32_t body) {
        if (!atmosphere_points_ready_) return;
        const rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(2, eclipse_ssbos_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(25, material_phases_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(27, atmospheres_.point_responses(frame)),
            eclipse_refraction_.write(ctx), refraction[0], refraction[1]};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, atmosphere_point_pipeline_.handle);
        ctx.push_descriptors(cmd, atmosphere_point_pipeline_, writes);
        ctx.push_constants(cmd, atmosphere_point_pipeline_, uint32_t(1));
        vkCmdDraw(cmd, 6, 1, 0, body);
    };
    atmospheres_.composite(cmd, ctx, color, depth,
        transparency_.active() ? std::span{transparency_writes} : std::span<const rhi::PushDescriptorWrite>{}, draw);
    transparency_.resolve(cmd, ctx, color, depth, physical_color);
}

} // namespace sf::rendering
