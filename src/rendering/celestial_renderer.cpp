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
static_assert(sizeof(SurfaceFluidHeaderGPU) == 32+sizeof(OceanCascadesGPU));
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

struct OceanFocus {
    int body = -1;
    double altitude_m = std::numeric_limits<double>::infinity();
    glm::dvec3 camera_fixed_m{0.0};
};

OceanFocus find_ocean_focus(const Camera& cam, const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, uint32_t count) {
    OceanFocus focus;
    if (!settings.oceans_enabled) return focus;
    double best = std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < count; ++i) {
        const auto& b = sys.body(i);
        const auto& d = b.definition;
        if (d.is_star || d.is_black_hole || !d.ocean.enabled) continue;
        glm::dvec3 camera_offset = -camera_relative(b.render.position, cam.eye_pos_render());
        double distance = glm::length(camera_offset);
        double altitude = distance / std::max(d.radius_au(), 1e-15) - 1.0;
        if (altitude >= best) continue;
        best = altitude;
        glm::dvec3 pole = glm::length(b.render.pole) > 1e-4 ? glm::normalize(b.render.pole) : glm::dvec3(0.0, 1.0, 0.0);
        focus.body = static_cast<int>(i);
        focus.altitude_m = (distance - d.radius_au()) * AU_TO_KM * 1000.0 - d.ocean.sea_level_km * 1000.0;
        focus.camera_fixed_m = glm::transpose(fluid_rotation(pole, b.render.spin_angle_rad)) * camera_offset * (AU_TO_KM * 1000.0);
    }
    return focus;
}

struct SphereVertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

} // namespace

void CelestialRenderer::init(rhi::VkContext& ctx, bool load_surface_assets) {
    build_sphere_mesh(ctx, 64, 32);
    cloud_maps_.init(ctx);
    cloud_volume_.init(ctx);
    ocean_.init(ctx);
    textures_.init(ctx);
    normals_.init(ctx, true, true);
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
        {31, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {39, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {47, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {48, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {49, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {50, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };

    planetshine_maps_ = ctx.create_image_2d(PLANETSHINE_MAP_WIDTH, PLANETSHINE_MAP_HEIGHT, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, PLANETSHINE_MAP_LAYERS, true);
    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        ctx.transition_image(cmd, planetshine_maps_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    rhi::DescriptorLayoutBinding planetshine_bindings[] = {
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
    };
    planetshine_map_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "celestial/planetshine_map.comp",
        planetshine_bindings, sizeof(glm::uvec2));
    for (auto& buffer : planetshine_bodies_)
        buffer = ctx.create_host_buffer(MAX_CELESTIAL_BODIES*sizeof(BodyInstanceGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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
    sphere_desc.capture_statistics = ctx.pipeline_statistics_enabled();

    sphere_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, sphere_desc);

    rhi::GraphicsPipelineDesc cloud_desc = sphere_desc;
    cloud_desc.vert_path = "celestial/cloud.vert";
    cloud_desc.frag_path = "celestial/cloud_sheet.frag";
    cloud_desc.capture_statistics = std::getenv("STELLAR_SHADER_STATS") != nullptr;
    cloud_desc.bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {29, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {39, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    cloud_desc.depth_write = false;
    cloud_desc.premultiplied_blend = true;
    cloud_desc.cull_mode = VK_CULL_MODE_FRONT_BIT;
    cloud_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, cloud_desc);
    if (ctx.fragment_stores_enabled()) {
        auto capture = cloud_desc;
        capture.macros = {{"TRANSPARENCY_CAPTURE", "1"}};
        capture.bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
        capture.bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
        cloud_transparency_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, capture);
        auto inline_desc = cloud_desc;
        inline_desc.macros = {{"CLOUD_SHEET_INLINE", "1"}};
        inline_desc.bindings.push_back({35, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
        inline_desc.bindings.push_back({36, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
        inline_desc.depth_test = false;
        cloud_inline_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, inline_desc);
    }
    cloud_desc.macros = {{"CLOUD_DEPTH_ONLY", "1"}};
    cloud_desc.depth_write = true;
    cloud_desc.premultiplied_blend = false;
    cloud_desc.additive_blend = true;
    cloud_depth_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, cloud_desc);

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
    cloud_volume_.shutdown(ctx);
    cloud_maps_.shutdown(ctx);
    ocean_.shutdown(ctx);
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
    rhi::ShaderCache::destroy_pipeline(ctx, fp32_pipelines_.sphere);
    rhi::ShaderCache::destroy_pipeline(ctx, fp32_pipelines_.cloud);
    rhi::ShaderCache::destroy_pipeline(ctx, fp32_pipelines_.cloud_depth);
    rhi::ShaderCache::destroy_pipeline(ctx, fp32_pipelines_.cloud_transparency);
    rhi::ShaderCache::destroy_pipeline(ctx, fp32_pipelines_.point);
    rhi::ShaderCache::destroy_pipeline(ctx, fp32_pipelines_.atmosphere_point);
    fp32_pipelines_.ready = false;
    rhi::ShaderCache::destroy_pipeline(ctx, planetshine_map_pipeline_);
    ctx.destroy_image(planetshine_maps_);
    for (auto& buffer : planetshine_bodies_) ctx.destroy_buffer(buffer);
    rhi::ShaderCache::destroy_pipeline(ctx, cloud_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, cloud_depth_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, cloud_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, cloud_inline_pipeline_);
    ctx.destroy_image(cloud_layer_color_);
    ctx.destroy_image(cloud_layer_depth_);
    for (auto& [image, serial] : retired_cloud_layers_) ctx.destroy_image(image);
    retired_cloud_layers_.clear();
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
    measure("cpu_normals_prepare_ms", [&] {
        normals_.prepare(cmd, ctx, cam, sys, extent,
            settings.terrain_imported_enabled && settings.terrain_normal_maps_enabled,
            settings.terrain_imported_enabled && settings.terrain_self_shadows_enabled,
            settings.oceans_enabled);
    });
    measure("cpu_rings_prepare_ms", [&] { rings_.prepare(cmd, ctx, cam, sys, count, extent, &refraction_.host()); });
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
    planetshine_albedos_.resize(count);
    for (uint32_t j = 0; j < count; ++j) {
        const auto& b = sys.body(j);
        const auto& d = b.definition;
        glm::dvec3 albedo = glm::clamp(glm::dvec3(d.color_rgb), glm::dvec3(0.0), glm::dvec3(1.0));
        if (settings.surface_textures_enabled) {
            if (auto tex = textures_.mean_albedo(d))
                albedo = glm::clamp(glm::dvec3(*tex), glm::dvec3(0.0), glm::dvec3(1.0));
        }
        if (settings.clouds_enabled && d.clouds.enabled) {
            double cloud_alpha = cloud_maps_.mean_disk_opacity(d.clouds);
            glm::dvec3 cloud_col = glm::clamp(glm::dvec3(d.clouds.color), glm::dvec3(0.0), glm::dvec3(1.0));
            albedo = glm::mix(albedo, cloud_col, cloud_alpha);
        }
        if (settings.show_atmospheres && d.atmosphere.enabled && b.render.atmosphere &&
            !d.atmosphere.pressure_reference && d.type != "Gas Giant" && d.type != "Ice Giant") {
            double aerosol_tau = 0.0;
            if (d.atmosphere.automatic_aerosols) {
                for (const auto& layer : d.atmosphere.aerosols) aerosol_tau += layer.optical_depth_550;
            } else {
                aerosol_tau = double(std::max({d.atmosphere.mie_extinction.r, d.atmosphere.mie_extinction.g,
                    d.atmosphere.mie_extinction.b})) * double(d.atmosphere.mie_scale_height_km);
            }
            if (aerosol_tau < 1.0) {
                glm::dvec3 tau_r = glm::clamp(glm::dvec3(b.render.atmosphere->rayleigh)
                    * std::max(b.render.atmosphere->surface.scale_height_km, 0.0), glm::dvec3(0.0), glm::dvec3(2.0));
                glm::dvec3 r_atm = (0.75 * tau_r) / (glm::dvec3(1.0) + 0.75 * tau_r);
                glm::dvec3 t_atm = glm::dvec3(1.0) / (glm::dvec3(1.0) + 0.75 * tau_r);
                albedo = r_atm + (t_atm * t_atm * albedo) / glm::max(glm::dvec3(1.0) - r_atm * albedo, glm::dvec3(1e-4));
            }
        }
        planetshine_albedos_[j] = glm::vec3(glm::clamp(albedo, glm::dvec3(0.0), glm::dvec3(1.0)));
    }
    planetshine_ = prepare_planetshine(sys, settings.planetshine_enabled && !settings.direct_sunlight_only,
        settings.hdr_enabled, MAX_CELESTIAL_BODIES, &materials_, planetshine_albedos_);
    rings_.update_ring_planetshine(cmd, ctx, cam, sys, settings, count, extent, materials_);
    });
    measure("cpu_atmosphere_prepare_ms", [&] { atmospheres_.prepare(cmd, ctx, cam, sys, settings, extent, atmosphere_minima_, rings_.hosts(), &rings_.shadow_lut(), rings_.shadow_revision(), shadow_diagnostics, planetshine_, &rings_, &refraction_.host()); });
    measure("cpu_planetshine_maps_ms", [&] { bake_planetshine_maps(cmd, ctx, cam, sys, extent, count); });
    measure("cpu_transparency_prepare_ms", [&] {
    bool visible_clouds = false;
    cloud_layer_active_ = false;
    bool inline_clouds = settings.clouds_enabled && settings.cloud_sheet_inline &&
        cloud_inline_pipeline_.handle != VK_NULL_HANDLE;
    if (settings.clouds_enabled) {
        double pixels = extent.height/(2.0*std::tan(cam.fov_deg*DEG_TO_RAD*0.5));
        for (uint32_t i = 0; i < count; ++i) {
            const auto& body = sys.body(i);
            const auto& cloud = body.definition.clouds;
            // Inline clouds blend in the atmosphere composite, so they need no transparent fragments.
            bool composited = inline_clouds && atmospheres_.composites_clouds(i);
            cloud_layer_active_ = cloud_layer_active_ || composited;
            if (visible_clouds || composited) continue;
            double distance = glm::length(camera_relative(body.render.position, cam.eye_pos_render()));
            visible_clouds = cloud.enabled && cloud.coverage > 0 && cloud.density > 0 && cloud.extinction_per_km > 0 &&
                body.definition.radius_au()/std::max(distance, 1e-15)*pixels > 1.5;
        }
    }
    prepare_cloud_layer(cmd, ctx, extent);
    uint32_t visible_rings = settings.show_rings ? rings_.visible_host_count() : 0u;
    bool orbits_active = settings.show_orbits && sys.size() > 1;
    transparency_has_list_ = visible_clouds || orbits_active || visible_rings > 1u;
    transparency_.prepare(cmd, ctx, extent, !settings.transparency_reference_legacy && !sys.empty() &&
        (transparency_has_list_ || visible_rings > 0u));
    });
    measure("cpu_eclipse_refraction_prepare_ms", [&] { eclipse_refraction_.prepare(cmd, ctx, sys, settings); });
    measure("cpu_cloud_volume_prepare_ms", [&] { cloud_volume_.prepare(ctx, cam, sys, settings, cloud_maps_, count); });
    measure("cpu_ocean_prepare_ms", [&] {
    auto focus = find_ocean_focus(cam, sys, settings, count);
    ocean_focus_ = -1;
    if (focus.body < 0 || focus.altitude_m > OCEAN_ACTIVE_ALTITUDE_M) {
        ocean_.reset();
        return;
    }
    const auto& body = sys.body(static_cast<uint32_t>(focus.body));
    const auto& d = body.definition;
    double g = surface_gravity(d);
    double u = std::max(d.ocean.wind_speed_ms, 0.5);
    double peak = std::max(22.0 * std::cbrt(g * g / (u * OCEAN_FETCH_M)), 0.855 * g / u);
    double peak_wavelength_m = TWO_PI * g / (peak * peak);
    double min_footprint_m = focus.altitude_m * 2.0 * std::tan(0.5 * cam.fov_deg * DEG_TO_RAD)
                           / std::max(double(extent.height), 1.0);
    if (min_footprint_m >= 8.0 * peak_wavelength_m) {
        ocean_.reset();
        return;
    }
    ocean_.track(body.revisions.identity, focus.camera_fixed_m);
    uint32_t timer = ctx.begin_gpu_timer(cmd, "gpu_ocean_waves");
    ocean_.update(cmd, ctx, {d.ocean.wind_speed_ms, d.ocean.wind_direction_deg * DEG_TO_RAD, surface_gravity(d)},
                  sys.time().value * SECONDS_PER_YEAR);
    ctx.end_gpu_timer(cmd, timer);
    ocean_focus_ = focus.body;
    });
}

void CelestialRenderer::bake_planetshine_maps(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                                              const ephemeris::SystemSnapshot& sys, VkExtent2D extent, uint32_t count) {
    planetshine_layers_.assign(count, 0u);
    uint32_t frame = ctx.frame_index();
    const auto* records = static_cast<const glm::vec4*>(atmospheres_.surface_records(frame).mapped);
    auto* bodies = static_cast<BodyInstanceGPU*>(planetshine_bodies_[frame].mapped);
    bool atmosphere_changed = atmospheres_.stats().surface_lut_updates > 0;
    double pixels = extent.height/(2.0*std::tan(cam.fov_deg*DEG_TO_RAD*0.5));
    std::vector<glm::uvec2> dirty;
    uint32_t layer = 0;
    for (uint32_t i = 0; i < std::min(count, uint32_t(planetshine_.size())) && layer < PLANETSHINE_MAP_LAYERS; ++i) {
        const auto& shine = planetshine_[i];
        const auto& body = sys.body(i);
        const auto& material = materials_.material(i);
        bool lit = std::any_of(shine.irradiance.begin(), shine.irradiance.end(),
            [](const glm::vec4& v) { return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f; });
        if (!lit || body.definition.is_star || material.grain.x > 0.5f) continue;
        double distance = glm::length(camera_relative(body.render.position, cam.eye_pos_render()));
        if (body.opaque_radius_au()/std::max(distance, 1e-15)*pixels <= 1.0) continue;
        PlanetshineMapKey key{};
        key.body = i;
        key.radius = float(body.opaque_radius_au());
        key.pole_oblateness = glm::vec4(glm::vec3(body.render.pole), float(body.definition.oblateness));
        key.atmosphere = records[i];
        key.shine = shine;
        key.material = material;
        BodyInstanceGPU instance{};
        instance.pos_radius.w = key.radius;
        instance.pole_oblateness = key.pole_oblateness;
        instance.planetshine = shine;
        instance.material = material;
        bodies[i] = instance;
        if (planetshine_keys_.size() <= layer) planetshine_keys_.resize(layer+1);
        if (atmosphere_changed || std::memcmp(&planetshine_keys_[layer], &key, sizeof(key)) != 0) {
            planetshine_keys_[layer] = key;
            dirty.push_back({i, layer});
        }
        planetshine_layers_[i] = ++layer;
    }
    if (dirty.empty()) return;
    ctx.flush_buffer(planetshine_bodies_[frame]);
    uint32_t timer = ctx.begin_gpu_timer(cmd, "planetshine_maps");
    ctx.transition_image(cmd, planetshine_maps_, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, planetshine_map_pipeline_.handle);
    ctx.push_descriptors(cmd, planetshine_map_pipeline_, {
        rhi::PushDescriptorWrite::storage_buffer(1, planetshine_bodies_[frame]),
        rhi::PushDescriptorWrite::storage_image(3, planetshine_maps_),
        rhi::PushDescriptorWrite::sampled_image(12, atmospheres_.surface_lut(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(13, atmospheres_.surface_records(frame))});
    for (auto bake : dirty) {
        ctx.push_constants(cmd, planetshine_map_pipeline_, bake);
        vkCmdDispatch(cmd, PLANETSHINE_MAP_WIDTH/8, PLANETSHINE_MAP_HEIGHT/8, 1);
    }
    ctx.transition_image(cmd, planetshine_maps_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ctx.end_gpu_timer(cmd, timer);
}

void CelestialRenderer::ensure_fp32_pipelines(rhi::VkContext& ctx) {
    if (fp32_pipelines_.ready) return;
    fp32_pipelines_.ready = true;
    const rhi::DescriptorLayoutBinding bindings[] = {
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
        {31, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {39, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {47, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {48, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {49, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {50, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    rhi::GraphicsPipelineDesc sphere_desc{};
    sphere_desc.vert_path = "celestial/sphere.vert";
    sphere_desc.frag_path = "celestial/sphere.frag";
    sphere_desc.bindings.assign(std::begin(bindings), std::end(bindings));
    sphere_desc.vertex_bindings = {{0, sizeof(SphereVertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    sphere_desc.vertex_attributes = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, normal)},
    };
    sphere_desc.color_formats = {VK_FORMAT_R32G32B32A32_SFLOAT};
    sphere_desc.depth_format = VK_FORMAT_D32_SFLOAT;
    sphere_desc.depth_test = true;
    sphere_desc.depth_write = true;
    sphere_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    sphere_desc.cull_mode = VK_CULL_MODE_BACK_BIT;
    sphere_desc.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    fp32_pipelines_.sphere = rhi::ShaderCache::create_graphics_pipeline(ctx, sphere_desc);

    rhi::GraphicsPipelineDesc cloud_desc = sphere_desc;
    cloud_desc.vert_path = "celestial/cloud.vert";
    cloud_desc.frag_path = "celestial/cloud_sheet.frag";
    cloud_desc.bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {29, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {39, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    cloud_desc.depth_write = false;
    cloud_desc.premultiplied_blend = true;
    cloud_desc.cull_mode = VK_CULL_MODE_FRONT_BIT;
    fp32_pipelines_.cloud = rhi::ShaderCache::create_graphics_pipeline(ctx, cloud_desc);
    if (ctx.fragment_stores_enabled()) {
        auto capture = cloud_desc;
        capture.macros = {{"TRANSPARENCY_CAPTURE", "1"}};
        capture.bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
        capture.bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
        fp32_pipelines_.cloud_transparency = rhi::ShaderCache::create_graphics_pipeline(ctx, capture);
    }
    cloud_desc.macros = {{"CLOUD_DEPTH_ONLY", "1"}};
    cloud_desc.depth_write = true;
    cloud_desc.premultiplied_blend = false;
    cloud_desc.additive_blend = true;
    fp32_pipelines_.cloud_depth = rhi::ShaderCache::create_graphics_pipeline(ctx, cloud_desc);

    rhi::GraphicsPipelineDesc point_desc{};
    point_desc.vert_path = "celestial/point_celestial.vert";
    point_desc.frag_path = "celestial/point_celestial.frag";
    point_desc.bindings.assign(std::begin(bindings), std::end(bindings));
    point_desc.bindings.push_back({27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1});
    point_desc.push_constant_size = sizeof(uint32_t);
    point_desc.color_formats = {VK_FORMAT_R32G32B32A32_SFLOAT};
    point_desc.depth_format = VK_FORMAT_D32_SFLOAT;
    point_desc.depth_test = false;
    point_desc.depth_write = false;
    point_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    point_desc.additive_blend = true;
    point_desc.cull_mode = VK_CULL_MODE_NONE;
    fp32_pipelines_.point = rhi::ShaderCache::create_graphics_pipeline(ctx, point_desc);
    point_desc.depth_format = VK_FORMAT_UNDEFINED;
    fp32_pipelines_.atmosphere_point = rhi::ShaderCache::create_graphics_pipeline(ctx, point_desc);
}

void CelestialRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                               const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent,
                               std::map<std::string, double>* timings, bool gpu_diagnostics,
                               VkFormat color_format) {
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
        if (settings.ephem_hide_outside && !b.render.spice_valid) {
            r_au = 0.0f;
            rel_eye = glm::vec3(1e12f);
        }

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
        if (i < planetshine_layers_.size()) inst.planetshine.irradiance[0].w = float(planetshine_layers_[i]);
        inst.material = materials_.material(i);
        inst_ptr[i] = inst;

        if (b.definition.is_star && ubo.num_stars < MAX_SCENE_STARS) {
            uint32_t s = ubo.num_stars++;
            star_bodies[s] = i;
            ubo.stars_pos_radius[s] = glm::vec4(rel_eye, r_au);
            ubo.stars_color_lum[s] = glm::vec4(b.definition.color_rgb, static_cast<float>(std::max(b.definition.lum_eq, 1e-4)));
        }
    }

    bool has_real_stars = ubo.num_stars > 0;
    if (!has_real_stars) {
        ubo.num_stars = 1;
        ubo.stars_pos_radius[0] = glm::vec4(glm::vec3(-eye_pos), static_cast<float>(SOLAR_RADII_TO_AU));
        ubo.stars_color_lum[0] = glm::vec4(1.0f, 0.95f, 0.90f, 1.0f);
    } else {
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t s = 0; s < ubo.num_stars; ++s)
                if (!sys.illuminates(star_bodies[s], i)) inst_ptr[i].surface_flags.y |= 1u << s;
    }
    for (uint32_t h = 0; h < rings_.host_count(); ++h) {
        uint32_t host_body = rings_.hosts()[h].counts_indices.x;
        if (host_body < count) inst_ptr[host_body].surface_flags.z = h + 1u;
    }

    uint32_t ring_count = settings.show_rings ? rings_.host_count() : 0u;
    static_assert(MAX_RING_HOSTS < 32);
    surface_ring_candidate_masks_.assign(ubo.num_ring_hosts > 0u ? count * ubo.num_stars : 0u, 0u);
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
            if (i < count) {
                for (uint32_t j = 0; j < count; ++j) {
                    if (i == j || inst_ptr[j].color_is_star.w > 0.5f) continue;
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
            } else {
                const auto& host = rings_.hosts()[i - count];
                glm::dvec3 ring_center(host.center_min_r);
                glm::dvec3 ring_pole(host.pole_max_r);
                glm::dvec3 ring_u(host.tangent_radius);
                glm::dvec3 ring_v(host.bitangent_oblateness);
                double inner_r = double(host.center_min_r.w);
                double outer_r = double(host.pole_max_r.w);
                std::vector<std::pair<glm::vec4, RingShadowFootprint>> accepted;
                for (uint32_t j = 0; j < count; ++j) {
                    if (inst_ptr[j].color_is_star.w > 0.5f || inst_ptr[j].pos_radius.w <= 0.0f) continue;
                    auto caster = inst_ptr[j].pos_radius;
                    if (j < eclipse_bounds.size() && eclipse_bounds[j].x > 0) {
                        double distance = glm::length(glm::dvec3(caster)-glm::dvec3(receiver_bounds))+receiver_bounds.w;
                        caster.w = float(eclipse_bounds[j].x+distance*eclipse_bounds[j].y);
                    }
                    RingShadowFootprint fp;
                    bool keep = false;
                    if (j == host.counts_indices.x) {
                        keep = eclipse_could_shadow(receiver_bounds, ubo.stars_pos_radius[s], caster);
                        if (keep) eclipse_ring_shadow_footprint(ring_center, ring_pole, inner_r, outer_r, ubo.stars_pos_radius[s], caster, fp);
                    } else {
                        keep = eclipse_ring_shadow_footprint(ring_center, ring_pole, inner_r, outer_r, ubo.stars_pos_radius[s], caster, fp);
                    }
                    if (keep) {
                        eclipse_words_.push_back(j);
                        accepted.emplace_back(caster, fp);
                    }
                }
                uint32_t cand_count = static_cast<uint32_t>(eclipse_words_.size()) - start;
                eclipse_words_[range] = start;
                eclipse_words_[range + 1] = cand_count;
                ring_eclipse_candidate_counts_[(i - count) * ubo.num_stars + s] = cand_count;

                if (cand_count > 0u) {
                    size_t grid_start = eclipse_words_.size();
                    eclipse_words_.resize(grid_start + 1024u, 0u);
                    uint32_t* grid = eclipse_words_.data() + grid_start;
                    double grid_r = std::max(outer_r * 1.04, 1e-12);
                    double cell_w = (2.0 * grid_r) / 32.0;
                    double cell_r = (0.5 * std::sqrt(2.0) + 1e-3) * cell_w;
                    double r_pad = 0.03 * std::max(outer_r, 1e-12);
                    double ann_min = std::max(0.0, inner_r - r_pad);
                    double ann_max = outer_r + r_pad;

                    auto stamp_disk = [&](glm::dvec3 rel_c, double disk_r, uint32_t bit) {
                        double cx_local = glm::dot(rel_c, ring_u);
                        double cy_local = glm::dot(rel_c, ring_v);
                        int ix0 = std::clamp(int(std::floor((cx_local - disk_r + grid_r) / cell_w)), 0, 31);
                        int ix1 = std::clamp(int(std::floor((cx_local + disk_r + grid_r) / cell_w)), 0, 31);
                        int iy0 = std::clamp(int(std::floor((cy_local - disk_r + grid_r) / cell_w)), 0, 31);
                        int iy1 = std::clamp(int(std::floor((cy_local + disk_r + grid_r) / cell_w)), 0, 31);
                        double r2 = disk_r * disk_r;
                        double half_c = 0.5 * cell_w;
                        for (int iy = iy0; iy <= iy1; ++iy) {
                            double cy = -grid_r + (double(iy) + 0.5) * cell_w;
                            double dy = std::max(0.0, std::abs(cy_local - cy) - half_c);
                            for (int ix = ix0; ix <= ix1; ++ix) {
                                double cx = -grid_r + (double(ix) + 0.5) * cell_w;
                                double dx = std::max(0.0, std::abs(cx_local - cx) - half_c);
                                if (dx * dx + dy * dy <= r2) grid[iy * 32 + ix] |= bit;
                            }
                        }
                    };

                    for (uint32_t c = 0; c < cand_count; ++c) {
                        uint32_t bit = c < 31u ? (1u << c) : 0x80000000u;
                        const auto& [caster, fp] = accepted[c];
                        bool wide_or_unbounded = !fp.bounded ||
                            (fp.has_cone && fp.cone_radius > grid_r) ||
                            (fp.has_sphere && fp.sphere_radius > grid_r);
                        if (!wide_or_unbounded) {
                            if (fp.has_sphere) stamp_disk(fp.sphere_center, fp.sphere_radius, bit);
                            if (fp.has_cone) stamp_disk(fp.cone_center, fp.cone_radius, bit);
                        } else {
                            for (int iy = 0; iy < 32; ++iy) {
                                double cy = -grid_r + (double(iy) + 0.5) * cell_w;
                                for (int ix = 0; ix < 32; ++ix) {
                                    double cx = -grid_r + (double(ix) + 0.5) * cell_w;
                                    double d_cent = std::hypot(cx, cy);
                                    if (d_cent + cell_r < ann_min || d_cent - cell_r > ann_max) continue;
                                    glm::vec3 cell_world = glm::vec3(ring_center + ring_u * cx + ring_v * cy);
                                    if (eclipse_could_shadow(glm::vec4(cell_world, float(cell_r)), ubo.stars_pos_radius[s], caster))
                                        grid[iy * 32 + ix] |= bit;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ubo.ring_shadow_params = {static_cast<uint32_t>(eclipse_words_.size()), 0u, 0u,
                              std::bit_cast<uint32_t>(settings.ring_shadow_softness)};
    ubo.eclipse_params.x = settings.eclipse_reference_smoothstep ? 1u : 0u;
    eclipse_words_.insert(eclipse_words_.end(), surface_ring_candidate_masks_.begin(), surface_ring_candidate_masks_.end());
    if (!eclipse_records.empty()) {
        ubo.eclipse_params.y = uint32_t(eclipse_words_.size());
        eclipse_words_.resize(eclipse_words_.size()+eclipse_records.size()*4);
        std::memcpy(eclipse_words_.data()+ubo.eclipse_params.y, eclipse_records.data(), eclipse_records.size_bytes());
    }
    ubo.eclipse_params.z = uint32_t(eclipse_words_.size());
    eclipse_words_.reserve(ubo.eclipse_params.z + size_t(count) * ubo.num_stars * 4u);
    for (uint32_t i = 0; i < count; ++i) {
        const glm::dvec3& body_pos = sys.body(i).render.position.value;
        for (uint32_t s = 0; s < ubo.num_stars; ++s) {
            glm::dvec3 star_pos = has_real_stars ? sys.body(star_bodies[s]).render.position.value : glm::dvec3(0.0);
            glm::dvec3 delta = star_pos - body_pos;
            double dist = glm::length(delta);
            glm::vec3 dir = dist > 1e-12 ? glm::vec3(delta / dist) : glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec4 packed(dir, static_cast<float>(dist));
            auto words = std::bit_cast<std::array<uint32_t, 4>>(packed);
            eclipse_words_.insert(eclipse_words_.end(), words.begin(), words.end());
        }
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
        rhi::PushDescriptorWrite::sampled_image(39, rings_.penumbra_table(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(10, normals_.image(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(11, normals_.pages(f_idx)),
        rhi::PushDescriptorWrite::sampled_image(47, normals_.horizon_image(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(12, atmospheres_.surface_lut(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(13, atmospheres_.surface_records(f_idx)),
        rhi::PushDescriptorWrite::sampled_image(16, rings_.ringshine_atlas(f_idx), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(25, phases, 0, phase_bytes),
        rhi::PushDescriptorWrite::storage_buffer(28, fluid_ssbos_[f_idx], 0, fluid_bytes),
        rhi::PushDescriptorWrite::sampled_image(29, cloud_maps_.image(), ctx.sampler_linear_repeat()),
        rhi::PushDescriptorWrite::sampled_image(31, planetshine_maps_, ctx.sampler_linear_repeat()),
        rhi::PushDescriptorWrite::sampled_image(32, ocean_.moments(), ctx.sampler_linear_repeat()),
        rhi::PushDescriptorWrite::storage_buffer(48, normals_.horizon_bounds()),
        rhi::PushDescriptorWrite::storage_buffer(49, ocean_.mip_summary()),
        rhi::PushDescriptorWrite::sampled_image(50, ocean_.wind_lut(), ctx.sampler_linear_clamp()),
        eclipse_refraction_.write(ctx),
        refraction[0],
        refraction[1],
    };

    bool fp32 = color_format == VK_FORMAT_R32G32B32A32_SFLOAT;
    if (fp32) ensure_fp32_pipelines(ctx);
    const auto& sphere_pipe = fp32 ? fp32_pipelines_.sphere : sphere_pipeline_;
    if (timings) (*timings)["cpu_surface_setup_ms"] = elapsed_ms(setup_start);
    auto draw_start = timings ? SteadyClock::now() : SteadyClock::time_point{};
    auto sphere_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_spheres") : UINT32_MAX;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sphere_pipe.handle);
    ctx.push_descriptors(cmd, sphere_pipe, writes);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vbo_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, sphere_ibo_.handle, 0, VK_INDEX_TYPE_UINT32);
    auto skip_surface_body = [&](uint32_t idx) {
        return terrain_.covers(idx) || (settings.ephem_hide_outside && !sys.body(idx).render.spice_valid);
    };
    for (uint32_t first = 0; first < count;) {
        if (skip_surface_body(first)) { ++first; continue; }
        uint32_t end = first+1;
        while (end < count && !skip_surface_body(end)) ++end;
        vkCmdDrawIndexed(cmd, sphere_index_count_, end-first, 0, 0, first);
        first = end;
    }

    ctx.end_gpu_timer(cmd, sphere_timer);
    body_shader_keys_.assign(count, 0u);
    const auto* atm_records = static_cast<const glm::vec4*>(atmospheres_.surface_records(f_idx).mapped);
    glm::dvec3 star0_pos_d = has_real_stars ? sys.body(star_bodies[0]).render.position.value : glm::dvec3(0.0);
    for (uint32_t i = 0; i < count; ++i) {
        if (!terrain_.covers(i)) continue;
        if (settings.terrain_debug_tiles) {
            body_shader_keys_[i] = terrain::SURFACE_BIT_DEBUG;
            continue;
        }
        uint32_t key = 0u;
        const auto& body = sys.body(i);
        const auto& inst = inst_ptr[i];
        if (atm_records[i].w > 0.5f) key |= terrain::SURFACE_BIT_ATMOSPHERE;
        bool has_ocean = settings.oceans_enabled && body.definition.ocean.enabled &&
            !body.definition.is_star && !body.definition.is_black_hole;
        if (has_ocean) {
            key |= terrain::SURFACE_BIT_OCEAN;
            if (ocean_focus_ == int(i)) key |= terrain::SURFACE_BIT_OCEAN_FFT;
        }
        if (textures_.body_active(i)) key |= terrain::SURFACE_BIT_ALBEDO_MAP;
        uint32_t nflags = normals_.body_flags(i);
        if ((nflags & 1u) != 0u) key |= terrain::SURFACE_BIT_NORMALS;
        if ((nflags & 2u) != 0u) key |= terrain::SURFACE_BIT_HORIZONS;
        bool has_baked_shine = i < planetshine_layers_.size() && planetshine_layers_[i] > 0u;
        if (has_baked_shine) key |= terrain::SURFACE_BIT_PLANETSHINE;
        bool has_eclipse = false;
        bool has_ring_shadow = false;
        for (uint32_t s = 0; s < ubo.num_stars; ++s) {
            uint32_t range = (i * ubo.num_stars + s) * 2u;
            if (eclipse_words_[range + 1u] > 0u) has_eclipse = true;
            if (!surface_ring_candidate_masks_.empty() && surface_ring_candidate_masks_[i * ubo.num_stars + s] != 0u)
                has_ring_shadow = true;
        }
        bool has_ringshine = ubo.ringshine_params.x != 0u && inst.surface_flags.z != 0u;
        if (has_eclipse) key |= terrain::SURFACE_BIT_ECLIPSE_SHADOW;
        if (has_ring_shadow) key |= terrain::SURFACE_BIT_RING_SHADOW;
        if (has_ringshine) key |= terrain::SURFACE_BIT_RINGSHINE;
        bool has_unbaked_shine = false;
        if (!has_baked_shine) {
            for (int k = 0; k < 4; ++k) {
                const auto& irr = inst.planetshine.irradiance[k];
                if (irr.x > 0.0f || irr.y > 0.0f || irr.z > 0.0f) {
                    has_unbaked_shine = true;
                    break;
                }
            }
        }
        bool large_star = false;
        if (ubo.num_stars == 1u) {
            double dist_d = glm::length(star0_pos_d - body.render.position.value);
            double outer_r = double(inst.pos_radius.w) + double(terrain_.relief_extent_au(i));
            if (double(ubo.stars_pos_radius[0].w) / std::max(dist_d - outer_r, 1e-12) >= 0.0049)
                large_star = true;
        }
        if (ubo.num_stars != 1u || inst.material.grain.x >= 0.5f || inst.spin_flags.z > 0.5f ||
            inst.color_is_star.w > 0.5f || has_unbaked_shine || large_star) {
            key |= terrain::SURFACE_BIT_FULL_LIGHTING;
        }
        if ((key & terrain::SURFACE_BIT_FULL_LIGHTING) == 0u && (key & terrain::SURFACE_BIT_ATMOSPHERE) != 0u) {
            double flattening = 1.0 / std::max(1.0 - body.definition.oblateness, 0.05) - 1.0;
            glm::dvec3 cam_rel = -camera_relative(body.render.position, cam.eye_pos_render());
            double d_cam = glm::length(cam_rel);
            double outer_r = double(inst.pos_radius.w) + double(terrain_.relief_extent_au(i));
            if (flattening < 0.0101 && d_cam > outer_r) {
                double theta_cap = std::acos(std::clamp(outer_r / d_cam, 0.0, 1.0));
                glm::dvec3 to_star = star0_pos_d - body.render.position.value;
                double d_star = glm::length(to_star);
                if (d_star > 1e-12) {
                    double sin_r = double(ubo.stars_pos_radius[0].w) / std::max(d_star - outer_r, 1e-12);
                    double theta_sun = std::acos(std::clamp(glm::dot(cam_rel / d_cam, to_star / d_star), -1.0, 1.0));
                    double worst_angle = theta_sun + theta_cap + 2.0 * body.definition.oblateness + 0.05;
                    if (worst_angle < 0.5 * PI && std::cos(worst_angle) > 4.5 * sin_r) {
                        key |= terrain::SURFACE_BIT_ATM_HORIZON_CLEAR;
                    }
                }
            }
        }
        body_shader_keys_[i] = key;
    }
    size_t patch_count = terrain_.prepared_count();
    patch_shader_keys_.resize(patch_count);
    for (size_t p = 0; p < patch_count; ++p) {
        auto info = terrain_.prepared_patch_info(p);
        uint32_t bi = info.body_index;
        uint32_t key = bi < body_shader_keys_.size() ? body_shader_keys_[bi] : 0u;
        if ((key & terrain::SURFACE_BIT_RING_SHADOW) != 0u) {
            bool any_ring = false;
            for (uint32_t s = 0; s < ubo.num_stars && !any_ring; ++s) {
                uint32_t mask = surface_ring_candidate_masks_[bi * ubo.num_stars + s];
                while (mask != 0u) {
                    uint32_t h = uint32_t(std::countr_zero(mask));
                    mask &= mask - 1u;
                    const auto& host = rings_.hosts()[h];
                    if (ring_could_shadow_surface(info.world_bounds, ubo.stars_pos_radius[s],
                            {glm::vec3(host.center_min_r), host.pole_max_r.w},
                            {glm::vec3(host.pole_max_r), host.center_min_r.w},
                            fb_h * fov_factor, settings.ring_shadow_softness)) {
                        any_ring = true;
                        break;
                    }
                }
            }
            if (!any_ring) key &= ~terrain::SURFACE_BIT_RING_SHADOW;
        }
        if ((key & terrain::SURFACE_BIT_ECLIPSE_SHADOW) != 0u) {
            bool any_eclipse = false;
            for (uint32_t s = 0; s < ubo.num_stars && !any_eclipse; ++s) {
                uint32_t range = (bi * ubo.num_stars + s) * 2u;
                uint32_t start = eclipse_words_[range];
                uint32_t cand_count = eclipse_words_[range + 1u];
                for (uint32_t k = 0; k < cand_count; ++k) {
                    uint32_t j = eclipse_words_[start + k];
                    auto caster = inst_ptr[j].pos_radius;
                    if (j < eclipse_bounds.size() && eclipse_bounds[j].x > 0) {
                        double distance = glm::length(glm::dvec3(caster) - glm::dvec3(info.world_bounds)) + info.world_bounds.w;
                        caster.w = float(eclipse_bounds[j].x + distance * eclipse_bounds[j].y);
                    }
                    if (eclipse_could_shadow(info.world_bounds, ubo.stars_pos_radius[s], caster)) {
                        any_eclipse = true;
                        break;
                    }
                }
            }
            if (!any_eclipse) key &= ~terrain::SURFACE_BIT_ECLIPSE_SHADOW;
        }
        if ((key & terrain::SURFACE_BIT_RINGSHINE) != 0u) {
            uint32_t h = inst_ptr[bi].surface_flags.z - 1u;
            const auto& host = rings_.hosts()[h];
            float r_outer_ratio = host.pole_max_r.w / std::max(host.tangent_radius.w, 1e-12f);
            float q = std::clamp(1.0f - float(sys.body(bi).definition.oblateness), 0.05f, 1.0f);
            float relief_ratio = terrain_.relief_extent_au(bi) / std::max(inst_ptr[bi].pos_radius.w, 1e-12f);
            if (info.max_cylindrical_ratio / std::max(q - relief_ratio, 0.05f) < (1.0f / std::max(r_outer_ratio, 1.0f)) - 0.05f)
                key &= ~terrain::SURFACE_BIT_RINGSHINE;
        }
        patch_shader_keys_[p] = key;
    }
    auto terrain_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_terrain_draw") : UINT32_MAX;
    terrain_.render(cmd, ctx, writes, body_shader_keys_, color_format, patch_shader_keys_);
    ctx.end_gpu_timer(cmd, terrain_timer);

    if (!cloud_bodies_.empty()) {
        auto cloud_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_clouds") : UINT32_MAX;
        std::vector<rhi::PushDescriptorWrite> cloud_writes = {
            rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
            rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
            rhi::PushDescriptorWrite::sampled_image(12, atmospheres_.surface_lut(), ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_buffer(13, atmospheres_.surface_records(f_idx)),
            refraction[0],
            refraction[1],
            rhi::PushDescriptorWrite::storage_buffer(28, fluid_ssbos_[f_idx], 0, fluid_bytes),
            rhi::PushDescriptorWrite::sampled_image(29, cloud_maps_.image(), ctx.sampler_linear_repeat()),
            rhi::PushDescriptorWrite::storage_buffer(2, eclipse_buffer, 0, eclipse_bytes),
            rhi::PushDescriptorWrite::storage_buffer(7, rings_.host_buffer(f_idx), 0, MAX_RING_HOSTS * sizeof(RingHostGPU)),
            rhi::PushDescriptorWrite::sampled_image(9, rings_.shadow_lut(), ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(39, rings_.penumbra_table(), ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(16, rings_.ringshine_atlas(f_idx), ctx.sampler_linear_clamp()),
            eclipse_refraction_.write(ctx),
        };
        std::vector<uint32_t> inline_bodies, layered_bodies;
        for (uint32_t body : cloud_bodies_)
            (cloud_layer_active_ && atmospheres_.composites_clouds(body) ? inline_bodies : layered_bodies).push_back(body);
        vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vbo_.handle, &offset);
        vkCmdBindIndexBuffer(cmd, sphere_ibo_.handle, 0, VK_INDEX_TYPE_UINT32);
        if (!inline_bodies.empty()) {
            auto inline_writes = cloud_writes;
            inline_writes.push_back(rhi::PushDescriptorWrite::storage_image(35, cloud_layer_color_));
            inline_writes.push_back(rhi::PushDescriptorWrite::storage_image(36, cloud_layer_depth_));
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cloud_inline_pipeline_.handle);
            ctx.push_descriptors(cmd, cloud_inline_pipeline_, inline_writes);
            for (uint32_t body : inline_bodies) vkCmdDrawIndexed(cmd, sphere_index_count_, 1, 0, 0, body);
        }
        if (!layered_bodies.empty()) {
            const auto& cloud_transp_pipe = fp32 ? fp32_pipelines_.cloud_transparency : cloud_transparency_pipeline_;
            const auto& cloud_pipe = fp32 ? fp32_pipelines_.cloud : cloud_pipeline_;
            const auto& cloud_depth_pipe = fp32 ? fp32_pipelines_.cloud_depth : cloud_depth_pipeline_;
            bool capture = transparency_.active() && cloud_transp_pipe.handle;
            const auto& pipeline = capture ? cloud_transp_pipe : cloud_pipe;
            if (capture) {
                auto capture_writes = transparency_.writes();
                cloud_writes.insert(cloud_writes.end(), capture_writes.begin(), capture_writes.end());
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
            ctx.push_descriptors(cmd, pipeline, cloud_writes);
            for (uint32_t body : layered_bodies) vkCmdDrawIndexed(cmd, sphere_index_count_, 1, 0, 0, body);
            if (!capture) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cloud_depth_pipe.handle);
                ctx.push_descriptors(cmd, cloud_depth_pipe, cloud_writes);
                for (uint32_t body : layered_bodies) vkCmdDrawIndexed(cmd, sphere_index_count_, 1, 0, 0, body);
            }
        }
        ctx.end_gpu_timer(cmd, cloud_timer);
    }

    auto points_timer = profile_gpu ? ctx.begin_gpu_timer(cmd, "gpu_points") : UINT32_MAX;
    const auto& point_pipe = fp32 ? fp32_pipelines_.point : point_pipeline_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipe.handle);
    std::vector<rhi::PushDescriptorWrite> point_writes(std::begin(writes), std::end(writes));
    point_writes.push_back(rhi::PushDescriptorWrite::storage_buffer(27, atmospheres_.point_responses(f_idx)));
    ctx.push_descriptors(cmd, point_pipe, point_writes);
    ctx.push_constants(cmd, point_pipe, uint32_t(0));
    atmosphere_points_ready_ = true;
    for (uint32_t first = 0; first < count;) {
        if (skip_surface_body(first)) { ++first; continue; }
        uint32_t end = first+1;
        while (end < count && !skip_surface_body(end)) ++end;
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
        refraction_.exit_write(ctx),
        eclipse_refraction_.write(ctx),
        rhi::PushDescriptorWrite::storage_buffer(25, phases, 0, phase_bytes),
    };
    auto transparency_writes = transparency_.slot_writes();
    ring_slot_active_ = transparency_.active() && rings_.drawn(settings);
    rings_.render(cmd, ctx, settings, ring_base_writes, {ubo.stars_pos_radius, ubo.num_stars},
        transparency_.active() ? std::span{transparency_writes} : std::span<const rhi::PushDescriptorWrite>{},
        color_format);
    if (timings) (*timings)["cpu_surface_draw_ms"] = elapsed_ms(draw_start);
}

void CelestialRenderer::prepare_fluids(uint32_t frame, const Camera& cam, const ephemeris::SystemSnapshot& sys,
                                       const RenderSettings& settings, uint32_t count, float pixel_scale) {
    auto* bytes = static_cast<std::byte*>(fluid_ssbos_[frame].mapped);
    auto* fluids = reinterpret_cast<SurfaceFluidGPU*>(bytes + sizeof(SurfaceFluidHeaderGPU));
    SurfaceFluidHeaderGPU header{};
    header.flags = {settings.oceans_enabled ? 1u : 0u, settings.clouds_enabled ? 1u : 0u, 0x40u, UINT32_MAX};
    double seconds = sys.time().value * SECONDS_PER_YEAR;
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
                float mode = d.ocean.mask == "specular" ? 4.0f
                           : d.ocean.mask == "global" ? 3.0f
                           : d.ocean.mask == "height" ? 2.0f : 1.0f;
                f.ocean_deep = glm::vec4(d.ocean.deep_color, mode);
                f.ocean_shallow = glm::vec4(d.ocean.shallow_color, static_cast<float>(d.ocean.foam));
                f.ocean_params = glm::vec4(static_cast<float>(d.ocean.wind_speed_ms),
                    static_cast<float>(cox_munk_variance(d.ocean.wind_speed_ms)),
                    static_cast<float>(radius_km + d.ocean.sea_level_km), static_cast<float>(radius_km));
            }
            if (d.clouds.enabled) {
                f.cloud_color = glm::vec4(d.clouds.color, 1.0f);
                f.cloud_shape = glm::vec4(static_cast<float>(d.clouds.coverage), static_cast<float>(b.render.cloud_base_altitude_km),
                    static_cast<float>(d.clouds.thickness_km), static_cast<float>(d.clouds.density * d.clouds.extinction_per_km));
                f.cloud_style = glm::vec4(static_cast<float>(cloud_type_index(d.clouds.type)), static_cast<float>(d.clouds.detail_size_km),
                    static_cast<float>(d.clouds.erosion), static_cast<float>(cloud_maps_.layer(d.clouds.coverage_map)));
                f.cloud_noise.x = static_cast<float>(d.clouds.shape_size_km);
                f.cloud_noise.y = static_cast<float>(atmospheres_.inline_cloud_slot(i));
                double drift = std::fmod(d.clouds.wind_speed_ms * seconds / (radius_km * 1000.0), TWO_PI);
                double evolution = std::fmod(d.clouds.evolution * seconds / 3600.0, 1000.0);
                f.cloud_motion = glm::vec4(static_cast<float>(d.clouds.feature_size_km), static_cast<float>(drift),
                    static_cast<float>(evolution), static_cast<float>(radius_km));
                double apparent_px = d.radius_au() / std::max(distance, 1e-15) * pixel_scale;
                if (settings.clouds_enabled && d.clouds.coverage > 0 && d.clouds.density > 0 &&
                    d.clouds.extinction_per_km > 0 && apparent_px > 1.5) clouds.emplace_back(distance, i);
            }
        }
        fluids[i] = f;
    }

    if (ocean_focus_ >= 0 && static_cast<uint32_t>(ocean_focus_) < count) {
        uint32_t fi = static_cast<uint32_t>(ocean_focus_);
        const auto& d = sys.body(fi).definition;
        double alt_m = std::max(glm::length(camera_relative(sys.body(fi).render.position, cam.eye_pos_render())) * AU_TO_KM * 1000.0
            - double(fluids[fi].ocean_params.z) * 1000.0, 0.0);
        double min_footprint_m = alt_m / std::max(double(pixel_scale), 1.0) * 0.95;
        uint32_t last_c = 0u;
        bool has_nonzero = false;
        for (uint32_t c = 0; c < OCEAN_CASCADE_COUNT; ++c) {
            if (ocean_.cascade_mean_square_slope(c) > 1e-18) {
                last_c = c;
                has_nonzero = true;
            }
        }
        uint32_t first_c = OCEAN_CASCADE_COUNT;
        if (has_nonzero) {
            first_c = 0u;
            while (first_c <= last_c && (ocean_.cascade_mean_square_slope(first_c) <= 1e-18 || min_footprint_m >= OCEAN_CASCADE_SIZES_M[first_c])) {
                ++first_c;
            }
        }
        header.flags.z = (first_c & 7u) | (last_c << 4u);
        header.flags.w = fi;
        header.ocean = ocean_.cascades();
        header.header.z = static_cast<float>(ocean_.residual_mean_square_slope(cox_munk_variance(d.ocean.wind_speed_ms)));
        header.header.w = static_cast<float>(OCEAN_CHOPPINESS * OCEAN_CHOPPINESS * ocean_.mean_square_slope());
    }
    std::memcpy(bytes, &header, sizeof(header));

    std::sort(clouds.begin(), clouds.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    cloud_bodies_.clear();
    for (const auto& entry : clouds) cloud_bodies_.push_back(entry.second);
}

void CelestialRenderer::prepare_cloud_layer(VkCommandBuffer cmd, rhi::VkContext& ctx, VkExtent2D extent) {
    std::erase_if(retired_cloud_layers_, [&](auto& entry) {
        if (entry.second > ctx.completed_serial()) return false;
        ctx.destroy_image(entry.first);
        return true;
    });
    if (!cloud_layer_active_) return;
    if (cloud_layer_depth_.extent.width != extent.width || cloud_layer_depth_.extent.height != extent.height) {
        for (auto* image : {&cloud_layer_color_, &cloud_layer_depth_})
            if (*image) retired_cloud_layers_.emplace_back(*image, ctx.recording_serial());
        cloud_layer_color_ = ctx.create_image_2d(extent.width, extent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        cloud_layer_depth_ = ctx.create_image_2d(extent.width, extent.height, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    // Only depth is cleared: colour is read only where the stored depth lies in front of the scene.
    ctx.transition_image(cmd, cloud_layer_color_, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ctx.transition_image(cmd, cloud_layer_depth_, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, cloud_layer_depth_.handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void CelestialRenderer::composite_atmospheres(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                                              rhi::Image* physical_color) {
    auto transparency_writes = transparency_.writes();
    auto refraction = refraction_.writes(ctx);
    uint32_t frame = ctx.frame_index();
    bool fp32 = color.format == VK_FORMAT_R32G32B32A32_SFLOAT;
    if (fp32) ensure_fp32_pipelines(ctx);
    auto draw = [&](uint32_t body) {
        if (!atmosphere_points_ready_) return;
        const rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(2, eclipse_ssbos_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(25, material_phases_[frame]),
            rhi::PushDescriptorWrite::storage_buffer(27, atmospheres_.point_responses(frame)),
            eclipse_refraction_.write(ctx), refraction[0], refraction[1]};
        const auto& atm_point_pipe = fp32 ? fp32_pipelines_.atmosphere_point : atmosphere_point_pipeline_;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, atm_point_pipe.handle);
        ctx.push_descriptors(cmd, atm_point_pipe, writes);
        ctx.push_constants(cmd, atm_point_pipe, uint32_t(1));
        vkCmdDraw(cmd, 6, 1, 0, body);
    };
    std::array<rhi::PushDescriptorWrite, 2> cloud_layer{};
    if (cloud_layer_active_) {
        for (auto* image : {&cloud_layer_color_, &cloud_layer_depth_})
            ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        cloud_layer = {rhi::PushDescriptorWrite::sampled_image(33, cloud_layer_color_, ctx.sampler_nearest_clamp()),
                       rhi::PushDescriptorWrite::sampled_image(34, cloud_layer_depth_, ctx.sampler_nearest_clamp())};
    }
    auto slot_writes = transparency_.slot_writes();
    bool ring_slot = ring_slot_active_ && transparency_.active();
    bool blended = atmospheres_.composite(cmd, ctx, color, depth,
        transparency_.active() ? std::span{transparency_writes} : std::span<const rhi::PushDescriptorWrite>{}, draw,
        cloud_layer_active_ ? std::span<const rhi::PushDescriptorWrite>{cloud_layer} : std::span<const rhi::PushDescriptorWrite>{},
        transparency_.active() ? std::span{slot_writes}.subspan(2) : std::span<const rhi::PushDescriptorWrite>{}, ring_slot,
        transparency_has_list_);
    bool should_resolve = transparency_.active() && (!blended || transparency_has_list_ || physical_color != nullptr);
    transparency_resolved_last_frame_ = should_resolve;
    if (should_resolve)
        transparency_.resolve(cmd, ctx, color, depth, physical_color, !ring_slot ? 0u : blended ? 1u : 2u);
    ring_slot_active_ = false;
}

} // namespace sf::rendering
