#pragma once

#include "ephemeris/spice_ephemeris.hpp"

void particle_optics_contracts() {
    auto sphere = physics::mie_sphere({1.55, 0}, 2*std::numbers::pi*.525/.6328);
    check(std::abs(sphere.extinction-3.1054255) < 1e-7 && std::abs(sphere.scattering-3.1054255) < 1e-7,
        "Mie efficiencies match the independent published nonabsorbing sphere benchmark");
    auto absorbing = physics::mie_sphere({1.5, -1}, 2);
    check(std::abs(absorbing.electric[0].real()-.546520203397) < 1e-10 &&
        std::abs(std::abs(absorbing.electric[0].imag())-.152373857258) < 1e-10 &&
        std::abs(absorbing.magnetic[0].real()-.389714727888) < 1e-10,
        "Absorbing Mie coefficients match the published benchmark with conjugate convention");
    auto tiny = physics::mie_sphere({1.5, 0}, .001);
    double rayleigh = 8.0/3*std::pow(.001, 4)*std::pow((2.25-1)/(2.25+2), 2);
    check(std::abs(tiny.scattering/rayleigh-1) < 1e-5, "Small particles converge to the Rayleigh cross section");

    AerosolLayer dust{"mineral_dust", .48, 1.5, .3, 12};
    auto prepared = physics::prepare_aerosol(dust);
    check(std::abs(prepared.extinction.g*12-.48) < 1e-7, "550 nm aerosol extinction conserves the specified column");
    glm::dvec3 integral(0);
    for (size_t i = 1; i < physics::AEROSOL_PHASE_SAMPLES; ++i) {
        double a = double(i-1)/(physics::AEROSOL_PHASE_SAMPLES-1), b = double(i)/(physics::AEROSOL_PHASE_SAMPLES-1);
        integral += std::numbers::pi*(std::cos(std::numbers::pi*a*a)-std::cos(std::numbers::pi*b*b))*
            glm::dvec3(prepared.phase[i-1]+prepared.phase[i]);
    }
    check(glm::length(integral-glm::dvec3(1)) < 1e-6, "Each spectral phase table integrates to one over solid angle");
    check(prepared.scattering.r <= prepared.extinction.r && prepared.scattering.b <= prepared.extinction.b &&
        prepared.scattering.b/prepared.extinction.b < prepared.scattering.r/prepared.extinction.r,
        "Absorption is nonnegative and mineral dust absorbs blue more strongly");
    check(prepared.phase[0].b > prepared.phase[0].r, "Particle size and wavelength determine different forward lobes");
    auto crystal = physics::prepare_aerosol({"ammonia_ice", 1, 1.5, .2, 12});
    for (size_t i = 1; i < physics::AEROSOL_PHASE_SAMPLES; ++i)
        check(crystal.phase[i].r <= crystal.phase[i-1].r && crystal.phase[i].b <= crystal.phase[i-1].b,
            "Crystal phase approximation removes spherical-droplet rainbow oscillations");
    dust.effective_radius_um = .05;
    check(physics::prepare_aerosol(dust).phase[0].r < prepared.phase[0].r, "Smaller particles broaden the forward lobe");
    AerosolLayer detached{"organic_haze", .02, .04, .1, 12, 500, 20};
    auto upper = physics::prepare_aerosol(detached);
    double column = 0;
    for (int i = 0; i < 10000; ++i) column += upper.extinction.g*physics::aerosol_density(upper.shape, (i+.5)*.1)*.1;
    check(std::abs(column-.02) < 1e-8, "Detached Gaussian haze conserves its column optical depth");

    AtmosphereDefinition a;
    a.enabled = a.physical = true;
    a.aerosols = {AerosolLayer{"mineral_dust", .48, 1.5, .3, 12}, detached};
    auto json = nlohmann::json(a);
    check(json.at("schema_version") == 2 && !json.contains("rayleigh_scattering") && !json.contains("mie_extinction") &&
        !json.contains("height_km") && !json.contains("automatic_aerosols"), "Physical schema saves inputs without redundant derived optics");
    check(json.get<AtmosphereDefinition>() == a, "Layered physical records roundtrip losslessly");
    auto first = physics::prepare_atmosphere(a, 3e-6, 6378);
    a.pressure_pa *= .5;
    a.composition = {{"CO2", 1}};
    auto second = physics::prepare_atmosphere(a, 3e-6, 6378);
    check(first->aerosols[0].extinction == second->aerosols[0].extinction &&
        first->aerosols[0].phase == second->aerosols[0].phase, "Authored particles do not change with gas pressure or composition");
    a.aerosols.clear();
    check(physics::prepare_atmosphere(a, 3e-6, 6378)->aerosols.empty(), "Explicit clear gas does not invent dust");
    auto clear = nlohmann::json{{"schema_version", 2}, {"pressure_pa", 146921.25}, {"temperature_k", 94},
        {"composition", {{"N2", .9435}, {"CH4", .0565}}}, {"aerosols", nlohmann::json::array()}}.get<AtmosphereDefinition>();
    check(clear.aerosols.empty(), "Version 2 methane atmospheres do not trigger legacy haze inference");
    ui::AtmosphereEditDraft draft;
    draft.values = first->inputs;
    draft.values.automatic_aerosols = false;
    draft.values.aerosols.clear();
    draft.set_gas_percentage("CO2", 90);
    check(!draft.values.automatic_aerosols, "Gas editing preserves custom manual optical ownership");
    draft.values = first->inputs;
    draft.set_gas_percentage("CO2", 90);
    check(draft.values.aerosols == first->inputs.aerosols, "Gas editing preserves authored particle layers");
    a.aerosols = {detached};
    a.aerosols[0].optical_depth_550 = -1;
    check(!valid_atmosphere(a), "Negative aerosol columns are rejected");
    a.aerosols = std::vector<AerosolLayer>(MAX_AEROSOL_LAYERS+1);
    check(!valid_atmosphere(a), "Layer capacity is validated before GPU packing");
    bool rejected = false;
    try { nlohmann::json{{"schema_version", 3}}.get<AtmosphereDefinition>(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Future schemas are rejected instead of silently misread");

    for (const char* name : {"Solar System", "Ephemeris Mode"}) {
        auto data = paths::read_json(paths::external_path("data", "systems", name, "system.json"));
        check(data.has_value(), "Bundled atmosphere fixtures exist");
        auto loaded = SystemManager::parse_system(*data, name);
        for (const char* target : {"Mars", "Titan"}) {
            const auto& atmosphere = loaded.definition.bodies[loaded.definition.find_body(target)].atmosphere;
            check(atmosphere.physical && atmosphere.automatic_aerosols && !atmosphere.aerosols.empty(),
                "Both bundled systems use explicit physical particle columns for Mars and Titan");
            check(atmosphere.aerosols[0].material == (std::string(target) == "Mars" ? "mineral_dust" : "organic_haze"),
                "Bundled worlds use their correct particle materials");
        }
    }
}

void giant_atmosphere_contracts() {
    SystemManager manager;
    auto solar = manager.load_system("Solar System");
    auto ephem = manager.load_system("Ephemeris Mode");
    check(ephem.definition.bodies.size() == solar.definition.bodies.size(), "Ephemeris template reuses the entire Solar System preset");
    for (const auto& b : solar.definition.bodies) {
        const auto& copy = ephem.definition.bodies[ephem.definition.find_body(b.name)];
        check(copy.atmosphere == b.atmosphere && copy.mass_msun == b.mass_msun && copy.radius_rsun == b.radius_rsun &&
            copy.rings == b.rings && copy.color_rgb == b.color_rgb, "Ephemeris physical definitions match the Solar System source");
    }
    const auto& venus = solar.definition.bodies[solar.definition.find_body("Venus")];
    auto at60 = physics::sample_atmosphere(venus.atmosphere, venus.mass_msun, venus.radius_au()*AU_TO_KM, 60);
    check(at60.temperature_k == 260 && at60.pressure_pa > 15000 && at60.pressure_pa < 35000,
        "Venus cloud-level temperature and pressure occupy the reference atmospheric range");
    check(venus.atmosphere.aerosols.size() == 3 && venus.atmosphere.automatic_aerosols,
        "Venus uses explicit sulfuric-acid cloud populations");
    for (const auto& cloud : venus.atmosphere.aerosols)
        check(cloud.material == "sulfate" && cloud.peak_km >= 48 && cloud.peak_km <= 70,
            "Venus cloud populations are elevated above the hot lower atmosphere");
    for (const char* name : {"Jupiter", "Saturn", "Uranus", "Neptune"}) {
        const auto& body = solar.definition.bodies[solar.definition.find_body(name)];
        const auto& a = body.atmosphere;
        double radius = body.radius_au()*AU_TO_KM;
        check(a.pressure_reference && a.pressure_pa == 100000 && a.bottom_altitude_km < 0,
            "Giant gas conditions use the explicit one-bar radius datum");
        check(body.opaque_radius_au() < body.radius_au() && body.valid_height_range(), "Giant opaque boundary lies below the reference radius");
        auto bottom = physics::sample_atmosphere(a, body.mass_msun, radius, a.bottom_altitude_km);
        check(bottom.pressure_pa > 9.8e5 && bottom.pressure_pa < 1.02e6 && bottom.temperature_k > a.temperature_k,
            "Signed hydrostatic thermal profiles reach approximately ten bar at the lower boundary");
        auto prepared = physics::prepare_atmosphere(a, body.mass_msun, radius);
        check(prepared->aerosols.size() == a.aerosols.size() && prepared->aerosols.size() >= 3,
            "Giant cloud and haze populations prepare independently");
        for (size_t i = 0; i < a.aerosols.size(); ++i) {
            const auto& layer = prepared->aerosols[i];
            double column = 0, step = (prepared->height_km-a.bottom_altitude_km)/40000;
            for (int k = 0; k < 40000; ++k)
                column += layer.extinction.g*physics::aerosol_density(layer.shape, a.bottom_altitude_km+(k+.5)*step)*step;
            check(std::abs(column/a.aerosols[i].optical_depth_550-1) < 1e-5,
                "Cloud columns are normalized above the modeled lower boundary, including negative altitudes");
        }
        check(nlohmann::json(a).get<AtmosphereDefinition>() == a, "Pressure datum, signed thermal knots and clouds roundtrip");
        for (double height : {a.bottom_altitude_km, a.bottom_altitude_km*.5, 0.0, 100.0}) {
            double index = std::clamp((std::asinh(height/prepared->profile_domain.z)-prepared->profile_domain.x)*prepared->profile_domain.y, 0.0, 1.0)*255;
            size_t lower = std::min(size_t(index), size_t(254));
            double density = std::exp(std::lerp(double(prepared->log_density[lower].x), double(prepared->log_density[lower+1].x), index-lower));
            auto actual = physics::sample_atmosphere(a, body.mass_msun, radius, height);
            check(std::abs(density*prepared->surface.number_density_m3/actual.number_density_m3-1) < .008,
                "Prepared molecular density resolves the signed giant domain");
        }
    }
#ifdef STELLAR_TEST_EPHEMERIS
    auto manifest_path = paths::external_path("data", "ephemeris.json");
    auto manifest = paths::read_text_file(manifest_path);
    SpiceKernelManager kernels;
    std::string error;
    auto dynamic = kernels.build_ephemeris_system(&solar.definition, solar.definition.epoch_utc, error);
    if (manifest) paths::write_binary_file(manifest_path, manifest->data(), manifest->size());
    check(!dynamic.definition.bodies.empty(), "SPICE system builds from the shared Solar System template");
    for (const char* name : {"Venus", "Jupiter", "Saturn", "Uranus", "Neptune"}) {
        int index = dynamic.definition.find_body(name);
        check(index >= 0, "Shared giant planet is available in SPICE playback");
        check(dynamic.definition.bodies[index].atmosphere == solar.definition.bodies[solar.definition.find_body(name)].atmosphere,
            "SPICE state replacement preserves each canonical atmosphere");
    }
    Simulation simulation;
    check(simulation.replace_system(std::move(dynamic)).accepted && simulation.advance(units::Seconds{60}).accepted,
        "Shared Solar System definitions support actual SPICE playback");
#endif
}
