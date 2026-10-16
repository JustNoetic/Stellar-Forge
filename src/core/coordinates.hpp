#pragma once

#include "core/constants.hpp"

namespace sf {

template <typename Frame, typename Kind>
struct FramedVector {
    glm::dvec3 value{0.0};
    explicit constexpr FramedVector(glm::dvec3 v = glm::dvec3{0.0}) noexcept : value(v) {}
};

using EclipticPosition = FramedVector<struct EclipticFrame, struct PositionAU>;
using RenderPosition = FramedVector<struct RenderFrame, PositionAU>;
using EclipticVelocity = FramedVector<EclipticFrame, struct VelocityAUPerYear>;
using RenderVelocity = FramedVector<RenderFrame, VelocityAUPerYear>;

constexpr RenderPosition ecliptic_to_render(EclipticPosition p) noexcept {
    return RenderPosition{ecliptic_to_render(p.value)};
}
constexpr RenderVelocity ecliptic_to_render(EclipticVelocity v) noexcept {
    return RenderVelocity{ecliptic_to_render(v.value)};
}
constexpr EclipticPosition render_to_ecliptic(RenderPosition p) noexcept {
    return EclipticPosition{render_to_ecliptic(p.value)};
}
constexpr EclipticVelocity render_to_ecliptic(RenderVelocity v) noexcept {
    return EclipticVelocity{render_to_ecliptic(v.value)};
}
constexpr glm::dvec3 camera_relative(RenderPosition body, RenderPosition eye) noexcept {
    return body.value - eye.value;
}

}
