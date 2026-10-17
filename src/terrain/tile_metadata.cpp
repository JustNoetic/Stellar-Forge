#include "terrain/tile_metadata.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace sf::terrain {
namespace {

struct Range { double low, high; };

Range sine_range(double phase, double radius) {
    constexpr double pi = std::numbers::pi, period = 2.0 * pi;
    if (radius >= pi) return {-1.0, 1.0};
    double low = phase - radius, high = phase + radius;
    double a = std::sin(low), b = std::sin(high);
    Range result{std::min(a, b), std::max(a, b)};
    if (std::ceil((low - pi * .5) / period) <= std::floor((high - pi * .5) / period)) result.high = 1.0;
    if (std::ceil((low + pi * .5) / period) <= std::floor((high + pi * .5) / period)) result.low = -1.0;
    return result;
}

Range product(Range a, Range b) {
    std::array values{a.low * b.low, a.low * b.high, a.high * b.low, a.high * b.high};
    auto [low, high] = std::minmax_element(values.begin(), values.end());
    return {*low, *high};
}

double magnitude(Range range) { return std::max(std::abs(range.low), std::abs(range.high)); }

}

TileMetadata tile_metadata(TileId tile) noexcept {
    struct Wave { glm::dvec3 a, b; double weight; };
    static const std::array waves{
        Wave{{18, 5, 0}, {0, 0, 14}, .55},
        Wave{{0, 37, -11}, {23, 0, 0}, .30},
        Wave{{19, 0, 61}, {0, 47, 0}, .15},
    };
    double width = std::ldexp(2.0, -int(tile.level));
    auto direction = glm::normalize(cube_vector(tile.face, -1.0 + (tile.x + .5) * width, -1.0 + (tile.y + .5) * width));
    double radius = width / std::sqrt(2.0);
    TileMetadata result{0, 0, 0, 0};
    for (const auto& wave : waves) {
        double a = glm::length(wave.a), b = glm::length(wave.b);
        double p = glm::dot(wave.a, direction), q = glm::dot(wave.b, direction);
        auto sp = sine_range(p, a * radius), cp = sine_range(p + std::numbers::pi * .5, a * radius);
        auto sq = sine_range(q, b * radius), cq = sine_range(q + std::numbers::pi * .5, b * radius);
        auto height = product(sp, cq);
        result.minimum_height += wave.weight * height.low;
        result.maximum_height += wave.weight * height.high;
        result.gradient_bound += wave.weight * (a * magnitude(product(cp, cq)) + b * magnitude(product(sp, sq)));
        result.curvature_bound += wave.weight * ((a * a + b * b) * magnitude(height) + 2.0 * a * b * magnitude(product(cp, sq)));
    }
    result.minimum_height -= RELIEF_ROUNDING;
    result.maximum_height += RELIEF_ROUNDING;
    return result;
}

TileBounds surface_bounds(TileId tile, double polar_ratio, double height_amplitude, const TileMetadata& metadata) noexcept {
    double width = std::ldexp(2.0, -int(tile.level));
    auto direction = glm::normalize(cube_vector(tile.face, -1.0 + (tile.x + .5) * width, -1.0 + (tile.y + .5) * width));
    auto scale = glm::dvec3(1, polar_ratio, 1);
    auto normal = glm::normalize(direction / scale);
    double low = metadata.minimum_height * height_amplitude, high = metadata.maximum_height * height_amplitude;
    double middle = (low + high) * .5;
    double radius = width / std::sqrt(2.0);
    double normal_variation = std::min(2.0, radius / polar_ratio);
    double maximum_height = std::max(std::abs(low), std::abs(high));
    double local_radius = radius + (high - low) * .5 + maximum_height * normal_variation;
    if (local_radius >= radius + maximum_height) return {direction * scale, radius + maximum_height};
    return {direction * scale + normal * middle, local_radius};
}

double surface_error_coefficient(double polar_ratio, double height_amplitude, const TileMetadata& metadata) noexcept {
    double maximum_height = std::max(std::abs(metadata.minimum_height), std::abs(metadata.maximum_height));
    return 4.0 + height_amplitude * (metadata.curvature_bound + (4.0 + 2.0 / polar_ratio) * metadata.gradient_bound
                                  + 4.0 * maximum_height / (polar_ratio * polar_ratio));
}

TileMetadata TileMetadataCache::get(TileId tile) {
    auto key = tile.key();
    if (auto found = entries_.find(key); found != entries_.end()) {
        ages_.splice(ages_.begin(), ages_, found->second.age);
        return found->second.metadata;
    }
    ++builds_;
    auto metadata = tile_metadata(tile);
    if (capacity_ == 0) return metadata;
    if (entries_.size() == capacity_) {
        entries_.erase(ages_.back());
        ages_.pop_back();
    }
    ages_.push_front(key);
    entries_.emplace(key, Entry{metadata, ages_.begin()});
    return metadata;
}

void TileMetadataCache::clear() {
    entries_.clear();
    ages_.clear();
    builds_ = 0;
}

}
