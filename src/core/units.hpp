#pragma once

#include "core/constants.hpp"
#include <compare>

namespace sf::units {

template <typename Tag>
struct Quantity {
    double value = 0.0;
    explicit constexpr Quantity(double v = 0.0) noexcept : value(v) {}
    auto operator<=>(const Quantity&) const = default;
};

using Seconds = Quantity<struct SecondsTag>;
using Years = Quantity<struct YearsTag>;
using AstronomicalUnits = Quantity<struct AstronomicalUnitsTag>;
using Kilometres = Quantity<struct KilometresTag>;
using SolarMasses = Quantity<struct SolarMassesTag>;

constexpr Years to_years(Seconds time) noexcept { return Years{time.value / SECONDS_PER_YEAR}; }
constexpr Seconds to_seconds(Years time) noexcept { return Seconds{time.value * SECONDS_PER_YEAR}; }
constexpr AstronomicalUnits to_au(Kilometres distance) noexcept { return AstronomicalUnits{distance.value / AU_TO_KM}; }
constexpr Kilometres to_km(AstronomicalUnits distance) noexcept { return Kilometres{distance.value * AU_TO_KM}; }

}
