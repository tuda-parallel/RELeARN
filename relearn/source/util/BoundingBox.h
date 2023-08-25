#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "util/RelearnException.h"
#include "util/Utility.h"

#include "fmt/ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <type_traits>
#include <utility>

/**
 * @brief A Vec3 holds three different values of type T and allows computations via operators.
 * @tparam T The type that shall be stored inside this class. Is required to fulfill std::is_arithmetic_v<T>
 */
template <typename T>
class BoundingBox {
public:
    using value_type = T;

    BoundingBox() = default;

    /**
     * @brief Constructs a new instance and initializes all values
     * @param _x The value for x
     * @param _y The value for y
     * @param _z The value for z
     */
    constexpr BoundingBox(const T& minimum, const T& maximum) noexcept
        : minimum(minimum)
        , maximum(maximum) {
    }

    constexpr BoundingBox(const BoundingBox<T>& other) = default;
    constexpr BoundingBox<T>& operator=(const BoundingBox<T>& other) = default;

    constexpr BoundingBox(BoundingBox<T>&& other) noexcept = default;
    constexpr BoundingBox<T>& operator=(BoundingBox<T>&& other) noexcept = default;

    [[nodiscard]] T get_minimum() const {
        return minimum;
    }

    [[nodiscard]] T get_maximum() const {
        return maximum;
    }

    /*[[nodiscard]] bool contains(const T & point) const {
        return minimum <= point && point <= maximum;
    }

    [[nodiscard]] bool contains(const BoundingBox<T>& other) const {
        return contains(other.minimum) && contains(other.maximum);
    }*/

    friend std::ostream& operator<<(std::ostream& output_stream, const BoundingBox<T>& bb) {
        const auto& [min, max] = bb;
        output_stream << min << " - " << max;

        return output_stream;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return minimum;
        }
        if constexpr (Index == 1) {
            return maximum;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto const& get() const& {
        if constexpr (Index == 0) {
            return minimum;
        }
        if constexpr (Index == 1) {
            return maximum;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return std::move(minimum);
        }
        if constexpr (Index == 1) {
            return std::move(maximum);
        }
    }

    [[nodiscard]] constexpr bool operator==(const BoundingBox<T>& other) const noexcept {
        return minimum == other.minimum && maximum == other.maximum;
    }

    /*
    [[nodiscard]] bool equals_eps(const BoundingBox<T>& other) {
        const auto diff_min = std::abs(other.minimum-minimum);
        const auto diff_max = std::abs(other.maximum-maximum);
        return diff_max.calculate_1_norm() + diff_min.calculate_1_norm() < Constants::eps;
    }*/

    [[nodiscard]] constexpr bool operator<(const BoundingBox<T>& other) const noexcept {
        if (minimum < other.minimum) {
            return true;
        }
        if (minimum == other.minimum && maximum < other.maximum) {
            return true;
        }
        return false;
    }

private:
    T minimum;
    T maximum;
};

template <typename T>
struct fmt::formatter<BoundingBox<T>> : ostream_formatter { };

namespace std {
template <typename T>
struct tuple_size<::BoundingBox<T>> {
    static constexpr size_t value = 2;
};

template <typename T>
struct tuple_element<0, ::BoundingBox<T>> {
    using type = typename BoundingBox<T>::value_type;
};

template <typename T>
struct tuple_element<1, ::BoundingBox<T>> {
    using type = typename BoundingBox<T>::value_type;
};
};