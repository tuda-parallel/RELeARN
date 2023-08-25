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
#include "util/Vec3.h"

#include <cstdint>
#include <type_traits>

/**
 * This class represents a MortonCurve in 3D.
 * This class does not perform argument checking. SpaceFillingCurve<Morton> should be used for that.
 */
class Morton {
public:
    using BoxCoordinates = Vec3s;

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr static BoxCoordinates map_1d_to_3d(const std::uint64_t idx) noexcept {
        auto extract_coordinate = [idx](std::uint8_t offset) noexcept {
            constexpr std::uint8_t loop_bound = Constants::max_lvl_subdomains * 3;

            std::uint64_t current_value = 0;

            std::uint8_t coords_bit = 0;

            // Takes every third bit from idx (starting at offset) and copies it to current_value
            for (std::uint8_t idx_bit = offset; idx_bit < loop_bound; idx_bit += 3) {
                const auto old = current_value;
                const auto new_val = copy_bit(idx, idx_bit, old, coords_bit);
                current_value = new_val;
                ++coords_bit;
            }

            return current_value;
        };

        // The index has structure: ...... z2 y2 x2 z1 y1 x1 z0 y0 x0
        const auto x_value = extract_coordinate(0);
        const auto y_value = extract_coordinate(1);
        const auto z_value = extract_coordinate(2);

        return { x_value, y_value, z_value };
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param idx The three dimensional index
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr std::uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept {
        std::uint64_t result = 0;
        for (std::uint8_t i = 0; i < refinement_level; ++i) {
            const auto& [x, y, z] = coords;

            const auto x_bit = select_bit(x, i);
            const auto y_bit = select_bit(y, i);
            const auto z_bit = select_bit(z, i);

            std::uint64_t block = (z_bit << 2U) + (y_bit << 1U) + x_bit;

            result |= block << (3U * i);
        }

        return result;
    }

    /**
     * @brief Returns the current refinement level
     * @return The current refinement level
     */
    [[nodiscard]] constexpr std::uint8_t get_random_refinement_level() const noexcept {
        return this->refinement_level;
    }

    /**
     * @brief Sets the new refinement level
     * @param refinement_level The new refinement level
     */
    constexpr void set_refinement_level(const std::uint8_t refinement_level) noexcept {
        this->refinement_level = refinement_level;
    }

private:
    [[nodiscard]] constexpr static std::uint64_t set_bit(const std::uint64_t variable, const std::uint8_t bit) noexcept {
        const auto val = variable | (static_cast<std::uint64_t>(1) << bit);
        return val;
    }

    [[nodiscard]] constexpr static std::uint64_t unset_bit(const std::uint64_t variable, const std::uint8_t bit) noexcept {
        const auto val = variable & ~(static_cast<std::uint64_t>(1) << bit);
        return val;
    }

    [[nodiscard]] constexpr static std::uint64_t select_bit(const std::uint64_t number, const std::uint8_t bit) noexcept {
        return ((number >> bit) & 1U);
    }

    [[nodiscard]] constexpr static std::uint64_t copy_bit(const std::uint64_t source, const std::uint8_t source_bit, const std::uint64_t destination, const std::uint8_t destination_bit) noexcept {
        // A simpler solution might be:
        // destination ^= (-select_bit(source, source_bit) ^ destination) & (1 << destination_bit);

        const std::uint64_t bit_in_source = select_bit(source, source_bit);
        if (1 == bit_in_source) {
            const auto return_value = set_bit(destination, destination_bit);
            return return_value;
        }

        const auto return_value = unset_bit(destination, destination_bit);
        return return_value;
    }

    std::uint8_t refinement_level{ 0 };
};

/**
 * This class represents a space filling curve in 3D.
 * It is parameterized by an actual implementation T, which must be nothrow {constructible, copy constructible, move constructible}.
 */
template <class T>
class SpaceFillingCurve {
    static_assert(std::is_nothrow_constructible_v<T>);
    static_assert(std::is_nothrow_copy_constructible_v<T>);
    static_assert(std::is_nothrow_move_constructible_v<T>);

public:
    using BoxCoordinates = Vec3s;

    /**
     * @brief Constructs a new instance of a space filling curve with the desired refinement level
     * @param refinement_level The desired refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    explicit SpaceFillingCurve(const std::uint8_t refinement_level = 0) {
        set_refinement_level(refinement_level);
    }

    /**
     * @brief Returns the current refinement level
     * @return The current refinement level
     */
    [[nodiscard]] size_t get_random_refinement_level() const noexcept {
        return curve.get_random_refinement_level();
    }

    /**
     * @brief Sets the new refinement level
     * @param refinement_level The new refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    void set_refinement_level(const std::uint8_t refinement_level) {
        // With 64-bit keys we can only support 20 subdivisions per
        // dimension (i.e, 2^20 boxes per dimension)
        RelearnException::check(refinement_level <= Constants::max_lvl_subdomains,
            "SpaceFillingCurve::set_refinement_level:Number of subdivisions is too large: {} vs {}", refinement_level, Constants::max_lvl_subdomains);

        curve.set_refinement_level(refinement_level);
    }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @return The three dimensional index
     */
    [[nodiscard]] BoxCoordinates map_1d_to_3d(const std::uint64_t idx) const noexcept {
        return curve.map_1d_to_3d(idx);
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param idx The three dimensional index
     * @return The one dimensional index
     */
    [[nodiscard]] std::uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept {
        return curve.map_3d_to_1d(coords);
    }

private:
    T curve{};
};
