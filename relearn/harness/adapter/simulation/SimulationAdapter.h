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

#include "adapter/random/RandomAdapter.h"

#include "Config.h"
#include "util/Vec3.h"
#include "Types.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <tuple>

class SimulationAdapter {
public:
    constexpr static inline double position_boundary = 10000.0;
    constexpr static unsigned short small_refinement_level = 5;
    constexpr static unsigned short max_refinement_level = Constants::max_lvl_subdomains;

    static RelearnTypes::bounding_box_type get_random_simulation_box_size(std::mt19937& mt) {
        const auto rand_x_1 = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);
        const auto rand_x_2 = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);

        const auto rand_y_1 = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);
        const auto rand_y_2 = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);

        const auto rand_z_1 = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);
        const auto rand_z_2 = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);

        return {
            { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
            { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
        };
    }

    static RelearnTypes::bounding_box_type round_bounding_box(RelearnTypes::bounding_box_type bb) {
        Vec3d min{static_cast<double>(static_cast<int>(bb.get_minimum().get_x())), static_cast<double>(static_cast<int>(bb.get_minimum().get_y())), static_cast<double>(static_cast<int>(bb.get_minimum().get_z()))};
        Vec3d max{static_cast<double>(static_cast<int>(bb.get_maximum().get_x())), static_cast<double>(static_cast<int>(bb.get_maximum().get_y())), static_cast<double>(static_cast<int>(bb.get_maximum().get_z()))};
        return { min,max };
    }

    static double get_random_position_element(std::mt19937& mt) {
        const auto val = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);
        return val;
    }

    static Vec3d get_random_position(std::mt19937& mt) {
        const auto x = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);
        const auto y = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);
        const auto z = RandomAdapter::get_random_double(-position_boundary, +position_boundary, mt);

        return { x, y, z };
    }

    static std::vector<Vec3d> get_random_positions(std::mt19937& mt, RelearnTypes::number_neurons_type num_neurons) {
        std::vector<Vec3d> pos;
        for (auto i = 0; i < num_neurons; i++) {
            pos.push_back(get_random_position(mt));
        }
        return pos;
    }

    static Vec3d get_minimum_position() {
        return { -position_boundary, -position_boundary, -position_boundary };
    }

    static Vec3d get_maximum_position() {
        return { position_boundary, position_boundary, position_boundary };
    }

    static Vec3d get_random_position_in_box(const Vec3d& min, const Vec3d& max, std::mt19937& mt) {
        const auto x = RandomAdapter::get_random_double(min.get_x(), max.get_x(), mt);
        const auto y = RandomAdapter::get_random_double(min.get_y(), max.get_y(), mt);
        const auto z = RandomAdapter::get_random_double(min.get_z(), max.get_z(), mt);

        return { x, y, z };
    }

    static Vec3d get_random_position_in_box(const RelearnTypes::bounding_box_type bb, std::mt19937& mt) {
        const auto& [min,max] = bb;
        return get_random_position_in_box(min,max,mt);
    }

    static std::uint8_t get_random_refinement_level(std::mt19937& mt) noexcept {
        return static_cast<uint8_t>(RandomAdapter::get_random_integer<size_t>(0, max_refinement_level, mt));
    }

    static std::uint8_t get_small_refinement_level(std::mt19937& mt) noexcept {
        return static_cast<uint8_t>(RandomAdapter::get_random_integer<size_t>(0, small_refinement_level, mt));
    }

    static std::uint8_t get_large_refinement_level(std::mt19937& mt) noexcept {
        return static_cast<uint8_t>(RandomAdapter::get_random_integer<size_t>(small_refinement_level + 1, max_refinement_level, mt));
    }
};
