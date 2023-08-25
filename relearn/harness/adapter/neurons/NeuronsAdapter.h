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

#include "adapter/simulation/SimulationAdapter.h"

#include "util/NeuronID.h"
#include "util/Vec3.h"
#include "util/ranges/Functional.hpp"
#include "util/shuffle/shuffle.h"

#include <random>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/zip.hpp>
#include <tuple>
#include <vector>

class NeuronsAdapter {
public:
    static std::vector<std::pair<Vec3d, NeuronID>> generate_random_neurons(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id, std::mt19937& mt) {
        auto ids = NeuronID::range(max_id) | ranges::to_vector | actions::shuffle(mt);

        return ranges::views::zip(
                   ranges::views::generate([&min, &max, &mt]() { return SimulationAdapter::get_random_position_in_box(min, max, mt); }),
                   ids)
            | ranges::to_vector;
    }
};
