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

#include <random>

class FMMAdapter {
public:
    static Vec3u get_random_multi_index(std::mt19937& mt) {
        const auto x = RandomAdapter::get_random_integer<Vec3u::value_type>(0, Constants::p, mt);
        const auto y = RandomAdapter::get_random_integer<Vec3u::value_type>(0, Constants::p, mt);
        const auto z = RandomAdapter::get_random_integer<Vec3u::value_type>(0, Constants::p, mt);

        return Vec3u{ x, y, z };
    }
};
