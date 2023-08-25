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

#include "util/Interval.h"

#include <climits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

class IntervalAdapter {
public:
    static Interval generate_random_interval(std::mt19937& mt) {
        using int_type = Interval::step_type;

        constexpr auto min = std::numeric_limits<int_type>::min();
        constexpr auto max = std::numeric_limits<int_type>::max();

        const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
        const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
        const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

        return Interval{ std::min(begin, end), std::max(begin, end), frequency };
    }

    static std::string codify_interval(const Interval& interval) {
        std::stringstream ss{};
        ss << interval.begin << '-' << interval.end << ':' << interval.frequency;
        return ss.str();
    }

    static std::pair<Interval, std::string> generate_random_interval_description(std::mt19937& mt) {
        auto interval = generate_random_interval(mt);
        auto description = codify_interval(interval);
        return { std::move(interval), std::move(description) };
    }
};
