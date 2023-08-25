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

#include "Types.h"
#include "util/Interval.h"

#include <vector>

#include <range/v3/algorithm/copy.hpp>

class StimulusAdapter {
public:
    static std::vector<Interval> get_random_non_overlapping_intervals(RelearnTypes::step_type num_intervals, RelearnTypes::step_type num_steps, std::mt19937& mt) {
        std::vector<Interval> intervals{};
        const auto interval_max_size = num_steps / num_intervals / 10;

        for (const auto _ : ranges::views::indices(num_intervals)) {
            std::vector<Interval> test_interval{};
            Interval interval;
            do {
                test_interval.clear();
                ranges::copy(intervals, std::back_inserter(test_interval));
                const auto begin = RandomAdapter::get_random_integer(0U, num_steps - interval_max_size, mt);
                const auto end = RandomAdapter::get_random_integer(begin, begin + interval_max_size, mt);
                interval = { begin, end, 1U };
                test_interval.emplace_back(interval);
            } while (Interval::check_intervals_for_intersection(test_interval));
            intervals.emplace_back(interval);
        }
        return intervals;
    }

    static Interval get_random_interval(RelearnTypes::step_type num_steps, RelearnTypes::step_type frequency, std::mt19937& mt) {
        const auto begin = RandomAdapter::get_random_integer(0U, num_steps, mt);
        const auto end = RandomAdapter::get_random_integer(begin, num_steps, mt);
        return Interval{ begin, end, frequency };
    }
};
