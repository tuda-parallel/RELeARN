/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_interval.h"

#include "adapter/interval/IntervalAdapter.h"
#include "adapter/random/RandomAdapter.h"

#include "io/parser/IntervalParser.h"

#include <sstream>

TEST_F(IntervalTest, testConstruction) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    Interval i{ begin, end, frequency };

    ASSERT_EQ(i.begin, begin);
    ASSERT_EQ(i.end, end);
    ASSERT_EQ(i.frequency, frequency);
}

TEST_F(IntervalTest, testHitsStep) {
    using int_type = Interval::step_type;

    constexpr auto min = 0;
    constexpr auto max = 100000;

    const auto first = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto second = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    const auto begin = first < second ? first : second;
    const auto end = first < second ? second : first;
    const auto frequency = RandomAdapter::get_random_integer<int_type>(1, 200, mt);

    Interval i{ begin, end, frequency };

    for (int_type step = begin; step != end; step++) {
        const auto diff = step - begin;
        if (diff % frequency == 0) {
            ASSERT_TRUE(i.hits_step(step)) << begin << '-' << end << ':' << frequency << '\t' << step;
        } else {
            ASSERT_FALSE(i.hits_step(step)) << begin << '-' << end << ':' << frequency << '\t' << step;
        }
    }
}

TEST_F(IntervalTest, testEquality) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    Interval i{ begin, end, frequency };

    ASSERT_EQ(i, i);
    ASSERT_EQ(i, Interval(begin, end, frequency));
    ASSERT_EQ(Interval(begin, end, frequency), i);

    ASSERT_NE(i, Interval(begin + 1, end, frequency));
    ASSERT_NE(Interval(begin + 1, end, frequency), i);

    ASSERT_NE(i, Interval(begin, end + 1, frequency));
    ASSERT_NE(Interval(begin, end + 1, frequency), i);

    ASSERT_NE(i, Interval(begin, end, frequency + 1));
    ASSERT_NE(Interval(begin, end, frequency + 1), i);

    const auto other_begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto other_end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto other_frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    Interval other_i{ other_begin, other_end, other_frequency };

    if (begin == other_begin && end == other_end && frequency == other_frequency) {
        ASSERT_EQ(i, other_i);
        ASSERT_EQ(other_i, i);
    } else {
        ASSERT_NE(i, other_i);
        ASSERT_NE(other_i, i);
    }
}

TEST_F(IntervalTest, testIntersection) {
    Interval i1{ 10, 20, 1 };
    Interval i2{ 30, 40, 1 };

    ASSERT_FALSE(i1.check_for_intersection(i2));
    ASSERT_FALSE(i2.check_for_intersection(i1));

    Interval i3{ 100, 200, 4 };
    Interval i4{ 200, 300, 6 };

    ASSERT_TRUE(i3.check_for_intersection(i4));
    ASSERT_TRUE(i4.check_for_intersection(i3));

    Interval i5{ 1000, 10000, 5 };
    Interval i6{ 4000, 5000, 3 };

    ASSERT_TRUE(i5.check_for_intersection(i6));
    ASSERT_TRUE(i6.check_for_intersection(i5));

    Interval i7{ 2000, 3000, 70 };
    Interval i8{ 2500, 3500, 80 };

    ASSERT_TRUE(i7.check_for_intersection(i8));
    ASSERT_TRUE(i8.check_for_intersection(i7));
}

TEST_F(IntervalTest, testIntersecions) {
    std::vector<Interval> intervals{};

    intervals.emplace_back(IntervalAdapter::generate_random_interval(mt));
    intervals.emplace_back(IntervalAdapter::generate_random_interval(mt));
    intervals.emplace_back(IntervalAdapter::generate_random_interval(mt));

    auto golden_intersect = false;

    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            if (i == j) {
                continue;
            }

            golden_intersect |= intervals[i].check_for_intersection(intervals[j]);
        }
    }

    const auto all_intersect = Interval::check_intervals_for_intersection(intervals);

    ASSERT_EQ(golden_intersect, all_intersect);
}
