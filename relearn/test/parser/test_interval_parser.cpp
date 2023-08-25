/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_interval_parser.h"

#include "adapter/interval/IntervalAdapter.h"
#include "adapter/random/RandomAdapter.h"

#include "io/parser/IntervalParser.h"

TEST_F(IntervalParserTest, testParseInterval) {
    const auto& [golden_interval, description] = IntervalAdapter::generate_random_interval_description(mt);

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_TRUE(opt_interval.has_value());

    const auto& interval = opt_interval.value();

    ASSERT_EQ(golden_interval.begin, interval.begin);
    ASSERT_EQ(golden_interval.end, interval.end);
    ASSERT_EQ(golden_interval.frequency, interval.frequency);
}

TEST_F(IntervalParserTest, testParseIntervalFail1) {
    const auto& opt_interval = IntervalParser::parse_interval({});

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalFail2) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << std::min(begin, end) << '-' << std::max(begin, end);

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalFail3) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << begin << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalFail4) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalFail5) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency << ':';

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalFail6) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << std::max(begin, end) << '-' << std::min(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervals1) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ';';
        }
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    for (auto i = 0; i < 10; i++) {
        ASSERT_EQ(golden_intervals[i], intervals[i]);
    }
}

TEST_F(IntervalParserTest, testParseInterval2) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}

TEST_F(IntervalParserTest, testParseIntervalsFail1) {
    const auto& intervals = IntervalParser::parse_description_as_intervals({});
    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsFail2) {
    const auto& intervals = IntervalParser::parse_description_as_intervals("sgahkllkrduf,'�.;f�lsa�df::SAfd--dfasdjf45");
    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsFail3) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ',';
        }
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsFail4) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ':';
        }
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsFail5) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    ss << "136546543135";

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}
