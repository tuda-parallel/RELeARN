/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_step_parser.h"

#include "io/parser/StepParser.h"

#include <sstream>

TEST_F(StepParserTest, testGenerateFunction1) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    auto function = StepParser::generate_step_check_function(std::vector<Interval>{});

    for (RelearnTypes::step_type step = 0; step < 10000; step++) {
        const auto result_1 = function(step);
        ASSERT_FALSE(result_1) << step;

        const auto random_step = RandomAdapter::get_random_integer<int_type>(min, max, mt);
        const auto result_2 = function(random_step);
        ASSERT_FALSE(result_2) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction2) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    Interval i{ min, max, 1 };

    auto function = StepParser::generate_step_check_function({ i });

    for (RelearnTypes::step_type step = 0; step < 10000; step++) {
        const auto result_1 = function(step);
        ASSERT_TRUE(result_1) << step;

        const auto random_step = RandomAdapter::get_random_integer<int_type>(min, max, mt);
        const auto result_2 = function(random_step);
        ASSERT_TRUE(result_2) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction3) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    Interval i1{ 0, 99, 10 };
    Interval i2{ 100, 999, 10 };
    Interval i3{ 1000, 2689, 10 };
    Interval i4{ 2690, 10000, 10 };

    auto function = StepParser::generate_step_check_function({ i1, i2, i3, i4 });

    for (RelearnTypes::step_type step = 0; step < 20000; step++) {
        const auto result_1 = function(step);
        ASSERT_EQ(result_1, (step <= 10000 && step % 10 == 0)) << step;

        const auto random_step = RandomAdapter::get_random_integer<int_type>(min, max, mt);
        const auto result_2 = function(random_step);
        ASSERT_EQ(result_2, (random_step <= 10000 && random_step % 10 == 0)) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction4) {
    using int_type = Interval::step_type;

    constexpr auto min = 10000;
    constexpr auto max = 90000;

    auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    Interval i1{ 0, 99, 7 };
    Interval i2{ 100, 999, 7 };
    Interval i3{ 1000, 2689, 7 };
    Interval i4{ 2690, 9999, 7 };
    Interval i5{ std::min(begin, end), std::max(begin, end), 11 };

    std::stringstream ss{};
    ss << codify_interval(i1) << ';';
    ss << codify_interval(i2) << ';';
    ss << codify_interval(i3) << ';';
    ss << codify_interval(i4) << ';';
    ss << codify_interval(i5);

    auto function_1 = StepParser::generate_step_check_function({ i1, i2, i3, i4, i5 });
    auto function_2 = StepParser::generate_step_check_function(ss.str());

    for (RelearnTypes::step_type step = 0; step < 90000; step++) {
        const auto result_1 = function_1(step);
        const auto result_2 = function_2(step);
        ASSERT_EQ(result_1, result_2) << step;
    }
}

TEST_F(StepParserTest, testGenerateFunction5) {
    using int_type = Interval::step_type;

    constexpr auto min = 10000;
    constexpr auto max = 90000;

    auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    Interval i1{ 0, 99, 7 };
    Interval i2{ 100, 999, 7 };
    Interval i3{ 400, 2689, 7 };
    Interval i4{ 2690, 9999, 7 };
    Interval i5{ std::min(begin, end), std::max(begin, end), 11 };

    std::stringstream ss{};
    ss << codify_interval(i1) << ';';
    ss << codify_interval(i2) << ';';
    ss << codify_interval(i3) << ';';
    ss << codify_interval(i4) << ';';
    ss << codify_interval(i5);

    auto function_1 = StepParser::generate_step_check_function({ i1, i2, i3, i4, i5 });
    ASSERT_FALSE(function_1.operator bool());
}
