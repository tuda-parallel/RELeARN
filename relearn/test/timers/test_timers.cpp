/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_timers.h"

#include "adapter/random/RandomAdapter.h"
#include "adapter/timers/TimersAdapter.h"

#include "util/Timers.h"

#include <thread>

TEST_F(TimersTest, testReset) {
    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed.count(), 0);
}

TEST_F(TimersTest, testStartStop) {
    using namespace std::chrono_literals;

    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    Timers::start(region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop(region);

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed.count(), 0);
}

TEST_F(TimersTest, testMultipleStartStop) {
    using namespace std::chrono_literals;

    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    const auto outer_iterations = RandomAdapter::get_random_integer<unsigned int>(2, 10, mt);

    for (auto outer = 0U; outer < outer_iterations; outer++) {
        const auto start_iterations = RandomAdapter::get_random_integer<unsigned int>(2, 10, mt);
        for (auto start = 0U; start < start_iterations; start++) {
            Timers::start(region);
        }

        const auto end_iterations = RandomAdapter::get_random_integer<unsigned int>(2, 10, mt);
        for (auto end = 0U; end < end_iterations; end++) {
            Timers::stop(region);
        }
    }

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed.count(), 0);
}

TEST_F(TimersTest, testResetZero) {
    using namespace std::chrono_literals;

    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    Timers::start(region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop(region);
    Timers::add_start_stop_diff_to_elapsed(region);

    Timers::reset_elapsed(region);

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed.count(), 0);
}

TEST_F(TimersTest, testResetZero2) {
    using namespace std::chrono_literals;

    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    Timers::start(region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop_and_add(region);

    Timers::reset_elapsed(region);

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed.count(), 0);
}

TEST_F(TimersTest, testAdd) {
    using namespace std::chrono_literals;

    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    Timers::start(region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop(region);
    Timers::add_start_stop_diff_to_elapsed(region);

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_GE(elapsed.count(), 10000);

    std::this_thread::sleep_for(10000ns);
    const auto elapsed_again = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed_again, elapsed);
}

TEST_F(TimersTest, testAdd2) {
    using namespace std::chrono_literals;

    const auto region = TimersAdapter::get_random_timer_region(mt);
    Timers::reset_elapsed(region);

    Timers::start(region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop_and_add(region);

    const auto elapsed = Timers::get_elapsed(region);
    ASSERT_GE(elapsed.count(), 10000);

    std::this_thread::sleep_for(10000ns);
    const auto elapsed_again = Timers::get_elapsed(region);
    ASSERT_EQ(elapsed_again, elapsed);
}

TEST_F(TimersTest, testNonInterference) {
    using namespace std::chrono_literals;

    auto get_two_timers = [this]() {
        auto first = TimersAdapter::get_random_timer_region(mt);
        while (true) {
            auto second = TimersAdapter::get_random_timer_region(mt);
            if (second != first) {
                return std::pair{ first, second };
            }
        }
    };
    const auto [first_region, second_region] = get_two_timers();

    Timers::reset_elapsed(first_region);
    Timers::reset_elapsed(second_region);

    Timers::start(first_region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop_and_add(first_region);

    const auto elapsed_0 = Timers::get_elapsed(first_region);
    ASSERT_GE(elapsed_0.count(), 10000);

    const auto elapsed_1 = Timers::get_elapsed(second_region);
    ASSERT_EQ(elapsed_1.count(), 0);

    Timers::reset_elapsed(second_region);

    const auto elapsed_2 = Timers::get_elapsed(first_region);
    ASSERT_EQ(elapsed_0, elapsed_2);

    Timers::start(second_region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop_and_add(second_region);

    const auto elapsed_3 = Timers::get_elapsed(first_region);
    ASSERT_EQ(elapsed_0, elapsed_3);

    Timers::start(second_region);
    std::this_thread::sleep_for(10000ns);
    Timers::stop(second_region);
    Timers::add_start_stop_diff_to_elapsed(second_region);

    const auto elapsed_4 = Timers::get_elapsed(first_region);
    ASSERT_EQ(elapsed_0, elapsed_4);
}

TEST_F(TimersTest, testNoThrowWallTime) {
    ASSERT_NO_THROW(auto val = Timers::wall_clock_time());
}
