/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_probability_picker.h"

#include "adapter/random/RandomAdapter.h"

#include "util/ProbabilityPicker.h"

#include <algorithm>
#include <climits>
#include <numeric>
#include <vector>

#include <range/v3/algorithm/lower_bound.hpp>

TEST_F(ProbabilityPickerTest, testPickTargetDoubleException) {
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector<double>{}, 0.0), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector<double>{}, 1.0), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 1.0 }, -0.056), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0 }, 1.0), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0 }, 1.0), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0, -1.0 }, 1.0), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0, -1.0, 1.0 }, 1.0), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 2.0, 1.421, -1.0, 1.0 }, 1.0), RelearnException);

    std::vector<double> v({ 2.0, 1.421, -1.0, 1.0 });
}

TEST_F(ProbabilityPickerTest, testPickTargetKeyException) {
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector<double>{}, RandomHolderKey::SynapseDeletionFinder), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0 }, RandomHolderKey::SynapseDeletionFinder), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0 }, RandomHolderKey::SynapseDeletionFinder), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0, -1.0 }, RandomHolderKey::SynapseDeletionFinder), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0, -1.0, 1.0 }, RandomHolderKey::SynapseDeletionFinder), RelearnException);
    ASSERT_THROW(auto val = ProbabilityPicker::pick_target(std::vector{ 2.0, 1.421, -1.0, 1.0 }, RandomHolderKey::SynapseDeletionFinder), RelearnException);

    std::vector<double> v({ 2.0, 1.421, -1.0, 1.0 });
}

TEST_F(ProbabilityPickerTest, testPickTargetDouble) {
    const auto number_probability = RandomAdapter::get_random_integer<size_t>(1, 100, mt);

    std::vector<double> prefix_sum(number_probability);
    std::vector<double> probabilities(number_probability);
    for (auto i = 0; i < number_probability; i++) {
        probabilities[i] = RandomAdapter::get_random_double(0.0, 3.0, mt);

        if (i == 0) {
            prefix_sum[0] = 0.0;
            probabilities[0] += eps;
        } else {
            prefix_sum[i] = prefix_sum[i - 1] + probabilities[i - 1];
        }
    }

    const auto sum = ranges::accumulate(probabilities, 0.0);

    for (auto i = 0; i < number_probability; i++) {
        const auto probability = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), sum, mt);
        const auto index = ProbabilityPicker::pick_target(probabilities, probability);

        const auto it = ranges::lower_bound(prefix_sum, probability);
        const auto dist = std::distance(prefix_sum.begin(), it) - 1;

        ASSERT_EQ(index, dist);
    }

    const auto first_index = ProbabilityPicker::pick_target(probabilities, 0.0);
    ASSERT_EQ(first_index, 0);

    const auto first_index_2 = ProbabilityPicker::pick_target(probabilities, std::numeric_limits<double>::min());
    ASSERT_EQ(first_index_2, 0);

    for (auto i = 0; i < 64; i++) {
        const auto probability = RandomAdapter::get_random_double(0.0, sum, mt) + sum;
        const auto index = ProbabilityPicker::pick_target(probabilities, probability);

        ASSERT_EQ(index, number_probability - 1);
    }
}

TEST_F(ProbabilityPickerTest, testPickTargetDoubleManual) {
    auto val1 = ProbabilityPicker::pick_target(std::vector{ 0.0, 2.0, 0.0, 1.421, 1.0 }, 1.0);
    ASSERT_EQ(val1, 1);

    auto val2 = ProbabilityPicker::pick_target(std::vector{ 0.0, 2.0, 0.0, 1.421, 1.0 }, 4.0);
    ASSERT_EQ(val2, 4);

    auto val3 = ProbabilityPicker::pick_target(std::vector{ 1.0, 0.0, 0.0, 1.0 }, 1.0);
    ASSERT_EQ(val3, 0);

    auto val4 = ProbabilityPicker::pick_target(std::vector{ 1.0, 0.0, 0.0, 1.0 }, std::nextafter(1.0, 2.0));
    ASSERT_EQ(val4, 3);
}

TEST_F(ProbabilityPickerTest, testPickTargetDoubleWithZerosInbetween) {
    const auto number_probability = RandomAdapter::get_random_integer<size_t>(1, 100, mt);

    std::vector<double> prefix_sum(number_probability);
    std::vector<double> probabilities(number_probability);
    for (auto i = 0; i < number_probability; i++) {
        probabilities[i] = RandomAdapter::get_random_double(0.0, 3.0, mt);

        if (i == 0) {
            prefix_sum[0] = 0.0;
            probabilities[0] += eps;
        } else {
            if (RandomAdapter::get_random_bool(mt)) {
                probabilities[i] = 0.0;
            }
            prefix_sum[i] = prefix_sum[i - 1] + probabilities[i - 1];
        }
    }

    const auto sum = ranges::accumulate(probabilities, 0.0);

    for (auto i = 0; i < number_probability; i++) {
        const auto probability = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), sum, mt);
        const auto index = ProbabilityPicker::pick_target(probabilities, probability);

        const auto it = ranges::lower_bound(prefix_sum, probability);
        const auto dist = std::distance(prefix_sum.begin(), it) - 1;

        ASSERT_EQ(index, dist);
        ASSERT_GT(probabilities[index], 0.0);
    }

    const auto first_index = ProbabilityPicker::pick_target(probabilities, 0.0);
    ASSERT_EQ(first_index, 0);

    const auto first_index_2 = ProbabilityPicker::pick_target(probabilities, std::numeric_limits<double>::min());
    ASSERT_EQ(first_index_2, 0);

    auto last_index = number_probability - 1;
    for (; last_index >= 0 && probabilities[last_index] == 0.0; last_index--) {
    }

    for (auto i = 0; i < 64; i++) {
        const auto probability = RandomAdapter::get_random_double(0.0, sum, mt) + sum;
        const auto index = ProbabilityPicker::pick_target(probabilities, probability);

        ASSERT_EQ(index, last_index);
    }
}

TEST_F(ProbabilityPickerTest, testPickTargetDoubleWithZerosAtEnd) {
    const auto number_probability = RandomAdapter::get_random_integer<size_t>(1, 100, mt);
    const auto number_zeros = RandomAdapter::get_random_integer<size_t>(1, 100, mt);

    std::vector<double> prefix_sum(number_probability + number_zeros);
    std::vector<double> probabilities(number_probability + number_zeros);
    for (auto i = 0; i < number_probability; i++) {
        probabilities[i] = RandomAdapter::get_random_double(0.0, 3.0, mt);

        if (i == 0) {
            prefix_sum[0] = 0.0;
            probabilities[0] += eps;
        } else {
            prefix_sum[i] = prefix_sum[i - 1] + probabilities[i - 1];
        }
    }

    for (auto i = number_probability; i < number_probability + number_zeros; i++) {
        probabilities[i] = 0.0;
        prefix_sum[i] = prefix_sum[i - 1] + probabilities[i - 1];
    }

    const auto sum = ranges::accumulate(probabilities, 0.0);

    for (auto i = 0; i < number_probability; i++) {
        const auto probability = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), sum, mt);
        const auto index = ProbabilityPicker::pick_target(probabilities, probability);

        const auto it = ranges::lower_bound(prefix_sum, probability);
        const auto dist = std::distance(prefix_sum.begin(), it) - 1;

        ASSERT_EQ(index, dist);
    }

    const auto first_index = ProbabilityPicker::pick_target(probabilities, 0.0);
    ASSERT_EQ(first_index, 0);

    const auto first_index_2 = ProbabilityPicker::pick_target(probabilities, std::numeric_limits<double>::min());
    ASSERT_EQ(first_index_2, 0);

    for (auto i = 0; i < 64; i++) {
        const auto probability = RandomAdapter::get_random_double(0.0, sum, mt) + sum;
        const auto index = ProbabilityPicker::pick_target(probabilities, probability);

        ASSERT_EQ(index, number_probability - 1);
    }
}

TEST_F(ProbabilityPickerTest, testPickTargetKeyWithZeros) {
    const auto number_probability = RandomAdapter::get_random_integer<size_t>(1, 100, mt);
    const auto number_zeros = RandomAdapter::get_random_integer<size_t>(1, 100, mt);

    std::vector<double> probabilities(number_probability + number_zeros);
    for (auto i = 0; i < number_probability; i++) {
        probabilities[i] = RandomAdapter::get_random_double(0.0, 3.0, mt);

        if (i == 0) {
            probabilities[0] += eps;
        } else if (RandomAdapter::get_random_bool(mt)) {
            probabilities[i] = 0.0;
        }
    }

    for (auto i = number_probability; i < number_probability + number_zeros; i++) {
        probabilities[i] = 0.0;
    }

    for (auto i = 0; i < 1024; i++) {
        const auto it = ProbabilityPicker::pick_target(probabilities, RandomHolderKey::SynapseDeletionFinder);
        ASSERT_GE(probabilities[it], 0.0);
    }
}

TEST_F(ProbabilityPickerTest, testPickTargetKeyManual) {
    auto val1 = ProbabilityPicker::pick_target(std::vector{ 1.0, 0.0, 0.0, 0.0 }, 0.5);
    ASSERT_EQ(val1, 0);

    auto val2 = ProbabilityPicker::pick_target(std::vector{ 0.0, 1.0, 0.0, 0.0 }, 0.5);
    ASSERT_EQ(val2, 1);

    auto val3 = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0, 1.0, 0.0 }, 0.5);
    ASSERT_EQ(val3, 2);

    auto val4 = ProbabilityPicker::pick_target(std::vector{ 0.0, 0.0, 0.0, 1.0 }, 0.5);
    ASSERT_EQ(val4, 3);
}
