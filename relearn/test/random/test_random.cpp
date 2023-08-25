/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_random.h"

#include "adapter/random/RandomAdapter.h"

#include "util/Random.h"

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>

#include <cmath>
#include <numeric>
#include <unordered_set>

TEST_F(RandomTest, testSeeding) {
    RandomHolder::seed(RandomHolderKey::Partition, 1234);

    std::vector<unsigned int> golden_numbers(1000);
    for (auto i = 0; i < iterations; i++) {
        golden_numbers[i] = RandomHolder::get_random_uniform_integer<unsigned int>(RandomHolderKey::Partition, 0, 1000);
    }

    for (auto it = 0; it < 5; it++) {
        RandomHolder::seed(RandomHolderKey::Partition, 1234);

        std::vector<unsigned int> repeated_numbers(1000);
        for (auto i = 0; i < iterations; i++) {
            repeated_numbers[i] = RandomHolder::get_random_uniform_integer<unsigned int>(RandomHolderKey::Partition, 0, 1000);
        }

        ASSERT_EQ(golden_numbers, repeated_numbers);
    }
}

TEST_F(RandomTest, testSeedingAll) {
    RandomHolder::seed_all(1234);

    std::vector<unsigned int> golden_numbers(1000);
    for (auto i = 0; i < iterations; i++) {
        golden_numbers[i] = RandomHolder::get_random_uniform_integer<unsigned int>(RandomHolderKey::Partition, 0, 1000);
    }

    for (auto it = 0; it < 5; it++) {
        RandomHolder::seed(RandomHolderKey::Partition, 1234);

        std::vector<unsigned int> repeated_numbers(1000);
        for (auto i = 0; i < iterations; i++) {
            repeated_numbers[i] = RandomHolder::get_random_uniform_integer<unsigned int>(RandomHolderKey::Partition, 0, 1000);
        }

        ASSERT_EQ(golden_numbers, repeated_numbers);
    }
}

TEST_F(RandomTest, testUniformIntegerRange) {
    const auto lower_inclusive = RandomAdapter::get_random_integer<unsigned int>(0, 1000, mt);
    const auto upper_inclusive = RandomAdapter::get_random_integer<unsigned int>(0, 1000, mt) + lower_inclusive;

    for (auto i = 0; i < iterations; i++) {
        const auto random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Partition, lower_inclusive, upper_inclusive);
        ASSERT_LE(lower_inclusive, random_number);
        ASSERT_LE(random_number, upper_inclusive);
    }
}

TEST_F(RandomTest, testUniformIntegerMinimumRange) {
    const auto lower_inclusive = RandomAdapter::get_random_integer<unsigned int>(0, 1000, mt);
    const auto upper_inclusive = lower_inclusive;

    for (auto i = 0; i < iterations; i++) {
        const auto random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Partition, lower_inclusive, upper_inclusive);
        ASSERT_EQ(lower_inclusive, random_number);
    }

    const auto zero = RandomHolder::get_random_uniform_integer(RandomHolderKey::Partition, 0, 0);
    ASSERT_EQ(zero, 0);
}

TEST_F(RandomTest, testUniformIntegerException) {
    const auto lower_inclusive = RandomAdapter::get_random_integer<unsigned int>(1001, 2000, mt);
    const auto upper_inclusive = RandomAdapter::get_random_integer<unsigned int>(0, 1000, mt);

    for (auto i = 0; i < iterations; i++) {
        ASSERT_THROW(const auto random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Partition, lower_inclusive, upper_inclusive);, RelearnException);
    }
}

TEST_F(RandomTest, testUniformDoubleRange) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);
    const auto upper_exclusive = RandomAdapter::get_random_double<double>(0.0001, 1000.0, mt) + lower_inclusive;

    for (auto i = 0; i < iterations; i++) {
        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Partition, lower_inclusive, upper_exclusive);
        ASSERT_LE(lower_inclusive, random_number);
        ASSERT_LT(random_number, upper_exclusive);
    }
}

TEST_F(RandomTest, testUniformDoubleMinimumRange) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);
    const auto upper_exclusive = std::nextafter(lower_inclusive, lower_inclusive * 2.0);

    for (auto i = 0; i < iterations; i++) {
        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Partition, lower_inclusive, upper_exclusive);
        ASSERT_EQ(lower_inclusive, random_number);
    }
}

TEST_F(RandomTest, testUniformDoubleException) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(1000.0001, 2000.0, mt);
    const auto upper_exclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);

    for (auto i = 0; i < iterations; i++) {
        ASSERT_THROW(const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Partition, lower_inclusive, upper_exclusive);, RelearnException);
    }
}

TEST_F(RandomTest, testUniformIndices) {
    const auto number_indices = RandomAdapter::get_random_integer<size_t>(0, 20, mt);
    const auto number_elements = RandomAdapter::get_random_integer<size_t>(0, 20, mt) + number_indices;

    for (auto i = 0; i < iterations; i++) {
        const auto indices = RandomHolder::get_random_uniform_indices(RandomHolderKey::Partition, number_indices, number_elements);
        ASSERT_EQ(indices.size(), number_indices);

        const std::unordered_set<size_t> hashed = indices | ranges::to<std::unordered_set>;

        ASSERT_EQ(indices.size(), hashed.size());

        for (const auto index : indices) {
            ASSERT_LT(index, number_elements);
        }
    }
}

TEST_F(RandomTest, testUniformIndicesAll) {
    const auto number_indices = RandomAdapter::get_random_integer<size_t>(0, 20, mt);

    const auto indices = RandomHolder::get_random_uniform_indices(RandomHolderKey::Partition, number_indices, number_indices);
    ASSERT_EQ(indices.size(), number_indices);

    const std::unordered_set<size_t> hashed = indices | ranges::to<std::unordered_set>;

    ASSERT_EQ(indices.size(), hashed.size());

    for (const auto index : indices) {
        ASSERT_LT(index, number_indices);
    }
}

TEST_F(RandomTest, testUniformIndicesNone) {
    const auto number_elements = RandomAdapter::get_random_integer<size_t>(0, 20, mt);

    for (auto i = 0; i < iterations; i++) {
        const auto indices = RandomHolder::get_random_uniform_indices(RandomHolderKey::Partition, 0, number_elements);
        ASSERT_TRUE(indices.empty());
    }
}

TEST_F(RandomTest, testUniformIndicesException) {
    const auto number_indices = RandomAdapter::get_random_integer<size_t>(1, 20, mt);
    const auto number_elements = RandomAdapter::get_random_integer<size_t>(1, number_indices, mt) - 1;

    ASSERT_THROW(auto val = RandomHolder::get_random_uniform_indices(RandomHolderKey::Partition, number_indices, number_elements), RelearnException);
}

TEST_F(RandomTest, testNormalDoubleRange) {
    const auto mean = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);
    const auto stddev = RandomAdapter::get_random_double<double>(0.0001, 1.0, mt);

    std::vector<double> random_numbers(iterations);
    for (auto i = 0; i < iterations; i++) {
        random_numbers[i] = RandomHolder::get_random_normal_double(RandomHolderKey::Partition, mean, stddev);
    }

    const auto sum = ranges::accumulate(random_numbers, 0.0);
    std::cerr << "Testing normal doubles. Mean: " << mean << " stddev: " << stddev << "\nValue: " << (sum / iterations) << '\n';
}

TEST_F(RandomTest, testFillRange) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);
    const auto upper_exclusive = RandomAdapter::get_random_double<double>(0.0001, 1000.0, mt) + lower_inclusive;

    std::vector<double> random_numbers(iterations);
    RandomHolder::fill(RandomHolderKey::Partition, random_numbers, lower_inclusive, upper_exclusive);

    for (auto i = 0; i < iterations; i++) {
        const auto random_number = random_numbers[i];
        ASSERT_LE(lower_inclusive, random_number);
        ASSERT_LT(random_number, upper_exclusive);
    }
}

TEST_F(RandomTest, testFillMinimumRange) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);
    const auto upper_exclusive = std::nextafter(lower_inclusive, lower_inclusive * 2.0);

    std::vector<double> random_numbers(iterations);
    RandomHolder::fill(RandomHolderKey::Partition, random_numbers, lower_inclusive, upper_exclusive);

    for (auto i = 0; i < iterations; i++) {
        const auto random_number = random_numbers[i];
        ASSERT_EQ(lower_inclusive, random_number);
    }
}

TEST_F(RandomTest, testFillException) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(1000.0001, 2000.0, mt);
    const auto upper_exclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);

    std::vector<double> random_numbers(iterations);
    ASSERT_THROW(RandomHolder::fill(RandomHolderKey::Partition, random_numbers, lower_inclusive, upper_exclusive);, RelearnException);
}

TEST_F(RandomTest, testShuffle) {
    const auto lower_inclusive = RandomAdapter::get_random_double<double>(0.0, 1000.0, mt);
    const auto upper_exclusive = RandomAdapter::get_random_double<double>(0.1, 1000.0, mt) + lower_inclusive;

    std::vector<double> random_numbers(iterations);
    RandomHolder::fill(RandomHolderKey::Partition, random_numbers, lower_inclusive, upper_exclusive);
    std::ranges::sort(random_numbers);

    auto copy = random_numbers;

    RandomHolder::shuffle(RandomHolderKey::Partition, random_numbers);
    std::ranges::sort(random_numbers);

    ASSERT_EQ(copy, random_numbers);
}
