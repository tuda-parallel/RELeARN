/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_mpi_rank.h"

#include "adapter/random/RandomAdapter.h"

#include "util/MPIRank.h"

#include <climits>
#include <concepts>
#include <type_traits>
#include <utility>

template <typename T>
concept verifier_get_rank = requires(T a) {
    { static_cast<T&>(a).get_rank() } -> std::same_as<int>;
    { static_cast<const T&>(a).get_rank() } -> std::same_as<int>;
    { std::move(a).get_rank() } -> std::same_as<int>;
};

template <typename T>
concept verifier_is_initialized = requires(T a) {
    { static_cast<T&>(a).is_initialized() } -> std::same_as<bool>;
    { static_cast<const T&>(a).is_initialized() } -> std::same_as<bool>;
    { std::move(a).is_initialized() } -> std::same_as<bool>;
};

static_assert(verifier_get_rank<MPIRank>, "MPIRank::get_rank() does not return int");
static_assert(verifier_is_initialized<MPIRank>, "MPIRank::is_initialized() does not return bool");

TEST_F(MPIRankTest, testUnitializedRank) {
    const auto uninitialized_rank = MPIRank::uninitialized_rank();
    ASSERT_FALSE(uninitialized_rank.is_initialized());
    ASSERT_THROW(auto val = uninitialized_rank.get_rank(), RelearnException);
}

TEST_F(MPIRankTest, testRootRank) {
    const auto root_rank = MPIRank::root_rank();
    ASSERT_TRUE(root_rank.is_initialized());
    ASSERT_EQ(root_rank.get_rank(), 0);
}

TEST_F(MPIRankTest, testDefaultConstructor) {
    const auto default_rank = MPIRank{};
    ASSERT_FALSE(default_rank.is_initialized());
    ASSERT_THROW(auto val = default_rank.get_rank(), RelearnException);
}

TEST_F(MPIRankTest, testConstructor) {
    const auto max_rank = 1 << 30;

    auto generator = [this] {
        constexpr auto min = std::numeric_limits<int>::min();
        constexpr auto max = std::numeric_limits<int>::max();
        return RandomAdapter::get_random_integer(min, max, mt);
    };

    for (auto it = 0ULL; it < 1000ULL; it++) {
        const auto rank_ = generator();

        if (rank_ < 0) {
            ASSERT_THROW(MPIRank rank{ rank_ }, RelearnException) << rank_;
        } else if (rank_ >= max_rank) {
            ASSERT_THROW(MPIRank rank{ rank_ }, RelearnException) << rank_;
        } else {
            MPIRank rank{ rank_ };
            ASSERT_TRUE(rank.is_initialized()) << rank_;
            ASSERT_EQ(rank.get_rank(), rank_) << rank_;
        }
    }
}
