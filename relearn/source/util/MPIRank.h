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

#include "util/RelearnException.h"
#include "util/ranges/Functional.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <climits>
#include <compare>
#include <ostream>
#include <vector>

/**
 * This type reflects an MPI rank in a type safe manner
 */
class MPIRank {
    static constexpr auto id_bit_count = sizeof(int) * 8 - 1;

public:
    /**
     * @brief Returns the uninitialized rank that can be used for debugging purposes.
     * @return The uninitialized rank
     */
    [[nodiscard]] static constexpr MPIRank uninitialized_rank() noexcept { return MPIRank{}; }

    /**
     * @brief Returns the root rank (0) in an initialized state.
     * @return The root rank
     */
    [[nodiscard]] static constexpr MPIRank root_rank() noexcept { return MPIRank{ 0 }; }

    /**
     * @brief Create a vector of MPIRanks within the range [begin, end)
     *
     * @param begin begin of the vector
     * @param end end of the vector
     * @return constexpr auto vector of MPIRanks
     */
    [[nodiscard]] static auto range(int begin, int end) {
        return ranges::views::iota(begin, end) | ranges::views::transform(construct<MPIRank>);
    }

    /**
     * @brief Create a vector of MPIRanks within the range [0, size)
     *
     * @param size size of the vector
     * @return constexpr auto vector of MPIRanks
     */
    [[nodiscard]] static auto range(int size) {
        return range(0U, size);
    }

    /**
     * @brief Default constructs an uninitialized rank
     */
    constexpr MPIRank() noexcept = default;

    /**
     * @brief Constructs an initialized rank with rank
     * @exception Throws a RelearnException if rank < 0
     */
    constexpr explicit MPIRank(int rank)
        : actual_rank_{ rank }
        , is_initialized_{ true } {

        constexpr auto maximum_rank = std::numeric_limits<int>::max() / 2;
        RelearnException::check(rank >= 0, "MPIRank::MPIRank: The actual rank must be >= 0: {}.", rank);
        RelearnException::check(rank < maximum_rank, "MPIRank::MPIRank: The actual rank must be < {}: {}.", maximum_rank, rank);
    }

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const MPIRank& first, const MPIRank& second) = default;

    /**
     * @brief Check if the rank is initialized
     * @return true iff the rank is initialized
     */
    [[nodiscard]] constexpr bool is_initialized() const noexcept { return is_initialized_; }

    /**
     * @brief Returns the stored MPI rank as int
     * @exception Throws a RelearnException if the object is not initialized
     * @return The stored MPI rank
     */
    [[nodiscard]] constexpr int get_rank() const {
        RelearnException::check(is_initialized_, "MPIRank::get_rank: This MPIRank is not initialized.");
        return actual_rank_;
    }

    /**
     * @brief Prints the object's represented MPI rank
     * @param os The out-stream in which the object is printed
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const MPIRank& rank) {
        if (rank.is_initialized_) {
            os << "MPIRank: " << rank.actual_rank_;
        } else {
            os << "MPIRank: uninitialized";
        }
        return os;
    }

private:
    int actual_rank_ : id_bit_count = -1;
    bool is_initialized_ : 1 = false;
};

template <>
struct fmt::formatter<MPIRank> : ostream_formatter { };

namespace std {
template <>
struct hash<MPIRank> {
    using argument_type = MPIRank;
    using result_type = std::size_t;

    result_type operator()(const argument_type& mpi_rank) const {
        constexpr auto max = std::numeric_limits<result_type>::max();
        if (!mpi_rank.is_initialized()) {
            // All bits are set
            return max;
        }

        const auto rank = mpi_rank.get_rank();
        return result_type(rank);
    }
};
} // namespace std
