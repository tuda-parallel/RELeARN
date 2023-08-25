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

#include "util/MPIRank.h"

#include <cmath>
#include <random>

class MPIRankAdapter {
public:
    constexpr static int upper_bound_num_ranks = 32;

    static int get_random_number_ranks(std::mt19937& mt) {
        return RandomAdapter::get_random_integer<int>(1, upper_bound_num_ranks, mt);
    }

    static int get_adjusted_random_number_ranks(std::mt19937& mt) {
        const auto random_rank = get_random_number_ranks(mt);
        return static_cast<int>(round_to_next_exponent(random_rank, 2));
    }

    static MPIRank get_random_mpi_rank(std::mt19937& mt) {
        const auto rank = RandomAdapter::get_random_integer<int>(0, upper_bound_num_ranks - 1, mt);
        return MPIRank(rank);
    }

    static MPIRank get_random_mpi_rank(size_t number_ranks, std::mt19937& mt) {
        const auto rank = RandomAdapter::get_random_integer<int>(0, int(number_ranks - 1), mt);
        return MPIRank(rank);
    }

    static MPIRank get_random_mpi_rank(size_t number_ranks, MPIRank except, std::mt19937& mt) {
        MPIRank mpi_rank{};
        do {
            const auto rank = RandomAdapter::get_random_integer<int>(0, int(number_ranks - 1), mt);
            mpi_rank = MPIRank(rank);
        } while (mpi_rank == except);

        return mpi_rank;
    }

    static MPIRank get_random_mpi_rank(int number_ranks, std::mt19937& mt) {
        const auto rank = RandomAdapter::get_random_integer<int>(0, number_ranks - 1, mt);
        return MPIRank(rank);
    }

    static MPIRank get_random_mpi_rank(int number_ranks, MPIRank except, std::mt19937& mt) {
        MPIRank mpi_rank{};
        do {
            const auto rank = RandomAdapter::get_random_integer<int>(0, number_ranks - 1, mt);
            mpi_rank = MPIRank(rank);
        } while (mpi_rank == except);

        return mpi_rank;
    }

private:
    static size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
        auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
        auto rounded_exp = std::ceil(log);
        auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
        return static_cast<size_t>(new_val);
    }
};
