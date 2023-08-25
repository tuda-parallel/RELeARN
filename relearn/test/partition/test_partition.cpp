/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_partition.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "structure/Partition.h"
#include "util/RelearnException.h"

#include <algorithm>
#include <cstddef>
#include <numeric>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/indices.hpp>

bool is_power_of_two(size_t number) {
    auto counter = 0;

    while (number != 0) {
        auto res = number & 1;
        if (res == 1) {
            counter++;
        }
        number >>= 1;
    }

    return counter == 1;
}

TEST_F(PartitionTest, testPartitionZeroRanks) {
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(mt);
    ASSERT_THROW(Partition part(0, my_rank), RelearnException) << my_rank;
}

TEST_F(PartitionTest, testPartitionConstructorArguments) {
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(mt);
    const auto num_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto is_rank_power_2 = is_power_of_two(num_ranks);

    if (!is_rank_power_2) {
        ASSERT_THROW(Partition part(num_ranks, my_rank), RelearnException) << num_ranks << ' ' << my_rank;
        return;
    }

    if (my_rank.get_rank() >= num_ranks) {
        ASSERT_THROW(Partition part(num_ranks, my_rank), RelearnException) << num_ranks << ' ' << my_rank;
        return;
    }

    ASSERT_NO_THROW(Partition part(num_ranks, my_rank)) << num_ranks << ' ' << my_rank;
}

TEST_F(PartitionTest, testPartitionConstructor) {
    const auto num_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);

    const auto my_subdomains = num_subdomains / num_ranks;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    const auto num_subdomains_per_dim = static_cast<size_t>(std::ceil(std::pow(static_cast<double>(num_subdomains), 1.0 / 3.0)));

    for (const auto my_rank : MPIRank::range(num_ranks)) {
        Partition partition(num_ranks, my_rank);

        ASSERT_EQ(partition.get_number_mpi_ranks(), num_ranks);
        ASSERT_EQ(partition.get_my_mpi_rank(), my_rank);

        ASSERT_EQ(partition.get_total_number_subdomains(), num_subdomains) << num_subdomains;
        ASSERT_EQ(partition.get_number_local_subdomains(), my_subdomains) << my_subdomains;

        ASSERT_EQ(partition.get_local_subdomain_id_start(), my_subdomains * my_rank.get_rank()) << my_subdomains << ' ' << my_rank;
        ASSERT_EQ(partition.get_local_subdomain_id_end(), my_subdomains * (my_rank.get_rank() + 1) - 1) << my_subdomains << ' ' << my_rank;

        ASSERT_EQ(partition.get_level_of_subdomain_trees(), oct_exponent) << oct_exponent;
        ASSERT_EQ(partition.get_number_subdomains_per_dimension(), num_subdomains_per_dim) << num_subdomains_per_dim;
    }
}

TEST_F(PartitionTest, testPartitionNumberNeurons) {
    const auto num_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    std::vector<std::vector<size_t>> number_local_neurons(num_ranks);
    size_t number_total_neurons = 0;

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        number_local_neurons[my_rank] = std::vector<size_t>(my_subdomains);

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            const auto num_local_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

            number_local_neurons[my_rank][my_subdomain] = num_local_neurons;
            number_total_neurons += num_local_neurons;
        }
    }

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        const auto& local_neurons = number_local_neurons[my_rank];

        std::vector<NeuronID> local_ids_start(my_subdomains, NeuronID{ 0 });
        std::vector<NeuronID> local_ids_ends(my_subdomains, NeuronID{ 0 });

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            if (my_subdomain > 0) {
                const auto local_start = local_ids_ends[static_cast<size_t>(my_subdomain) - 1].get_neuron_id() + 1;
                local_ids_start[my_subdomain] = NeuronID(false, local_start);
            }

            const auto local_end = local_ids_start[my_subdomain].get_neuron_id() + local_neurons[my_subdomain] - 1;
            local_ids_ends[my_subdomain] = NeuronID(false, local_end);
        }

        Partition partition(num_ranks, MPIRank(my_rank));

        ASSERT_THROW(auto val = partition.get_total_number_neurons(), RelearnException);
        partition.set_total_number_neurons(number_total_neurons);
        ASSERT_EQ(partition.get_total_number_neurons(), number_total_neurons);

        const auto num_local_neurons = ranges::accumulate(local_neurons, size_t{ 0 });
        partition.set_number_local_neurons(num_local_neurons);

        ASSERT_EQ(partition.get_number_local_neurons(), num_local_neurons);
    }
}

TEST_F(PartitionTest, testPartitionSubdomainIndices) {
    const auto num_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));

    if (oct_exponent >= std::numeric_limits<uint8_t>::max()) {
        return;
    }

    const auto oct_exponent_cast = static_cast<uint8_t>(oct_exponent);
    SpaceFillingCurve<Morton> sfc(oct_exponent_cast);

    std::vector<bool> found_indices(num_subdomains, false);

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        Partition partition(num_ranks, MPIRank(my_rank));

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            const auto index_1 = partition.get_1d_index_of_subdomain(my_subdomain);
            const auto index_3 = partition.get_3d_index_of_subdomain(my_subdomain);

            const auto translated_index_3 = sfc.map_1d_to_3d(index_1);
            const auto translated_index_1 = sfc.map_3d_to_1d(index_3);

            ASSERT_EQ(index_1, translated_index_1) << index_1 << ' ' << translated_index_1;
            ASSERT_EQ(index_3, translated_index_3) << index_3 << ' ' << translated_index_3;

            ASSERT_FALSE(found_indices[index_1]) << index_1;
            ASSERT_TRUE(index_1 < num_subdomains) << index_1 << ' ' << num_subdomains;

            found_indices[index_1] = true;
        }

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            ASSERT_THROW(auto val = partition.get_1d_index_of_subdomain(my_subdomain + num_subdomains), RelearnException);
            ASSERT_THROW(auto val = partition.get_3d_index_of_subdomain(my_subdomain + num_subdomains), RelearnException);
        }
    }
}

TEST_F(PartitionTest, testPartitionSubdomainBoundaries) {
    const auto num_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    if (oct_exponent >= std::numeric_limits<uint8_t>::max()) {
        return;
    }

    const auto oct_exponent_cast = static_cast<uint8_t>(oct_exponent);
    const auto num_subdomains_per_dim = std::ceil(std::pow(static_cast<double>(num_subdomains), 1.0 / 3.0));

    SpaceFillingCurve<Morton> sfc(oct_exponent_cast);

    const auto& [simulation_box_minimum, simulation_box_maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto& simulation_box_dimensions = simulation_box_maximum - simulation_box_minimum;
    const auto& subdomain_box_dimensions = simulation_box_dimensions / num_subdomains_per_dim;

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        Partition partition(num_ranks, MPIRank(my_rank));

        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_maximum, simulation_box_minimum), RelearnException);

        ASSERT_THROW(partition.set_simulation_box_size({ simulation_box_minimum.get_x() + Constants::uninitialized, simulation_box_minimum.get_y(), simulation_box_minimum.get_z() }, simulation_box_maximum), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size({ simulation_box_minimum.get_x(), simulation_box_minimum.get_y() + Constants::uninitialized, simulation_box_minimum.get_z() }, simulation_box_maximum), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size({ simulation_box_minimum.get_x(), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() + Constants::uninitialized }, simulation_box_maximum), RelearnException);

        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_minimum, { simulation_box_maximum.get_x() + Constants::uninitialized, simulation_box_maximum.get_y(), simulation_box_maximum.get_z() }), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y() + Constants::uninitialized, simulation_box_maximum.get_z() }), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() + Constants::uninitialized }), RelearnException);

        ASSERT_THROW(partition.set_simulation_box_size({ simulation_box_minimum.get_x() - Constants::uninitialized, simulation_box_minimum.get_y(), simulation_box_minimum.get_z() }, simulation_box_maximum), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size({ simulation_box_minimum.get_x(), simulation_box_minimum.get_y() - Constants::uninitialized, simulation_box_minimum.get_z() }, simulation_box_maximum), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size({ simulation_box_minimum.get_x(), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() - Constants::uninitialized }, simulation_box_maximum), RelearnException);

        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_minimum, { simulation_box_maximum.get_x() - Constants::uninitialized, simulation_box_maximum.get_y(), simulation_box_maximum.get_z() }), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y() - Constants::uninitialized, simulation_box_maximum.get_z() }), RelearnException);
        ASSERT_THROW(partition.set_simulation_box_size(simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() - Constants::uninitialized }), RelearnException);

        ASSERT_THROW(auto val = partition.get_simulation_box_size(), RelearnException);

        partition.set_simulation_box_size(simulation_box_minimum, simulation_box_maximum);
        const auto& [retrieved_min, retrieved_max] = partition.get_simulation_box_size();

        ASSERT_EQ(simulation_box_minimum, retrieved_min);
        ASSERT_EQ(simulation_box_maximum, retrieved_max);
    }

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        Partition partition(num_ranks, MPIRank(my_rank));
        partition.set_simulation_box_size(simulation_box_minimum, simulation_box_maximum);

        for (size_t subdomain_index_1 = 0; subdomain_index_1 < num_subdomains; subdomain_index_1++) {
            const auto& subdomain_index_3 = sfc.map_1d_to_3d(subdomain_index_1);

            const auto& [min_1, max_1] = partition.calculate_subdomain_boundaries(subdomain_index_1);
            const auto& [min_3, max_3] = partition.calculate_subdomain_boundaries(subdomain_index_3);

            ASSERT_EQ(min_1, min_3) << min_1 << min_3;
            ASSERT_EQ(max_1, max_3) << max_1 << max_3;

            Vec3d subdomain_expected_min = Vec3d{
                subdomain_box_dimensions.get_x() * subdomain_index_3.get_x(),
                subdomain_box_dimensions.get_y() * subdomain_index_3.get_y(),
                subdomain_box_dimensions.get_z() * subdomain_index_3.get_z()
            } + simulation_box_minimum;

            Vec3d subdomain_expected_max = subdomain_expected_min + subdomain_box_dimensions;

            const auto& difference_min = min_1 - subdomain_expected_min;
            const auto& difference_max = max_1 - subdomain_expected_max;

            ASSERT_NEAR(difference_min.calculate_p_norm(1.0), 0.0, eps) << min_1 << subdomain_expected_min;
            ASSERT_NEAR(difference_max.calculate_p_norm(1.0), 0.0, eps) << max_1 << subdomain_expected_max;
        }

        partition.calculate_and_set_subdomain_boundaries();

        std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries{};

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            const auto& [min, max] = partition.get_subdomain_boundaries(my_subdomain);
            const auto index_1 = partition.get_1d_index_of_subdomain(my_subdomain);

            const auto& [min1, max1] = partition.calculate_subdomain_boundaries(index_1);

            ASSERT_EQ(min, min1) << min << min1;
            ASSERT_EQ(max, max1) << max << max1;

            local_subdomain_boundaries.emplace_back(min, max);
        }

        auto partition_local_subdomain_boundaries = partition.get_all_local_subdomain_boundaries();

        struct {
            bool operator()(RelearnTypes::bounding_box_type& a, RelearnTypes::bounding_box_type& b) const { return a < b; }
        } customLess;

        ranges::sort(local_subdomain_boundaries, customLess);
        ranges::sort(partition_local_subdomain_boundaries, customLess);

        ASSERT_EQ(local_subdomain_boundaries, partition_local_subdomain_boundaries);

        for (const auto my_subdomain : ranges::views::indices(num_ranks)) {
            ASSERT_THROW(auto val = partition.get_subdomain_boundaries(my_subdomain + num_subdomains), RelearnException) << my_subdomain << num_ranks;
        }
    }
}

TEST_F(PartitionTest, testPartitionPositionToMpi) {
    const auto num_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    const auto num_subdomains_per_dim = std::ceil(std::pow(static_cast<double>(num_subdomains), 1.0 / 3.0));
    if (oct_exponent >= std::numeric_limits<uint8_t>::max()) {
        return;
    }

    const auto oct_exponent_cast = static_cast<uint8_t>(oct_exponent);
    SpaceFillingCurve<Morton> sfc(oct_exponent_cast);

    const auto& [simulation_box_minimum, simulation_box_maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto& simulation_box_dimensions = simulation_box_maximum - simulation_box_minimum;
    const auto& subdomain_box_dimensions = simulation_box_dimensions / num_subdomains_per_dim;

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        Partition partition(num_ranks, MPIRank(my_rank));

        for (const auto j : ranges::views::indices(iterations)) {
            const auto& position = SimulationAdapter::get_random_position_in_box(simulation_box_minimum, simulation_box_maximum, mt);
            ASSERT_THROW(auto val = partition.get_mpi_rank_from_position(position), RelearnException);
        }

        partition.set_simulation_box_size(simulation_box_minimum, simulation_box_maximum);

        for (const auto j : ranges::views::indices(iterations)) {
            const auto& position = SimulationAdapter::get_random_position_in_box(simulation_box_minimum, simulation_box_maximum, mt);
            const auto proposed_rank = partition.get_mpi_rank_from_position(position);

            const auto index_1_start = proposed_rank * my_subdomains;

            auto correct = false;

            for (auto subdomain_id = index_1_start; subdomain_id < index_1_start + my_subdomains; subdomain_id++) {
                const auto& [min, max] = partition.calculate_subdomain_boundaries(subdomain_id);
                const auto& is_in_subdomain = position.check_in_box(min, max);
                correct |= is_in_subdomain;
            }

            ASSERT_TRUE(correct);
        }
    }
}
