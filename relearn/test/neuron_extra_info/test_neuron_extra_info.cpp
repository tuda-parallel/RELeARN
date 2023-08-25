/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_extra_info.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "neurons/NeuronsExtraInfo.h"

#include <algorithm>

void NeuronsExtraInfoTest::assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons) {
    const auto& positions = nei.get_positions();

    const auto& positions_size = positions.size();

    ASSERT_EQ(0, positions_size) << positions_size;

    for (const auto i : NeuronID::range_id(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, 1, mt);

        ASSERT_THROW(const auto& tmp = nei.get_position(neuron_id), RelearnException) << "assert empty position" << neuron_id;
    }
}

void NeuronsExtraInfoTest::assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check, const std::vector<Vec3d>& expected_positions) {

    const auto& expected_positions_size = expected_positions.size();

    ASSERT_EQ(num_neurons_check, expected_positions_size) << num_neurons_check << ' ' << expected_positions_size;

    const auto& actual_positions = nei.get_positions();

    const auto& positions_size = actual_positions.size();

    ASSERT_EQ(positions_size, number_neurons) << positions_size << ' ' << number_neurons;

    for (auto neuron_id : NeuronID::range(num_neurons_check)) {

        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], actual_positions[neuron_id.get_neuron_id()]) << neuron_id;
        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], nei.get_position(neuron_id)) << neuron_id;
    }

    for (const auto i : ranges::views::indices(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(
            number_neurons, number_neurons, mt);

        ASSERT_THROW(const auto& tmp = nei.get_position(neuron_id), RelearnException) << neuron_id;
    }
}

TEST_F(NeuronsExtraInfoTest, testConstructor) {
    NeuronsExtraInfo nei{};

    assert_empty(nei, NeuronIdAdapter::upper_bound_num_neurons);

    ASSERT_THROW(nei.set_positions(std::vector<NeuronsExtraInfo::position_type>{}), RelearnException);

    assert_empty(nei, NeuronIdAdapter::upper_bound_num_neurons);

    const auto new_size = NeuronIdAdapter::get_random_number_neurons(mt);

    ASSERT_THROW(nei.set_positions(std::vector<NeuronsExtraInfo::position_type>(new_size)), RelearnException);

    assert_empty(nei, NeuronIdAdapter::upper_bound_num_neurons);
}

TEST_F(NeuronsExtraInfoTest, testInit) {
    NeuronsExtraInfo nei{};

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    nei.init(number_neurons);
    assert_empty(nei, number_neurons);

    auto num_neurons_wrong = NeuronIdAdapter::get_random_number_neurons(mt);
    if (num_neurons_wrong == number_neurons) {
        num_neurons_wrong++;
    }

    std::vector<Vec3d> positions_wrong(num_neurons_wrong);

    ASSERT_THROW(nei.set_positions(positions_wrong), RelearnException);

    assert_empty(nei, number_neurons);

    std::vector<Vec3d> positions_right(number_neurons);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        positions_right[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right);

    assert_contains(nei, number_neurons, number_neurons, positions_right);

    std::vector<Vec3d> positions_right_2(number_neurons);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        positions_right_2[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right_2);

    assert_contains(nei, number_neurons, number_neurons, positions_right_2);
}

TEST_F(NeuronsExtraInfoTest, testCreate) {
    NeuronsExtraInfo nei{};

    const auto num_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_neurons_create_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_neurons_create_2 = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto num_neurons_total_1 = num_neurons_init + num_neurons_create_1;
    const auto num_neurons_total_2 = num_neurons_total_1 + num_neurons_create_2;

    nei.init(num_neurons_init);

    ASSERT_THROW(nei.create_neurons(num_neurons_create_1), RelearnException);

    assert_empty(nei, num_neurons_init);

    std::vector<Vec3d> positions_right(num_neurons_init);

    for (const auto neuron_id : NeuronID::range_id(num_neurons_init)) {
        positions_right[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right);

    nei.create_neurons(num_neurons_create_1);

    assert_contains(nei, num_neurons_total_1, num_neurons_init, positions_right);

    std::vector<Vec3d> positions_right_2(num_neurons_total_1);

    for (const auto neuron_id : NeuronID::range_id(num_neurons_total_1)) {
        positions_right_2[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right_2);

    assert_contains(nei, num_neurons_total_1, num_neurons_total_1, positions_right_2);

    nei.create_neurons(num_neurons_create_2);

    assert_contains(nei, num_neurons_total_2, num_neurons_total_1, positions_right_2);

    std::vector<Vec3d> positions_right_3(num_neurons_total_2);

    for (const auto neuron_id : NeuronID::range_id(num_neurons_total_2)) {
        positions_right_3[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right_3);

    assert_contains(nei, num_neurons_total_2, num_neurons_total_2, positions_right_3);
}

TEST_F(NeuronsExtraInfoTest, testSetStatus) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NeuronsExtraInfo nei{};
    nei.init(number_neurons);

    std::vector<NeuronID> enabled_neurons{};
    std::vector<NeuronID> disabled_neurons{};
    std::vector<NeuronID> static_neurons{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto random_number = RandomAdapter::get_random_integer(0, 5, mt);
        if (random_number == 0) {
            static_neurons.emplace_back(neuron_id);
        } else if (random_number == 1) {
            disabled_neurons.emplace_back(neuron_id);
        } else {
            enabled_neurons.emplace_back(neuron_id);
        }
    }

    ASSERT_THROW(nei.set_enabled_neurons(enabled_neurons), RelearnException);
    ASSERT_NO_THROW(nei.set_disabled_neurons(disabled_neurons));
    ASSERT_NO_THROW(nei.set_static_neurons(static_neurons));

    const auto status_flags = nei.get_disable_flags();
    ASSERT_EQ(status_flags.size(), number_neurons);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto index = neuron_id.get_neuron_id();

        if (std::ranges::binary_search(enabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Enabled);
            ASSERT_TRUE(nei.does_update_electrical_actvity(neuron_id));
            ASSERT_TRUE(nei.does_update_plasticity(neuron_id));
        } else if (std::ranges::binary_search(disabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Disabled);
            ASSERT_FALSE(nei.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(nei.does_update_plasticity(neuron_id));
        } else {
            ASSERT_EQ(status_flags[index], UpdateStatus::Static);
            ASSERT_TRUE(nei.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(nei.does_update_plasticity(neuron_id));
        }
    }
}

TEST_F(NeuronsExtraInfoTest, testSetStatusShuffle) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NeuronsExtraInfo nei{};
    nei.init(number_neurons);

    std::vector<NeuronID> enabled_neurons{};
    std::vector<NeuronID> disabled_neurons{};
    std::vector<NeuronID> static_neurons{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto random_number = RandomAdapter::get_random_integer(0, 5, mt);
        if (random_number == 0) {
            static_neurons.emplace_back(neuron_id);
        } else if (random_number == 1) {
            disabled_neurons.emplace_back(neuron_id);
        } else {
            enabled_neurons.emplace_back(neuron_id);
        }
    }

    RandomAdapter::shuffle(enabled_neurons, mt);
    RandomAdapter::shuffle(disabled_neurons, mt);
    RandomAdapter::shuffle(static_neurons, mt);

    ASSERT_THROW(nei.set_enabled_neurons(enabled_neurons), RelearnException);
    ASSERT_NO_THROW(nei.set_disabled_neurons(disabled_neurons));
    ASSERT_NO_THROW(nei.set_static_neurons(static_neurons));

    std::ranges::sort(enabled_neurons);
    std::ranges::sort(disabled_neurons);
    std::ranges::sort(static_neurons);

    const auto status_flags = nei.get_disable_flags();
    ASSERT_EQ(status_flags.size(), number_neurons);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto index = neuron_id.get_neuron_id();

        if (std::ranges::binary_search(enabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Enabled);
            ASSERT_TRUE(nei.does_update_electrical_actvity(neuron_id));
            ASSERT_TRUE(nei.does_update_plasticity(neuron_id));
        } else if (std::ranges::binary_search(disabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Disabled);
            ASSERT_FALSE(nei.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(nei.does_update_plasticity(neuron_id));
        } else {
            ASSERT_EQ(status_flags[index], UpdateStatus::Static);
            ASSERT_TRUE(nei.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(nei.does_update_plasticity(neuron_id));
        }
    }
}

TEST_F(NeuronsExtraInfoTest, testSetStatusOutOfBounds) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NeuronsExtraInfo nei{};
    nei.init(number_neurons);

    std::vector<NeuronID> enabled_neurons{};
    std::vector<NeuronID> disabled_neurons{};
    std::vector<NeuronID> static_neurons{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto random_number = RandomAdapter::get_random_integer(0, 5, mt);
        if (random_number == 0) {
            static_neurons.emplace_back(neuron_id);
        } else if (random_number == 1) {
            disabled_neurons.emplace_back(neuron_id);
        } else {
            enabled_neurons.emplace_back(neuron_id);
        }
    }

    static_neurons.emplace_back(number_neurons);
    enabled_neurons.emplace_back(number_neurons + 1);
    disabled_neurons.emplace_back(number_neurons + 2);

    ASSERT_THROW(nei.set_enabled_neurons(enabled_neurons), RelearnException);
    ASSERT_THROW(nei.set_disabled_neurons(disabled_neurons), RelearnException);
    ASSERT_THROW(nei.set_static_neurons(static_neurons), RelearnException);
}

TEST_F(NeuronsExtraInfoTest, testSetStatusRepeated) {
    const auto number_neurons = 5;

    NeuronsExtraInfo nei{};
    nei.init(number_neurons);

    const auto status_flags = nei.get_disable_flags();

    nei.set_disabled_neurons(std::vector{ NeuronID(2) });
    nei.set_enabled_neurons(std::vector{ NeuronID(2) });

    for (auto i = 0; i < number_neurons; i++) {
        ASSERT_EQ(status_flags[i], UpdateStatus::Enabled);
    }

    nei.set_static_neurons(std::vector{ NeuronID(2), NeuronID(3) });

    ASSERT_THROW(nei.set_enabled_neurons(std::vector{ NeuronID(3) }), RelearnException);
    ASSERT_THROW(nei.set_disabled_neurons(std::vector{ NeuronID(2) }), RelearnException);

    ASSERT_EQ(status_flags[0], UpdateStatus::Enabled);
    ASSERT_EQ(status_flags[1], UpdateStatus::Enabled);
    ASSERT_EQ(status_flags[2], UpdateStatus::Static);
    ASSERT_EQ(status_flags[3], UpdateStatus::Static);
    ASSERT_EQ(status_flags[4], UpdateStatus::Enabled);
}

TEST_F(NeuronsExtraInfoTest, testGetPositionsFor) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NeuronsExtraInfo nei{};
    nei.init(number_neurons);

    std::vector<RelearnTypes::position_type> positions(number_neurons);
    for (auto i = 0; i < number_neurons; i++) {
        positions[i] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions);

    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);

    CommunicationMap<NeuronID> cm(number_ranks, NeuronIdAdapter::upper_bound_num_neurons);

    for (const auto rank : MPIRank::range(number_ranks)) {
        const auto number_neurons_for_rank = NeuronIdAdapter::get_random_number_neurons(mt);
        for (auto it = 0; it < number_neurons_for_rank; it++) {
            cm.emplace_back(rank, NeuronIdAdapter::get_random_neuron_id(number_neurons, mt));
        }
    }

    auto results = nei.get_positions_for(cm);

    ASSERT_EQ(cm.size(), results.size());

    for (auto outer_it = 0; outer_it < cm.size(); outer_it++) {
        const auto rank = MPIRank(outer_it);
        const auto& requests = cm.get_requests(rank);
        const auto& responses = results.get_requests(rank);

        ASSERT_EQ(requests.size(), responses.size());

        for (auto inner_it = 0; inner_it < requests.size(); inner_it++) {
            const auto& expected_position = nei.get_position(requests[inner_it]);
            ASSERT_EQ(expected_position, responses[inner_it]);
        }
    }
}

TEST_F(NeuronsExtraInfoTest, testGetPositionsForException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NeuronsExtraInfo nei{};
    nei.init(number_neurons);

    std::vector<RelearnTypes::position_type> positions(number_neurons);
    for (auto i = 0; i < number_neurons; i++) {
        positions[i] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions);

    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);

    CommunicationMap<NeuronID> cm(number_ranks, NeuronIdAdapter::upper_bound_num_neurons);

    for (const auto rank : MPIRank::range(number_ranks)) {
        const auto number_neurons_for_rank = NeuronIdAdapter::get_random_number_neurons(mt);
        for (auto it = 0; it < number_neurons_for_rank; it++) {
            cm.emplace_back(rank, NeuronIdAdapter::get_random_neuron_id(number_neurons, mt));
        }
    }

    const auto faulty_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);
    cm.emplace_back(faulty_rank, number_neurons);

    ASSERT_THROW(auto val = nei.get_positions_for(cm), RelearnException);
}
