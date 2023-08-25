/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_io.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "io/NeuronIO.h"
#include "neurons/LocalAreaTranslator.h"

#include <random>

#include <range/v3/action/sort.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/repeat.hpp>

TEST_F(IOTest, testNeuronIOWriteComponentwiseSizeExceptions) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto correct_ids = std::vector<NeuronID>{ number_neurons };
    const auto correct_position = std::vector<RelearnTypes::position_type>{ number_neurons };
    const auto correct_area_ids = std::vector<RelearnTypes::area_id>{ number_neurons };
    const auto correct_area_names = std::vector<RelearnTypes::area_name>{ 1 };
    const auto correct_signal_types = std::vector<SignalType>{ number_neurons };

    const auto faulty_ids = std::vector<NeuronID>{ number_neurons + 1 };
    const auto faulty_position = std::vector<RelearnTypes::position_type>{ number_neurons + 1 };
    const auto faulty_area_ids = std::vector<RelearnTypes::area_id>{ number_neurons + 1 };
    const auto faulty_signal_types = std::vector<SignalType>{ number_neurons + 1 };

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), correct_signal_types, path), RelearnException);

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), correct_signal_types, path), RelearnException);

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, std::make_shared<LocalAreaTranslator>(correct_area_names, faulty_area_ids), faulty_signal_types, path), RelearnException);
}

TEST_F(IOTest, testNeuronIOWriteComponentwiseFileNotFound) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position(mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    std::filesystem::path path{ "" };

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path), RelearnException);
}

TEST_F(IOTest, testNeuronIOWriteComponentwise) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position(mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path));
}

TEST_F(IOTest, testNeuronIOReadComponentwise) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<RelearnTypes::area_name>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);

    const auto& [read_ids, read_positions, read_area_ids, read_area_names, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_componentwise(path);

    const auto& preliminary_full_area_names = NeuronAssignmentAdapter::get_neuron_id_vs_area_name(preliminary_area_ids, preliminary_area_names);
    const auto& read_full_area_names = NeuronAssignmentAdapter::get_neuron_id_vs_area_name(read_area_ids, read_area_names);
    ASSERT_EQ(preliminary_full_area_names, read_full_area_names);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto& diff = preliminary_position[neuron_id] - read_positions[neuron_id];
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = ranges::count(preliminary_signal_types, SignalType::Excitatory);
    const auto number_inhibitory = ranges::count(preliminary_signal_types, SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
}

TEST_F(IOTest, testNeuronIOReadComponentwiseFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path), RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwiseIDException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);
    auto idx2 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 2, mt);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);

    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionXException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_x(-preliminary_position[idx1].get_x());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionYException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_y(-preliminary_position[idx1].get_y());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionZException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_z(-preliminary_position[idx1].get_z());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOWrite1) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_area_names = std::vector<RelearnTypes::area_name>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position(mt));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id], preliminary_area_ids[neuron_id]);
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, path, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids)));
}

TEST_F(IOTest, testNeuronIOWrite2) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_area_names = std::vector<RelearnTypes::area_name>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id], preliminary_area_ids[neuron_id]);
    }

    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, path, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids)));

    const auto& [read_ids, read_positions, read_area_ids, read_area_names, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_componentwise(path);

    const auto& preliminary_full_area_names = NeuronAssignmentAdapter::get_neuron_id_vs_area_name(preliminary_area_ids, preliminary_area_names);
    const auto& read_full_area_names = NeuronAssignmentAdapter::get_neuron_id_vs_area_name(read_area_ids, read_area_names);
    ASSERT_EQ(preliminary_full_area_names, read_full_area_names);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto& diff = preliminary_position[neuron_id] - read_positions[neuron_id];
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = ranges::count(preliminary_signal_types, SignalType::Excitatory);
    const auto number_inhibitory = ranges::count(preliminary_signal_types, SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
}

TEST_F(IOTest, testNeuronIOWriteFileNotFound) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_area_names = std::vector<RelearnTypes::area_name>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position(mt));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id], preliminary_area_ids[neuron_id]);
    }

    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);

    std::filesystem::path path{ "" };

    ASSERT_THROW(NeuronIO::write_neurons(preliminary_neurons, path, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids)), RelearnException);
}

TEST_F(IOTest, testNeuronIORead) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();
    std::vector<RelearnTypes::area_name> area_names{};
    std::vector<RelearnTypes::area_id> area_ids{};

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_neurons.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt), NeuronID{ false, neuron_id }, NeuronTypesAdapter::get_random_signal_type(mt), neuron_id);
        area_names.emplace_back("area_" + std::to_string(neuron_id));
        area_ids.emplace_back(neuron_id);
    }
    auto rng = std::default_random_engine{};
    shuffle(area_names, rng);

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, path, std::make_shared<LocalAreaTranslator>(area_names, area_ids)));

    const auto& [read_neurons, area_id_to_area_name, additional_infos, _]
        = NeuronIO::read_neurons(path);

    ASSERT_EQ(read_neurons.size(), preliminary_neurons.size());

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto& preliminary_neuron = preliminary_neurons[neuron_id];
        const auto& read_neuron = read_neurons[neuron_id];

        const auto& diff = preliminary_neuron.pos - read_neuron.pos;
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);

        ASSERT_EQ(read_neuron.id, preliminary_neuron.id);
        ASSERT_EQ(read_neuron.signal_type, preliminary_neuron.signal_type);
        ASSERT_EQ(read_neuron.area_id, preliminary_neuron.area_id);

        minimum.calculate_componentwise_minimum(read_neuron.pos);
        maximum.calculate_componentwise_maximum(read_neuron.pos);

        if (read_neuron.signal_type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
}

TEST_F(IOTest, testNeuronIOReadFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path), RelearnException);
}

TEST_F(IOTest, testNeuronIOReadIDException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 3;

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);
    auto idx2 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 2, mt);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);

    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionXException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_x(-preliminary_position[idx1].get_x());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionYException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_y(-preliminary_position[idx1].get_y());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionZException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_z(-preliminary_position[idx1].get_z());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadIDs) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);

    const auto& read_ids = NeuronIO::read_neuron_ids(path);

    ASSERT_EQ(read_ids, preliminary_ids);
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty1) {
    const auto number_neurons = RandomAdapter::get_random_integer<NeuronID::value_type>(2, NeuronIdAdapter::upper_bound_num_neurons, mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    const auto idx1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 1, mt);
    auto idx2 = RandomAdapter::get_random_integer<size_t>(0, number_neurons - 2, mt);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path);

    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty2) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};
    auto preliminary_area_ids = std::vector<RelearnTypes::area_id>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationAdapter::get_maximum_position();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_area_names.emplace_back("area_" + std::to_string(neuron_id));
        preliminary_area_ids.emplace_back(neuron_id);
        preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }
    auto rng = std::default_random_engine{};
    shuffle(preliminary_area_names, rng);
    shuffle(preliminary_area_ids, rng);

    preliminary_ids.emplace_back(false, number_neurons + 1);
    preliminary_position.emplace_back(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));
    preliminary_area_names.emplace_back("area_" + std::to_string(number_neurons + 1));
    preliminary_area_ids.emplace_back(number_neurons);
    preliminary_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, std::make_shared<LocalAreaTranslator>(preliminary_area_names, preliminary_area_ids), preliminary_signal_types, path), RelearnException);
    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadIDsFileNotFound) {
    std::filesystem::path path{ "" };
    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadCommentsFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& comments = NeuronIO::read_comments(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComments1) {
    std::filesystem::path path{ "./comments.tmp" };

    {
        std::ofstream out_file{ path };
        out_file << "# 1\n# 2\n# #\n#";
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_EQ(comments.size(), 4);
    ASSERT_EQ(comments[0], std::string("# 1"));
    ASSERT_EQ(comments[1], std::string("# 2"));
    ASSERT_EQ(comments[2], std::string("# #"));
    ASSERT_EQ(comments[3], std::string("#"));
}

TEST_F(IOTest, testNeuronIOReadComments2) {
    std::filesystem::path path{ "./comments.tmp" };

    {
        std::ofstream out_file{ path };
        out_file << "Hallo\n# 1\n# 2\n# #\n#";
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_TRUE(comments.empty());
}

TEST_F(IOTest, testNeuronIOReadComments3) {
    std::filesystem::path path{ "./comments.tmp" };

    {
        std::ofstream out_file{ path };
        for (auto i = 0; i < 10; i++) {
            out_file << "# 1\n";
            out_file << "Hallo\n";
        }
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_EQ(comments.size(), 1);
    ASSERT_EQ(comments[0], std::string("# 1"));
}

TEST_F(IOTest, testReadInSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_in_synapses(path, 1, MPIRank(1), 2), RelearnException);
}

TEST_F(IOTest, testReadInSynapses) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = NetworkGraphAdapter::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    StaticLocalSynapses preliminary_local_synapses_static{};
    StaticDistantInSynapses preliminary_distant_synapses_static{};
    PlasticLocalSynapses preliminary_local_synapses_plastic{};
    PlasticDistantInSynapses preliminary_distant_synapses_plastic{};

    std::filesystem::path path{ "./in_network.tmp" };
    std::ofstream ofstream(path);

    for (const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto source_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto static_weight = NetworkGraphAdapter::get_random_static_synapse_weight(mt);

        const bool plastic = RandomAdapter::get_random_bool(mt);
        const char flag = plastic ? '1' : '0';

        if (source_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(target_id, RankNeuronId(source_rank, source_id), plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(target_id, RankNeuronId(source_rank, source_id), static_weight);
            }
        }

        if (plastic) {
            ofstream << my_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << source_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << plastic_weight << '\t' << flag << '\n';
        } else {
            ofstream << my_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << source_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << static_weight << '\t' << flag << '\n';
        }
    }

    ofstream.flush();
    ofstream.close();

    auto [synapses_static, synapses_plastic] = NeuronIO::read_in_synapses(path, number_neurons, my_rank, number_ranks);
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
}

TEST_F(IOTest, testReadOutSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_out_synapses(path, 1, MPIRank(1), 2), RelearnException);
}

TEST_F(IOTest, testReadOutSynapses) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = NetworkGraphAdapter::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    StaticLocalSynapses preliminary_local_synapses_static{};
    StaticDistantOutSynapses preliminary_distant_synapses_static{};
    PlasticLocalSynapses preliminary_local_synapses_plastic{};
    PlasticDistantOutSynapses preliminary_distant_synapses_plastic{};

    std::filesystem::path path{ "./out_network.tmp" };
    std::ofstream ofstream(path);

    for (const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto target_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto static_weight = NetworkGraphAdapter::get_random_static_synapse_weight(mt);

        const bool plastic = RandomAdapter::get_random_bool(mt);
        const char flag = plastic ? '1' : '0';

        if (target_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(RankNeuronId(target_rank, target_id), source_id, plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(RankNeuronId(target_rank, target_id), source_id, static_weight);
            }
        }

        if (plastic) {
            ofstream << target_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << my_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << plastic_weight << '\t' << flag << '\n';
        } else {
            ofstream << target_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << my_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << static_weight << '\t' << flag << '\n';
        }
    }

    ofstream.flush();
    ofstream.close();

    auto [synapses_static, synapses_plastic] = NeuronIO::read_out_synapses(path, number_neurons, my_rank, static_cast<int>(number_ranks));
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
}

TEST_F(IOTest, testWriteInSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(NeuronIO::write_in_synapses({}, {}, {}, {}, MPIRank(0), 0, path);, RelearnException);
}

TEST_F(IOTest, testWriteInSynapses) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = NetworkGraphAdapter::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    StaticLocalSynapses preliminary_local_synapses_static{};
    StaticDistantInSynapses preliminary_distant_synapses_static{};
    PlasticLocalSynapses preliminary_local_synapses_plastic{};
    PlasticDistantInSynapses preliminary_distant_synapses_plastic{};

    std::filesystem::path path{ "./in_network.tmp" };

    for (const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto source_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto static_weight = NetworkGraphAdapter::get_random_static_synapse_weight(mt);

        const bool plastic = RandomAdapter::get_random_bool(mt);
        const char flag = plastic ? '1' : '0';

        if (source_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(target_id, RankNeuronId(source_rank, source_id), plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(target_id, RankNeuronId(source_rank, source_id), static_weight);
            }
        }
    }

    NeuronIO::write_in_synapses(preliminary_local_synapses_static, preliminary_distant_synapses_static, preliminary_local_synapses_plastic, preliminary_distant_synapses_plastic, my_rank, number_neurons, path);

    auto [synapses_static, synapses_plastic] = NeuronIO::read_in_synapses(path, number_neurons, my_rank, static_cast<int>(number_ranks));
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
}

TEST_F(IOTest, testWriteOutSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(NeuronIO::write_out_synapses({}, {}, {}, {}, MPIRank(0), 0, path);, RelearnException);
}

TEST_F(IOTest, testWriteOutSynapses) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = NetworkGraphAdapter::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    StaticLocalSynapses preliminary_local_synapses_static{};
    StaticDistantOutSynapses preliminary_distant_synapses_static{};
    PlasticLocalSynapses preliminary_local_synapses_plastic{};
    PlasticDistantOutSynapses preliminary_distant_synapses_plastic{};

    std::filesystem::path path{ "./out_network.tmp" };

    for (const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto target_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto static_weight = NetworkGraphAdapter::get_random_static_synapse_weight(mt);

        const bool plastic = RandomAdapter::get_random_bool(mt);
        const char flag = plastic ? '1' : '0';

        if (target_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(RankNeuronId(target_rank, target_id), source_id, plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(RankNeuronId(target_rank, target_id), source_id, static_weight);
            }
        }
    }

    NeuronIO::write_out_synapses(preliminary_local_synapses_static, preliminary_distant_synapses_static, preliminary_local_synapses_plastic, preliminary_distant_synapses_plastic, my_rank, number_neurons, path);

    auto [synapses_static, synapses_plastic] = NeuronIO::read_out_synapses(path, number_neurons, my_rank, static_cast<int>(number_ranks));
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
}

TEST_F(IOTest, additionalPositionInformationTest) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto total_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + number_neurons;
    const auto num_subdomains = RandomAdapter::get_random_integer(1, 10, mt);

    std::vector<NeuronID> correct_ids;
    std::vector<RelearnTypes::position_type> correct_position;
    std::vector<RelearnTypes::area_id> correct_area_ids;
    std::vector<RelearnTypes::area_name> correct_area_names;
    std::vector<SignalType> correct_signal_types;

    auto sim_box = SimulationAdapter::round_bounding_box(SimulationAdapter::get_random_simulation_box_size(mt));
    std::vector<RelearnTypes::bounding_box_type> subdomain_boxes{};
    for (auto i = 0; i < num_subdomains; i++) {
        subdomain_boxes.push_back(SimulationAdapter::round_bounding_box(SimulationAdapter::get_random_simulation_box_size(mt)));
    }

    for (auto i = 0; i < number_neurons; i++) {
        correct_ids.push_back(NeuronID{ i });
        correct_position.emplace_back(SimulationAdapter::get_random_position_in_box(sim_box, mt));
        correct_area_names.emplace_back("area_" + std::to_string(i));
        correct_area_ids.emplace_back(i);
        correct_signal_types.emplace_back(NeuronTypesAdapter::get_random_signal_type(mt));
    }

    auto path = get_relearn_path() / "neurons.tmp";

    NeuronIO::write_neurons_componentwise(correct_ids, correct_position, std::make_shared<LocalAreaTranslator>(correct_area_names, correct_area_ids), correct_signal_types, path, total_neurons, sim_box, subdomain_boxes);

    auto comments = NeuronIO::read_comments(path);
    auto infos = NeuronIO::parse_additional_position_information(comments);

    ASSERT_EQ(infos.total_neurons, total_neurons);
    ASSERT_EQ(infos.local_neurons, number_neurons);
    ASSERT_EQ(sim_box, infos.sim_size);
    ASSERT_EQ(subdomain_boxes, infos.subdomain_sizes);
}