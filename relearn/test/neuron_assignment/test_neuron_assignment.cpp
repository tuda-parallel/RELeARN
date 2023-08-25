/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_assignment.h"

#include "adapter/random/RandomAdapter.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "sim/Essentials.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/file/MultipleSubdomainsFromFile.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/random/SubdomainFromNeuronPerRank.h"
#include "structure/Partition.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

double calculate_excitatory_fraction(const std::vector<SignalType>& types) {
    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto& type : types) {
        if (type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto ratio = static_cast<double>(number_excitatory) / static_cast<double>(number_excitatory + number_inhibitory);
    return ratio;
}

void write_synapses_to_file(const std::vector<PlasticLocalSynapse>& synapses, std::filesystem::path path) {
    std::ofstream in_of(path / "rank_0_in_network.txt");
    std::ofstream out_of(path / "rank_0_out_network.txt");

    for (const auto& [target, source, weight] : synapses) {
        in_of << "0 " << (target.get_neuron_id() + 1) << '\t' << "0 " << (source.get_neuron_id() + 1) << '\t' << weight << '\t' << '1' << '\n';
        out_of << "0 " << (target.get_neuron_id() + 1) << '\t' << "0 " << (source.get_neuron_id() + 1) << '\t' << weight << '\t' << '1' << '\n';
    }
}

TEST_F(NeuronAssignmentTest, testDensityTooManyRanks) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt) * 2;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(SubdomainFromNeuronDensity sfnd(golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part), RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testDensityConstructor) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto number_neurons = sfnd.get_requested_number_neurons();
    const auto fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, number_neurons);
    ASSERT_EQ(golden_fraction_excitatory_neurons, fraction_excitatory_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    ASSERT_EQ(0, sfnd.get_number_placed_neurons());
    ASSERT_EQ(0.0, sfnd.get_ratio_placed_excitatory_neurons());
}

TEST_F(NeuronAssignmentTest, testDensityInitialize) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    sfnd.initialize();

    const auto requested_number_neurons = sfnd.get_requested_number_neurons();
    const auto placed_number_neurons = sfnd.get_number_placed_neurons();

    const auto requested_fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();
    const auto placed_fraction_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, requested_number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_EQ(requested_number_neurons, placed_number_neurons);
    ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
}

TEST_F(NeuronAssignmentTest, testDensityNeuronAttributesSizes) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnd.initialize();

    const auto placed_number_neurons = sfnd.get_number_placed_neurons();

    const auto& positions = sfnd.get_neuron_positions_in_subdomains();
    const auto& types = sfnd.get_neuron_types_in_subdomains();
    const auto placed_number_neurons_in_subdomain = sfnd.get_number_neurons_in_subdomains();

    ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
    ASSERT_EQ(placed_number_neurons, positions.size());
    ASSERT_EQ(placed_number_neurons, types.size());
    ASSERT_EQ(placed_number_neurons, sfnd.get_local_area_translator()->get_number_neurons_in_total());
}

TEST_F(NeuronAssignmentTest, testDensityNeuronAttributesSemantic) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnd.initialize();

    const auto placed_ratio_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnd.get_neuron_positions_in_subdomains();
    const auto& types = sfnd.get_neuron_types_in_subdomains();

    const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
    ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1.0 / golden_number_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto neurons_per_dimension = pow(golden_number_neurons, 1. / 3);
    const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

    std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

    for (const auto& position : positions) {
        ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
        auto cast_position = Vec3s{ position / golden_um_per_neuron };

        const auto x = cast_position.get_x();
        const auto y = cast_position.get_y();
        const auto z = cast_position.get_z();

        ASSERT_LE(x, number_boxes);
        ASSERT_LE(y, number_boxes);
        ASSERT_LE(z, number_boxes);

        const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

        ASSERT_FALSE(flag);
        box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
    }
}

TEST_F(NeuronAssignmentTest, testDensityWriteToFile) {
    std::vector<Vec3d> positions{};
    std::vector<RelearnTypes::area_id> area_ids{};
    std::vector<RelearnTypes::area_name> area_names{};
    std::vector<SignalType> types{};

    NeuronAssignmentAdapter::generate_random_neurons(positions, area_ids, area_names, types, mt);

    const auto number_neurons = positions.size();

    std::ifstream file("neurons.tmp", std::ios::binary | std::ios::in);

    std::vector<std::string> lines{};

    std::string str{};
    while (std::getline(file, str)) {
        if (str[0] == '#') {
            continue;
        }

        lines.emplace_back(str);
    }

    file.close();

    ASSERT_EQ(number_neurons, lines.size());

    std::vector<bool> is_there(number_neurons, false);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const Vec3d& desired_position = positions[neuron_id];
        const RelearnTypes::area_id& desired_area_id = area_ids[neuron_id];
        const RelearnTypes::area_name& desired_area_name = area_names[desired_area_id];
        const SignalType& desired_signal_type = types[neuron_id];

        const std::string& current_line = lines[neuron_id];

        std::stringstream sstream(current_line);

        size_t id;
        double x;
        double y;
        double z;
        std::string area;
        std::string type_string;

        sstream
            >> id
            >> x
            >> y
            >> z
            >> area
            >> type_string;

        ASSERT_TRUE(0 < id);
        ASSERT_TRUE(id <= number_neurons);

        ASSERT_FALSE(is_there[id - 1]);
        is_there[id - 1] = true;

        ASSERT_NEAR(x, desired_position.get_x(), eps);
        ASSERT_NEAR(y, desired_position.get_y(), eps);
        ASSERT_NEAR(z, desired_position.get_z(), eps);

        SignalType type;
        if (type_string == "ex") {
            type = SignalType::Excitatory;
        } else if (type_string == "in") {
            type = SignalType::Inhibitory;
        } else {
            ASSERT_TRUE(false);
        }

        ASSERT_TRUE(area == desired_area_name);
        ASSERT_TRUE(type == desired_signal_type);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankTooFewNeurons) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(SubdomainFromNeuronPerRank sfnpr(0, golden_fraction_excitatory_neurons, golden_um_per_neuron, part), RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankSingleSubdomain) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto number_neurons = sfnpr.get_requested_number_neurons();
    const auto fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, number_neurons);
    ASSERT_EQ(golden_fraction_excitatory_neurons, fraction_excitatory_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    ASSERT_EQ(0, sfnpr.get_number_placed_neurons());
    ASSERT_EQ(0.0, sfnpr.get_ratio_placed_excitatory_neurons());
}

TEST_F(NeuronAssignmentTest, testPerRankConstructorMultipleSubdomains) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        const auto number_neurons = sfnpr.get_requested_number_neurons();
        const auto fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();

        ASSERT_EQ(golden_number_neurons * golden_number_ranks, number_neurons);

        ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / golden_number_neurons);

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        const auto box_length = (sim_box_max - sim_box_min).get_maximum();

        const auto number_neurons_per_box_max = static_cast<size_t>(ceil(static_cast<double>(golden_number_neurons) / part->get_number_local_subdomains()));

        const auto golden_box_length = calculate_box_length(number_neurons_per_box_max, golden_um_per_neuron) * part->get_number_subdomains_per_dimension();
        ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

        ASSERT_EQ(0, sfnpr.get_number_placed_neurons());
        ASSERT_EQ(0.0, sfnpr.get_ratio_placed_excitatory_neurons());
    }
}

TEST_F(NeuronAssignmentTest, testPerRankInitializeSingleSubdomain) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    sfnpr.initialize();

    const auto requested_number_neurons = sfnpr.get_requested_number_neurons();
    const auto placed_number_neurons = sfnpr.get_number_placed_neurons();

    const auto requested_fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();
    const auto placed_fraction_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, requested_number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_EQ(requested_number_neurons, placed_number_neurons);
    ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
}

TEST_F(NeuronAssignmentTest, testPerRankInitializeMultipleSubdomains) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    auto accumulated_placed_neurons = 0;
    auto accumulated_ratio_excitatory_neurons = 0.0;

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

        ASSERT_EQ(placed_number_neurons, golden_number_neurons);
        ASSERT_NEAR(golden_fraction_excitatory_neurons, placed_ratio_excitatory_neurons, static_cast<double>(number_subdomains) / golden_number_neurons);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSizesSingleSubdomain) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnpr.initialize();

    const auto placed_number_neurons = sfnpr.get_number_placed_neurons();

    const auto& positions = sfnpr.get_neuron_positions_in_subdomains();
    const auto& types = sfnpr.get_neuron_types_in_subdomains();
    const auto placed_number_neurons_in_subdomain = sfnpr.get_number_neurons_in_subdomains();

    ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
    ASSERT_EQ(placed_number_neurons, positions.size());
    ASSERT_EQ(placed_number_neurons, types.size());
    ASSERT_EQ(placed_number_neurons, sfnpr.get_local_area_translator()->get_number_neurons_in_total());
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSizeMultipleSubdomains) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    size_t accumulated_placed_neurons = 0;

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        accumulated_placed_neurons += placed_number_neurons;

        const auto& all_positions = sfnpr.get_neuron_positions_in_subdomains();
        const auto& all_types = sfnpr.get_neuron_types_in_subdomains();
        const auto all_placed_neurons_in_subdomains = sfnpr.get_number_neurons_in_subdomains();

        ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
        ASSERT_EQ(placed_number_neurons, all_positions.size());
        ASSERT_EQ(placed_number_neurons, all_types.size());
        ASSERT_EQ(placed_number_neurons, sfnpr.get_local_area_translator()->get_number_neurons_in_total());
    }

    ASSERT_EQ(accumulated_placed_neurons, golden_number_ranks * golden_number_neurons);
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSemanticSingleSubdomain) {
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnpr.initialize();

    const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnpr.get_neuron_positions_in_subdomains();
    const auto& types = sfnpr.get_neuron_types_in_subdomains();

    const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
    ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1.0 / golden_number_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto neurons_per_dimension = pow(golden_number_neurons, 1. / 3);
    const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

    std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

    for (const auto& position : positions) {
        ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
        auto cast_position = Vec3s{ position / golden_um_per_neuron };

        const auto x = cast_position.get_x();
        const auto y = cast_position.get_y();
        const auto z = cast_position.get_z();

        ASSERT_LE(x, number_boxes);
        ASSERT_LE(y, number_boxes);
        ASSERT_LE(z, number_boxes);

        const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

        ASSERT_FALSE(flag);
        box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
    }
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSemanticMultipleSubdomains) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto golden_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100;

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

        const auto& all_positions = sfnpr.get_neuron_positions_in_subdomains();
        const auto& all_types = sfnpr.get_neuron_types_in_subdomains();

        const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(all_types);
        ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, static_cast<double>(number_subdomains) / golden_number_neurons) << golden_number_neurons;

        const auto& positions = sfnpr.get_neuron_positions_in_subdomains();
        const auto& types = sfnpr.get_neuron_types_in_subdomains();

        ASSERT_EQ(placed_number_neurons, golden_number_neurons);
        ASSERT_EQ(positions.size(), golden_number_neurons);
        ASSERT_EQ(types.size(), golden_number_neurons);
        ASSERT_EQ(sfnpr.get_local_area_translator()->get_number_neurons_in_total(), golden_number_neurons);
    }
}

TEST_F(NeuronAssignmentTest, testFileLoadSingleSubdomain) {
    std::vector<Vec3d> positions{};
    std::vector<RelearnTypes::area_id> area_ids{};
    std::vector<RelearnTypes::area_name> area_names{};
    std::vector<SignalType> types{};

    NeuronAssignmentAdapter::generate_random_neurons(positions, area_ids, area_names, types, mt);

    const auto number_neurons = positions.size();

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    MultipleSubdomainsFromFile sff{ "neurons.tmp", {}, part };

    sff.initialize();

    const auto& loaded_positions = sff.get_neuron_positions_in_subdomains();
    const auto& loaded_types = sff.get_neuron_types_in_subdomains();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto& curr_pos = positions[neuron_id];
        const auto& curr_loaded_pos = loaded_positions[neuron_id];

        ASSERT_NEAR(curr_pos.get_x(), curr_loaded_pos.get_x(), eps);
        ASSERT_NEAR(curr_pos.get_y(), curr_loaded_pos.get_y(), eps);
        ASSERT_NEAR(curr_pos.get_z(), curr_loaded_pos.get_z(), eps);

        const auto& curr_id = area_ids[neuron_id];
        const auto& curr_loaded_id = sff.get_local_area_translator()->get_area_id_for_neuron_id(neuron_id);

        ASSERT_EQ(curr_id, curr_loaded_id);

        const auto& curr_name = area_names[curr_id];
        const auto& curr_loaded_name = sff.get_local_area_translator()->get_area_name_for_neuron_id(curr_loaded_id);

        ASSERT_EQ(curr_name, curr_loaded_name);

        const auto& curr_type = types[neuron_id];
        const auto& curr_loaded_type = loaded_types[neuron_id];

        ASSERT_EQ(curr_type, curr_loaded_type);
    }
}

TEST_F(NeuronAssignmentTest, testFileLoadNetworkSingleSubdomain) {
    std::vector<Vec3d> positions{};
    std::vector<RelearnTypes::area_id> area_ids{};
    std::vector<RelearnTypes::area_name> area_names{};
    std::vector<SignalType> types{};

    NeuronAssignmentAdapter::generate_random_neurons(positions, area_ids, area_names, types, mt);

    const auto number_neurons = positions.size();

    const auto& synapses = NetworkGraphAdapter::generate_local_synapses(number_neurons, mt);

    write_synapses_to_file(synapses, ".");

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    MultipleSubdomainsFromFile sff{ "neurons.tmp", ".", part };

    sff.initialize();

    const auto loader = sff.get_synapse_loader();

    const auto& [static_synapses, plastic_synapses] = loader->load_synapses(std::make_unique<Essentials>());

    const auto& [local_synapses, in_synapses, out_synapses] = plastic_synapses;

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::plastic_synapse_weight> synapse_map{};

    for (const auto& [target, source, weight] : local_synapses) {
        synapse_map[{ target, source }] += weight;
    }

    for (const auto& [target, source, weight] : synapses) {
        synapse_map[{ target, source }] -= weight;
    }

    for (const auto& [_, weight] : synapse_map) {
        ASSERT_NEAR(weight, 0.0, eps);
    }
}

TEST_F(NeuronAssignmentTest, testFileGivenInputONCE) {
    auto path_to_neurons = get_relearn_path() / "input/positions.txt";
    auto path_to_synapses = get_relearn_path() / "input/";

    const auto part = std::make_shared<Partition>(1, MPIRank::root_rank());
    MultipleSubdomainsFromFile sff{ path_to_neurons, path_to_synapses, part };

    sff.initialize();

    const auto& positions = sff.get_neuron_positions_in_subdomains();

    ASSERT_EQ(positions.size(), 51842);

    const auto sl = sff.get_synapse_loader();

    const auto& [static_synapses, plastic_synapses] = sl->load_synapses(std::make_unique<Essentials>());

    const auto& [local_synapses, in_synapses, out_synapses] = plastic_synapses;

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    for (const auto& [source_id, target_id, weight] : local_synapses) {
        ASSERT_EQ(weight, 1);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesEmptyPositionPath) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(MultipleSubdomainsFromFile msff(std::filesystem::path(""), {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesNonExistentPositionPath) {
    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(MultipleSubdomainsFromFile msff(std::filesystem::path("./asfhasdfb�aslidhsdjfnasd"), {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesNonExistentFiles) {
    namespace fs = std::filesystem;

    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto directory = fs::path("./temp_dir");

    if (fs::exists(directory)) {
        for (const auto& path : directory) {
            if (path.string()[0] == '.') {
                continue;
            }
            fs::remove_all(path);
        }
    } else {
        fs::create_directory(directory);
    }

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(MultipleSubdomainsFromFile msff(directory, {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesEmptyFiles) {
    namespace fs = std::filesystem;

    const auto golden_number_ranks = MPIRankAdapter::get_adjusted_random_number_ranks(mt);
    const auto directory = fs::path("./temp_dir");

    if (fs::exists(directory)) {
        for (const auto& path : directory) {
            if (path.string()[0] == '.') {
                continue;
            }

            fs::remove_all(path);
        }
    } else {
        fs::create_directory(directory);
    }

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        auto position_path = directory / ("rank_" + std::to_string(rank.get_rank()) + "_positions.txt");
        std::ofstream out_file{ position_path };
        out_file.flush();
    }

    for (const auto rank : MPIRank::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(MultipleSubdomainsFromFile msff(directory, {}, partition);, RelearnException);
    }
}
