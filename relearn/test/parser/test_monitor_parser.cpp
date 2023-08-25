/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_monitor_parser.h"

#include "adapter/helper/RankNeuronIdAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "io/parser/MonitorParser.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"
#include "util/shuffle/shuffle.h"

#include <memory>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/generate.hpp>
#include <range/v3/view/map.hpp>
#include <sstream>
#include <vector>

TEST_F(MonitorParserTest, testParseIds) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    size_t my_number_neurons = 0;

    const auto random_number_neurons = [this]() { return NeuronIdAdapter::get_random_number_neurons(mt); };

    const auto create_rank_neuron_ids = [this, my_rank, &my_number_neurons](const auto& rank_num_neurons_pair) {
        const auto& rank = std::get<0>(rank_num_neurons_pair);
        const auto& number_neurons = std::get<1>(rank_num_neurons_pair);

        if (rank == my_rank) {
            my_number_neurons = number_neurons;
        }

        return NeuronID::range(number_neurons)
            | ranges::views::transform([rank](const NeuronID& neuron_id) -> RankNeuronId { return { rank, neuron_id }; });
    };

    const auto rank_neuron_ids = ranges::views::zip(
                                     MPIRank::range(number_ranks),
                                     ranges::views::generate(random_number_neurons))
        | ranges::views::for_each(create_rank_neuron_ids)
        | ranges::to_vector
        | actions::shuffle(mt);

    std::stringstream ss{};
    ss << "0:1";

    for (const auto& rni : rank_neuron_ids) {
        if (rni.get_rank() != my_rank) {
            ss << ';' << RankNeuronIdAdapter::codify_rank_neuron_id(rni);
            continue;
        }

        const auto use_default = RandomAdapter::get_random_bool(mt);

        if (use_default) {
            ss << ";-1:" << rni.get_neuron_id().get_neuron_id() + 1;
        } else {
            ss << ';' << RankNeuronIdAdapter::codify_rank_neuron_id(rni);
        }
    }

    auto translator = std::make_shared<LocalAreaTranslator>(std::vector<RelearnTypes::area_name>({ "random" }), std::vector<RelearnTypes::area_id>({ 0 }));

    const auto& parsed_ids = MonitorParser::parse_my_ids(ss.str(), my_rank, translator);

    ASSERT_EQ(parsed_ids.size(), my_number_neurons);

    for (const auto neuron_id : NeuronID::range_id(my_number_neurons)) {
        ASSERT_EQ(parsed_ids[neuron_id], NeuronID(neuron_id));
    }
}

TEST_F(MonitorParserTest, testParseAreas) {

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto& area_names = NeuronAssignmentAdapter::get_random_area_names_specific(3, mt);
    const auto& area_ids = NeuronAssignmentAdapter::get_random_area_ids(3, number_neurons, mt);
    auto translator = std::make_shared<LocalAreaTranslator>(area_names, area_ids);

    std::string str = area_names[0] + ";" + area_names[1] + ";" + NeuronAssignmentAdapter::get_random_area_name(mt);

    const auto& parsed_ids = MonitorParser::parse_my_ids(str, MPIRank{ 0 }, translator);

    const auto num_neurons_in_areas = translator->get_number_neurons_in_area(0) + translator->get_number_neurons_in_area(1);
    ASSERT_EQ(parsed_ids.size(), num_neurons_in_areas);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto it = std::find(parsed_ids.begin(), parsed_ids.end(), NeuronID{ neuron_id });

        ASSERT_TRUE(area_ids[neuron_id] == 2 && it == parsed_ids.end() || area_ids[neuron_id] != 2 && it != parsed_ids.end());
    }
}

TEST_F(MonitorParserTest, testParseAreasRegex) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_regex_areas = RandomAdapter::get_random_integer(0, 10, mt);

    std::vector<RelearnTypes::area_name> area_names = NeuronAssignmentAdapter::get_random_area_names(10, mt);
    const auto random_areas = area_names.size();
    for (auto i = 0; i < num_regex_areas; i++) {
        area_names.push_back("REG" + RandomAdapter::get_random_string(RandomAdapter::get_random_integer(1, 20, mt), mt) + "EX");
    }

    const auto num_areas = area_names.size();
    const auto& area_ids = NeuronAssignmentAdapter::get_random_area_ids(num_areas, number_neurons, mt);
    auto translator = std::make_shared<LocalAreaTranslator>(area_names, area_ids);

    auto regex_neurons = 0;
    for (auto area_id = random_areas; area_id < area_names.size(); area_id++) {
        regex_neurons += translator->get_number_neurons_in_area(area_id);
    }

    std::string str = "REG.+EX";
    const auto& parsed_ids = MonitorParser::parse_my_ids(str, MPIRank{ 0 }, translator);

    ASSERT_EQ(parsed_ids.size(), regex_neurons);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto it = std::find(parsed_ids.begin(), parsed_ids.end(), NeuronID{ neuron_id });

        ASSERT_TRUE(area_ids[neuron_id] < random_areas && it == parsed_ids.end() || area_ids[neuron_id] >= random_areas && it != parsed_ids.end());
    }
}
