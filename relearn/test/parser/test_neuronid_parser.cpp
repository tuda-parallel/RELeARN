/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuronid_parser.h"

#include "adapter/helper/RankNeuronIdAdapter.h"
#include "adapter/random/RandomAdapter.h"

#include "adapter/mpi/MpiRankAdapter.h"

#include "io/parser/NeuronIdParser.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/ranges/Functional.hpp"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

TEST_F(NeuronIdParserTest, testParseDescriptionFixed) {
    auto checker = [](std::string_view description, MPIRank rank, NeuronID::value_type neuron_id) {
        auto opt_rni = NeuronIdParser::parse_description(description, rank);
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        RankNeuronId rni{ rank, NeuronID(neuron_id) };
        ASSERT_EQ(rni, parsed_rni);
    };

    checker("0:1", MPIRank(0), 0);
    checker("2:1", MPIRank(2), 0);
    checker("155:377", MPIRank(155), 376);
    checker("-1:17", MPIRank(5), 16);
}

TEST_F(NeuronIdParserTest, testUninitRank) {
    auto val1 = NeuronIdParser::parse_description("", MPIRank());
    ASSERT_FALSE(val1.has_value());

    auto val2 = NeuronIdParser::parse_description("155:377", MPIRank());
    ASSERT_FALSE(val2.has_value());

    auto val3 = NeuronIdParser::parse_description("-1:17", MPIRank());
    ASSERT_FALSE(val3.has_value());
}

TEST_F(NeuronIdParserTest, testParseDescriptionFail) {
    auto checker = [](std::string_view description, MPIRank default_rank) {
        auto opt_rni = NeuronIdParser::parse_description(description, default_rank);
        ASSERT_FALSE(opt_rni.has_value());
    };

    checker("0:1:0", MPIRank::root_rank());
    checker("5:-4", MPIRank::root_rank());
    checker("+0:1", MPIRank::root_rank());
    checker("AB:1", MPIRank::root_rank());
    checker("-5:2", MPIRank::root_rank());
    checker("0:", MPIRank::root_rank());
    checker("5;2", MPIRank::root_rank());
    checker("", MPIRank::root_rank());
}

TEST_F(NeuronIdParserTest, testParseDescriptionException) {
    auto checker = [](std::string_view description, MPIRank default_rank) {
        ASSERT_THROW(auto opt_rni = NeuronIdParser::parse_description(description, default_rank);, RelearnException);
    };

    checker("0:0", MPIRank::root_rank());
    checker("1:0", MPIRank::root_rank());
    checker("-1:0", MPIRank::root_rank());
    checker("24575:0", MPIRank::root_rank());
}

TEST_F(NeuronIdParserTest, testParseDescriptionRandom) {
    for (auto i = 0; i < 10000; i++) {
        const auto& [rni, descr] = RankNeuronIdAdapter::generate_random_rank_neuron_id_description(mt);

        auto opt_rni = NeuronIdParser::parse_description(descr, MPIRank(0));
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        ASSERT_EQ(rni, parsed_rni);
    }
}

TEST_F(NeuronIdParserTest, testParseDescriptions) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_neurons + 1);

    std::stringstream ss{};

    const auto& [first_rni, first_description] = RankNeuronIdAdapter::generate_random_rank_neuron_id_description(mt);
    rank_neuron_ids.emplace_back(first_rni);

    ss << first_description;

    for (auto i = 0; i < number_neurons; i++) {
        const auto& [new_rni, new_description] = RankNeuronIdAdapter::generate_random_rank_neuron_id_description(mt);
        ss << ';' << new_description;

        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto& parsed_rnis = NeuronIdParser::parse_multiple_description(ss.str(), MPIRank(3));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis);
}

TEST_F(NeuronIdParserTest, testParseDescriptionsFixed) {
    std::vector<RankNeuronId> rank_neuron_ids{
        { MPIRank(2), NeuronID(100) },
        { MPIRank(5), NeuronID(6) },
        { MPIRank(0), NeuronID(122) },
        { MPIRank(2), NeuronID(100) },
        { MPIRank(1674), NeuronID(1) },
        { MPIRank(89512), NeuronID(6) },
        { MPIRank(0), NeuronID(1) },
        { MPIRank(0), NeuronID(1) },
    };

    constexpr auto description_1 = "2:101;5:7;0:123;2:101;1674:2;89512:7;0:2;0:2";
    constexpr auto description_2 = "2:101;-1:7;0:123;2:101;1674:2;89512:7;0:2;0:2";
    constexpr auto description_3 = "2:101;5:7;-1:123;2:101;1674:2;89512:7;-1:2;0:2";
    constexpr auto description_4 = "2:101;5:7;-1:123;-8:801;2:101;6:;1674:2;-999:6;89512:7;-1:2;0:2";

    const auto& parsed_rnis_1 = NeuronIdParser::parse_multiple_description(description_1, MPIRank(3));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_1);

    const auto& parsed_rnis_2 = NeuronIdParser::parse_multiple_description(description_2, MPIRank(5));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_2);

    const auto& parsed_rnis_3 = NeuronIdParser::parse_multiple_description(description_3, MPIRank(0));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_3);

    const auto& parsed_rnis_4 = NeuronIdParser::parse_multiple_description(description_4, MPIRank(0));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_4);
}

TEST_F(NeuronIdParserTest, testExtractNeuronIDs) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_neurons + 2);

    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(100, mt);

    for (auto i = 0; i < number_neurons; i++) {
        const auto& [new_rni, _] = RankNeuronIdAdapter::generate_random_rank_neuron_id_description(mt);
        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto position_1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
    const auto position_2 = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);

    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_1, RankNeuronId(my_rank, NeuronID(42)));
    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_2, RankNeuronId(my_rank, NeuronID(9874)));

    const std::vector<NeuronID> golden_ids = rank_neuron_ids | ranges::views::filter(equal_to(my_rank), &RankNeuronId::get_rank) | ranges::views::transform([](const RankNeuronId& rni) {
        const auto& [rank, id] = rni;
        return NeuronID(id.get_neuron_id());
    }) | ranges::to_vector;

    const auto& extracted_ids = NeuronIdParser::extract_my_ids(rank_neuron_ids, my_rank);

    ASSERT_EQ(golden_ids, extracted_ids);
}

TEST_F(NeuronIdParserTest, testRemoveAndSort) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(
            NeuronIdAdapter::get_random_neuron_id(number_neurons, mt));
    }

    const auto& unique_and_filtered = NeuronIdParser::remove_duplicates_and_sort(neuron_ids);

    for (auto i = 0; i < unique_and_filtered.size() - 1; i++) {
        ASSERT_LE(unique_and_filtered[i].get_neuron_id(), unique_and_filtered[i + 1].get_neuron_id());
    }

    for (const auto& original_id : neuron_ids) {
        const auto pos = std::ranges::find(unique_and_filtered, original_id);
        ASSERT_NE(pos, unique_and_filtered.end());
    }

    for (const auto& new_id : unique_and_filtered) {
        const auto pos = std::ranges::find(neuron_ids, new_id);
        ASSERT_NE(pos, neuron_ids.end());
    }
}

TEST_F(NeuronIdParserTest, testRemoveAndSortException1) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(
            NeuronIdAdapter::get_random_neuron_id(number_neurons, mt));
    }

    const auto virtual_rma = RandomAdapter::get_random_integer<NeuronID::value_type>(0, 100000, mt);
    const auto position = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID(true, virtual_rma));

    ASSERT_THROW(const auto& unique_and_filtered = NeuronIdParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}

TEST_F(NeuronIdParserTest, testRemoveAndSortException2) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(
            NeuronIdAdapter::get_random_neuron_id(number_neurons, mt));
    }

    const auto position = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID{});

    ASSERT_THROW(const auto& unique_and_filtered = NeuronIdParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}
