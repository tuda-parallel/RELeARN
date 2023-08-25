/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_background_activity_io.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "io/NeuronIO.h"
#include "neurons/LocalAreaTranslator.h"
#include "io/BackgroundActivityIO.h"

#include <tuple>
#include "range/v3/action/sort.hpp"

TEST_F(BackgroundActivityIOTest, testRead) {
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_steps = RandomAdapter::get_random_integer(1000, 999999, mt);
    const auto num_ranks = RandomAdapter::get_random_integer(1, 10, mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);

    std::filesystem::path file_path{ "./background_activity.tmp" };

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    const auto num_changes = RandomAdapter::get_random_integer(10, 50, mt);

    std::vector<std::tuple<RelearnTypes::step_type, std::string, std::vector<NeuronID>>> gold;

    const auto area_names = NeuronAssignmentAdapter::get_random_area_names(num_neurons, mt);
    const auto num_areas = area_names.size();
    const auto neuron_to_area_name = NeuronAssignmentAdapter::get_random_area_ids(num_areas, num_neurons, mt);
    const auto local_area_translator = std::make_shared<LocalAreaTranslator>(area_names, neuron_to_area_name);

    for (auto i = 0; i < num_changes; i++) {
        const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, num_steps, mt);

        const auto p = RandomAdapter::get_random_string(RandomAdapter::get_random_integer(1, 20, mt), mt);
        const auto num_selected_neurons = RandomAdapter::get_random_integer<size_t>(1, num_neurons, mt);
        const auto& neurons = NeuronIdAdapter::get_random_rank_neuron_ids(num_neurons, num_ranks, num_selected_neurons, mt);

        const auto num_fake_area_names = RandomAdapter::get_random_integer(0, 10, mt);
        const auto num_real_area_names = RandomAdapter::get_random_integer<size_t>(0, num_areas, mt);

        auto local_neurons = neurons | ranges::view::filter([my_rank](const auto& rni) { return rni.get_rank() == my_rank; }) | ranges::view::transform([](const auto& rni) { return rni.get_neuron_id(); })
            | ranges::to_vector;

        of << step << " " << p << " ";

        for (auto i = 0; i < num_fake_area_names; i++) {
            of << NeuronAssignmentAdapter::get_random_area_name(mt) << " ";
        }

        auto real_area_names = area_names;
        real_area_names = actions::shuffle(real_area_names, mt);
        for (auto i = 0; i < num_real_area_names; i++) {
            of << real_area_names[i] << " ";
            const auto& neurons_in_area = local_area_translator->get_neuron_ids_in_area(local_area_translator->get_area_id_for_area_name(real_area_names[i]));
            std::copy(neurons_in_area.begin(), neurons_in_area.end(), std::back_inserter(local_neurons));
        }

        for (const auto& neuron_id : neurons) {
            of << neuron_id.get_rank().get_rank() << ":" << neuron_id.get_neuron_id().get_neuron_id() + 1 << " ";
        }
        of << "\n";
        std::sort(local_neurons.begin(), local_neurons.end());
        local_neurons.erase(unique(local_neurons.begin(), local_neurons.end()), local_neurons.end());

        gold.emplace_back(step, p, local_neurons);
    }
    of.close();

    std::sort(gold.begin(), gold.end(), [](const auto& t1, const auto& t2) {
        return std::get<0>(t1) < std::get<0>(t2);
    });

    ASSERT_EQ(num_changes, gold.size());
    const auto loaded = BackgroundActivityIO::load_background_activity(file_path, my_rank, local_area_translator);

    ASSERT_EQ(gold.size(), loaded.size());

    auto last_step = 0;

    for (auto i = 0; i < num_changes; i++) {
        const auto& [loaded_step, loaded_type, loaded_neurons] = loaded[i];
        const auto& [step, type, neurons] = gold[i];

        ASSERT_TRUE(last_step <= loaded_step);
        last_step = loaded_step;

        ASSERT_EQ(step, loaded_step);
        ASSERT_EQ(type, loaded_type);
        ASSERT_EQ(neurons, loaded_neurons);
    }
}