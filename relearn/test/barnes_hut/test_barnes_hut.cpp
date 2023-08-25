/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_barnes_hut.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "neurons/enums/UpdateStatus.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"
#include "util/ranges/Functional.hpp"

#include <memory>
#include <range/v3/view/filter.hpp>
#include <stack>
#include <tuple>
#include <vector>

TEST_F(BarnesHutTest, testBarnesHutGetterSetter) {
    using additional_cell_attributes = BarnesHutCell;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    auto octree = std::make_shared<OctreeImplementation<additional_cell_attributes>>(min, max, 0);

    ASSERT_NO_THROW(BarnesHut algorithm(octree););

    BarnesHut algorithm(octree);

    ASSERT_EQ(algorithm.get_acceptance_criterion(), Constants::bh_default_theta);

    const auto random_acceptance_criterion = RandomAdapter::get_random_double<double>(0.0, Constants::bh_max_theta, mt);

    ASSERT_NO_THROW(algorithm.set_acceptance_criterion(random_acceptance_criterion));
    ASSERT_EQ(algorithm.get_acceptance_criterion(), random_acceptance_criterion);
}

TEST_F(BarnesHutTest, testUpdateFunctor) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) * 0 + 4;
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto& axons = SynapticElementsAdapter::create_axons(number_neurons, mt);
    const auto& excitatory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Excitatory, mt);
    const auto& inhibitory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Inhibitory, mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons, mt);

    auto octree = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, 0);

    std::map<NeuronID::value_type, Vec3d> positions{};
    for (const auto& [position, id] : neurons_to_place) {
        octree->insert(position, id);
        positions[id.get_neuron_id()] = position;
    }

    octree->initializes_leaf_nodes(number_neurons);

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);

    BarnesHut barnes_hut(octree);
    barnes_hut.set_synaptic_elements(axons, excitatory_dendrites, inhibitory_dendrites);
    barnes_hut.set_neuron_extra_infos(extra_infos);

    const auto update_status = NeuronTypesAdapter::get_update_status(number_neurons, mt);

    const auto disabled_neurons = NeuronID::range(number_neurons)
        | ranges::views::filter(equal_to(UpdateStatus::Disabled), lookup(update_status, &NeuronID::get_neuron_id))
        | ranges::to_vector;

    extra_infos->set_disabled_neurons(disabled_neurons);

    ASSERT_NO_THROW(barnes_hut.update_octree());

    std::stack<OctreeNode<BarnesHutCell>*> stack{};
    stack.push(octree->get_root());

    while (!stack.empty()) {
        auto* node = stack.top();
        stack.pop();

        const auto& cell = node->get_cell();

        if (node->is_leaf()) {
            const auto id = cell.get_neuron_id();
            const auto local_id = id.get_neuron_id();

            ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
            ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());

            const auto& golden_position = positions[local_id];

            ASSERT_EQ(cell.get_excitatory_dendrites_position().value(), golden_position);
            ASSERT_EQ(cell.get_inhibitory_dendrites_position().value(), golden_position);

            if (update_status[local_id] == UpdateStatus::Disabled) {
                ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
            } else {
                const auto& golden_excitatory_dendrites = excitatory_dendrites->get_free_elements(id);
                const auto& golden_inhibitory_dendrites = inhibitory_dendrites->get_free_elements(id);

                ASSERT_EQ(cell.get_number_excitatory_dendrites(), golden_excitatory_dendrites);
                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), golden_inhibitory_dendrites);
            }
        } else {
            auto total_number_excitatory_dendrites = 0;
            auto total_number_inhibitory_dendrites = 0;

            Vec3d excitatory_dendrites_position = { 0, 0, 0 };
            Vec3d inhibitory_dendrites_position = { 0, 0, 0 };

            for (auto* child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                const auto number_excitatory_dendrites = child_cell.get_number_excitatory_dendrites();
                const auto number_inhibitory_dendrites = child_cell.get_number_inhibitory_dendrites();

                total_number_excitatory_dendrites += number_excitatory_dendrites;
                total_number_inhibitory_dendrites += number_inhibitory_dendrites;

                if (number_excitatory_dendrites != 0) {
                    const auto& opt = child_cell.get_excitatory_dendrites_position();
                    ASSERT_TRUE(opt.has_value());
                    const auto& position = opt.value();

                    excitatory_dendrites_position += (position * number_excitatory_dendrites);
                }

                if (number_inhibitory_dendrites != 0) {
                    const auto& opt = child_cell.get_inhibitory_dendrites_position();
                    ASSERT_TRUE(opt.has_value());
                    const auto& position = opt.value();

                    inhibitory_dendrites_position += (position * number_inhibitory_dendrites);
                }

                stack.push(child);
            }

            ASSERT_EQ(total_number_excitatory_dendrites, cell.get_number_excitatory_dendrites());
            ASSERT_EQ(total_number_inhibitory_dendrites, cell.get_number_inhibitory_dendrites());

            if (total_number_excitatory_dendrites == 0) {
                ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
            } else {
                const auto& opt = cell.get_excitatory_dendrites_position();
                ASSERT_TRUE(opt.has_value());
                const auto& position = opt.value();

                const auto& diff = (excitatory_dendrites_position / total_number_excitatory_dendrites) - position;
                const auto& norm = diff.calculate_2_norm();

                ASSERT_NEAR(norm, 0.0, eps);
            }

            if (total_number_inhibitory_dendrites == 0) {
                ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
            } else {
                const auto& opt = cell.get_inhibitory_dendrites_position();
                ASSERT_TRUE(opt.has_value());
                const auto& position = opt.value();

                const auto& diff = (inhibitory_dendrites_position / total_number_inhibitory_dendrites) - position;
                const auto& norm = diff.calculate_2_norm();

                ASSERT_NEAR(norm, 0.0, eps);
            }
        }
    }
}
