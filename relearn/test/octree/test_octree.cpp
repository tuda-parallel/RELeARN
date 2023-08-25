/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_octree.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/Cells.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/map.hpp>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, NaiveCell>;
TYPED_TEST_SUITE(OctreeTest, test_types);

TYPED_TEST(OctreeTest, testConstructor) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
    ASSERT_EQ(octree.get_simulation_box_minimum(), min);
    ASSERT_EQ(octree.get_simulation_box_maximum(), max);

    const auto virtual_neurons = OctreeAdapter::extract_virtual_neurons(octree.get_root());

    std::map<size_t, size_t> level_to_count{};

    for (const auto& id : virtual_neurons | ranges::views::values) {
        level_to_count[id]++;
    }

    ASSERT_EQ(level_to_count.size(), level_of_branch_nodes + 1);

    for (auto level = 0U; level <= level_of_branch_nodes; level++) {
        const auto expected_elements = std::pow(8U, level);

        if (level == level_of_branch_nodes) {
            ASSERT_EQ(octree.get_num_local_trees(), expected_elements);
        }

        ASSERT_EQ(level_to_count[level], expected_elements);
    }
}

TYPED_TEST(OctreeTest, testConstructorExceptions) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    ASSERT_THROW(OctreeImplementation<TypeParam> octree(max, min, level_of_branch_nodes), RelearnException);
}

TYPED_TEST(OctreeTest, testInsertNeurons) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    std::vector<std::pair<Vec3d, NeuronID>> placed_neurons = OctreeAdapter::template extract_neurons_tree<TypeParam>(octree);

    ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

    ranges::sort(neurons_to_place, std::greater{}, element<1>);
    ranges::sort(placed_neurons, std::greater{}, element<1>);

    for (auto i = 0; i < neurons_to_place.size(); i++) {
        const auto& expected_neuron = neurons_to_place[i];
        const auto& found_neuron = placed_neurons[i];

        ASSERT_EQ(expected_neuron, found_neuron);
    }
}

TYPED_TEST(OctreeTest, testInsertNeuronsExceptions) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        const auto rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, this->mt);

        const Vec3d pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        const Vec3d pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(octree.insert(position, NeuronID::uninitialized_id()), RelearnException);

        ASSERT_THROW(octree.insert(pos_invalid_x_max, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_y_max, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_z_max, id), RelearnException);

        ASSERT_THROW(octree.insert(pos_invalid_x_min, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_y_min, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_z_min, id), RelearnException);
    }
}

TYPED_TEST(OctreeTest, testStructure) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    auto* root = octree.get_root();

    std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};
    octree_nodes.emplace(root, size_t(0));

    while (!octree_nodes.empty()) {
        const auto [current_node, level] = octree_nodes.top();

        octree_nodes.pop();
        ASSERT_EQ(level, current_node->get_level());
        ASSERT_TRUE(current_node->get_mpi_rank() == my_rank);

        if (current_node->is_parent()) {
            const auto& childs = current_node->get_children();
            auto one_child_exists = false;

            for (auto i = 0; i < 8; i++) {
                const auto child = childs[i];
                if (child != nullptr) {
                    octree_nodes.emplace(child, level + 1);

                    const auto& subcell_size = child->get_cell().get_size();
                    const auto& expected_subcell_size = current_node->get_cell().get_size_for_octant(i);

                    ASSERT_EQ(expected_subcell_size, subcell_size);

                    one_child_exists = true;
                }
            }

            ASSERT_TRUE(one_child_exists);
        } else {
            const auto& cell = current_node->get_cell();
            const auto& opt_position = cell.get_neuron_position();

            ASSERT_TRUE(opt_position.has_value());

            const auto& position = opt_position.value();

            const auto& cell_size = cell.get_size();
            const auto& cell_min = std::get<0>(cell_size);
            const auto& cell_max = std::get<1>(cell_size);

            ASSERT_LE(cell_min.get_x(), position.get_x());
            ASSERT_LE(cell_min.get_y(), position.get_y());
            ASSERT_LE(cell_min.get_z(), position.get_z());

            ASSERT_LE(position.get_x(), cell_max.get_x());
            ASSERT_LE(position.get_y(), cell_max.get_y());
            ASSERT_LE(position.get_z(), cell_max.get_z());

            const auto neuron_id = cell.get_neuron_id();

            if (!neuron_id.is_initialized()) {
                ASSERT_LE(neuron_id, NeuronID{ number_neurons + num_additional_ids });
            }
        }
    }
}

TYPED_TEST(OctreeTest, testMemoryStructure) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    const auto root = octree.get_root();

    std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};

    while (!octree_nodes.empty()) {
        const auto* current_node = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->is_leaf()) {
            continue;
        }

        const auto& children = current_node->get_children();

        OctreeNode<AdditionalCellAttributes>* child_pointer = nullptr;
        int child_id = -1;

        for (auto i = 0; i < 8; i++) {
            const auto child = children[i];
            if (child == nullptr) {
                continue;
            }

            octree_nodes.emplace(child);

            if (child_pointer == nullptr) {
                child_pointer = child;
                child_id = i;
            }

            auto ptr = child_pointer + i - child_id;
            ASSERT_EQ(ptr, child);
        }
    }
}
