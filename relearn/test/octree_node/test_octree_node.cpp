/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_octree_node.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/Cells.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Cell.h"
#include "structure/OctreeNode.h"
#include "structure/OctreeNodeHelper.h"
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

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, NaiveCell>;
TYPED_TEST_SUITE(OctreeNodeTest, test_types);

TYPED_TEST(OctreeNodeTest, testReset) {
    using AdditionalCellAttributes = TypeParam;

    OctreeNode<AdditionalCellAttributes> node{};

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_mpi_rank() == MPIRank::uninitialized_rank());
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        ASSERT_TRUE(children[i] == nullptr);
    }

    node.set_parent();

    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(this->mt);
    const auto rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, this->mt);

    node.set_rank(rank);

    std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct> other_nodes{};
    for (auto i = 0; i < Constants::number_oct; i++) {
        node.set_child(&(other_nodes[i]), i);
    }

    node.reset();

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_mpi_rank() == MPIRank::uninitialized_rank());
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& new_children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        ASSERT_TRUE(new_children[i] == nullptr);
    }
}

TYPED_TEST(OctreeNodeTest, testSetterGetter) {
    using AdditionalCellAttributes = TypeParam;

    OctreeNode<AdditionalCellAttributes> node{};

    node.set_parent();

    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(this->mt);
    const auto rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    node.set_rank(rank);
    node.set_level(level);

    std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct> other_nodes{};
    for (auto i = 0; i < Constants::number_oct; i++) {
        node.set_child(&(other_nodes[i]), i);
    }

    ASSERT_TRUE(node.is_parent());
    ASSERT_TRUE(node.get_mpi_rank() == rank);
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);
    ASSERT_EQ(node.get_level(), level);

    const auto& children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == &(other_nodes[i]));
        ASSERT_TRUE(children[i] == &(other_nodes[i]));
    }

    const auto ub = Constants::number_oct * 100 + 100;

    for (auto i = 0; i < ub; i++) {
        if (i < Constants::number_oct) {
            continue;
        }

        ASSERT_THROW(node.set_child(nullptr, i), RelearnException);
        ASSERT_THROW(node.set_child(&node, i), RelearnException);
        ASSERT_THROW(auto tmp = node.get_child(i), RelearnException);
    }
}

TYPED_TEST(OctreeNodeTest, testLocal) {
    using AdditionalCellAttributes = TypeParam;

    OctreeNode<AdditionalCellAttributes> node{};
    const auto my_rank = MPIWrapper::get_my_rank();

    for (const auto rank : MPIRank::range(1000)) {
        node.set_rank(rank);

        if (rank == my_rank) {
            ASSERT_TRUE(node.is_local());
        } else {
            ASSERT_FALSE(node.is_local());
        }
    }
}

TYPED_TEST(OctreeNodeTest, testInsert) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    const auto num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        auto tmp = node.insert(pos, id);
    }

    std::vector<std::pair<Vec3d, NeuronID>> placed_neurons = OctreeAdapter::template extract_neurons<AdditionalCellAttributes>(&node);

    ranges::sort(neurons_to_place, std::greater{}, element<1>);
    ranges::sort(placed_neurons, std::greater{}, element<1>);

    ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

    for (auto i = 0; i < neurons_to_place.size(); i++) {
        const auto& expected_neuron = neurons_to_place[i];
        const auto& found_neuron = placed_neurons[i];

        ASSERT_EQ(expected_neuron, found_neuron);
    }
}

TYPED_TEST(OctreeNodeTest, testInsertByHand) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const Vec3d min{ 0.0, 0.0, 0.0 };
    const Vec3d max{ 100.0, 100.0, 100.0 };

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(Vec3d{ 25.0, 25.0, 25.0 });

    auto* _1 = node.insert(Vec3d{ 25.0, 25.0, 75.0 }, NeuronID::virtual_id());
    auto* _2 = node.insert(Vec3d{ 25.0, 75.0, 25.0 }, NeuronID::virtual_id());
    auto* _3 = node.insert(Vec3d{ 75.0, 25.0, 25.0 }, NeuronID::virtual_id());
    auto* _4 = node.insert(Vec3d{ 25.0, 75.0, 75.0 }, NeuronID::virtual_id());
    auto* _5 = node.insert(Vec3d{ 75.0, 25.0, 75.0 }, NeuronID::virtual_id());
    auto* _6 = node.insert(Vec3d{ 75.0, 75.0, 25.0 }, NeuronID::virtual_id());
    auto* _7 = node.insert(Vec3d{ 75.0, 75.0, 75.0 }, NeuronID::virtual_id());

    ASSERT_TRUE(node.is_parent());
    ASSERT_FALSE(node.is_leaf());
    ASSERT_EQ(node.get_level(), 0);

    for (auto child_id = 0; child_id < Constants::number_oct; child_id++) {
        auto* child = node.get_child(child_id);
        ASSERT_NE(nullptr, child);

        ASSERT_FALSE(child->is_parent());
        ASSERT_TRUE(child->is_leaf());

        for (auto i = 0; i < Constants::number_oct; i++) {
            ASSERT_EQ(nullptr, child->get_child(i));
        }
    }

    auto* _10 = node.insert(Vec3d{ 24.0, 24.0, 24.0 }, NeuronID(11));
    auto* _11 = node.insert(Vec3d{ 24.0, 24.0, 76.0 }, NeuronID(22));
    auto* _12 = node.insert(Vec3d{ 24.0, 76.0, 24.0 }, NeuronID(33));
    auto* _13 = node.insert(Vec3d{ 76.0, 24.0, 24.0 }, NeuronID(44));
    auto* _14 = node.insert(Vec3d{ 24.0, 76.0, 76.0 }, NeuronID(55));
    auto* _15 = node.insert(Vec3d{ 76.0, 24.0, 76.0 }, NeuronID(66));
    auto* _16 = node.insert(Vec3d{ 76.0, 76.0, 24.0 }, NeuronID(77));
    auto* _17 = node.insert(Vec3d{ 76.0, 76.0, 76.0 }, NeuronID(88));

    ASSERT_TRUE(node.is_parent());
    ASSERT_FALSE(node.is_leaf());
    ASSERT_EQ(node.get_level(), 0);

    for (auto child_id = 0; child_id < Constants::number_oct; child_id++) {
        auto* child = node.get_child(child_id);
        ASSERT_NE(nullptr, child);

        ASSERT_FALSE(child->is_parent());
        ASSERT_TRUE(child->is_leaf());

        for (auto i = 0; i < Constants::number_oct; i++) {
            ASSERT_EQ(nullptr, child->get_child(i));
        }
    }
}

TYPED_TEST(OctreeNodeTest, testContains) {
    const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(100, this->mt);
    const auto mpi_rank = MPIRankAdapter::get_random_mpi_rank(20, this->mt);

    RankNeuronId rni{ mpi_rank, neuron_id };

    using AdditionalCellAttributes = TypeParam;

    for (const auto id_iterator : NeuronID::range(100)) {
        for (const auto rank_iterator : MPIRank::range(20)) {
            OctreeNode<AdditionalCellAttributes> node{};
            node.set_cell_neuron_id(id_iterator);
            node.set_rank(rank_iterator);

            const auto flag = node.contains(rni);

            if (node.is_leaf() && node.get_mpi_rank() == mpi_rank && node.get_cell_neuron_id() == neuron_id) {
                ASSERT_TRUE(flag);
            } else {
                ASSERT_FALSE(flag);
            }

            node.set_parent();

            ASSERT_FALSE(node.contains(rni));
        }
    }
}

TYPED_TEST(OctreeNodeTest, testLevel) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    std::uint16_t level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(level);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        auto tmp = node.insert(pos, id /*, my_rank */);
    }

    ASSERT_EQ(node.get_level(), level);
    Stack<std::pair<const OctreeNode<AdditionalCellAttributes>*, const OctreeNode<AdditionalCellAttributes>*>> stack;
    for (const auto* child : node.get_children()) {
        if (child != nullptr) {
            stack.emplace_back(&node, child);
        }
    }

    while (!stack.empty()) {
        const auto& [parent, child] = stack.pop_back();
        const auto parent_level = parent->get_level();
        const auto child_level = child->get_level();
        ASSERT_EQ(parent_level + 1, child_level);
        if (child->is_parent()) {
            for (const auto* new_child : child->get_children()) {
                if (new_child != nullptr) {
                    stack.emplace_back(child, new_child);
                }
            }
        }
    }
}

TYPED_TEST(OctreeNodeTest, testUpdateNode) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto midpoint = (max - min) / 2.0;
    const auto [mid_x, mid_y, mid_z] = midpoint;

    auto* child_1 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt), NeuronID(1));
    auto* child_2 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(mid_x, 0, 0), NeuronID(2));
    auto* child_3 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(0, mid_y, 0), NeuronID(3));
    auto* child_4 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(0, 0, mid_z), NeuronID(4));
    auto* child_5 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(0, mid_y, mid_z), NeuronID(5));
    auto* child_6 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(mid_x, 0, mid_z), NeuronID(6));
    auto* child_7 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(mid_x, mid_y, 0), NeuronID(7));
    auto* child_8 = node.insert(SimulationAdapter::get_random_position_in_box(min, min + midpoint, this->mt) + Vec3d(mid_x, mid_y, mid_z), NeuronID(8));

    auto golden_number_excitatory_dendrites = 0;
    auto golden_number_inhibitory_dendrites = 0;
    auto golden_number_excitatory_axons = 0;
    auto golden_number_inhibitory_axons = 0;

    Vec3d golden_position_excitatory_dendrites{ 0 };
    Vec3d golden_position_inhibitory_dendrites{ 0 };
    Vec3d golden_position_excitatory_axons{ 0 };
    Vec3d golden_position_inhibitory_axons{ 0 };

    if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_excitatory_dendrites += vacant_elements;
            child->set_cell_number_excitatory_dendrites(vacant_elements);

            golden_position_excitatory_dendrites += child->get_cell().get_neuron_position().value() * vacant_elements;
        }
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_inhibitory_dendrites += vacant_elements;
            child->set_cell_number_inhibitory_dendrites(vacant_elements);

            golden_position_inhibitory_dendrites += child->get_cell().get_neuron_position().value() * vacant_elements;
        }
    }

    if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_excitatory_axons += vacant_elements;
            child->set_cell_number_excitatory_axons(vacant_elements);

            golden_position_excitatory_axons += child->get_cell().get_neuron_position().value() * vacant_elements;
        }
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_inhibitory_axons += vacant_elements;
            child->set_cell_number_inhibitory_axons(vacant_elements);

            golden_position_inhibitory_axons += child->get_cell().get_neuron_position().value() * vacant_elements;
        }
    }

    OctreeNodeUpdater<AdditionalCellAttributes>::update_node(&node);

    const auto& cell = node.get_cell();

    if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
        ASSERT_EQ(cell.get_number_excitatory_dendrites(), golden_number_excitatory_dendrites);

        auto position = cell.get_excitatory_dendrites_position().value();
        auto scaled_position = position * golden_number_excitatory_dendrites;

        auto difference = scaled_position - golden_position_excitatory_dendrites;
        auto norm = difference.calculate_2_norm();

        ASSERT_NEAR(norm, 0.0, this->eps);
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
        ASSERT_EQ(cell.get_number_inhibitory_dendrites(), golden_number_inhibitory_dendrites);

        auto position = cell.get_inhibitory_dendrites_position().value();
        auto scaled_position = position * golden_number_inhibitory_dendrites;

        auto difference = scaled_position - golden_position_inhibitory_dendrites;
        auto norm = difference.calculate_2_norm();

        ASSERT_NEAR(norm, 0.0, this->eps);
    }

    if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
        ASSERT_EQ(cell.get_number_excitatory_axons(), golden_number_excitatory_axons);

        auto position = cell.get_excitatory_axons_position().value();
        auto scaled_position = position * golden_number_excitatory_axons;

        auto difference = scaled_position - golden_position_excitatory_axons;
        auto norm = difference.calculate_2_norm();

        ASSERT_NEAR(norm, 0.0, this->eps);
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
        ASSERT_EQ(cell.get_number_inhibitory_axons(), golden_number_inhibitory_axons);

        auto position = cell.get_inhibitory_axons_position().value();
        auto scaled_position = position * golden_number_inhibitory_axons;

        auto difference = scaled_position - golden_position_inhibitory_axons;
        auto norm = difference.calculate_2_norm();

        ASSERT_NEAR(norm, 0.0, this->eps);
    }
}

TYPED_TEST(OctreeNodeTest, testUpdateTree) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    const auto num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        auto tmp = node.insert(pos, id);
    }

    std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
    stack.push(&node);

    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type> vacant_excitatory_dendrites{};
    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type> vacant_inhibitory_dendrites{};
    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type> vacant_excitatory_axons{};
    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type> vacant_inhibitory_axons{};

    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, Vec3d> position_excitatory_dendrites{};
    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, Vec3d> position_inhibitory_dendrites{};
    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, Vec3d> position_excitatory_axons{};
    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, Vec3d> position_inhibitory_axons{};

    while (!stack.empty()) {
        OctreeNode<AdditionalCellAttributes>* current = stack.top();
        stack.pop();

        if (current->is_leaf()) {
            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_dendrites(vacant_elements);

                vacant_excitatory_dendrites[current] = vacant_elements;
                position_excitatory_dendrites[current] = current->get_cell().get_neuron_position().value();
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_dendrites(vacant_elements);

                vacant_inhibitory_dendrites[current] = vacant_elements;
                position_inhibitory_dendrites[current] = current->get_cell().get_neuron_position().value();
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_axons(vacant_elements);

                vacant_excitatory_axons[current] = vacant_elements;
                position_excitatory_axons[current] = current->get_cell().get_neuron_position().value();
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_axons(vacant_elements);

                vacant_inhibitory_axons[current] = vacant_elements;
                position_inhibitory_axons[current] = current->get_cell().get_neuron_position().value();
            }

            continue;
        }

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }

    OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&node);
    stack.push(&node);

    while (!stack.empty()) {
        OctreeNode<AdditionalCellAttributes>* current = stack.top();
        stack.pop();

        const auto& cell = current->get_cell();

        if (current->is_leaf()) {
            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                auto expected_vacant_elements = vacant_excitatory_dendrites[current];
                auto expected_position = position_excitatory_dendrites[current];

                auto vacant_elements = cell.get_number_excitatory_dendrites();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
                    auto position = cell.get_excitatory_dendrites_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                auto expected_vacant_elements = vacant_inhibitory_dendrites[current];
                auto expected_position = position_inhibitory_dendrites[current];

                auto vacant_elements = cell.get_number_inhibitory_dendrites();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
                    auto position = cell.get_inhibitory_dendrites_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                auto expected_vacant_elements = vacant_excitatory_axons[current];
                auto expected_position = position_excitatory_axons[current];

                auto vacant_elements = cell.get_number_excitatory_axons();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
                    auto position = cell.get_excitatory_axons_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                auto expected_vacant_elements = vacant_inhibitory_axons[current];
                auto expected_position = position_inhibitory_axons[current];

                auto vacant_elements = cell.get_number_inhibitory_axons();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
                    auto position = cell.get_inhibitory_axons_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            continue;
        }

        auto golden_number_excitatory_dendrites = 0;
        auto golden_number_inhibitory_dendrites = 0;
        auto golden_number_excitatory_axons = 0;
        auto golden_number_inhibitory_axons = 0;

        Vec3d golden_position_excitatory_dendrites{ 0 };
        Vec3d golden_position_inhibitory_dendrites{ 0 };
        Vec3d golden_position_excitatory_axons{ 0 };
        Vec3d golden_position_inhibitory_axons{ 0 };

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);

                const auto& child_cell = child->get_cell();

                if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                    auto vacant_elements = child_cell.get_number_excitatory_dendrites();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_excitatory_dendrites_position().has_value());
                        auto position = child_cell.get_excitatory_dendrites_position().value();

                        golden_number_excitatory_dendrites += vacant_elements;
                        golden_position_excitatory_dendrites += position * vacant_elements;
                    }
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                    auto vacant_elements = child_cell.get_number_inhibitory_dendrites();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_inhibitory_dendrites_position().has_value());
                        auto position = child_cell.get_inhibitory_dendrites_position().value();

                        golden_number_inhibitory_dendrites += vacant_elements;
                        golden_position_inhibitory_dendrites += position * vacant_elements;
                    }
                }

                if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                    auto vacant_elements = child_cell.get_number_excitatory_axons();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_excitatory_axons_position().has_value());
                        auto position = child_cell.get_excitatory_axons_position().value();

                        golden_number_excitatory_axons += vacant_elements;
                        golden_position_excitatory_axons += position * vacant_elements;
                    }
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                    auto vacant_elements = child_cell.get_number_inhibitory_axons();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_inhibitory_axons_position().has_value());
                        auto position = child_cell.get_inhibitory_axons_position().value();

                        golden_number_inhibitory_axons += vacant_elements;
                        golden_position_inhibitory_axons += position * vacant_elements;
                    }
                }
            }
        }

        if (golden_number_excitatory_dendrites != 0) {
            golden_position_excitatory_dendrites /= golden_number_excitatory_dendrites;
        }

        if (golden_number_inhibitory_dendrites != 0) {
            golden_position_inhibitory_dendrites /= golden_number_inhibitory_dendrites;
        }

        if (golden_number_excitatory_axons != 0) {
            golden_position_excitatory_axons /= golden_number_excitatory_axons;
        }

        if (golden_number_inhibitory_axons != 0) {
            golden_position_inhibitory_axons /= golden_number_inhibitory_axons;
        }

        if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
            auto vacant_elements = cell.get_number_excitatory_dendrites();
            ASSERT_EQ(vacant_elements, golden_number_excitatory_dendrites);

            if (vacant_elements != 0) {
                ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
                auto position = cell.get_excitatory_dendrites_position().value();
                ASSERT_EQ(position, golden_position_excitatory_dendrites);
            } else {
                ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
            }
        }

        if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
            auto vacant_elements = cell.get_number_inhibitory_dendrites();
            ASSERT_EQ(vacant_elements, golden_number_inhibitory_dendrites);

            if (vacant_elements != 0) {

                ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
                auto position = cell.get_inhibitory_dendrites_position().value();
                ASSERT_EQ(position, golden_position_inhibitory_dendrites);
            } else {
                ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
            }
        }

        if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
            auto vacant_elements = cell.get_number_excitatory_axons();
            ASSERT_EQ(vacant_elements, golden_number_excitatory_axons);

            if (vacant_elements != 0) {
                ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
                auto position = cell.get_excitatory_axons_position().value();
                ASSERT_EQ(position, golden_position_excitatory_axons);
            } else {
                ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
            }
        }

        if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
            auto vacant_elements = cell.get_number_inhibitory_axons();
            ASSERT_EQ(vacant_elements, golden_number_inhibitory_axons);

            if (vacant_elements != 0) {
                ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
                auto position = cell.get_inhibitory_axons_position().value();
                ASSERT_EQ(position, golden_position_inhibitory_axons);
            } else {
                ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
            }
        }
    }
}

TYPED_TEST(OctreeNodeTest, testMemoryLayout) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;
    const auto size_of_node = sizeof(OctreeNode<AdditionalCellAttributes>);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(this->mt);

    auto root = OctreeAdapter::get_standard_tree<AdditionalCellAttributes>(number_neurons, minimum, maximum, this->mt);

    std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, OctreeNode<AdditionalCellAttributes>*>> stack{};
    stack.emplace(&root, nullptr);

    std::vector<std::uint64_t> touched_rmas{};

    while (!stack.empty()) {
        auto [current_node, parent] = stack.top();
        stack.pop();

        if (current_node->is_leaf()) {
            continue;
        }

        for (auto* child : current_node->get_children()) {
            if (child != nullptr) {
                stack.emplace(child, current_node);
            }
        }

        const NeuronID saved_neuron_id = current_node->get_cell_neuron_id();
        ASSERT_TRUE(saved_neuron_id.is_virtual());
        const auto saved_rma_offset = saved_neuron_id.get_rma_offset();

        ASSERT_EQ(saved_rma_offset % size_of_node, 0);
        const auto virtual_node_index = saved_rma_offset / size_of_node;

        touched_rmas.emplace_back(virtual_node_index);

        if (parent == nullptr) {
            ASSERT_EQ(&root, current_node);
            ASSERT_EQ(saved_rma_offset, 0);

            const auto mh_offset = MH::get_offset_from_parent(current_node);
            ASSERT_EQ(mh_offset, 0);

            auto* mh_ptr = MH::get_node_from_offset(0);

            for (auto child_id = 0; child_id < Constants::number_oct; child_id++) {
                auto* child = current_node->get_child(child_id);
                if (child == nullptr) {
                    continue;
                }

                auto* expected_ptr = mh_ptr + child_id;
                ASSERT_EQ(expected_ptr, child);
            }

            continue;
        }

        const auto mh_offset = MH::get_offset_from_parent(current_node);

        ASSERT_EQ(mh_offset, saved_rma_offset);

        auto* mh_ptr = MH::get_node_from_offset(virtual_node_index);

        for (auto child_id = 0; child_id < Constants::number_oct; child_id++) {
            auto* child = current_node->get_child(child_id);
            if (child == nullptr) {
                continue;
            }

            auto* expected_ptr = mh_ptr + child_id;
            ASSERT_EQ(expected_ptr, child);
        }
    }

    std::ranges::sort(touched_rmas);

    for (auto idx = 0; idx < touched_rmas.size(); idx++) {
        ASSERT_EQ(idx * Constants::number_oct, touched_rmas[idx]);
    }
}

TYPED_TEST(OctreeNodeTest, testNodeExtractor) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    const auto num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        auto tmp = node.insert(pos, id);
    }

    std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
    stack.push(&node);

    std::vector<std::pair<Vec3d, typename OctreeNode<AdditionalCellAttributes>::counter_type>> excitatory_dendrites{};
    std::vector<std::pair<Vec3d, typename OctreeNode<AdditionalCellAttributes>::counter_type>> inhibitory_dendrites{};
    std::vector<std::pair<Vec3d, typename OctreeNode<AdditionalCellAttributes>::counter_type>> excitatory_axons{};
    std::vector<std::pair<Vec3d, typename OctreeNode<AdditionalCellAttributes>::counter_type>> inhibitory_axons{};

    while (!stack.empty()) {
        OctreeNode<AdditionalCellAttributes>* current = stack.top();
        stack.pop();

        if (current->is_leaf()) {
            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_dendrites(vacant_elements);

                if (vacant_elements != 0) {
                    excitatory_dendrites.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_dendrites(vacant_elements);

                if (vacant_elements != 0) {
                    inhibitory_dendrites.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_axons(vacant_elements);

                if (vacant_elements != 0) {
                    excitatory_axons.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto vacant_elements = RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_axons(vacant_elements);

                if (vacant_elements != 0) {
                    inhibitory_axons.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            continue;
        }

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }

    OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&node);

    ranges::sort(excitatory_dendrites);
    ranges::sort(inhibitory_dendrites);
    ranges::sort(excitatory_axons);
    ranges::sort(inhibitory_axons);

    using TT = OctreeNodeExtractor<AdditionalCellAttributes>;

    if (!AdditionalCellAttributes::has_excitatory_dendrite) {
        ASSERT_THROW(auto val = TT::get_all_positions_for(&node, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, ElementType::Dendrite, SignalType::Excitatory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, excitatory_dendrites);
    }

    if (!AdditionalCellAttributes::has_inhibitory_dendrite) {
        ASSERT_THROW(auto val = TT::get_all_positions_for(&node, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, ElementType::Dendrite, SignalType::Inhibitory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, inhibitory_dendrites);
    }

    if (!AdditionalCellAttributes::has_excitatory_axon) {
        ASSERT_THROW(auto val = TT::get_all_positions_for(&node, ElementType::Axon, SignalType::Excitatory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, ElementType::Axon, SignalType::Excitatory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, excitatory_axons);
    }

    if (!AdditionalCellAttributes::has_inhibitory_axon) {
        ASSERT_THROW(auto val = TT::get_all_positions_for(&node, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, ElementType::Axon, SignalType::Inhibitory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, inhibitory_axons);
    }
}
