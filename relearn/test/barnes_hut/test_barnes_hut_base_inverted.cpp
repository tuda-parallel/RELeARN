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
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/node_cache/NodeCacheAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "structure/OctreeNode.h"
#include "util/Vec3.h"

#include <stack>
#include <tuple>
#include <vector>

TEST_F(BarnesHutInvertedBaseTest, testACException) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto source_position = Vec3d{ 15.0, 15.0, 15.0 };

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta),
        RelearnException);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Axon, searched_signal_type, 0.0),
        RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion),
        RelearnException);

    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 1000.0, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion),
        RelearnException);
}

TEST_F(BarnesHutInvertedBaseTest, testACLeafDendrites) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationAdapter::get_random_position(mt);
        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(number_free_elements, 0);
        } else {
            node.set_cell_number_axons(0, number_free_elements);
        }

        const auto accept = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(accept, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(0, number_free_elements);
        } else {
            node.set_cell_number_axons(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testACLeafSamePositionDendrites) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);

    for (auto it = 0; it < 1000; it++) {
        const auto number_free_elements_excitatory = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 1000, mt);
        const auto number_free_elements_inhibitory = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 1000, mt);

        node.set_cell_number_axons(number_free_elements_excitatory, number_free_elements_inhibitory);

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(node_position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testACParentDendrite) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto& scaled_minimum = minimum / 10.0;
    const auto& scaled_maximum = maximum / 10.0;

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(scaled_minimum, scaled_maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);
    node.set_parent();

    node.set_cell_size(scaled_minimum, scaled_maximum);
    node.set_cell_neuron_position(node_position);

    const auto& cell_dimensions = scaled_maximum - scaled_minimum;
    const auto& maximum_cell_dimension = cell_dimensions.get_maximum();

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const auto distance = (node_position - position).calculate_2_norm();
        const auto quotient = maximum_cell_dimension / distance;

        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(number_free_elements, 0);
        } else {
            node.set_cell_number_axons(0, number_free_elements);
        }

        const auto status = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);

        if (acceptance_criterion > quotient) {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);
        } else {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand);
        }

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(0, number_free_elements);
        } else {
            node.set_cell_number_axons(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderException) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const Vec3d position{ 0.0 };
    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 10000.0, mt);

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, 0.0), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, Constants::bh_max_theta + eps), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, 0.0, true), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, Constants::bh_max_theta + eps, true), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion, true), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion, true), RelearnException);

    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, nullptr, ElementType::Axon, searched_signal_type, acceptance_criterion), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta, true), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, nullptr, ElementType::Axon, searched_signal_type, acceptance_criterion, true), RelearnException);
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderLeaf) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationAdapter::get_random_position(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(number_free_elements, 0);
        } else {
            node.set_cell_number_axons(0, number_free_elements);
        }

        const auto accept_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(1, accept_nodes.size());
        ASSERT_EQ(&node, accept_nodes[0]);

        const auto accept_nodes_early = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion, true);
        ASSERT_EQ(1, accept_nodes_early.size());
        ASSERT_EQ(&node, accept_nodes_early[0]);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(0, number_free_elements);
        } else {
            node.set_cell_number_axons(number_free_elements, 0);
        }

        const auto discard_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(discard_nodes.empty());

        const auto discard_nodes_early = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(discard_nodes_early.empty());
    }

    node.set_cell_number_axons(0, 0);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationAdapter::get_random_position(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);

        const auto accept_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(accept_nodes.empty());

        const auto discard_nodes_early = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(discard_nodes_early.empty());
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderNoDendrites) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_axons<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes_axon = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_axon.empty());
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderNoElements) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_synaptic_elements<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes_axon = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_axon.empty());
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsider) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderNoAxons) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_dendrites<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderDistributedTree) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderEarlyReturn) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, true);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderEarlyReturnDistributedTree) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, true);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand && current->get_mpi_rank() != MPIRank(0)) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronException) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const NeuronID neuron_id(1000000);
    const Vec3d position{ 0.0 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 10000.0, mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Axon, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Axon, searched_signal_type, Constants::bh_max_theta + eps);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion);, RelearnException);
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronNoAxons) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_axons<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const RankNeuronId searching_id{ MPIRank::root_rank(), neuron_id };
        auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron(searching_id, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_FALSE(found_target.has_value());
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronNoChoice) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_axons(2);
    root.set_cell_number_inhibitory_axons(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    for (auto it = 0; it < 1000; it++) {
        const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(0) }, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_FALSE(first_target_opt.has_value());

        auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(1) }, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(second_target_opt.has_value());

        auto [second_rank, second_id] = second_target_opt.value();
        ASSERT_EQ(second_rank, MPIRank::root_rank());
        ASSERT_EQ(second_id, NeuronID(0));
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronOneChoice) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(2));

    const auto first_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);
    auto _1 = root.insert(first_position, NeuronID(0));

    const auto second_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);
    auto _2 = root.insert(second_position, NeuronID(1));

    auto* first_node = OctreeAdapter::find_node<additional_cell_attributes>({ MPIRank::root_rank(), NeuronID(0) }, &root);
    auto* second_node = OctreeAdapter::find_node<additional_cell_attributes>({ MPIRank::root_rank(), NeuronID(1) }, &root);

    first_node->set_cell_number_excitatory_axons(1);
    first_node->set_cell_number_inhibitory_axons(1);
    second_node->set_cell_number_excitatory_axons(2);
    second_node->set_cell_number_inhibitory_axons(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
    const auto& position = SimulationAdapter::get_random_position(mt);

    auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(0) }, first_position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);
    ASSERT_TRUE(first_target_opt.has_value());

    auto [first_rank, first_id] = first_target_opt.value();
    ASSERT_EQ(first_rank, MPIRank(0));
    ASSERT_EQ(first_id, NeuronID(1));

    auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(1) }, second_position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);
    ASSERT_TRUE(second_target_opt.has_value());

    auto [second_rank, second_id] = second_target_opt.value();
    ASSERT_EQ(second_rank, MPIRank(0));
    ASSERT_EQ(second_id, NeuronID(0));
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronFullChoice) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const RankNeuronId searching_id{ MPIRank::root_rank(), neuron_id };
        auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron(searching_id, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(found_target.has_value());

        const auto found_id = found_target.value();
        ASSERT_NE(searching_id, found_id);

        ASSERT_TRUE(nodes.contains(found_id));
        ASSERT_GE(nodes.at(found_id)->get_cell().get_number_axons_for(searched_signal_type), 0);

        auto [target_rank, target_id] = found_target.value();

        ASSERT_EQ(target_rank, MPIRank::root_rank());
        ASSERT_NE(neuron_id, target_id);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronNoChoiceDistributed) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank(1));
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_axons(2);
    root.set_cell_number_inhibitory_axons(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    for (auto it = 0; it < 1000; it++) {
        const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(0) }, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(first_target_opt.has_value());

        auto [first_rank, first_id] = first_target_opt.value();
        ASSERT_EQ(first_rank, MPIRank(1));
        ASSERT_EQ(first_id, NeuronID(0));

        auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(1) }, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(second_target_opt.has_value());

        auto [second_rank, second_id] = second_target_opt.value();
        ASSERT_EQ(second_rank, MPIRank(1));
        ASSERT_EQ(second_id, NeuronID(0));
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronFullChoiceDistributed) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const RankNeuronId searching_id{ MPIRank::root_rank(), neuron_id };
        const auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron(searching_id, position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(found_target.has_value());

        const auto found_id = found_target.value();
        ASSERT_NE(searching_id, found_id);

        ASSERT_TRUE(nodes.contains(found_id));
        ASSERT_GE(nodes.at(found_id)->get_cell().get_number_axons_for(searched_signal_type), 0);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronsException) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const NeuronID neuron_id(1000000);
    const Vec3d position{ 0.0 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 10000.0, mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), neuron_id }, position, 1, nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Axon, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Axon, searched_signal_type, Constants::bh_max_theta + eps);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion);, RelearnException);
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronsNoChoice) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_axons(2);
    root.set_cell_number_inhibitory_axons(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    for (auto it = 0; it < 1000; it++) {
        const auto number_vacant_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto first_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), NeuronID(0) }, position, number_vacant_elements,
            &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(first_targets.empty());

        auto second_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), NeuronID(1) }, position, number_vacant_elements,
            &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(second_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = second_targets[i];
            ASSERT_EQ(rank, MPIRank::root_rank());

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_EQ(target_id, NeuronID(0));
            ASSERT_EQ(source_id, NeuronID(1));
            ASSERT_EQ(signal_type, searched_signal_type);
        }
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronsOneChoice) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(2));

    const auto first_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);
    auto _1 = root.insert(first_position, NeuronID(0));

    const auto second_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);
    auto _2 = root.insert(second_position, NeuronID(1));

    auto* first_node = OctreeAdapter::find_node<additional_cell_attributes>({ MPIRank::root_rank(), NeuronID(0) }, &root);
    auto* second_node = OctreeAdapter::find_node<additional_cell_attributes>({ MPIRank::root_rank(), NeuronID(1) }, &root);

    first_node->set_cell_number_excitatory_axons(1);
    first_node->set_cell_number_inhibitory_axons(1);
    second_node->set_cell_number_excitatory_axons(2);
    second_node->set_cell_number_inhibitory_axons(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto number_vacant_elements_1 = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
    const auto number_vacant_elements_2 = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
    const auto& position = SimulationAdapter::get_random_position(mt);

    auto first_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), NeuronID(0) }, first_position, number_vacant_elements_1,
        &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

    ASSERT_EQ(first_targets.size(), number_vacant_elements_1);

    for (RelearnTypes::counter_type i = 0; i < number_vacant_elements_1; i++) {
        const auto& [rank, creation_request] = first_targets[i];
        ASSERT_EQ(rank, MPIRank::root_rank());

        const auto& [target_id, source_id, signal_type] = creation_request;

        ASSERT_EQ(target_id, NeuronID(1));
        ASSERT_EQ(source_id, NeuronID(0));
        ASSERT_EQ(signal_type, searched_signal_type);
    }

    auto second_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), NeuronID(1) }, second_position, number_vacant_elements_2,
        &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

    ASSERT_EQ(second_targets.size(), number_vacant_elements_2);

    for (RelearnTypes::counter_type i = 0; i < number_vacant_elements_2; i++) {
        const auto& [rank, creation_request] = second_targets[i];
        ASSERT_EQ(rank, MPIRank::root_rank());

        const auto& [target_id, source_id, signal_type] = creation_request;

        ASSERT_EQ(target_id, NeuronID(0));
        ASSERT_EQ(source_id, NeuronID(1));
        ASSERT_EQ(signal_type, searched_signal_type);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronsFullChoice) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto number_vacant_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const RankNeuronId searching_id{ MPIRank::root_rank(), neuron_id };
        auto found_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(searching_id, position, number_vacant_elements, &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(found_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = found_targets[i];
            ASSERT_EQ(rank, MPIRank::root_rank());

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_NE(target_id, neuron_id);
            ASSERT_EQ(source_id, neuron_id);
            ASSERT_EQ(signal_type, searched_signal_type);
        }
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronsNoChoiceDistributed) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank(1));
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_axons(2);
    root.set_cell_number_inhibitory_axons(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    for (auto it = 0; it < 1000; it++) {
        const auto number_vacant_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto first_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), NeuronID(0) }, position, number_vacant_elements,
            &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(first_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = first_targets[i];
            ASSERT_EQ(rank, MPIRank(1));

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_TRUE((target_id == NeuronID(0) || target_id == NeuronID(1)));
            ASSERT_EQ(source_id, NeuronID(0));
            ASSERT_EQ(signal_type, searched_signal_type);
        }

        auto second_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons({ MPIRank::root_rank(), NeuronID(1) }, position, number_vacant_elements,
            &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(second_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = second_targets[i];
            ASSERT_EQ(rank, MPIRank(1));

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_TRUE((target_id == NeuronID(0) || target_id == NeuronID(1)));
            ASSERT_EQ(source_id, NeuronID(1));
            ASSERT_EQ(signal_type, searched_signal_type);
        }
    }
}

TEST_F(BarnesHutInvertedBaseTest, testFindTargetNeuronsFullChoiceDistributed) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto number_vacant_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const RankNeuronId searching_id{ MPIRank::root_rank(), neuron_id };
        const auto found_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(searching_id, position, number_vacant_elements,
            &root, ElementType::Axon, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(found_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = found_targets[i];
            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_EQ(source_id, neuron_id);
            ASSERT_EQ(signal_type, searched_signal_type);

            const RankNeuronId found_id(rank, target_id);

            ASSERT_NE(found_id, searching_id);
            ASSERT_TRUE(nodes.contains(found_id));
            ASSERT_GE(nodes.at(found_id)->get_cell().get_number_axons_for(searched_signal_type), 0);
        }
    }
}

TEST_F(BarnesHutInvertedBaseTest, testConvertTargetNodeException) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const NeuronID neuron_id(1000000);
    const Vec3d position{ 0.0 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::convert_target_node({ MPIRank::root_rank(), neuron_id }, position, nullptr, searched_signal_type, branching_level), RelearnException);
}

TEST_F(BarnesHutInvertedBaseTest, testConvertTargetNodeLeaf) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const NeuronID source(1000000);
    const NeuronID target(1000001);

    const Vec3d source_position{ 0.2 };
    const Vec3d target_position{ 0.5 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;
    const auto target_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    OctreeNode<additional_cell_attributes> target_node{};
    target_node.set_cell_neuron_id(target);
    target_node.set_cell_size(Vec3d{ -1.0 }, Vec3d{ 1.0 });
    target_node.set_level(target_level);
    target_node.set_cell_neuron_position(target_position);
    target_node.set_rank(MPIRank::root_rank());

    for (const auto mpi_rank : MPIRank::range(1000)) {
        const RankNeuronId rni{ mpi_rank, source };

        const auto& val = BarnesHutBase<additional_cell_attributes>::convert_target_node(rni, source_position, &target_node, searched_signal_type, branching_level);

        ASSERT_TRUE(val.has_value());

        const auto& [found_rank, distant_neuron_request] = val.value();

        ASSERT_EQ(found_rank, MPIRank::root_rank());

        const auto& [_source_id, _source_position, _target_identifier, _target_neuron_type, _searched_signal_type] = distant_neuron_request;

        ASSERT_EQ(_source_id, source);
        ASSERT_EQ(_source_position, source_position);
        ASSERT_EQ(_target_identifier, target.get_neuron_id());
        ASSERT_EQ(_target_neuron_type, DistantNeuronRequest::TargetNeuronType::Leaf);
        ASSERT_EQ(_searched_signal_type, searched_signal_type);
    }
}

TEST_F(BarnesHutInvertedBaseTest, testConvertTargetNodeTooHigh) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const NeuronID source(1000000);

    const Vec3d source_position{ 0.2 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto target_level = SimulationAdapter::get_small_refinement_level(mt) + 1;
    const auto branching_level = target_level + 1;

    OctreeNode<additional_cell_attributes> target_node{};
    target_node.set_cell_size(Vec3d{ -1.0 }, Vec3d{ 1.0 });
    target_node.set_level(target_level);
    target_node.set_rank(MPIRank::root_rank());
    target_node.set_cell_neuron_id(NeuronID(0));
    target_node.set_cell_neuron_position(Vec3d{ 0.0 });

    auto _1 = target_node.insert(Vec3d{ 0.3 }, NeuronID(1));
    auto _2 = target_node.insert(Vec3d{ 0.5 }, NeuronID(2));
    auto _3 = target_node.insert(Vec3d{ 0.7 }, NeuronID(3));

    const auto& val_0 = BarnesHutBase<additional_cell_attributes>::convert_target_node({ MPIRank(0), source }, source_position, &target_node, searched_signal_type, branching_level);
    ASSERT_FALSE(val_0.has_value());

    const auto& val_1 = BarnesHutBase<additional_cell_attributes>::convert_target_node({ MPIRank(1), source }, source_position, &target_node, searched_signal_type, branching_level);
    ASSERT_FALSE(val_1.has_value());
}

TEST_F(BarnesHutInvertedBaseTest, testConvertTargetNodeVirtual) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const NeuronID source(1000000);

    const Vec3d source_position{ 0.2 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;
    const auto target_level = branching_level + 1;

    OctreeNode<additional_cell_attributes> target_node{};
    target_node.set_cell_size(Vec3d{ -1.0 }, Vec3d{ 1.0 });
    target_node.set_level(target_level);
    target_node.set_cell_neuron_id(NeuronID(0));
    target_node.set_cell_neuron_position(Vec3d{ 0.0 });
    target_node.set_rank(MPIRank::root_rank());

    auto _1 = target_node.insert(Vec3d{ 0.3 }, NeuronID(1));
    auto _2 = target_node.insert(Vec3d{ 0.5 }, NeuronID(2));
    auto _3 = target_node.insert(Vec3d{ 0.7 }, NeuronID(3));

    target_node.set_cell_neuron_id(NeuronID(true, 10101010));

    for (const auto mpi_rank : MPIRank::range(1000)) {
        const RankNeuronId rni{ mpi_rank, source };

        const auto& val = BarnesHutBase<additional_cell_attributes>::convert_target_node(rni, source_position, &target_node, searched_signal_type, branching_level);

        ASSERT_TRUE(val.has_value());

        const auto& [found_rank, distant_neuron_request] = val.value();

        ASSERT_EQ(found_rank, MPIRank::root_rank());

        const auto& [_source_id, _source_position, _target_identifier, _target_neuron_type, _searched_signal_type] = distant_neuron_request;

        ASSERT_EQ(_source_id, source);
        ASSERT_EQ(_source_position, source_position);
        ASSERT_EQ(_target_identifier, 10101010);
        ASSERT_EQ(_target_neuron_type, DistantNeuronRequest::TargetNeuronType::VirtualNode);
        ASSERT_EQ(_searched_signal_type, searched_signal_type);
    }
}
