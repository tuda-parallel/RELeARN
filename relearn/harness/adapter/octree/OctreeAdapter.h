#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "adapter/random/RandomAdapter.h"

#include "adapter/simulation/SimulationAdapter.h"

#include "neurons/helper/RankNeuronId.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"

#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <range/v3/algorithm/for_each.hpp>

class OctreeAdapter {
public:
    template <typename AdditionalCellAttributes>
    static std::vector<std::pair<Vec3d, size_t>> extract_virtual_neurons(const OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<std::pair<Vec3d, size_t>> return_value{};

        std::stack<std::pair<const OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};
        octree_nodes.emplace(root, 0);

        while (!octree_nodes.empty()) {
            // Don't change this to a reference
            const auto [current_node, level] = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->get_cell().get_neuron_id().is_virtual()) {
                return_value.emplace_back(current_node->get_cell().get_neuron_position().value(), level);
            }

            if (current_node->is_parent()) {
                const auto& childs = current_node->get_children();
                for (auto i = 0; i < 8; i++) {
                    const auto child = childs[i];
                    if (child != nullptr) {
                        octree_nodes.emplace(child, level + 1);
                    }
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<const OctreeNode<AdditionalCellAttributes>*> extract_leaf_nodes(const OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<const OctreeNode<AdditionalCellAttributes>*> return_value{};

        std::stack<const OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            const OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_leaf()) {
                return_value.emplace_back(current_node);
                continue;
            }

            const auto& childs = current_node->get_children();
            for (auto* child : childs) {
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<std::pair<Vec3d, NeuronID>> extract_neurons(const OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<std::pair<Vec3d, NeuronID>> return_value{};

        std::stack<const OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            const OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_parent()) {
                const auto& childs = current_node->get_children();
                for (auto* child : childs) {
                    if (child != nullptr) {
                        octree_nodes.push(child);
                    }
                }
            } else {
                const Cell<AdditionalCellAttributes>& cell = current_node->get_cell();
                const auto neuron_id = cell.get_neuron_id();
                const auto& opt_position = cell.get_neuron_position();

                EXPECT_TRUE(opt_position.has_value());

                const auto& position = opt_position.value();

                if (neuron_id.is_initialized() && !neuron_id.is_virtual()) {
                    return_value.emplace_back(position, neuron_id);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<std::pair<Vec3d, NeuronID>> extract_neurons_tree(const OctreeImplementation<AdditionalCellAttributes>& octree) {
        const auto root = octree.get_root();
        if (root == nullptr) {
            return {};
        }

        return extract_neurons<AdditionalCellAttributes>(root);
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_standard_tree(const RelearnTypes::number_neurons_type number_neurons, const Vec3d& min_pos, const Vec3d& max_pos, std::mt19937& mt) {
        auto get_synaptic_count = [&mt]() { return RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(1, 2, mt); };

        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(MPIRank::root_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronID::range(1, number_neurons)) {
            auto* ptr = root.insert(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt), id);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(get_synaptic_count());
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_tree_no_axons(const RelearnTypes::number_neurons_type number_neurons, const Vec3d& min_pos, const Vec3d& max_pos, std::mt19937& mt) {
        auto get_synaptic_count = [&mt]() { return RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 1, mt); };

        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(MPIRank::root_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronID::range(1, number_neurons)) {
            auto* ptr = root.insert(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt), id);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(0);
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_tree_no_dendrites(const RelearnTypes::number_neurons_type number_neurons, const Vec3d& min_pos, const Vec3d& max_pos, std::mt19937& mt) {
        auto get_synaptic_count = [&mt]() { return RandomAdapter::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 1, mt); };

        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(MPIRank::root_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronID::range(1, number_neurons)) {
            auto* ptr = root.insert(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt), id);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(get_synaptic_count());
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_tree_no_synaptic_elements(const RelearnTypes::number_neurons_type number_neurons, const Vec3d& min_pos, const Vec3d& max_pos, std::mt19937& mt) {
        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(MPIRank::root_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronID::range(1, number_neurons)) {
            auto* ptr = root.insert(SimulationAdapter::get_random_position_in_box(min_pos, max_pos, mt), id);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(0);
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static void mark_node_as_distributed(OctreeNode<AdditionalCellAttributes>* root, const std::uint16_t level_of_branch_nodes) {
        std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};
        branch_nodes.reserve(static_cast<size_t>(std::pow(8.0, level_of_branch_nodes) * 2));

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->get_level() == level_of_branch_nodes) {
                branch_nodes.emplace_back(current);
                continue;
            }

            if (current->get_level() < level_of_branch_nodes) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
                continue;
            }
        }

        auto current_rank = 0;
        for (auto* node : branch_nodes) {
            node->set_rank(MPIRank(current_rank));
            current_rank++;
        }

        auto mark_children = [](OctreeNode<AdditionalCellAttributes>* node) {
            auto rank = node->get_mpi_rank();

            std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
            stack.push(node);

            while (!stack.empty()) {
                auto* current = stack.top();
                stack.pop();

                current->set_rank(rank);

                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        };

        ranges::for_each(branch_nodes, mark_children);
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes>* find_node(const RankNeuronId& rank_neuron_id, OctreeNode<AdditionalCellAttributes>* root) {
        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf() && current->get_mpi_rank() == rank_neuron_id.get_rank() && current->get_cell_neuron_id() == rank_neuron_id.get_neuron_id()) {
                return current;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        return nullptr;
    }

    template <typename AdditionalCellAttributes>
    static std::unordered_map<RankNeuronId, OctreeNode<AdditionalCellAttributes>*> find_nodes(OctreeNode<AdditionalCellAttributes>* root) {
        std::unordered_map<RankNeuronId, OctreeNode<AdditionalCellAttributes>*> mapping{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                RankNeuronId rni{ current->get_mpi_rank(), current->get_cell_neuron_id() };
                mapping.emplace(rni, current);
                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        return mapping;
    }

    template <typename AdditionalCellAttributes>
    static std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*> find_child_offsets(OctreeNode<AdditionalCellAttributes>* root) {
        std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*> mapping{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }

            const auto virtual_id = current->get_cell().get_neuron_id().get_rma_offset();
            mapping.emplace(virtual_id, current);
        }

        return mapping;
    }
};
