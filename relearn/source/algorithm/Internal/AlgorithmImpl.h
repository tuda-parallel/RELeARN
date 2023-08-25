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

#include "algorithm/Algorithm.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <memory>
#include <vector>

/**
 * This class captures the common functionality of updating the Octree according
 * to the necessities of the respective algorithm.
 * @tparam AdditionalCellAttributes The additional cell attributes of the nodes stored in the octree
 */
template <typename AdditionalCellAttributes>
class AlgorithmImpl : public Algorithm {
public:
    /**
     * @brief Constructs the object and sets the necessary octree
     * @param octree The octree that is used for the algorithm, not nullptr
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit AlgorithmImpl(const std::shared_ptr<OctreeImplementation<AdditionalCellAttributes>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "AlgorithmImpl::AlgorithmImpl: octree was null");
    }

    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      Performs communication via MPI
     * @exception Can throw a RelearnException
     */
    void update_octree() override {
        // Update my leaf nodes
        Timers::start(TimerRegion::UPDATE_LEAF_NODES);
        update_leaf_nodes();
        Timers::stop_and_add(TimerRegion::UPDATE_LEAF_NODES);

        // Update the octree
        global_tree->synchronize_tree();
    }

protected:
    /**
     * @brief Returns the stored octree
     * @return The octree
     */
    [[nodiscard]] constexpr const std::shared_ptr<OctreeImplementation<AdditionalCellAttributes>>& get_octree() const noexcept {
        return global_tree;
    }

    /**
     * @brief Returns the root of the stored octree
     * @return The root
     */
    [[nodiscard]] constexpr OctreeNode<AdditionalCellAttributes>* get_octree_root() const noexcept {
        return global_tree->get_root();
    }

    /**
     * @brief Returns the level of branch nodes of the stored octree
     * @return The level of branch nodes
     */
    [[nodiscard]] constexpr std::uint16_t get_level_of_branch_nodes() const noexcept {
        return global_tree->get_level_of_branch_nodes();
    }

private:
    /**
     * @brief Updates all leaf nodes in the octree by the algorithm if the extra infos specify so.
     * @exception Throws a RelearnException if the number of flags is different than the number of leaf nodes, or if there is an internal error
     */
    void update_leaf_nodes() {
        const auto& dendrites_excitatory_counts = excitatory_dendrites->get_grown_elements();
        const auto& dendrites_excitatory_connected_counts = excitatory_dendrites->get_connected_elements();

        const auto& dendrites_inhibitory_counts = inhibitory_dendrites->get_grown_elements();
        const auto& dendrites_inhibitory_connected_counts = inhibitory_dendrites->get_connected_elements();

        const auto& axons_counts = axons->get_grown_elements();
        const auto& axons_connected_counts = axons->get_connected_elements();

        const auto& leaf_nodes = global_tree->get_leaf_nodes();
        const auto num_leaf_nodes = leaf_nodes.size();
        const auto num_disable_flags = extra_infos->get_size();
        const auto num_dendrites_excitatory_counts = dendrites_excitatory_counts.size();
        const auto num_dendrites_excitatory_connected_counts = dendrites_excitatory_connected_counts.size();
        const auto num_dendrites_inhibitory_counts = dendrites_inhibitory_counts.size();
        const auto num_dendrites_inhibitory_connected_counts = dendrites_inhibitory_connected_counts.size();

        const auto all_same_size = num_leaf_nodes == num_disable_flags
            && num_leaf_nodes == num_dendrites_excitatory_counts
            && num_leaf_nodes == num_dendrites_excitatory_connected_counts
            && num_leaf_nodes == num_dendrites_inhibitory_counts
            && num_leaf_nodes == num_dendrites_inhibitory_connected_counts;

        RelearnException::check(all_same_size, "AlgorithmImpl::update_leaf_nodes: The vectors were of different sizes");

        for (const auto& neuron_id : NeuronID::range(num_leaf_nodes)) {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            auto* node = leaf_nodes[local_neuron_id];
            RelearnException::check(node != nullptr, "AlgorithmImpl::update_leaf_nodes: node was nullptr: {}", neuron_id);

            const auto& cell = node->get_cell();
            const auto other_neuron_id = cell.get_neuron_id();
            RelearnException::check(neuron_id == other_neuron_id, "AlgorithmImpl::update_leaf_nodes: The nodes are not in order {} != {}", neuron_id, other_neuron_id);

            if (!extra_infos->does_update_plasticity(neuron_id)) {
                if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                    node->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                    node->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                    node->set_cell_number_excitatory_axons(0);
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                    node->set_cell_number_inhibitory_axons(0);
                }
                continue;
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto number_vacant_excitatory_dendrites = excitatory_dendrites->get_free_elements(neuron_id);
                node->set_cell_number_excitatory_dendrites(number_vacant_excitatory_dendrites);
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto number_vacant_inhibitory_dendrites = inhibitory_dendrites->get_free_elements(neuron_id);
                node->set_cell_number_inhibitory_dendrites(number_vacant_inhibitory_dendrites);
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto signal_type = axons->get_signal_type(neuron_id);

                if (signal_type == SignalType::Excitatory) {
                    const auto number_vacant_axons = axons->get_free_elements(neuron_id);
                    node->set_cell_number_excitatory_axons(number_vacant_axons);
                } else {
                    node->set_cell_number_excitatory_axons(0);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto signal_type = axons->get_signal_type(neuron_id);

                if (signal_type == SignalType::Inhibitory) {
                    const auto number_vacant_axons = axons->get_free_elements(neuron_id);
                    node->set_cell_number_inhibitory_axons(number_vacant_axons);
                } else {
                    node->set_cell_number_inhibitory_axons(0);
                }
            }
        }
    }

    std::shared_ptr<OctreeImplementation<AdditionalCellAttributes>> global_tree{};
};