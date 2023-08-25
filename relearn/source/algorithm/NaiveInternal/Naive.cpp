/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Naive.h"

#include "algorithm/Connector.h"
#include "algorithm/Kernel/Kernel.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/NeuronID.h"
#include "util/Timers.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>
#include <array>
#include <stack>

#include <range/v3/view/filter.hpp>

std::optional<RankNeuronId> Naive::find_target_neuron(const NeuronID& src_neuron_id, const position_type& axon_position, const SignalType dendrite_type_needed) {
    OctreeNode<NaiveCell>* node_selected = nullptr;
    OctreeNode<NaiveCell>* root_of_subtree = get_octree_root();

    RelearnException::check(root_of_subtree != nullptr, "Naive::find_target_neuron: root_of_subtree was nullptr");

    while (true) {
        /**
         * Create vector with nodes that have at least one dendrite and are
         * precise enough given the position of an axon
         */
        const auto& vector = get_nodes_for_interval(axon_position, root_of_subtree, dendrite_type_needed);

        /**
         * Assign a probability to each node in the vector.
         * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
         * Nodes with 0 probability are removed.
         * The probabilities of all vector elements sum up to 1.
         */
        auto* node_selected = Kernel<AdditionalCellAttributes>::pick_target({ MPIWrapper::get_my_rank(), src_neuron_id }, axon_position, vector, ElementType::Dendrite, dendrite_type_needed);
        if (node_selected == nullptr) {
            return {};
        }

        const auto done = !node_selected->is_parent();

        if (done) {
            break;
        }

        // Update root of subtree
        root_of_subtree = node_selected;
    }

    RankNeuronId rank_neuron_id{ node_selected->get_mpi_rank(), node_selected->get_cell_neuron_id() };
    return rank_neuron_id;
}

CommunicationMap<SynapseCreationRequest> Naive::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(number_neurons_type(number_ranks), number_neurons);
    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    // For my neurons
    for (const auto id : NeuronID::range(number_neurons) | ranges::views::filter(equal_to(UpdateStatus::Enabled), lookup(disable_flags, &NeuronID::get_neuron_id))) {
        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto dendrite_type_needed = axons->get_signal_type(id);
        const auto& axon_position = extra_infos->get_position(id);

        // For all vacant axons of neuron "neuron_id"
        for (unsigned int j = 0; j < number_vacant_axons; j++) {
            /**
             * Find target neuron for connecting and
             * connect if target neuron has still dendrite available.
             *
             * The target neuron might not have any dendrites left
             * as other axons might already have connected to them.
             * Right now, those collisions are handled in a first-come-first-served fashion.
             */
            std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(id, axon_position, dendrite_type_needed);
            if (!rank_neuron_id.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            const auto& [target_rank, target_id] = rank_neuron_id.value();
            const SynapseCreationRequest creation_request(target_id, id, dendrite_type_needed);

            /**
             * Append request for synapse creation to rank "target_rank"
             * Note that "target_rank" could also be my own rank.
             */
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<NaiveCell>::clear();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::tuple<bool, bool> Naive::acceptance_criterion_test(const position_type& /*axon_position*/, const OctreeNode<NaiveCell>* const node_with_dendrite,
    const SignalType dendrite_type_needed) {
    RelearnException::check(node_with_dendrite != nullptr, "Naive::acceptance_criterion_test: node_with_dendrite was nullptr");

    const auto& cell = node_with_dendrite->get_cell();
    const auto has_vacant_dendrites = cell.get_number_dendrites_for(dendrite_type_needed) != 0;
    const auto is_parent = node_with_dendrite->is_parent();

    // Accept leaf only
    return std::make_tuple(!is_parent, has_vacant_dendrites);
}

std::vector<OctreeNode<NaiveCell>*> Naive::get_nodes_for_interval(const position_type& axon_position, OctreeNode<NaiveCell>* const root,
    const SignalType dendrite_type_needed) {
    if (root == nullptr) {
        return {};
    }

    if (root->get_cell().get_number_dendrites_for(dendrite_type_needed) == 0) {
        return {};
    }

    if (root->is_leaf()) {
        /**
         * The root node is a leaf and thus contains the target neuron.
         *
         * NOTE: Root is not intended to be a leaf but we handle this as well.
         * Without pushing root onto the stack, it would not make it into the "vector" of nodes.
         */

        const auto [accept, _] = acceptance_criterion_test(axon_position, root, dendrite_type_needed);
        if (accept) {
            return { root };
        }

        return {};
    }

    std::stack<OctreeNode<NaiveCell>*> stack{};

    const auto add_children_to_stack = [&stack](OctreeNode<NaiveCell>* node) {
        std::array<OctreeNode<NaiveCell>*, Constants::number_oct> children{ nullptr };

        // Node is owned by this rank
        if (node->is_local()) {
            // Node is owned by this rank, so the pointers are good
            children = node->get_children();
        } else {
            // Node owned by different rank, so we have to download the data to local nodes
            children = NodeCache<NaiveCell>::download_children(node);
        }

        for (auto* it : children) {
            if (it != nullptr) {
                stack.push(it);
            }
        }
    };

    // The algorithm expects that root is not considered directly, rather its children
    add_children_to_stack(root);

    std::vector<OctreeNode<NaiveCell>*> nodes_to_consider{};
    nodes_to_consider.reserve(Constants::number_oct);

    while (!stack.empty()) {
        // Get top-of-stack node and remove it from stack
        auto* stack_elem = stack.top();
        stack.pop();

        /**
         * Should node be used for probability interval?
         *
         * Only take those that have dendrites available
         */
        const auto [accept, has_vacant_dendrites] = acceptance_criterion_test(axon_position, stack_elem, dendrite_type_needed);

        if (accept) {
            // Insert node into vector
            nodes_to_consider.emplace_back(stack_elem);
            continue;
        }

        if (!has_vacant_dendrites) {
            continue;
        }

        add_children_to_stack(stack_elem);
    } // while

    return nodes_to_consider;
}

std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
Naive::process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);
}

PlasticDistantOutSynapses Naive::process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
    const CommunicationMap<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
}
