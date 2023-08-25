/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHut.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Connector.h"
#include "io/Event.h"
#include "neurons/NeuronsExtraInfo.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"
#include "util/Timers.h"

#include <algorithm>

void BarnesHut::set_acceptance_criterion(const double acceptance_criterion) {
    RelearnException::check(acceptance_criterion > 0.0, "BarnesHut::set_acceptance_criterion: acceptance_criterion was less than or equal to 0 ({})", acceptance_criterion);
    this->acceptance_criterion = acceptance_criterion;
}

CommunicationMap<SynapseCreationRequest> BarnesHut::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = MPIWrapper::get_num_ranks();
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto size_hint = std::min(number_neurons, number_neurons_type(number_ranks));
    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    auto* const root = get_octree_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, my_rank, number_neurons, disable_flags, synapse_creation_requests_outgoing)
    for (NeuronID::value_type neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = axons->get_signal_type(id);

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons({ my_rank, id }, axon_position, number_vacant_axons, root, ElementType::Dendrite, dendrite_type_needed, acceptance_criterion);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        Event::create_and_print_counter_event("BH", {}, { { "ID:", std::to_string(neuron_id) }, { "Cache:", std::to_string(NodeCache<BarnesHutCell>::get_cache_size()) }, { "Memory:", std::to_string(NodeCache<BarnesHutCell>::get_memory_size()) } }, true);
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<BarnesHutCell>::clear();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
BarnesHut::process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);
}

PlasticDistantOutSynapses BarnesHut::process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
    const CommunicationMap<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
}
