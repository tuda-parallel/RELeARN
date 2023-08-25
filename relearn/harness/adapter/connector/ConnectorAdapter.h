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

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "mpi/CommunicationMap.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/enums/SignalType.h"
#include "util/NeuronID.h"

#include <random>
#include <tuple>
#include <vector>

class ConnectorAdapter {
public:
    static std::tuple<CommunicationMap<SynapseCreationRequest>, std::vector<size_t>, std::vector<size_t>> create_incoming_requests(size_t number_ranks, int current_rank,
        size_t number_neurons, size_t number_requests_lower_bound, size_t number_requests_upper_bound, std::mt19937& mt) {

        CommunicationMap<SynapseCreationRequest> cm(static_cast<int>(number_ranks));
        std::vector<size_t> number_excitatory_requests(number_neurons, 0);
        std::vector<size_t> number_inhibitory_requests(number_neurons, 0);

        for (const auto& target_id : NeuronID::range(number_neurons)) {
            const auto number_requests = RandomAdapter::get_random_integer<size_t>(number_requests_lower_bound, number_requests_upper_bound, mt);

            const auto id = target_id.get_neuron_id();

            for (const auto r : ranges::views::indices(number_requests)) {
                const auto source_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);
                const auto source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
                const auto fixed_source_id = (source_id.get_neuron_id() == target_id.get_neuron_id() && current_rank == source_rank.get_rank()) ? NeuronID{ (source_id.get_neuron_id() + 1) % number_neurons } : source_id;

                const auto signal_type = RandomAdapter::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;

                const SynapseCreationRequest scr{ target_id, fixed_source_id, signal_type };

                if (signal_type == SignalType::Excitatory) {
                    number_excitatory_requests[id]++;
                } else {
                    number_inhibitory_requests[id]++;
                }

                cm.append(source_rank, scr);
            }
        }

        return { cm, number_excitatory_requests, number_inhibitory_requests };
    }
};
