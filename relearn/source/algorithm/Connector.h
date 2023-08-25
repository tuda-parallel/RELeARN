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

#include "Types.h"
#include "mpi/CommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/models/SynapticElements.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/ranges/Functional.hpp"

#include <concepts>
#include <memory>
#include <utility>
#include <vector>

#include <range/v3/functional/compose.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/zip.hpp>

/**
 * This class commits SynapseCreationRequests and SynapseCreationResponses to the synaptic elements.
 * It assumes that a request is made by an axon and targets a dendrite. It does not perform communication with MPI.
 */
class ForwardConnector {
public:
    /**
     * @brief Connects as many of the creation requests as possible locally, and commits the changes to the synaptic elements.
     *      Picks the order of the requests randomly. A request is from an axon to a dendrite
     * @param creation_requests The creation requests from all MPI ranks to the current rank
     * @param excitatory_dendrites The excitatory dendrites
     * @param inhibitory_dendrites The inhibitory dendrites
     * @exception Throws a RelearnException if (a) One of the pointers is empty, (b) They have different sizes, (c) One target has an id larger than the number of elements in the pointers
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] static std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const std::shared_ptr<SynapticElements>& excitatory_dendrites, const std::shared_ptr<SynapticElements>& inhibitory_dendrites) {

        const auto excitatory_empty = excitatory_dendrites.operator bool();
        RelearnException::check(excitatory_empty, "ForwardConnector::process_requests: The excitatory dendrites are empty");

        const auto inhibitory_empty = inhibitory_dendrites.operator bool();
        RelearnException::check(inhibitory_empty, "ForwardConnector::process_requests: The inhibitory dendrites are empty");

        const auto number_neurons = excitatory_dendrites->get_size();
        const auto number_neurons_2 = inhibitory_dendrites->get_size();

        RelearnException::check(number_neurons == number_neurons_2,
            "ForwardConnector::process_requests: The sizes of the synaptic elements don't match: {} and {}", number_neurons, number_neurons_2);

        const auto my_rank = MPIWrapper::get_my_rank();
        const auto number_ranks = creation_requests.get_number_ranks();

        const auto size_hint = creation_requests.size();
        CommunicationMap<SynapseCreationResponse> responses(number_ranks, size_hint);

        if (creation_requests.empty()) {
            return { responses, {} };
        }

        responses.resize(creation_requests.get_request_sizes());

        const auto total_number_requests = creation_requests.get_total_number_requests();

        PlasticLocalSynapses local_synapses{};
        local_synapses.reserve(total_number_requests);

        PlasticDistantInSynapses distant_synapses{};
        distant_synapses.reserve(total_number_requests);

        const auto indices = creation_requests
            | ranges::views::for_each([](const auto& request) {
                  const auto& [source_rank, requests] = request;
                  return ranges::views::zip(
                      ranges::views::repeat(source_rank),
                      ranges::views::indices(ranges::size(requests)));
              })
            | ranges::to<std::vector<std::pair<MPIRank, unsigned int>>>
            | RandomHolder::shuffleAction(RandomHolderKey::Connector);
        // We need to shuffle the request indices so we do not prefer those from smaller MPI ranks and lower neuron ids

        for (const auto& [source_rank, request_index] : indices) {
            const auto& [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(source_rank, request_index);

            if (source_rank == my_rank && target_neuron_id == source_neuron_id) {
                responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
                continue;
            }

            RelearnException::check(target_neuron_id.get_neuron_id() < number_neurons, "ForwardConnector::process_requests: target_neuron_id exceeds my neurons");

            const auto& dendrites = (SignalType::Inhibitory == dendrite_type_needed) ? inhibitory_dendrites : excitatory_dendrites;

            const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : 1;
            const auto number_free_elements = dendrites->get_free_elements(target_neuron_id);

            if (number_free_elements == 0) {
                // Other axons were faster and came first
                responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
                continue;
            }

            // Increment number of connected dendrites
            dendrites->update_connected_elements(target_neuron_id, 1);

            // Set response to "connected" (success)
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Succeeded);

            if (source_rank == my_rank) {
                local_synapses.emplace_back(target_neuron_id, source_neuron_id, weight);
                continue;
            }

            distant_synapses.emplace_back(target_neuron_id, RankNeuronId{ source_rank, source_neuron_id }, weight);
        }

        return { responses, { local_synapses, distant_synapses } };
    }

    /**
     * @brief Processes all incoming responses from the MPI ranks locally, and commits the changes to the synaptic elements.
     *      A response is from a request from an axon to a dendrite
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @param axons The axons
     * @exception Throws a RelearnException if (a) The axons are empty, (b) The requests and responses don't have the same size,
     *      (c) One of the source ids that are accepted are too large, (d) An accepted request targets an axon with not enough vacant elements
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] static PlasticDistantOutSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses, const std::shared_ptr<SynapticElements>& axons) {

        const auto axons_empty = axons.operator bool();
        RelearnException::check(axons_empty, "ForwardConnector::process_responses: The axons are empty");

        RelearnException::check(creation_requests.size() == creation_responses.size(),
            "ForwardConnector::process_responses: Requests and Responses had different sizes: requests: {} vs responses: {}", creation_requests.size(), creation_responses.size());

        for (const auto rank : MPIRank::range(creation_requests.size())) {
            RelearnException::check(creation_requests.size(rank) == creation_responses.size(rank),
                "ForwardConnector::process_responses: Requests and Responses for rank {} had different sizes", rank);
        }

        const auto number_neurons = axons->get_size();
        const auto my_rank = MPIWrapper::get_my_rank();

        const auto total_number_responses = creation_requests.get_total_number_requests();

        PlasticDistantOutSynapses synapses{};
        synapses.reserve(total_number_responses);

        // Process the responses of all mpi ranks
        for (const auto& [target_rank, requests] : creation_responses) {
            const auto num_requests = requests.size();

            // All responses from a rank
            for (auto request_index = 0; request_index < num_requests; request_index++) {
                const auto connected = requests[request_index];
                if (connected == SynapseCreationResponse::Failed) {
                    continue;
                }

                const auto& [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(target_rank, request_index);

                RelearnException::check(source_neuron_id.get_neuron_id() < number_neurons,
                    "ForwardConnector::process_responses: The source neuron id was too large: {} vs {}", source_neuron_id.get_neuron_id(), number_neurons);

                RelearnException::check(axons->get_free_elements(source_neuron_id) > 0,
                    "ForwardConnector::process_responses: The source neuron did not have a vacant element: {}", source_neuron_id);

                // Increment number of connected axons
                axons->update_connected_elements(source_neuron_id, 1);

                if (target_rank == my_rank) {
                    // I have already created the synapse in the network if the response comes from myself
                    continue;
                }

                // Mark this synapse for later use (must be added to the network graph)
                const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : +1;
                synapses.emplace_back(RankNeuronId{ target_rank, target_neuron_id }, source_neuron_id, weight);
            }
        }

        return synapses;
    }
};

/**
 * This class commits SynapseCreationRequests and SynapseCreationResponses to the synaptic elements.
 * It assumes that a request is made by a dendrite and targets an axon. It does not perform communication with MPI.
 */
class BackwardConnector {
public:
    /**
     * @brief Connects as many of the creation requests as possible locally, and commits the changes to the synaptic elements.
     *      Picks the order of the requests randomly. A request is from a dendrite to an axon
     * @param creation_requests The creation requests from all MPI ranks to the current rank
     * @param axons The axons
     * @exception Throws a RelearnException if (a) One of the pointers is empty, (b) They have different sizes, (c) One target has an id larger than the number of elements in the pointers
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses from the local rank
     */
    [[nodiscard]] static std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantOutSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests, const std::shared_ptr<SynapticElements>& axons) {

        const auto axons_empty = axons.operator bool();
        RelearnException::check(axons_empty, "BackwardConnector::process_requests: The axons are empty");

        const auto number_neurons = axons->get_size();
        const auto my_rank = MPIWrapper::get_my_rank();
        const auto number_ranks = MPIWrapper::get_num_ranks();

        const auto size_hint = creation_requests.size();
        CommunicationMap<SynapseCreationResponse> responses(number_ranks, size_hint);
        if (creation_requests.empty()) {
            return { responses, {} };
        }

        responses.resize(creation_requests.get_request_sizes());

        const auto total_number_requests = creation_requests.get_total_number_requests();

        PlasticLocalSynapses local_synapses{};
        local_synapses.reserve(total_number_requests);

        PlasticDistantOutSynapses distant_synapses{};
        distant_synapses.reserve(total_number_requests);

        const auto indices = creation_requests
            | ranges::views::for_each([](const auto& request) {
                  const auto& [source_rank, requests] = request;
                  return ranges::views::zip(
                      ranges::views::repeat(source_rank),
                      ranges::views::iota(0U, ranges::size(requests)));
              })
            | ranges::to<std::vector<std::pair<MPIRank, unsigned int>>>
            | RandomHolder::shuffleAction(RandomHolderKey::Connector);
        // We need to shuffle the request indices so we do not prefer those from smaller MPI ranks and lower neuron ids

        const auto& signal_types = axons->get_signal_types();

        for (const auto& [source_rank, request_index] : indices) {
            const auto& [target_neuron_id, source_neuron_id, axon_type_needed] = creation_requests.get_request(source_rank, request_index);

            RelearnException::check(target_neuron_id.get_neuron_id() < number_neurons, "ForwardConnector::process_requests: target_neuron_id exceeds my neurons");
            RelearnException::check(signal_types[target_neuron_id.get_neuron_id()] == axon_type_needed, "ForwardConnector::process_requests: Request had the wrong signal type");

            const auto weight = (SignalType::Inhibitory == axon_type_needed) ? -1 : 1;
            const auto number_free_elements = axons->get_free_elements(target_neuron_id);

            if (number_free_elements == 0) {
                // Other axons were faster and came first
                responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
                continue;
            }

            // Increment number of connected dendrites
            axons->update_connected_elements(target_neuron_id, 1);

            // Set response to "connected" (success)
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Succeeded);

            if (source_rank == my_rank) {
                local_synapses.emplace_back(source_neuron_id, target_neuron_id, weight);
                continue;
            }

            distant_synapses.emplace_back(RankNeuronId{ source_rank, source_neuron_id }, target_neuron_id, weight);
        }

        return { responses, { local_synapses, distant_synapses } };
    }

    /**
     * @brief Processes all incoming responses from the MPI ranks locally, and commits the changes to the synaptic elements.
     *      A response is from a request from a dendrite to an axon
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @param excitatory_dendrites The excitatory dendrites
     * @param inhibitory_dendrites The inhibitory dendrites
     * @exception Throws a RelearnException if (a) The axons are empty, (b) One of the source ids that are accepted are too large
     * @return All synapses to this MPI rank from other MPI ranks
     */
    [[nodiscard]] static PlasticDistantInSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests, const CommunicationMap<SynapseCreationResponse>& creation_responses,
        const std::shared_ptr<SynapticElements>& excitatory_dendrites, const std::shared_ptr<SynapticElements>& inhibitory_dendrites) {

        const auto excitatory_empty = excitatory_dendrites.operator bool();
        RelearnException::check(excitatory_empty, "ForwardConnector::process_requests: The excitatory dendrites are empty");

        const auto inhibitory_empty = inhibitory_dendrites.operator bool();
        RelearnException::check(inhibitory_empty, "ForwardConnector::process_requests: The inhibitory dendrites are empty");

        const auto number_neurons = excitatory_dendrites->get_size();
        const auto number_neurons_2 = inhibitory_dendrites->get_size();

        RelearnException::check(number_neurons == number_neurons_2, "ForwardConnector::process_requests: The sizes of the synaptic elements don't match: {} and {}", number_neurons, number_neurons_2);

        const auto my_rank = MPIWrapper::get_my_rank();

        const auto total_number_responses = creation_requests.get_total_number_requests();

        PlasticDistantInSynapses synapses{};
        synapses.reserve(total_number_responses);

        // Process the responses of all mpi ranks
        for (const auto& [target_rank, requests] : creation_responses) {
            const auto num_requests = requests.size();

            // All responses from a rank
            for (auto request_index = 0; request_index < num_requests; request_index++) {
                const auto connected = requests[request_index];
                if (connected == SynapseCreationResponse::Failed) {
                    continue;
                }

                const auto& [target_neuron_id, source_neuron_id, axon_type_needed] = creation_requests.get_request(target_rank, request_index);

                const auto& dendrites = (SignalType::Inhibitory == axon_type_needed) ? inhibitory_dendrites : excitatory_dendrites;

                RelearnException::check(source_neuron_id.get_neuron_id() < number_neurons,
                    "ForwardConnector::process_responses: The source neuron id was too large: {} vs {}", source_neuron_id.get_neuron_id(), number_neurons);

                // Increment number of connected axons
                dendrites->update_connected_elements(source_neuron_id, 1);

                if (target_rank == my_rank) {
                    // I have already created the synapse in the network if the response comes from myself
                    continue;
                }

                // Mark this synapse for later use (must be added to the network graph)
                const auto weight = (SignalType::Inhibitory == axon_type_needed) ? -1 : +1;
                synapses.emplace_back(source_neuron_id, RankNeuronId{ target_rank, target_neuron_id }, weight);
            }
        }

        return synapses;
    }
};
