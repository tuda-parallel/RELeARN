/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_connector.h"

#include "adapter/connector/ConnectorAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"
#include "adapter/connector/ConnectorAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "algorithm/Connector.h"
#include "mpi/CommunicationMap.h"
#include "neurons/models/SynapticElements.h"
#include "util/NeuronID.h"

#include <map>
#include <vector>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/transform.hpp>

TEST_F(ConnectorTest, testForwardConnectorExceptions) {
    const auto number_neurons_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_2 = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto final_number_neurons = number_neurons_1 == number_neurons_2 ? number_neurons_2 + 1 : number_neurons_2;

    const auto number_ranks_1 = MPIRankAdapter::get_random_number_ranks(mt) + 1;
    const auto number_ranks_2 = MPIRankAdapter::get_random_number_ranks(mt) + 1;

    const auto final_number_ranks = number_ranks_1 == number_neurons_2 ? number_neurons_2 + 1 : number_ranks_2;

    const auto& excitatory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons_1, SignalType::Excitatory, mt);
    const auto& inhibitory_dendrites = SynapticElementsAdapter::create_dendrites(final_number_neurons, SignalType::Inhibitory, mt);

    std::shared_ptr<SynapticElements> empty = nullptr;

    CommunicationMap<SynapseCreationRequest> incoming_requests{ static_cast<size_t>(number_ranks_1) };

    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, empty, empty), RelearnException);
    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, empty), RelearnException);
    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, empty, inhibitory_dendrites), RelearnException);
    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites), RelearnException);

    CommunicationMap<SynapseCreationResponse> incoming_responses{ static_cast<size_t>(number_ranks_1) };
    ASSERT_THROW(auto val = ForwardConnector::process_responses(incoming_requests, incoming_responses, empty), RelearnException);

    CommunicationMap<SynapseCreationResponse> wrong_incoming_responses{ static_cast<size_t>(final_number_ranks) };
    ASSERT_THROW(auto val = ForwardConnector::process_responses(incoming_requests, wrong_incoming_responses, empty), RelearnException);
}

TEST_F(ConnectorTest, testForwardConnectorEmptyMap) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt) + 1;

    const auto& excitatory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Excitatory, mt);
    const auto& inhibitory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Inhibitory, mt);

    // The following copies are intentional
    const auto previous_connected_excitatory = vectorify_span(excitatory_dendrites->get_connected_elements());
    const auto previous_grown_excitatory = vectorify_span(excitatory_dendrites->get_grown_elements());
    const auto previous_deltas_excitatory = vectorify_span(excitatory_dendrites->get_deltas());
    const auto previous_types_excitatory = vectorify_span(excitatory_dendrites->get_signal_types());

    const auto previous_connected_inhibitory = vectorify_span(excitatory_dendrites->get_connected_elements());
    const auto previous_grown_inhibitory = vectorify_span(excitatory_dendrites->get_grown_elements());
    const auto previous_deltas_inhibitory = vectorify_span(excitatory_dendrites->get_deltas());
    const auto previous_types_inhibitory = vectorify_span(excitatory_dendrites->get_signal_types());

    CommunicationMap<SynapseCreationRequest> incoming_requests{ static_cast<size_t>(number_ranks) };

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(responses.size(), incoming_requests.size());
    ASSERT_EQ(responses.get_number_ranks(), incoming_requests.get_number_ranks());
    ASSERT_EQ(responses.get_total_number_requests(), 0);

    ASSERT_TRUE(local_synapses.empty());
    ASSERT_TRUE(distant_in_synapses.empty());

    const auto& now_connected_excitatory = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_excitatory = excitatory_dendrites->get_grown_elements();
    const auto& now_deltas_excitatory = excitatory_dendrites->get_deltas();
    const auto& now_types_excitatory = excitatory_dendrites->get_signal_types();

    const auto& now_connected_inhibitory = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_inhibitory = excitatory_dendrites->get_grown_elements();
    const auto& now_deltas_inhibitory = excitatory_dendrites->get_deltas();
    const auto& now_types_inhibitory = excitatory_dendrites->get_signal_types();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_connected_excitatory[neuron_id], now_connected_excitatory[neuron_id]);
        ASSERT_EQ(previous_grown_excitatory[neuron_id], now_grown_excitatory[neuron_id]);
        ASSERT_EQ(previous_deltas_excitatory[neuron_id], now_deltas_excitatory[neuron_id]);
        ASSERT_EQ(previous_types_excitatory[neuron_id], now_types_excitatory[neuron_id]);

        ASSERT_EQ(previous_connected_inhibitory[neuron_id], now_connected_inhibitory[neuron_id]);
        ASSERT_EQ(previous_grown_inhibitory[neuron_id], now_grown_inhibitory[neuron_id]);
        ASSERT_EQ(previous_deltas_inhibitory[neuron_id], now_deltas_inhibitory[neuron_id]);
        ASSERT_EQ(previous_types_inhibitory[neuron_id], now_types_inhibitory[neuron_id]);
    }
}

TEST_F(ConnectorTest, testForwardConnectorMatchingRequests) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt) + 1;

    const auto& excitatory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Excitatory, mt);
    const auto& inhibitory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Inhibitory, mt);

    CommunicationMap<SynapseCreationRequest> incoming_requests{ static_cast<size_t>(number_ranks) };

    auto number_excitatory_requests = 0;
    auto number_inhibitory_requests = 0;

    std::map<NeuronID, std::vector<SynapseCreationRequest>> excitatory_requests{};
    std::map<NeuronID, std::vector<SynapseCreationRequest>> inhibitory_requests{};

    for (const auto& id : NeuronID::range(number_neurons)) {
        const auto number_vacant_excitatory = excitatory_dendrites->get_free_elements(id);
        number_excitatory_requests += number_vacant_excitatory;

        for (const auto neuron_id : NeuronID::range(number_vacant_excitatory)) {
            SynapseCreationRequest scr(id, neuron_id, SignalType::Excitatory);
            incoming_requests.append(MPIRank(1), scr);

            excitatory_requests[id].emplace_back(scr);
        }

        const auto number_vacant_inhibitory = inhibitory_dendrites->get_free_elements(id);
        number_inhibitory_requests += number_vacant_inhibitory;

        for (const auto neuron_id : NeuronID::range(number_vacant_inhibitory)) {
            SynapseCreationRequest scr(id, neuron_id, SignalType::Inhibitory);
            incoming_requests.append(MPIRank(1), scr);

            inhibitory_requests[id].emplace_back(scr);
        }
    }

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each saved rank: The number of responses matches the number of requests
    ASSERT_EQ(request_sizes.size(), response_sizes.size());
    for (const auto& [rank, size] : request_sizes) {
        const auto found_in_responses = response_sizes.contains(rank);
        ASSERT_TRUE(found_in_responses);

        ASSERT_EQ(size, response_sizes.at(rank));
    }

    for (const auto& [rank, resps] : responses) {
        for (const auto resp : resps) {
            ASSERT_EQ(resp, SynapseCreationResponse::Succeeded);
        }
    }

    ASSERT_EQ(local_synapses.size(), 0);
    ASSERT_EQ(distant_in_synapses.size(), number_excitatory_requests + number_inhibitory_requests);

    for (const auto& id : NeuronID::range(number_neurons)) {
        const auto number_vacant_excitatory = excitatory_dendrites->get_free_elements(id);
        ASSERT_EQ(number_vacant_excitatory, 0);

        const auto number_vacant_inhibitory = inhibitory_dendrites->get_free_elements(id);
        ASSERT_EQ(number_vacant_inhibitory, 0);
    }

    for (const auto& [target_id, source_id, weight] : distant_in_synapses) {
        const auto& [source_rank, source_neuron_id] = source_id;

        ASSERT_EQ(source_rank, MPIRank(1));
        ASSERT_EQ(std::abs(weight), 1);
    }
}

TEST_F(ConnectorTest, testForwardConnectorIncoming) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt) + 1;

    const auto& axons = SynapticElementsAdapter::create_axons(number_neurons, mt);
    const auto& excitatory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Excitatory, mt);
    const auto& inhibitory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Inhibitory, mt);

    // The following copies are intentional
    const auto previous_connected_excitatory_counts = vectorify_span(excitatory_dendrites->get_connected_elements());
    const auto previous_grown_excitatory_counts = vectorify_span(excitatory_dendrites->get_grown_elements());
    const auto previous_connected_inhibitory_counts = vectorify_span(inhibitory_dendrites->get_connected_elements());
    const auto previous_grown_inhibitory_counts = vectorify_span(inhibitory_dendrites->get_grown_elements());

    const auto& [incoming_requests, number_excitatory_requests, number_inhibitory_requests]
        = ConnectorAdapter::create_incoming_requests(number_ranks, 0, number_neurons, 0, 9, mt);

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    // There are as many requests as responses
    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each saved rank: The number of responses matches the number of requests
    ASSERT_EQ(request_sizes.size(), response_sizes.size());
    for (const auto& [rank, size] : request_sizes) {
        const auto found_in_responses = response_sizes.contains(rank);
        ASSERT_TRUE(found_in_responses);

        ASSERT_EQ(size, response_sizes.at(rank));
    }

    const auto& now_connected_excitatory_counts = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_excitatory_counts = excitatory_dendrites->get_grown_elements();
    const auto& now_connected_inhibitory_counts = inhibitory_dendrites->get_connected_elements();
    const auto& now_grown_inhibitory_counts = inhibitory_dendrites->get_grown_elements();

    std::vector<unsigned int> newly_connected_excitatory_dendrites(number_neurons, 0);
    std::vector<unsigned int> newly_connected_inhibitory_dendrites(number_neurons, 0);

    // The grown elements did not change. There are now not less connected then before, and not more than grown
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_grown_excitatory_counts[neuron_id], now_grown_excitatory_counts[neuron_id]) << neuron_id;
        ASSERT_EQ(previous_grown_inhibitory_counts[neuron_id], now_grown_inhibitory_counts[neuron_id]) << neuron_id;

        ASSERT_GE(now_connected_excitatory_counts[neuron_id], previous_connected_excitatory_counts[neuron_id]) << neuron_id;
        ASSERT_GE(now_connected_inhibitory_counts[neuron_id], previous_connected_inhibitory_counts[neuron_id]) << neuron_id;

        ASSERT_LE(now_connected_excitatory_counts[neuron_id], static_cast<unsigned int>(now_grown_excitatory_counts[neuron_id])) << neuron_id;
        ASSERT_LE(now_connected_inhibitory_counts[neuron_id], static_cast<unsigned int>(now_grown_inhibitory_counts[neuron_id])) << neuron_id;

        newly_connected_excitatory_dendrites[neuron_id] = now_connected_excitatory_counts[neuron_id] - previous_connected_excitatory_counts[neuron_id];
        newly_connected_inhibitory_dendrites[neuron_id] = now_connected_inhibitory_counts[neuron_id] - previous_connected_inhibitory_counts[neuron_id];
    }

    // If there are still vacant elements, then all requests are connected
    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto vacant_excitatory_elements = excitatory_dendrites->get_free_elements(neuron_id);
        if (vacant_excitatory_elements > 0) {
            ASSERT_EQ(newly_connected_excitatory_dendrites[neuron_id.get_neuron_id()], number_excitatory_requests[neuron_id.get_neuron_id()]) << neuron_id;
        }

        if (const auto vacant_inhibitory_elements = inhibitory_dendrites->get_free_elements(neuron_id);
            vacant_inhibitory_elements > 0) {
            ASSERT_EQ(newly_connected_inhibitory_dendrites[neuron_id.get_neuron_id()], number_inhibitory_requests[neuron_id.get_neuron_id()]) << neuron_id;
        }
    }

    std::vector<unsigned int> accepted_excitatory_requests(number_neurons, 0);
    std::vector<unsigned int> accepted_inhibitory_requests(number_neurons, 0);

    PlasticLocalSynapses expected_local_synapses{};
    PlasticDistantInSynapses expected_distant_in_synapses{};

    const auto my_rank = MPIWrapper::get_my_rank();

    // Extract things from the return value
    for (const auto rank : MPIRank::range(number_ranks)) {
        const auto found_in_requests = request_sizes.contains(rank);
        if (!found_in_requests) {
            continue;
        }

        for (auto index = 0; index < request_sizes.at(rank); index++) {
            const auto& [target_index, source_index, signal_type] = incoming_requests.get_request(rank, index);
            const auto& response = responses.get_request(rank, index);

            if (response == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& target_id = target_index.get_neuron_id();
            const auto& source_id = source_index.get_neuron_id();

            if (signal_type == SignalType::Excitatory) {
                accepted_excitatory_requests[target_id]++;
            } else {
                accepted_inhibitory_requests[target_id]++;
            }

            const auto weight = signal_type == SignalType::Excitatory ? 1 : -1;

            if (rank == my_rank) {
                expected_local_synapses.emplace_back(target_index, source_index, weight);
            } else {
                expected_distant_in_synapses.emplace_back(target_index, RankNeuronId{ rank, source_index }, weight);
            }
        }
    }

    // The sizes of the return values match the number of accepted responses
    ASSERT_EQ(local_synapses.size(), expected_local_synapses.size());
    ASSERT_EQ(distant_in_synapses.size(), expected_distant_in_synapses.size());

    ranges::sort(local_synapses);
    ranges::sort(distant_in_synapses);
    ranges::sort(expected_local_synapses);
    ranges::sort(expected_distant_in_synapses);

    // All and only the accepted local synapses are returned
    for (const auto neuron_id : ranges::views::indices(local_synapses.size())) {
        const auto& [target_1, source_1, weight_1] = local_synapses[neuron_id];
        const auto& [target_2, source_2, weight_2] = expected_local_synapses[neuron_id];

        ASSERT_EQ(target_1, target_2) << neuron_id;
        ASSERT_EQ(source_1, source_2) << neuron_id;
        ASSERT_EQ(weight_1, weight_2) << neuron_id;
    }

    // All and only the accepted distant in synapses are returned
    for (const auto neuron_id : ranges::views::indices(distant_in_synapses.size())) {
        const auto& [target_1, source_1, weight_1] = distant_in_synapses[neuron_id];
        const auto& [target_2, source_2, weight_2] = expected_distant_in_synapses[neuron_id];

        ASSERT_EQ(target_1, target_2) << neuron_id;
        ASSERT_EQ(source_1, source_2) << neuron_id;
        ASSERT_EQ(weight_1, weight_2) << neuron_id;
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(accepted_excitatory_requests[neuron_id], newly_connected_excitatory_dendrites[neuron_id]) << neuron_id;
        ASSERT_EQ(accepted_inhibitory_requests[neuron_id], newly_connected_inhibitory_dendrites[neuron_id]) << neuron_id;
    }
}
