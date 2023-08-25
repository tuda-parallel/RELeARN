/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_network_graph.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "Types.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <cstddef>
#include <map>
#include <numeric>
#include <random>
#include <range/v3/functional/compose.hpp>
#include <range/v3/range/traits.hpp>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/map.hpp>

void NetworkGraphTest::assert_local_plastic_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    auto [all_local_in_edges, _1] = network_graph.get_all_local_in_edges();
    auto [all_local_out_edges, _2] = network_graph.get_all_local_out_edges();

    for (const auto& neighborhood : all_local_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_local_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& [local_in_edges, _3] = network_graph.get_local_in_edges(neuron_id);
        const auto& [all_local_out_edges, _4] = network_graph.get_local_out_edges(neuron_id);

        ASSERT_TRUE(local_in_edges.empty());
        ASSERT_TRUE(all_local_out_edges.empty());
    }
}

void NetworkGraphTest::assert_distant_plastic_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    auto [all_distant_in_edges, _1] = network_graph.get_all_distant_in_edges();
    auto [all_distant_out_edges, _2] = network_graph.get_all_distant_out_edges();

    for (const auto& neighborhood : all_distant_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_distant_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& [distant_in_edges, _3] = network_graph.get_distant_in_edges(neuron_id);
        const auto& [distant_out_edges, _4] = network_graph.get_distant_out_edges(neuron_id);

        ASSERT_TRUE(distant_in_edges.empty());
        ASSERT_TRUE(distant_out_edges.empty());

        const auto& incoming_excitatory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Excitatory);
        const auto& incoming_inhibitory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Inhibitory);
        const auto& outgoing_partners = network_graph.get_all_plastic_partners_outgoing(neuron_id);

        ASSERT_TRUE(incoming_excitatory_partners.empty());
        ASSERT_TRUE(incoming_inhibitory_partners.empty());
        ASSERT_TRUE(outgoing_partners.empty());
    }
}

void NetworkGraphTest::assert_plastic_empty(const NetworkGraph& network_graph) {
    assert_local_plastic_empty(network_graph);
    assert_distant_plastic_empty(network_graph);

    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto [number_excitatory_in_edges, _1] = network_graph.get_number_excitatory_in_edges(neuron_id);
        const auto [number_inhibitiry_in_edges, _2] = network_graph.get_number_inhibitory_in_edges(neuron_id);
        const auto [number_out_edges, _3] = network_graph.get_number_out_edges(neuron_id);

        ASSERT_EQ(number_excitatory_in_edges, 0);
        ASSERT_EQ(number_inhibitiry_in_edges, 0);
        ASSERT_EQ(number_out_edges, 0);
    }
}

void NetworkGraphTest::assert_local_static_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    auto [_1, all_local_in_edges] = network_graph.get_all_local_in_edges();
    auto [_2, all_local_out_edges] = network_graph.get_all_local_out_edges();

    for (const auto& neighborhood : all_local_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_local_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& [_3, local_in_edges] = network_graph.get_local_in_edges(neuron_id);
        const auto& [_4, all_local_out_edges] = network_graph.get_local_out_edges(neuron_id);

        ASSERT_TRUE(local_in_edges.empty());
        ASSERT_TRUE(all_local_out_edges.empty());
    }
}

void NetworkGraphTest::assert_distant_static_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    auto [_1, all_distant_in_edges] = network_graph.get_all_distant_in_edges();
    auto [_2, all_distant_out_edges] = network_graph.get_all_distant_out_edges();

    for (const auto& neighborhood : all_distant_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_distant_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& [_3, distant_in_edges] = network_graph.get_distant_in_edges(neuron_id);
        const auto& [_4, distant_out_edges] = network_graph.get_distant_out_edges(neuron_id);

        ASSERT_TRUE(distant_in_edges.empty());
        ASSERT_TRUE(distant_out_edges.empty());
    }
}

void NetworkGraphTest::assert_static_empty(const NetworkGraph& network_graph) {
    assert_local_static_empty(network_graph);
    assert_distant_static_empty(network_graph);

    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto [_1, number_excitatory_in_edges] = network_graph.get_number_excitatory_in_edges(neuron_id);
        const auto [_2, number_inhibitiry_in_edges] = network_graph.get_number_inhibitory_in_edges(neuron_id);
        const auto [_3, number_out_edges] = network_graph.get_number_out_edges(neuron_id);

        ASSERT_EQ(number_excitatory_in_edges, 0);
        ASSERT_EQ(number_inhibitiry_in_edges, 0);
        ASSERT_EQ(number_out_edges, 0);
    }
}

void NetworkGraphTest::assert_plastic_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type expected_number_neurons) {
    const auto number_neurons = network_graph.get_number_neurons();
    ASSERT_EQ(number_neurons, expected_number_neurons);

    auto [all_distant_in_edges, _1] = network_graph.get_all_distant_in_edges();
    auto [all_distant_out_edges, _2] = network_graph.get_all_distant_out_edges();
    auto [all_local_in_edges, _3] = network_graph.get_all_local_in_edges();
    auto [all_local_out_edges, _4] = network_graph.get_all_local_out_edges();

    ASSERT_EQ(expected_number_neurons, all_distant_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_distant_out_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_out_edges.size());

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_NO_THROW(const auto& incoming_excitatory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Excitatory););
        ASSERT_NO_THROW(const auto& incoming_inhibitory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Inhibitory););
        ASSERT_NO_THROW(const auto& outgoing_partners = network_graph.get_all_plastic_partners_outgoing(neuron_id););
    }

    for (const auto neuron_id : NeuronID::range(number_neurons, number_neurons + number_neurons_out_of_scope)) {
        ASSERT_THROW(const auto& incoming_excitatory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Excitatory);, RelearnException);
        ASSERT_THROW(const auto& incoming_inhibitory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Inhibitory);, RelearnException);
        ASSERT_THROW(const auto& outgoing_partners = network_graph.get_all_plastic_partners_outgoing(neuron_id);, RelearnException);
    }
}

void NetworkGraphTest::assert_static_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type expected_number_neurons) {
    const auto number_neurons = network_graph.get_number_neurons();
    ASSERT_EQ(number_neurons, expected_number_neurons);

    auto [_1, all_distant_in_edges] = network_graph.get_all_distant_in_edges();
    auto [_2, all_distant_out_edges] = network_graph.get_all_distant_out_edges();
    auto [_3, all_local_in_edges] = network_graph.get_all_local_in_edges();
    auto [_4, all_local_out_edges] = network_graph.get_all_local_out_edges();

    ASSERT_EQ(expected_number_neurons, all_distant_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_distant_out_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_out_edges.size());
}

TEST_F(NetworkGraphTest, testNetworkGraphConstructor) {
    const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);

    NetworkGraph ng(rank);

    ASSERT_EQ(ng.get_number_neurons(), 0);

    assert_plastic_empty(ng);
    assert_static_empty(ng);
}

TEST_F(NetworkGraphTest, testNetworkGraphConstructorException) {
    ASSERT_THROW(NetworkGraph ng_exception(MPIRank::uninitialized_rank());, RelearnException);
}

TEST_F(NetworkGraphTest, testNetworkGraphInit) {
    const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NetworkGraph ng(rank);
    ng.init(number_neurons);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, number_neurons);
    assert_static_size(ng, number_neurons);
}

TEST_F(NetworkGraphTest, testNetworkGraphInitException) {
    const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    NetworkGraph ng(rank);

    ASSERT_THROW(ng.init(0), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, 0);
    assert_static_size(ng, 0);

    ng.init(number_neurons);

    ASSERT_THROW(ng.init(0), RelearnException);
    ASSERT_THROW(ng.init(number_neurons), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, number_neurons);
    assert_static_size(ng, number_neurons);
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeurons) {
    const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto number_neurons = number_neurons_init + number_neurons_create;

    NetworkGraph ng(rank);
    ng.init(number_neurons_init);
    ng.create_neurons(number_neurons_create);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, number_neurons);
    assert_static_size(ng, number_neurons);
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeuronsException) {
    const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto number_neurons = number_neurons_init + number_neurons_create;

    NetworkGraph ng(rank);

    ASSERT_THROW(ng.create_neurons(0), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, 0);
    assert_static_size(ng, 0);

    ASSERT_THROW(ng.create_neurons(number_neurons_create), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, 0);
    assert_static_size(ng, 0);

    ASSERT_THROW(ng.create_neurons(0), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, 0);
    assert_static_size(ng, 0);

    ng.init(number_neurons_init);

    ASSERT_THROW(ng.create_neurons(0), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, number_neurons_init);
    assert_static_size(ng, number_neurons_init);

    ng.create_neurons(number_neurons_create);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, number_neurons);
    assert_static_size(ng, number_neurons);

    ASSERT_THROW(ng.create_neurons(0), RelearnException);

    assert_plastic_empty(ng);
    assert_static_empty(ng);

    assert_plastic_size(ng, number_neurons);
    assert_static_size(ng, number_neurons);
}

TEST_F(NetworkGraphTest, testNetworkGraphLocalEdges) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_synapses = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    NetworkGraph ng(MPIRank::root_rank());
    ng.init(number_neurons);

    std::map<NeuronID::value_type, std::map<NeuronID::value_type, RelearnTypes::plastic_synapse_weight>> incoming_edges{};
    std::map<NeuronID::value_type, std::map<NeuronID::value_type, RelearnTypes::plastic_synapse_weight>> outgoing_edges{};

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto target_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        ng.add_synapse(PlasticLocalSynapse(target_id, source_id, weight));
        incoming_edges[target_id.get_neuron_id()][source_id.get_neuron_id()] += weight;
        outgoing_edges[source_id.get_neuron_id()][target_id.get_neuron_id()] += weight;
    }

    erase_empties(incoming_edges);
    erase_empties(outgoing_edges);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& golden_in_edges = incoming_edges[neuron_id.get_neuron_id()];
        const auto& golden_out_edges = outgoing_edges[neuron_id.get_neuron_id()];

        const auto [exc_in_edges_count, _1] = ng.get_number_excitatory_in_edges(neuron_id);
        const auto [inh_in_edges_count, _2] = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto [out_edges_count, _3] = ng.get_number_out_edges(neuron_id);

        const auto golden_excitatory_in_edges_count = ranges::accumulate(
            golden_in_edges
                | ranges::views::values
                | ranges::views::filter(greater_equal(0)),
            ranges::range_value_t<decltype(golden_in_edges)>::second_type{ 0 });

        const auto golden_inhibitory_in_edges_count = ranges::accumulate(
            golden_in_edges
                | ranges::views::values
                | ranges::views::filter(less_equal(0))
                | ranges::views::transform(as_abs),
            ranges::range_value_t<decltype(golden_in_edges)>::second_type{ 0 });

        const auto golden_out_edges_count = ranges::accumulate(
            golden_out_edges
                | ranges::views::values
                | ranges::views::transform(as_abs),
            ranges::range_value_t<decltype(golden_out_edges)>::second_type{ 0 });

        ASSERT_EQ(exc_in_edges_count, golden_excitatory_in_edges_count);
        ASSERT_EQ(inh_in_edges_count, golden_inhibitory_in_edges_count);
        ASSERT_EQ(out_edges_count, golden_out_edges_count);

        const auto& [local_in_edges, _4] = ng.get_local_in_edges(neuron_id);
        const auto& [distant_in_edges, _5] = ng.get_distant_in_edges(neuron_id);
        const auto& [local_out_edges, _6] = ng.get_local_out_edges(neuron_id);
        const auto& [distant_out_edges, _7] = ng.get_distant_out_edges(neuron_id);

        const auto golden_local_in_edges = incoming_edges[neuron_id.get_neuron_id()].size();
        const auto golden_local_out_edges = outgoing_edges[neuron_id.get_neuron_id()].size();

        ASSERT_EQ(local_in_edges.size(), golden_local_in_edges);
        ASSERT_EQ(distant_in_edges.size(), 0);
        ASSERT_EQ(local_out_edges.size(), golden_local_out_edges);
        ASSERT_EQ(distant_out_edges.size(), 0);

        for (const auto& [other_neuron_id, weight] : local_in_edges) {
            ASSERT_EQ(weight, incoming_edges[neuron_id.get_neuron_id()][other_neuron_id.get_neuron_id()]);
        }

        for (const auto& [other_neuron_id, weight] : local_out_edges) {
            ASSERT_EQ(weight, outgoing_edges[neuron_id.get_neuron_id()][other_neuron_id.get_neuron_id()]);
        }

        const auto& all_in_edges_excitatory = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Excitatory);
        const auto& all_in_edges_inhibitory = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Inhibitory);
        const auto& all_out_edges_excitatory = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Excitatory);
        const auto& all_out_edges_inhibitory = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Inhibitory);

        ASSERT_EQ(all_in_edges_excitatory.size() + all_in_edges_inhibitory.size(), golden_local_in_edges);
        ASSERT_EQ(all_out_edges_excitatory.size() + all_out_edges_inhibitory.size(), golden_local_out_edges);

        for (const auto& [rank_neuron_id, weight] : all_in_edges_excitatory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, incoming_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        for (const auto& [rank_neuron_id, weight] : all_in_edges_inhibitory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, incoming_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        for (const auto& [rank_neuron_id, weight] : all_out_edges_excitatory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, outgoing_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        for (const auto& [rank_neuron_id, weight] : all_out_edges_inhibitory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, outgoing_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        const auto& in_edges = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id);
        const auto& out_edges = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id);

        ASSERT_EQ(in_edges.size(), golden_local_in_edges);
        ASSERT_EQ(out_edges.size(), golden_local_out_edges);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdges) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_synapses = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    NetworkGraph ng(MPIRank::root_rank());
    ng.init(number_neurons);

    std::map<size_t, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>> in_edges{};
    std::map<size_t, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>> out_edges{};

    for (size_t edge_id = 0; edge_id < number_synapses; edge_id++) {
        const auto other_rank = MPIRankAdapter::get_random_mpi_rank(32, MPIRank::root_rank(), mt);
        const auto my_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto other_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const RelearnTypes::plastic_synapse_weight weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ MPIRank::root_rank(), my_neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(PlasticDistantInSynapse(my_neuron_id, other_id, weight));
            in_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] += weight;

            if (in_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] == 0) {
                in_edges[my_neuron_id.get_neuron_id()].erase({ other_rank, other_neuron_id });
            }
        } else {
            ng.add_synapse(PlasticDistantOutSynapse(other_id, my_neuron_id, weight));
            out_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] += weight;

            if (out_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] == 0) {
                out_edges[my_neuron_id.get_neuron_id()].erase({ other_rank, other_neuron_id });
            }
        }
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto [exc_in_edges_count_ng, _1] = ng.get_number_excitatory_in_edges(neuron_id);
        const auto [inh_in_edges_count_ng, _2] = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto [out_edges_count_ng, _3] = ng.get_number_out_edges(neuron_id);

        const auto& in_edges_ng = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id);
        const auto& out_edges_ng = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id);

        auto exc_in_edges_count_meta = 0;
        auto inh_in_edges_count_meta = 0;
        auto out_edges_count_meta = 0;

        for (const auto& it : in_edges[neuron_id.get_neuron_id()]) {
            if (it.second > 0) {
                exc_in_edges_count_meta += it.second;
            } else {
                inh_in_edges_count_meta += -it.second;
            }
        }

        for (const auto& it : out_edges[neuron_id.get_neuron_id()]) {
            out_edges_count_meta += std::abs(it.second);
        }

        ASSERT_EQ(exc_in_edges_count_ng, exc_in_edges_count_meta);
        ASSERT_EQ(inh_in_edges_count_ng, inh_in_edges_count_meta);
        ASSERT_EQ(out_edges_count_ng, out_edges_count_meta);

        for (const auto& [key, weight_meta] : in_edges[neuron_id.get_neuron_id()]) {
            ASSERT_TRUE(ranges::contains(in_edges_ng, std::make_pair(key, weight_meta)));
        }

        for (const auto& [key, weight_meta] : out_edges[neuron_id.get_neuron_id()]) {
            ASSERT_TRUE(ranges::contains(out_edges_ng, std::make_pair(key, weight_meta)));
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdges2) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_synapses = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    NetworkGraph ng_golden(MPIRank::root_rank());
    ng_golden.init(number_neurons);

    NetworkGraph ng(MPIRank::root_rank());
    ng.init(number_neurons);

    std::vector<PlasticLocalSynapse> local_synapses{};
    std::vector<PlasticDistantInSynapse> distant_in_synapses{};
    std::vector<PlasticDistantOutSynapse> distant_out_synapses{};

    for (size_t edge_id = 0; edge_id < number_synapses; edge_id++) {
        const auto other_rank = MPIRankAdapter::get_random_mpi_rank(32, MPIRank::root_rank(), mt);
        const auto my_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto other_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ MPIRank::root_rank(), my_neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng_golden.add_synapse(PlasticDistantInSynapse(my_neuron_id, other_id, weight));
            distant_in_synapses.emplace_back(my_neuron_id, other_id, weight);
        } else {
            ng_golden.add_synapse(PlasticDistantOutSynapse(other_id, my_neuron_id, weight));
            distant_out_synapses.emplace_back(other_id, my_neuron_id, weight);
        }
    }

    for (size_t synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto source_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto target_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        ng_golden.add_synapse(PlasticLocalSynapse(target_id, source_id, weight));
        local_synapses.emplace_back(target_id, source_id, weight);
    }

    ng.add_edges(local_synapses, distant_in_synapses, distant_out_synapses);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        auto [local_in_golden_ref, _1] = ng_golden.get_local_in_edges(neuron_id);
        auto [local_out_golden_ref, _2] = ng_golden.get_local_out_edges(neuron_id);
        auto [distant_in_golden_ref, _3] = ng_golden.get_distant_in_edges(neuron_id);
        auto [distant_out_golden_ref, _4] = ng_golden.get_distant_out_edges(neuron_id);

        auto local_in_golden = local_in_golden_ref;
        auto local_out_golden = local_out_golden_ref;
        auto distant_in_golden = distant_in_golden_ref;
        auto distant_out_golden = distant_out_golden_ref;

        std::ranges::sort(local_in_golden);
        std::ranges::sort(local_out_golden);
        std::ranges::sort(distant_in_golden);
        std::ranges::sort(distant_out_golden);

        auto [local_in_ref, _5] = ng.get_local_in_edges(neuron_id);
        auto [local_out_ref, _6] = ng.get_local_out_edges(neuron_id);
        auto [distant_in_ref, _7] = ng.get_distant_in_edges(neuron_id);
        auto [distant_out_ref, _8] = ng.get_distant_out_edges(neuron_id);

        auto local_in = local_in_ref;
        auto local_out = local_out_ref;
        auto distant_in = distant_in_ref;
        auto distant_out = distant_out_ref;

        std::ranges::sort(local_in);
        std::ranges::sort(local_out);
        std::ranges::sort(distant_in);
        std::ranges::sort(distant_out);

        ASSERT_EQ(local_in_golden, local_in);
        ASSERT_EQ(local_out_golden, local_out);
        ASSERT_EQ(distant_in_golden, distant_in);
        ASSERT_EQ(distant_out_golden, distant_out);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdgesSplit) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_edges = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    NetworkGraph ng(MPIRank::root_rank());
    ng.init(number_neurons);

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const auto other_rank = MPIRankAdapter::get_random_mpi_rank(32, MPIRank::root_rank(), mt);
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto other_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ MPIRank::root_rank(), neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(PlasticDistantInSynapse(neuron_id, other_id, weight));
        } else {
            ng.add_synapse(PlasticDistantOutSynapse(other_id, neuron_id, weight));
        }
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& in_edges_ng = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id);
        const auto& out_edges_ng = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id);

        auto in_edges_ng_ex = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Excitatory);
        const auto& in_edges_ng_in = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Inhibitory);
        const auto& out_edges_ng_ex = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Excitatory);
        auto out_edges_ng_in = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id, SignalType::Inhibitory);

        ASSERT_EQ(in_edges_ng.size(), in_edges_ng_ex.size() + in_edges_ng_in.size());
        ASSERT_EQ(out_edges_ng.size(), out_edges_ng_ex.size() + out_edges_ng_in.size());

        for (const auto& [_, edge_val] : in_edges_ng) {
            ASSERT_TRUE(edge_val < 0);
        }

        for (const auto& [_, edge_val] : out_edges_ng_in) {
            ASSERT_TRUE(edge_val < 0);
        }

        for (const auto& [_, edge_val] : in_edges_ng_ex) {
            ASSERT_TRUE(edge_val > 0);
        }

        for (const auto& [_, edge_val] : out_edges_ng_ex) {
            ASSERT_TRUE(edge_val > 0);
        }

        for (const auto& val : in_edges_ng_in) {
            in_edges_ng_ex.emplace_back(val);
        }

        for (const auto& val : out_edges_ng_ex) {
            out_edges_ng_in.emplace_back(val);
        }

        ASSERT_EQ(in_edges_ng.size(), in_edges_ng_ex.size());
        ASSERT_EQ(out_edges_ng.size(), out_edges_ng_ex.size());

        for (const auto& [edge_key, edge_val] : in_edges_ng) {
            ASSERT_TRUE(ranges::contains(in_edges_ng_ex, std::make_pair(edge_key, edge_val)));
        }

        for (const auto& [edge_key, edge_val] : out_edges_ng) {
            ASSERT_TRUE(ranges::contains(out_edges_ng_ex, std::make_pair(edge_key, edge_val)));
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdgesRemoval) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_edges = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    NetworkGraph ng(MPIRank::root_rank());
    ng.init(number_neurons);

    std::vector<std::tuple<NeuronID, MPIRank, NeuronID, MPIRank, RelearnTypes::plastic_synapse_weight>> synapses(num_edges);

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const auto other_rank = MPIRankAdapter::get_random_mpi_rank(32, MPIRank::root_rank(), mt);
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto other_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ MPIRank::root_rank(), neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (other_rank == MPIRank::root_rank()) {
            ng.add_synapse(PlasticLocalSynapse(neuron_id, other_neuron_id, weight));
            synapses[edge_id] = std::make_tuple(neuron_id, MPIRank::root_rank(), other_neuron_id, MPIRank::root_rank(), weight);
            continue;
        }

        if (is_in_synapse) {
            ng.add_synapse(PlasticDistantInSynapse(neuron_id, other_id, weight));
            synapses[edge_id] = std::make_tuple(neuron_id, MPIRank::root_rank(), other_neuron_id, other_rank, weight);
        } else {
            ng.add_synapse(PlasticDistantOutSynapse(other_id, neuron_id, weight));
            synapses[edge_id] = std::make_tuple(other_neuron_id, other_rank, neuron_id, MPIRank::root_rank(), weight);
        }
    }

    shuffle(synapses, mt);

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const auto& current_synapse = synapses[edge_id];
        const auto& [target_neuron_id, target_rank, source_neuron_id, source_rank, weight] = current_synapse;

        RankNeuronId target_id{ target_rank, target_neuron_id };
        RankNeuronId source_id{ source_rank, source_neuron_id };

        if (source_rank == MPIRank::root_rank() && target_rank == MPIRank::root_rank()) {
            ng.add_synapse(PlasticLocalSynapse(target_neuron_id, source_neuron_id, -weight));
            continue;
        }

        if (source_rank == MPIRank::root_rank()) {
            ng.add_synapse(PlasticDistantOutSynapse(target_id, source_neuron_id, -weight));
            continue;
        }

        ng.add_synapse(PlasticDistantInSynapse(target_neuron_id, source_id, -weight));
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto [exc_in_edges_count_1, exc_in_edges_count_2] = ng.get_number_excitatory_in_edges(neuron_id);
        const auto [inh_in_edges_count_1, inh_in_edges_count_2] = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto [out_edges_count_1, out_edges_count_2] = ng.get_number_out_edges(neuron_id);

        ASSERT_EQ(exc_in_edges_count_1, 0);
        ASSERT_EQ(inh_in_edges_count_1, 0);
        ASSERT_EQ(out_edges_count_1, 0);
        ASSERT_EQ(exc_in_edges_count_2, 0);
        ASSERT_EQ(inh_in_edges_count_2, 0);
        ASSERT_EQ(out_edges_count_2, 0);

        const auto& in_edges = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id);
        const auto& out_edges = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id);

        ASSERT_EQ(in_edges.size(), 0);
        ASSERT_EQ(out_edges.size(), 0);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreate) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_edges = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    NetworkGraph ng(MPIRank::root_rank());
    ng.init(number_neurons);

    std::map<RankNeuronId, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>> in_edges;
    std::map<RankNeuronId, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>> out_edges;

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const auto other_rank = MPIRankAdapter::get_random_mpi_rank(32, MPIRank::root_rank(), mt);
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto other_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ MPIRank::root_rank(), neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(PlasticDistantInSynapse(neuron_id, other_id, weight));
            in_edges[my_id][other_id] += weight;
        } else {
            ng.add_synapse(PlasticDistantOutSynapse(other_id, neuron_id, weight));
            out_edges[my_id][other_id] += weight;
        }
    }

    const auto num_new_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_new_edges = NetworkGraphAdapter::get_random_number_synapses(mt);

    const auto total_number_neurons = number_neurons + num_new_neurons;
    const auto total_num_edges = num_edges + num_new_edges;

    ng.create_neurons(num_new_neurons);

    for (size_t edge_id = num_edges; edge_id < total_num_edges; edge_id++) {
        const auto other_rank = MPIRankAdapter::get_random_mpi_rank(32, MPIRank::root_rank(), mt);
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto other_neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);

        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ MPIRank::root_rank(), neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(PlasticDistantInSynapse(neuron_id, other_id, weight));
            in_edges[my_id][other_id] += weight;
        } else {
            ng.add_synapse(PlasticDistantOutSynapse(other_id, neuron_id, weight));
            out_edges[my_id][other_id] += weight;
        }
    }

    for (auto neuron_id : NeuronID::range(total_number_neurons)) {
        const auto [exc_in_edges_count_ng, _1] = ng.get_number_excitatory_in_edges(neuron_id);
        const auto [inh_in_edges_count_ng, _2] = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto [out_edges_count_ng, _3] = ng.get_number_out_edges(neuron_id);

        const auto& in_edges_ng = NetworkGraphAdapter::get_all_plastic_in_edges(ng, MPIRank::root_rank(), neuron_id);
        const auto& out_edges_ng = NetworkGraphAdapter::get_all_plastic_out_edges(ng, MPIRank::root_rank(), neuron_id);

        auto exc_in_edges_count_meta = 0;
        auto inh_in_edges_count_meta = 0;
        auto out_edges_count_meta = 0;

        for (const auto& it : in_edges[{ MPIRank::root_rank(), neuron_id }]) {
            if (it.second > 0) {
                exc_in_edges_count_meta += it.second;
            } else {
                inh_in_edges_count_meta += -it.second;
            }
        }

        for (const auto& it : out_edges[{ MPIRank::root_rank(), neuron_id }]) {
            out_edges_count_meta += std::abs(it.second);
        }

        ASSERT_EQ(exc_in_edges_count_ng, exc_in_edges_count_meta);
        ASSERT_EQ(inh_in_edges_count_ng, inh_in_edges_count_meta);
        ASSERT_EQ(out_edges_count_ng, out_edges_count_meta);

        for (const auto& [key, weight_meta] : in_edges[{ MPIRank::root_rank(), neuron_id }]) {
            ASSERT_TRUE(ranges::contains(in_edges_ng, std::make_pair(key, weight_meta)));
        }

        for (const auto& [key, weight_meta] : out_edges[{ MPIRank::root_rank(), neuron_id }]) {
            ASSERT_TRUE(ranges::contains(out_edges_ng, std::make_pair(key, weight_meta)));
        }
    }
}
