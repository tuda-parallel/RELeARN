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

#include "neurons/NetworkGraph.h"
#include "util/ranges/Functional.hpp"

#include <memory>
#include <random>
#include <vector>

#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>

class NetworkGraphFactory {
public:
    static std::shared_ptr<NetworkGraph> construct_network_graph(RelearnTypes::number_neurons_type number_neurons) {
        auto network_graph = std::make_shared<NetworkGraph>(MPIRank::root_rank());
        network_graph->init(number_neurons);
        return network_graph;
    }

    static PlasticLocalSynapses generate_local_synapses(RelearnTypes::number_neurons_type number_neurons, RelearnTypes::number_neurons_type number_synapses_per_neuron) {
        std::vector<PlasticLocalSynapse> synapses{};
        synapses.reserve(number_neurons * number_synapses_per_neuron);

        std::mt19937 mt{};
        std::uniform_int_distribution<uint64_t> uid(0, number_neurons - 1);

        for (auto neuron_id = 0ULL; neuron_id < number_neurons; neuron_id++) {
            for (auto synapse_id = 0ULL; synapse_id < number_synapses_per_neuron; synapse_id++) {
                auto random_id = uid(mt);

                const NeuronID source_id{ neuron_id };
                const NeuronID target_id{ random_id };

                const auto weight = 1;

                synapses.emplace_back(target_id, source_id, weight);
            }
        }

        return synapses;
    }

    static PlasticDistantInSynapses generate_distant_in_synapses(RelearnTypes::number_neurons_type number_neurons, RelearnTypes::number_neurons_type number_synapses_per_neuron) {
        std::vector<PlasticDistantInSynapse> synapses{};
        synapses.reserve(number_neurons * number_synapses_per_neuron);

        std::mt19937 mt{};
        std::uniform_int_distribution<uint64_t> uid(0, number_neurons - 1);
        std::uniform_int_distribution<int> uid_rank(1, 32);

        for (auto neuron_id = 0ULL; neuron_id < number_neurons; neuron_id++) {
            for (auto synapse_id = 0ULL; synapse_id < number_synapses_per_neuron; synapse_id++) {
                auto random_id = uid(mt);
                auto random_rank = uid_rank(mt);

                const NeuronID source_id{ random_id };
                const NeuronID target_id{ neuron_id };

                const RankNeuronId rni{ MPIRank(random_rank), source_id };

                const auto weight = 1;

                synapses.emplace_back(target_id, rni, weight);
            }
        }

        return synapses;
    }

    static PlasticDistantOutSynapses generate_distant_out_synapses(RelearnTypes::number_neurons_type number_neurons, RelearnTypes::number_neurons_type number_synapses_per_neuron) {
        std::vector<PlasticDistantOutSynapse> synapses{};
        synapses.reserve(number_neurons * number_synapses_per_neuron);

        std::mt19937 mt{};
        std::uniform_int_distribution<uint64_t> uid(0, number_neurons - 1);
        std::uniform_int_distribution<int> uid_rank(1, 32);

        for (auto neuron_id = 0ULL; neuron_id < number_neurons; neuron_id++) {
            for (auto synapse_id = 0ULL; synapse_id < number_synapses_per_neuron; synapse_id++) {
                auto random_id = uid(mt);
                auto random_rank = uid_rank(mt);

                const NeuronID source_id{ neuron_id };
                const NeuronID target_id{ random_id };

                const RankNeuronId rni{ MPIRank(random_rank), target_id };

                const auto weight = 1;

                synapses.emplace_back(rni, source_id, weight);
            }
        }

        return synapses;
    }

    template <typename SynapseType>
    static std::vector<SynapseType> invert_synapses(const std::vector<SynapseType>& synapses) {
        static constexpr auto to_weight = element<2>;
        return synapses | ranges::views::transform([](SynapseType synapse) {
                 to_weight(synapse) = -to_weight(synapse);
                 return synapse;
               }) |
               ranges::to_vector;
    }

    template <typename SynapseType>
    static void add_synapses(NetworkGraph& ng, const std::vector<SynapseType>& synapses) {
        for (const auto& synapse : synapses) {
            ng.add_synapse(synapse);
        }
    }
};
