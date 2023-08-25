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

#include "util/NeuronID.h"
#include "neurons/helper/RankNeuronId.h"

#include <random>
#include <unordered_set>

class NeuronIdAdapter {
public:
    constexpr static int upper_bound_num_neurons = 1000;

    static NeuronID::value_type get_random_number_neurons(std::mt19937& mt) {
        return RandomAdapter::get_random_integer<NeuronID::value_type>(1, upper_bound_num_neurons, mt);
    }

    static NeuronID get_random_neuron_id(std::mt19937& mt) {
        const auto value = RandomAdapter::get_random_integer<NeuronID::value_type>(0, upper_bound_num_neurons - 1, mt);
        return NeuronID{ value };
    }

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, std::mt19937& mt) {
        const auto value = RandomAdapter::get_random_integer<NeuronID::value_type>(0, number_neurons - 1, mt);
        return NeuronID{ value };
    }

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID::value_type offset, std::mt19937& mt) {
        const auto value = RandomAdapter::get_random_integer<NeuronID::value_type>(offset, offset + number_neurons - 1, mt);
        return NeuronID{ value };
    }

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID except, std::mt19937& mt) {
        NeuronID nid{};
        do {
            nid = get_random_neuron_id(number_neurons, mt);
        } while (nid == except);
        return nid;
    }

    static std::unordered_set<NeuronID> get_random_neuron_ids(NeuronID::value_type number_neurons_in_sim, NeuronID::value_type number_neurons_in_sample, std::mt19937& mt) {
        std::unordered_set<NeuronID> set;
        for (const auto _ : NeuronID::range_id(number_neurons_in_sample)) {
            NeuronID neuron_id;
            do {
                neuron_id = get_random_neuron_id(number_neurons_in_sim, mt);
            } while (set.contains(neuron_id));
            set.insert(neuron_id);
        }
        return set;
    }

    static std::unordered_set<RankNeuronId> get_random_rank_neuron_ids(NeuronID::value_type number_neurons_per_rank, size_t num_ranks, NeuronID::value_type number_neurons_in_sample, std::mt19937& mt) {
        std::unordered_set<RankNeuronId> set;
        for (const auto _ : NeuronID::range_id(number_neurons_in_sample)) {
            RankNeuronId rni;
            do {
                const auto neuron_id = get_random_neuron_id(number_neurons_per_rank, mt);
                const auto rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);
                rni = RankNeuronId(rank, neuron_id);
            } while (set.contains(rni));
            set.insert(rni);
        }
        return set;
    }
};
