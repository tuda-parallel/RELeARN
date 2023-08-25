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
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "neurons/helper/RankNeuronId.h"

#include <random>
#include <sstream>
#include <string>
#include <utility>

class RankNeuronIdAdapter {
public:
    static RankNeuronId generate_random_rank_neuron_id(std::mt19937& mt) {
        const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(mt);

        return { rank, neuron_id };
    }

    static std::string codify_rank_neuron_id(const RankNeuronId& rni) {
        std::stringstream ss{};
        ss << rni.get_rank().get_rank() << ':' << (rni.get_neuron_id().get_neuron_id() + 1);
        return ss.str();
    }

    static RankNeuronId add_one_to_neuron_id(const RankNeuronId& rni) {
        const auto& [rank, neuron_id] = rni;
        const auto id = neuron_id.get_neuron_id();
        return RankNeuronId(rank, NeuronID(id + 1));
    }

    static RankNeuronId substract_one_from_neuron_id(const RankNeuronId& rni) {
        const auto& [rank, neuron_id] = rni;
        const auto id = neuron_id.get_neuron_id();
        return RankNeuronId(rank, NeuronID(id - 1));
    }

    static std::pair<RankNeuronId, std::string> generate_random_rank_neuron_id_description(std::mt19937& mt) {
        auto rank_neuron_id = RankNeuronIdAdapter::generate_random_rank_neuron_id(mt);
        auto description = codify_rank_neuron_id(rank_neuron_id);
        return { std::move(rank_neuron_id), std::move(description) };
    }
};
