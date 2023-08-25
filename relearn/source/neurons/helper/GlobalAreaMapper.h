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

#include "mpi/MPIWrapper.h"
#include "neurons/LocalAreaTranslator.h"
#include "Types.h"

#include <unordered_map>

/**
 * Class finds the area id of neurons on other ranks through mpi communication
 */
class GlobalAreaMapper {
public:
    /**
     * Constructor
     * @param local_area_translator The local area translator
     * @param num_ranks Number of mpi ranks
     * @param my_rank Current mp9i rank
     */
    GlobalAreaMapper(const std::shared_ptr<LocalAreaTranslator>& local_area_translator, int num_ranks, MPIRank my_rank)
        : local_area_translator(local_area_translator)
        , num_ranks(num_ranks)
        , my_rank(my_rank) {
        next_request.resize(num_ranks, {});
    }

    /**
     * Indicates that someone wants to know the area id for a neuron. Area id will be requested with the next communication
     * @param rni The neuron whose area id we want to know
     */
    void request_area_id(const RankNeuronId& rni) {
        if (my_rank == rni.get_rank() || known_mappings.contains(rni)) {
            return;
        }
        next_request[rni.get_rank().get_rank()].emplace_back(rni.get_neuron_id().get_neuron_id());
    }

    void check_cache() {
        known_mappings.clear();
    }

    /**
     * Returns the area id for a neuron. Must have been requested before the last communication
     * @param rni The neuron whose area id we want to know
     * @return Area id
     */
    RelearnTypes::area_id get_area_id(const RankNeuronId& rni) {
        if (rni.get_rank() == my_rank) {
            return local_area_translator->get_area_id_for_neuron_id(rni.get_neuron_id().get_neuron_id());
        }
        RelearnException::check(known_mappings.contains(rni), "GlobalAreaMapper::get_area_id: Unknown rank neuron id {}", rni);
        return known_mappings[rni];
    }

    /**
     * Start communication with other mpi ranks and exchange requested area ids
     */
    void exchange_requests() {
        send_requests();
        next_request.clear();
        next_request.resize(MPIWrapper::get_num_ranks(), {});
    }

private:
    std::shared_ptr<LocalAreaTranslator> local_area_translator{};
    std::unordered_map<RankNeuronId, RelearnTypes::area_id> known_mappings{};
    std::vector<std::vector<NeuronID>> next_request{};

    int num_ranks;
    MPIRank my_rank;

    static constexpr auto max_size_map = 10000;

    void send_requests() {
        const auto& received_requested_data = MPIWrapper::exchange_values<NeuronID>(next_request);
        answer_requests(received_requested_data);
    }

    void answer_requests(const std::vector<std::vector<NeuronID>>& received_data) {
        std::vector<std::vector<RelearnTypes::area_id>> answer_data{};
        answer_data.resize(received_data.size(), {});
        for (auto requesting_rank = 0; requesting_rank < received_data.size(); requesting_rank++) {
            for (const auto& neuron_id : received_data[requesting_rank]) {
                const auto& area_id = local_area_translator->get_area_id_for_neuron_id(neuron_id.get_neuron_id());
                answer_data[requesting_rank].emplace_back(area_id);
            }
        }
        const auto& received_answer_data = MPIWrapper::exchange_values<RelearnTypes::area_id>(answer_data);
        parse_answer(received_answer_data);
    }

    void parse_answer(const std::vector<std::vector<RelearnTypes::area_id>>& answer) {

        for (auto rank = 0; rank < answer.size(); rank++) {
            for (auto i = 0; i < answer[rank].size(); i++) {
                auto area_id = answer[rank][i];
                auto neuron_id = next_request[rank][i];
                const RankNeuronId rank_neuron_id{ MPIRank{ rank }, NeuronID{ neuron_id } };
                known_mappings.insert(std::make_pair(rank_neuron_id, area_id));
            }
        }
    }
};