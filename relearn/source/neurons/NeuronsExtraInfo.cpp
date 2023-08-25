/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronsExtraInfo.h"

#include "mpi/MPIWrapper.h"
#include "util/Random.h"

void NeuronsExtraInfo::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(!positions.empty(), "NeuronsExtraInfo::create_neurons: Was not initialized");
    RelearnException::check(creation_count != 0, "Cannot add 0 neurons");

    const auto num_ranks = MPIWrapper::get_num_ranks();

    RelearnException::check(num_ranks == 1, "NeuronsExtraInfo::create_neurons: Cannot create neurons if more than 1 MPI rank is computing");

    const auto current_size = size;
    const auto new_size = current_size + creation_count;

    update_status.resize(new_size, UpdateStatus::Enabled);
    deletions_log.resize(new_size, {});
    positions.resize(new_size);
    fire_history.resize(new_size);

    for (number_neurons_type i = current_size; i < new_size; i++) {
        const auto x_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);
        const auto y_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);
        const auto z_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);

        const auto random_pos_x = static_cast<number_neurons_type>(x_it * static_cast<double>(current_size));
        const auto random_pos_y = static_cast<number_neurons_type>(y_it * static_cast<double>(current_size));
        const auto random_pos_z = static_cast<number_neurons_type>(z_it * static_cast<double>(current_size));

        const auto x_pos = positions[random_pos_x].get_x();
        const auto y_pos = positions[random_pos_y].get_y();
        const auto z_pos = positions[random_pos_z].get_z();

        positions[i] = { x_pos, y_pos, z_pos };
    }

    size = new_size;
}

CommunicationMap<RelearnTypes::position_type> NeuronsExtraInfo::get_positions_for(const CommunicationMap<NeuronID>& local_neurons) {
    const auto number_ranks = local_neurons.get_number_ranks();
    const auto size_hint = local_neurons.size();

    CommunicationMap<RelearnTypes::position_type> positions(number_ranks, size_hint);

    if (local_neurons.empty()) {
        return positions;
    }

    positions.resize(local_neurons.get_request_sizes());

    for (const auto& [source_rank, requests] : local_neurons) {
        for (auto i = 0; i < requests.size(); i++) {
            const auto neuron_id = requests[i];
            const auto& position = get_position(neuron_id);
            positions.set_request(source_rank, i, position);
        }
    }

    return positions;
}
