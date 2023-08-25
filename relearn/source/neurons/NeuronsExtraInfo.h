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
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "neurons/enums/FiredStatus.h"
#include "mpi/MPIWrapper.h"
#include "util/Timers.h"

#include <bitset>
#include <span>
#include <string>
#include <vector>

#include <range/v3/algorithm/fill.hpp>
#include <range/v3/view/transform.hpp>

/**
 * An object of type NeuronsExtraInfo additional information of neurons.
 * For a single neuron, these additional information are: its x-, y-, and z- position and the name of the area the neuron is in.
 * It further stores which neurons update their electrical activity and their plasticity.
 */
class NeuronsExtraInfo {
public:
    using position_type = RelearnTypes::position_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;
    static constexpr unsigned int fire_history_length = 1000;
    constexpr static bool fire_history_enabled = true;

    /**
     * @brief Initializes a NeuronsExtraInfo that holds at most the given number of neurons.
     *      Must only be called once. Sets up all neurons so that they update, but does not initialize the positions.
     * @param number_neurons The number of neurons, greater than 0
     * @exception Throws an RelearnException if number_neurons is 0 or if called multiple times.
     */
    void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_neurons > 0, "NeuronsExtraInfo::init: number_neurons must be larger than 0.");
        RelearnException::check(size == 0, "NeuronsExtraInfo::init: NeuronsExtraInfo initialized two times, its size is already {}", size);

        size = number_neurons;
        update_status.resize(number_neurons, UpdateStatus::Enabled);
        deletions_log.resize(number_neurons, {});
        fire_history.resize(size);
    }

    /**
     * @brief Inserts additional neurons with x-, y-, z- positions randomly picked from already existing ones.
     *      Sets all neurons to update. Only works with one MPI rank.
     * @param creation_count The number of new neurons, greater than 0
     * @exception Throws an RelearnException if creation_count is 0, if the positions are empty, or if more than one MPI rank is active
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Marks the specified neurons as enabled
     * @param enabled_neurons The neuron ids from the now enabled neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_enabled_neurons(const std::span<const NeuronID> enabled_neurons) {
        const auto get_update_status = [this](const auto& neuron_id) -> UpdateStatus& {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_enabled_neurons: NeuronID {} is too large: {}", neuron_id);
            RelearnException::check(update_status[local_neuron_id] == UpdateStatus::Disabled, "NeuronsExtraInformation::set_enabled_neurons: Cannot enable a not disabled neuron");

            return update_status[local_neuron_id];
        };
        RelearnException::check(!enabled_neurons.empty(), "NeuronsExtraInformation::set_enabled_neurons: enabled_neurons is empty");

        ranges::fill(enabled_neurons | ranges::views::transform(get_update_status), UpdateStatus::Enabled);
    }

    /**
     * @brief Marks the specified neurons as disabled
     * @param disabled_neurons The neuron ids from the now disabled neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_disabled_neurons(const std::span<const NeuronID> disabled_neurons) {
        const auto get_update_status = [this](const auto& neuron_id) -> UpdateStatus& {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_disabled_neurons: NeuronID {} is too large: {}", neuron_id, size);

            auto& status = update_status[local_neuron_id];

            RelearnException::check(status != UpdateStatus::Static,
                "NeuronsExtraInformation::set_disabled_neurons: Cannot disable a static neuron");
            RelearnException::check(status != UpdateStatus::Disabled,
                "NeuronsExtraInformation::set_disabled_neurons: Cannot disable a disabled neuron");

            return status;
        };

        ranges::fill(disabled_neurons | ranges::views::transform(get_update_status), UpdateStatus::Disabled);
    }

    /**
     * @brief Marks the specified neurons as static
     * @param static_neurons The neuron ids from the now static neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_static_neurons(const std::span<const NeuronID> static_neurons) {
        const auto get_update_status = [this](const auto& neuron_id) -> UpdateStatus& {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            RelearnException::check(local_neuron_id < this->size, "NeuronsExtraInformation::set_static_neurons: NeuronID {} is too large", neuron_id);

            return update_status[local_neuron_id];
        };

        ranges::fill(static_neurons | ranges::views::transform(get_update_status), UpdateStatus::Static);
    }

    /**
     * @brief Overwrites the current positions with the supplied ones
     * @param names The new positions, must have the same size as neurons are stored
     * @exception Throws an RelearnException if pos.empty() or if the number of supplied elements does not match the number of stored neurons
     */
    void set_positions(std::vector<position_type> pos) {
        RelearnException::check(!pos.empty(), "NeuronsExtraInformation::set_positions: New positions are empty");
        RelearnException::check(size == pos.size(), "NeuronsExtraInformation::set_positions: Size does not match area names count");
        positions = std::move(pos);
    }

    /**
     * @brief Returns the currently stored positions as a vector
     * @return The currently stored positions
     */
    [[nodiscard]] std::span<const position_type> get_positions() const noexcept {
        return positions;
    }

    /**
     * @brief Returns a position_type with the x-, y-, and z- positions for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] position_type get_position(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::get_position: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < positions.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than positions.size() but was {}", neuron_id);
        return positions[local_neuron_id];
    }

    /**
     * @brief Checks for a neuron if it updates its electrical activity
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return True iff the neuron updates its electrical activity
     */
    [[nodiscard]] bool does_update_electrical_actvity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::does_update_electrical_actvity: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < update_status.size(), "NeuronsExtraInfo::does_update_electrical_actvity: neuron_id must be smaller than update_status.size() but was {}", neuron_id);

        return update_status[local_neuron_id] != UpdateStatus::Disabled;
    }

    /**
     * @brief Checks for a neuron if it updates its plasticity
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return True iff the neuron updates its plasticity
     */
    [[nodiscard]] bool does_update_plasticity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::does_update_plasticity: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < update_status.size(), "NeuronsExtraInfo::does_update_plasticity: neuron_id must be smaller than update_status.size() but was {}", neuron_id);

        return update_status[local_neuron_id] == UpdateStatus::Enabled;
    }

    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] const std::span<const UpdateStatus> get_disable_flags() const noexcept {
        return update_status;
    }

    /**
     * @brief Returns the number of stored neurons
     * @return The number of neurons
     */
    [[nodiscard]] number_neurons_type get_size() const noexcept {
        return size;
    }

    /**
     * @brief Translates a collection of local neuron ids to their positions
     * @param local_neurons The local neuron ids
     * @return The position of the local neurons, has the same size as the argument
     */
    [[nodiscard]] CommunicationMap<RelearnTypes::position_type> get_positions_for(const CommunicationMap<NeuronID>& local_neurons);

    void reset_deletion_log() {
        deletions_log.clear();
        deletions_log.resize(size, {});
    }

    void set_fired(const NeuronID& neuron_id, const FiredStatus& fired) {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        fire_history[local_neuron_id] <<= 1;
        fire_history[local_neuron_id][0] = static_cast<bool>(fired);
    }

    void mark_deletion(const NeuronID& target_neuron_id, const RankNeuronId& source_neuron, const RelearnTypes::plastic_synapse_weight weight) {
        deletions_log[target_neuron_id.get_neuron_id()].push_back(std::make_pair(source_neuron, weight));
    }

    [[nodiscard]] const std::bitset<fire_history_length>& get_fire_history(const NeuronID& neuron_id) const {
        return fire_history[neuron_id.get_neuron_id()];
    }

    [[nodiscard]] std::bitset<fire_history_length> get_fire_history(const RankNeuronId& neuron_id) const {
        if (neuron_id.get_rank() == MPIWrapper::get_my_rank()) {
            return get_fire_history(neuron_id.get_neuron_id());
        }
        const auto data = MPIWrapper::get_from_window<std::bitset<fire_history_length>>(MPIWindow::FireHistory, neuron_id.get_rank().get_rank(), neuron_id.get_neuron_id().get_neuron_id(), 1);
        return data[0];
    }

    void publish_fire_history() const {
        if (!fire_history_enabled) {
            return;
        }
        Timers::start(TimerRegion::UPDATE_FIRE_HISTORY);

        MPIWrapper::set_in_window(MPIWindow::FireHistory, 0, fire_history);

        Timers::stop_and_add(TimerRegion::UPDATE_FIRE_HISTORY);
    }

    const std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>& get_deletions_log(const NeuronID& neuron_id) const {
        return deletions_log[neuron_id.get_neuron_id()];
    }

private:
    number_neurons_type size{ 0 };

    std::vector<position_type> positions{};
    std::vector<UpdateStatus> update_status{};

    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> deletions_log{};
    std::vector<std::bitset<fire_history_length>> fire_history{};
};
