/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElements.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/Random.h"

#include <range/v3/view/drop.hpp>

void SynapticElements::init(const number_neurons_type number_neurons) {
    RelearnException::check(size == 0, "SynapticElements::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "SynapticElements::init: Cannot initialize with 0 neurons");

    size = number_neurons;

    grown_elements.resize(size);

    if (initial_vacant_elements_lower_bound < initial_vacant_elements_upper_bound) {
        RandomHolder::fill(RandomHolderKey::SynapticElements, grown_elements, initial_vacant_elements_lower_bound, initial_vacant_elements_upper_bound);
    } else if (initial_vacant_elements_lower_bound == initial_vacant_elements_upper_bound) {
        std::ranges::fill(grown_elements, initial_vacant_elements_lower_bound);
    } else {
        RelearnException::fail("SynapticElements::init: Should initialize synaptic elements with values between in the wrong order (lower is larger than upper)");
    }

    connected_elements.resize(size, 0);
    deltas_since_last_update.resize(size, 0.0);
    signal_types.resize(size);
}

void SynapticElements::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(size > 0, "SynapticElements::create_neurons: Was not initialized");
    RelearnException::check(creation_count > 0, "SynapticElements::create_neurons: Cannot create 0 neurons");

    const auto current_size = size;
    const auto new_size = current_size + creation_count;

    grown_elements.resize(new_size);

    if (initial_vacant_elements_lower_bound < initial_vacant_elements_upper_bound) {
        RandomHolder::fill(RandomHolderKey::SynapticElements, grown_elements | ranges::views::drop(current_size), initial_vacant_elements_lower_bound, initial_vacant_elements_upper_bound);
    } else if (initial_vacant_elements_lower_bound == initial_vacant_elements_upper_bound) {
        ranges::fill(grown_elements | ranges::views::drop(current_size), initial_vacant_elements_lower_bound);
    } else {
        RelearnException::fail("SynapticElements::create_neurons: Should initialize synaptic elements with values between in the wrong order (lower is larger than upper)");
    }

    connected_elements.resize(new_size, 0);
    deltas_since_last_update.resize(new_size, 0.0);
    signal_types.resize(new_size);

    size = new_size;
}

std::pair<unsigned int, std::vector<unsigned int>> SynapticElements::commit_updates() {
    const auto& disable_flags = extra_infos->get_disable_flags();

    RelearnException::check(disable_flags.size() == size, ":SynapticElements::commit_updates: disable_flags was not of the right size");

    auto current_additions = 0.0;
    auto current_deletions = 0.0;

#pragma omp parallel for reduction(+ : current_additions, current_deletions) shared(disable_flags) default(none)
    for (NeuronID::value_type neuron_id = 0U; neuron_id < size; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto delta = deltas_since_last_update[neuron_id];

        const auto adds = delta > 0.0 ? delta : 0.0;
        auto dels = delta < 0.0 ? std::abs(delta) : 0.0;
        if (dels > grown_elements[neuron_id]) {
            dels = grown_elements[neuron_id];
        }

        current_additions += adds;
        current_deletions += dels;
    }

    total_additions += current_additions;
    total_deletions += current_deletions;

    std::vector<unsigned int> number_deletions(size, 0);
    unsigned int sum_to_delete = 0;

#pragma omp parallel for reduction(+ : sum_to_delete) shared(number_deletions, disable_flags) default(none)
    for (NeuronID::value_type neuron_id = 0U; neuron_id < size; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        NeuronID converted_id{ neuron_id };
        const auto num_synapses_to_delete = update_number_elements(converted_id);

        number_deletions[neuron_id] = num_synapses_to_delete;
        sum_to_delete += num_synapses_to_delete;
    }

    return std::make_pair(sum_to_delete, number_deletions);
}

void SynapticElements::update_number_elements_delta(const std::span<const double> calcium, const std::span<const double> target_calcium) {
    if (nu <= 0.0) {
        return;
    }

    const auto& disable_flags = extra_infos->get_disable_flags();

    RelearnException::check(calcium.size() == size, "SynapticElements::update_number_elements_delta: calcium was not of the right size");
    RelearnException::check(target_calcium.size() == size, "SynapticElements::update_number_elements_delta: target_calcium was not of the right size");
    RelearnException::check(disable_flags.size() == size, "SynapticElements::update_number_elements_delta: disable_flags was not of the right size");

#pragma omp parallel for shared(calcium, target_calcium, disable_flags) default(none)
    for (NeuronID::value_type neuron_id = 0U; neuron_id < size; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto target_calcium_value = target_calcium[neuron_id];
        const auto current_calcium_value = calcium[neuron_id];

        const auto clamped_target = std::max(target_calcium_value, min_C_level_to_grow);

        const auto inc = gaussian_growth_curve(current_calcium_value, min_C_level_to_grow, clamped_target, nu);

        deltas_since_last_update[neuron_id] += inc;
    }
}

unsigned int SynapticElements::update_number_elements(const NeuronID neuron_id) {
    const auto local_neuron_id = neuron_id.get_neuron_id();

    RelearnException::check(local_neuron_id < size, "SynapticElements::update_number_elements: {} is too large! {}", neuron_id, size);

    const auto current_count = grown_elements[local_neuron_id];
    const auto current_connected_count_integral = connected_elements[local_neuron_id];
    const auto current_connected_count = static_cast<double>(current_connected_count_integral);
    const auto current_vacant = current_count - current_connected_count;
    const auto current_delta = deltas_since_last_update[local_neuron_id];

    RelearnException::check(current_count >= 0.0, "SynapticElements::update_number_elements: {}", current_count);
    RelearnException::check(current_connected_count >= 0.0, "SynapticElements::update_number_elements: {}", current_connected_count);
    RelearnException::check(current_vacant >= 0.0, "SynapticElements::update_number_elements: {}", current_count - current_connected_count);

    // The vacant portion after caring for the delta
    // No deletion of bound synaptic elements required, connected_elements stays the same
    if (const auto new_vacant = current_vacant + current_delta; new_vacant >= 0.0) {
        const auto new_count = (1 - vacant_retract_ratio) * new_vacant + current_connected_count;
        RelearnException::check(new_count >= current_connected_count, "SynapticElements::update_number_elements: new count is smaller than connected count");

        grown_elements[local_neuron_id] = new_count;
        deltas_since_last_update[local_neuron_id] = 0.0;
        return 0;
    }

    /**
     * More bound elements should be deleted than are available.
     * Now, neither vacant (see if branch above) nor bound elements are left.
     */
    if (current_count + current_delta < 0.0) {
        connected_elements[local_neuron_id] = 0;
        grown_elements[local_neuron_id] = 0.0;
        deltas_since_last_update[local_neuron_id] = 0.0;

        return current_connected_count_integral;
    }

    const auto new_count = current_count + current_delta;
    const auto new_connected_count = floor(new_count);
    const auto num_vacant = new_count - new_connected_count;

    const auto retracted_new_count = (1 - vacant_retract_ratio) * num_vacant + new_connected_count;

    RelearnException::check(num_vacant >= 0, "SynapticElements::update_number_elements: num vacant is neg");

    connected_elements[local_neuron_id] = static_cast<unsigned int>(new_connected_count);
    grown_elements[local_neuron_id] = retracted_new_count;
    deltas_since_last_update[local_neuron_id] = 0.0;

    const auto deleted_counts = current_connected_count - new_connected_count;

    RelearnException::check(deleted_counts >= 0.0, "SynapticElements::update_number_elements: deleted was negative");
    const auto num_delete_connected = static_cast<unsigned int>(deleted_counts);

    return num_delete_connected;
}
