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

#include "adapter/random/RandomAdapter.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/ElementType.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SignalType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "util/shuffle/shuffle.h"

#include <random>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <vector>

class NeuronTypesAdapter {
public:
    static ElementType get_random_element_type(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_bool(mt) ? ElementType::Axon : ElementType::Dendrite;
    }

    static SignalType get_random_signal_type(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
    }

    static DistantNeuronRequest::TargetNeuronType get_random_target_neuron_type(std::mt19937& mt) {
        const auto drawn = RandomAdapter::get_random_bool(mt);

        if (drawn) {
            return DistantNeuronRequest::TargetNeuronType::Leaf;
        }

        return DistantNeuronRequest::TargetNeuronType::VirtualNode;
    }

    static std::vector<FiredStatus> get_fired_status(size_t number_neurons, std::mt19937& mt) {
        const auto number_disabled = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
        return get_fired_status(number_neurons, number_disabled, mt);
    }

    static std::vector<FiredStatus> get_fired_status(size_t number_neurons, size_t number_inactive, std::mt19937& mt) {
        return ranges::views::concat(
                   ranges::views::repeat_n(FiredStatus::Inactive, number_inactive),
                   ranges::views::repeat_n(FiredStatus::Fired, number_neurons - number_inactive))
            | ranges::to_vector | actions::shuffle(mt);
    }

    static std::vector<UpdateStatus> get_update_status(size_t number_neurons, std::mt19937& mt) {
        const auto number_disabled = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
        return get_update_status(number_neurons, number_disabled, mt);
    }

    static std::vector<UpdateStatus> get_update_status(size_t number_neurons, size_t number_disabled, std::mt19937& mt) {
        return ranges::views::concat(
                   ranges::views::repeat_n(UpdateStatus::Disabled, number_disabled),
                   ranges::views::repeat_n(UpdateStatus::Enabled, number_neurons - number_disabled))
            | ranges::to_vector | actions::shuffle(mt);
    }

    static void disable_neurons(size_t number_neurons, std::shared_ptr<NeuronsExtraInfo> extra_infos, std::mt19937& mt) {
        const auto number_disabled = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
        disable_neurons(number_neurons, number_disabled, std::move(extra_infos), mt);
    }

    static void disable_neurons(size_t number_neurons, size_t number_disabled, std::shared_ptr<NeuronsExtraInfo> extra_infos, std::mt19937& mt) {
        const auto neuron_ids = NeuronID::range(number_neurons) | ranges::to_vector | actions::shuffle(mt);
        extra_infos->set_disabled_neurons(std::span<const NeuronID>{ neuron_ids.data(), number_disabled });
    }
};
