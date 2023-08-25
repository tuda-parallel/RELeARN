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

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <memory>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>

class NeuronsExtraInfoFactory {
public:
    static std::shared_ptr<NeuronsExtraInfo> construct_extra_info() {
        return std::make_shared<NeuronsExtraInfo>();
    }

    static void enable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
        const auto disable_flags = extra_info->get_disable_flags();
        const auto ids_to_enable =
            NeuronID::range(disable_flags.size()) |
            ranges::views::filter(
                equal_to(UpdateStatus::Disabled),
                lookup(disable_flags, &NeuronID::get_neuron_id)) |
            ranges::to_vector;

        extra_info->set_enabled_neurons(ids_to_enable);
    }

    static void disable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
        const auto disable_flags = extra_info->get_disable_flags();

        const auto ids_to_disable =
            NeuronID::range(disable_flags.size()) |
            ranges::views::filter(
                equal_to(UpdateStatus::Enabled),
                lookup(disable_flags, &NeuronID::get_neuron_id)) |
            ranges::to_vector;

        extra_info->set_disabled_neurons(ids_to_disable);
    }
};
