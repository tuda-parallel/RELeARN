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
#include "io/InteractiveNeuronIO.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/LocalAreaTranslator.h"
#include "util/MPIRank.h"
#include "util/NeuronID.h"
#include "util/Timers.h"
#include "util/ranges/Functional.hpp"

#include <filesystem>
#include <memory>
#include <vector>

#include <range/v3/algorithm/fill.hpp>
#include <range/v3/view/transform.hpp>

class Stimulus {
public:
    Stimulus() = default;

    Stimulus(RelearnTypes::stimuli_function_type stimulus_function)
        : stimulus_function(std::move(stimulus_function)) {
    }

    Stimulus(const std::filesystem::path& stimulus_file, const MPIRank mpi_rank, std::shared_ptr<LocalAreaTranslator> local_area_translator) {
        stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(stimulus_file, mpi_rank, std::move(local_area_translator));
    }

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info.operator bool();
        RelearnException::check(is_filled, "Stimulus::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    void init(const RelearnTypes::number_neurons_type number_neurons) {
        RelearnException::check(stimulus.empty(), "Stimulus::init: Was already initialized");
        RelearnException::check(number_neurons > 0, "Stimulus::init: Cannot initialize with 0 neurons");
        stimulus.resize(number_neurons, 0.0);
    }

    void create_neurons(const RelearnTypes::number_neurons_type creation_count) {
        RelearnException::check(stimulus.size() > 0, "Stimulus::create_neurons: Was not initialized");
        RelearnException::check(creation_count > 0, "Stimulus::create_neurons: Cannot create 0 neurons");

        const auto current_size = stimulus.size();
        const auto new_size = current_size + creation_count;

        stimulus.resize(new_size, 0.0);
    }

    void update_stimulus(const RelearnTypes::step_type step) {
        if (!stimulus_function.operator bool()) {
            return;
        }

        Timers::start(TimerRegion::CALC_STIMULUS);
        ranges::fill(stimulus, 0.0);
        const auto& stimuli = stimulus_function(step);

        for (const auto& [neuron_ids, intensity] : stimuli) {
            ranges::fill(
                neuron_ids | ranges::views::transform(lookup(stimulus, &NeuronID::get_neuron_id)),
                intensity);
        }
        Timers::stop_and_add(TimerRegion::CALC_STIMULUS);
    }

    [[nodiscard]] double get_stimulus(const NeuronID neuron_id) const {
        return stimulus[neuron_id.get_neuron_id()];
    }

    [[nodiscard]] std::unique_ptr<Stimulus> clone() const {
        return std::make_unique<Stimulus>(stimulus_function);
    }

private:
    std::vector<double> stimulus{};
    std::shared_ptr<NeuronsExtraInfo> extra_infos{};
    RelearnTypes::stimuli_function_type stimulus_function{};
};
