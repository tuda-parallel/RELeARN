/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModels.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/Random.h"

using models::PoissonModel;

PoissonModel::PoissonModel(
    const unsigned int h,
    std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
    std::unique_ptr<Stimulus>&& stimulus_calculator,
    const double x_0,
    const double tau_x,
    const unsigned int refractory_period)
    : NeuronModel{ h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator) }
    , x_0{ x_0 }
    , tau_x{ tau_x }
    , refractory_period{ refractory_period } {
}

[[nodiscard]] std::unique_ptr<NeuronModel> PoissonModel::clone() const {
    return std::make_unique<PoissonModel>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(),
        get_stimulus_calculator()->clone(), x_0, tau_x, refractory_period);
}

[[nodiscard]] std::vector<ModelParameter> PoissonModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
    res.emplace_back(Parameter<double>{ "x_0", x_0, PoissonModel::min_x_0, PoissonModel::max_x_0 });
    res.emplace_back(Parameter<double>{ "tau_x", tau_x, PoissonModel::min_tau_x, PoissonModel::max_tau_x });
    res.emplace_back(Parameter<unsigned int>{ "refractory_time", refractory_period, PoissonModel::min_refractory_time, PoissonModel::max_refractory_time });
    return res;
}

[[nodiscard]] std::string PoissonModel::name() {
    return "PoissonModel";
}

void PoissonModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    refractory_time.resize(number_neurons, 0);
    init_neurons(0, number_neurons);
}

void PoissonModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    refractory_time.resize(old_size + creation_count, 0);
    init_neurons(old_size, creation_count);
}

void PoissonModel::update_activity_benchmark(const NeuronID neuron_id) {
    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto stimulus = get_stimulus(neuron_id);
    const auto input = synaptic_input + background + stimulus;

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto tau_x_inverse = 1.0 / tau_x;

    const auto local_neuron_id = neuron_id.get_neuron_id();

    auto x_val = get_x(neuron_id);

    for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
        x_val += ((x_0 - x_val) * tau_x_inverse + input) * scale;
    }

    if (refractory_time[local_neuron_id] == 0) {
        const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        const auto f = x_val >= threshold;
        if (f) {
            set_fired(neuron_id, FiredStatus::Fired);
            refractory_time[local_neuron_id] = refractory_period;
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
        }
    } else {
        set_fired(neuron_id, FiredStatus::Inactive);
        --refractory_time[local_neuron_id];
    }

    set_x(neuron_id, x_val);
}

void PoissonModel::update_activity_benchmark() {
    const auto number_local_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

#pragma omp parallel for shared(disable_flags, number_local_neurons) default(none)
    for (NeuronID::value_type neuron_id = 0U; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID converted_id{ neuron_id };
        update_activity_benchmark(converted_id);
    }
}

void PoissonModel::update_activity() {
    const auto number_local_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto tau_x_inverse = 1.0 / tau_x;

#pragma omp parallel for shared(disable_flags, number_local_neurons, h, scale, tau_x_inverse) default(none)
    for (NeuronID::value_type neuron_id = 0U; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID converted_id{ neuron_id };

        const auto synaptic_input = get_synaptic_input(converted_id);
        const auto background = get_background_activity(converted_id);
        const auto stimulus = get_stimulus(converted_id);
        const auto input = synaptic_input + background + stimulus;

        auto x_val = get_x(converted_id);

        for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + input) * scale;
        }

        if (refractory_time[neuron_id] == 0) {
            const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
            const auto f = x_val >= threshold;
            if (f) {
                set_fired(converted_id, FiredStatus::Fired);
                refractory_time[neuron_id] = refractory_period;
            } else {
                set_fired(converted_id, FiredStatus::Inactive);
            }
        } else {
            set_fired(converted_id, FiredStatus::Inactive);
            --refractory_time[neuron_id];
        }

        set_x(converted_id, x_val);
    }
}
