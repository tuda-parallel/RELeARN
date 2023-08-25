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
#include "util/NeuronID.h"

using models::FitzHughNagumoModel;

FitzHughNagumoModel::FitzHughNagumoModel(
    const unsigned int h,
    std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
    std::unique_ptr<Stimulus>&& stimulus_calculator,
    const double a,
    const double b,
    const double phi)
    : NeuronModel{ h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator) }
    , a{ a }
    , b{ b }
    , phi{ phi } {
}

std::unique_ptr<NeuronModel> FitzHughNagumoModel::clone() const {
    return std::make_unique<FitzHughNagumoModel>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(),
        get_stimulus_calculator()->clone(), a, b, phi);
}

std::vector<ModelParameter> FitzHughNagumoModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
    res.emplace_back(Parameter<double>{ "a", a, FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a });
    res.emplace_back(Parameter<double>{ "b", b, FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b });
    res.emplace_back(Parameter<double>{ "phi", phi, FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi });
    return res;
}

std::string FitzHughNagumoModel::name() {
    return "FitzHughNagumoModel";
}

void FitzHughNagumoModel::init(number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    w.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void FitzHughNagumoModel::create_neurons(number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void FitzHughNagumoModel::update_activity_benchmark(const NeuronID neuron_id) {
    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto stimulus = get_stimulus(neuron_id);
    const auto input = synaptic_input + background + stimulus;

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto local_neuron_id = neuron_id.get_neuron_id();

    auto x_val = get_x(neuron_id);
    auto w_val = w[local_neuron_id];

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        const auto x_increase = x_val - x_val * x_val * x_val * (1.0 / 3.0) - w_val + input;
        const auto w_increase = phi * (x_val + a - b * w_val);

        x_val += x_increase * scale;
        w_val += w_increase * scale;
    }

    const auto spiked = w_val > x_val - x_val * x_val * x_val * (1.0 / 3.0) && x_val > 1.0;

    if (spiked) {
        set_fired(neuron_id, FiredStatus::Fired);
    } else {
        set_fired(neuron_id, FiredStatus::Inactive);
    }

    set_x(neuron_id, x_val);
    w[local_neuron_id] = w_val;
}

void FitzHughNagumoModel::update_activity_benchmark() {
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

void FitzHughNagumoModel::update_activity() {
    const auto number_local_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

#pragma omp parallel for shared(disable_flags, number_local_neurons, h, scale) default(none)
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
        auto w_val = w[neuron_id];

        for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
            const auto x_increase = x_val - x_val * x_val * x_val * (1.0 / 3.0) - w_val + input;
            const auto w_increase = phi * (x_val + a - b * w_val);

            x_val += x_increase * scale;
            w_val += w_increase * scale;
        }

        const auto spiked = w_val > x_val - x_val * x_val * x_val * (1.0 / 3.0) && x_val > 1.0;

        if (spiked) {
            set_fired(converted_id, FiredStatus::Fired);
        } else {
            set_fired(converted_id, FiredStatus::Inactive);
        }

        set_x(converted_id, x_val);
        w[neuron_id] = w_val;
    }
}

void FitzHughNagumoModel::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    for (const auto neuron_id : NeuronID::range(start_id, end_id)) {
        w[neuron_id.get_neuron_id()] = FitzHughNagumoModel::init_w;
        set_x(neuron_id, FitzHughNagumoModel::init_x);
    }
}

double FitzHughNagumoModel::iter_x(const double x, const double w, const double input) noexcept {
    return x - x * x * x / 3 - w + input;
}

double FitzHughNagumoModel::iter_refraction(const double w, const double x) const noexcept {
    return phi * (x + a - b * w);
}

bool FitzHughNagumoModel::spiked(const double x, const double w) noexcept {
    return w > iter_x(x, 0, 0) && x > 1.;
}
