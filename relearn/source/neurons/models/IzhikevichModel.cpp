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

using models::IzhikevichModel;

IzhikevichModel::IzhikevichModel(
    const unsigned int h,
    std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
    std::unique_ptr<Stimulus>&& stimulus_calculator,
    const double a,
    const double b,
    const double c,
    const double d,
    const double V_spike,
    const double k1,
    const double k2,
    const double k3)
    : NeuronModel{ h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator) }
    , a{ a }
    , b{ b }
    , c{ c }
    , d{ d }
    , V_spike{ V_spike }
    , k1{ k1 }
    , k2{ k2 }
    , k3{ k3 } {
}

[[nodiscard]] std::unique_ptr<NeuronModel> IzhikevichModel::clone() const {
    return std::make_unique<IzhikevichModel>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(),
        get_stimulus_calculator()->clone(), a, b, c, d, V_spike, k1, k2, k3);
}

[[nodiscard]] std::vector<ModelParameter> IzhikevichModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
    res.emplace_back(Parameter<double>{ "a", a, IzhikevichModel::min_a, IzhikevichModel::max_a });
    res.emplace_back(Parameter<double>{ "b", b, IzhikevichModel::min_b, IzhikevichModel::max_b });
    res.emplace_back(Parameter<double>{ "c", c, IzhikevichModel::min_c, IzhikevichModel::max_c });
    res.emplace_back(Parameter<double>{ "d", d, IzhikevichModel::min_d, IzhikevichModel::max_d });
    res.emplace_back(Parameter<double>{ "V_spike", V_spike, IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike });
    res.emplace_back(Parameter<double>{ "k1", k1, IzhikevichModel::min_k1, IzhikevichModel::max_k1 });
    res.emplace_back(Parameter<double>{ "k2", k2, IzhikevichModel::min_k2, IzhikevichModel::max_k2 });
    res.emplace_back(Parameter<double>{ "k3", k3, IzhikevichModel::min_k3, IzhikevichModel::max_k3 });
    return res;
}

[[nodiscard]] std::string IzhikevichModel::name() {
    return "IzhikevichModel";
}

void IzhikevichModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    u.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void IzhikevichModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    u.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void IzhikevichModel::update_activity_benchmark(const NeuronID neuron_id) {
    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto stimulus = get_stimulus(neuron_id);
    const auto input = synaptic_input + background + stimulus;

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto local_neuron_id = neuron_id.get_neuron_id();

    auto x_val = get_x(neuron_id);
    auto u_val = u[local_neuron_id];

    auto has_spiked = FiredStatus::Inactive;

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        const auto x_increase = k1 * x_val * x_val + k2 * x_val + k3 - u_val + input;
        const auto u_increase = a * (b * x_val - u_val);

        x_val += x_increase * scale;
        u_val += u_increase * scale;

        const auto spiked = x_val >= V_spike;

        if (spiked) {
            x_val = c;
            u_val += d;
            has_spiked = FiredStatus::Fired;
            break;
        }
    }

    set_fired(neuron_id, has_spiked);
    set_x(neuron_id, x_val);
    u[local_neuron_id] = u_val;
}

void IzhikevichModel::update_activity_benchmark() {
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

void IzhikevichModel::update_activity() {
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
        auto u_val = u[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
            const auto x_increase = k1 * x_val * x_val + k2 * x_val + k3 - u_val + input;
            const auto u_increase = a * (b * x_val - u_val);

            x_val += x_increase * scale;
            u_val += u_increase * scale;

            const auto spiked = x_val >= V_spike;

            if (spiked) {
                x_val = c;
                u_val += d;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }

        set_fired(converted_id, has_spiked);
        set_x(converted_id, x_val);
        u[neuron_id] = u_val;
    }
}

void IzhikevichModel::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    for (const auto neuron_id : NeuronID::range(start_id, end_id)) {
        u[neuron_id.get_neuron_id()] = iter_refraction(b * c, c);
        set_x(neuron_id, c);
    }
}

double IzhikevichModel::iter_x(const double x, const double u, const double input) const noexcept {
    return k1 * x * x + k2 * x + k3 - u + input;
}

double IzhikevichModel::iter_refraction(const double u, const double x) const noexcept {
    return a * (b * x - u);
}

bool IzhikevichModel::spiked(const double x) const noexcept {
    return x >= V_spike;
}
