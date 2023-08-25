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
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <vector>

class GrowthrateCalculator {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    virtual void init(number_neurons_type number_neurons) = 0;

    virtual void create_neurons(number_neurons_type creation_count) = 0;

    [[nodiscard]] virtual double get_growth_rate(NeuronID neuron_id) const = 0;

    static constexpr double default_growth_rate{ 1e-5 }; // In Sebastian's work: 1e-5
    static constexpr double min_growth_rate{ 0.0 };
    static constexpr double max_growth_rate{ 1.0 };
};

class ConstantGrowthrateCalculator : public GrowthrateCalculator {
public:
    ConstantGrowthrateCalculator(const double growth_rate)
        : intended_growth_rate(growth_rate) {
        RelearnException::check(min_growth_rate <= growth_rate,
            "ConstantGrowthrateCalculator::ConstantGrowthrateCalculator: growth_rate is too small: {}", growth_rate);
        RelearnException::check(growth_rate <= max_growth_rate,
            "ConstantGrowthrateCalculator::ConstantGrowthrateCalculator: growth_rate is too large: {}", growth_rate);
    }

    void init([[maybe_unused]] const number_neurons_type number_neurons) override { }

    void create_neurons([[maybe_unused]] const number_neurons_type creat_count) override { }

    [[nodiscard]] double get_growth_rate([[maybe_unused]] const NeuronID neuron_id) const noexcept override {
        return intended_growth_rate;
    }

private:
    double intended_growth_rate{};
};

class AdaptiveGrowthrateCalculator : public GrowthrateCalculator {
public:
    AdaptiveGrowthrateCalculator(const double growth_rate, const double decay)
        : intended_growth_rate(growth_rate)
        , tau(decay) {
        RelearnException::check(min_growth_rate <= growth_rate,
            "AdaptiveGrowthrateCalculator::AdaptiveGrowthrateCalculator: growth_rate is too small: {}", growth_rate);
        RelearnException::check(growth_rate <= max_growth_rate,
            "AdaptiveGrowthrateCalculator::AdaptiveGrowthrateCalculator: growth_rate is too large: {}", growth_rate);
        RelearnException::check(min_decay <= growth_rate,
            "AdaptiveGrowthrateCalculator::AdaptiveGrowthrateCalculator: decay is too small: {}", decay);
        RelearnException::check(growth_rate <= max_decay,
            "AdaptiveGrowthrateCalculator::AdaptiveGrowthrateCalculator: decay is too large: {}", decay);
    }

    void init(const number_neurons_type number_neurons) override {
        RelearnException::check(growth_rates.empty(), "AdaptiveGrowthrateCalculator::init: Was already initialized");
        RelearnException::check(number_neurons > 0, "AdaptiveGrowthrateCalculator::init: Cannot initialize with 0 neurons");

        growth_rates.resize(number_neurons, intended_growth_rate);
    }

    void create_neurons(const number_neurons_type creation_count) override {
        RelearnException::check(!growth_rates.empty(), "AdaptiveGrowthrateCalculator::create_neurons: Was not initialized previously");
        RelearnException::check(creation_count > 0, "AdaptiveGrowthrateCalculator::create_neurons: Cannot create 0 neurons");

        growth_rates.resize(growth_rates.size() + creation_count, intended_growth_rate);
    }

    [[nodiscard]] double get_growth_rate(const NeuronID neuron_id) const override {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < growth_rates.size(), "AdaptiveGrowthrateCalculator::get_growth_rate: NeuronID {} is larger than the number of neurons {}", neuron_id, growth_rates.size());

        return growth_rates[local_neuron_id];
    }

    static constexpr double default_decay{ 1.0e+3 };
    static constexpr double min_decay{ 0.0 };
    static constexpr double max_decay{ 1.0e+7 };

private:
    double intended_growth_rate{};
    double tau{};

    std::vector<double> growth_rates{};
};
