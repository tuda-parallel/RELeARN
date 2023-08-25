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

#include "adapter/neurons/NeuronTypesAdapter.h"

#include "neurons/models/SynapticElements.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "util/NeuronID.h"

#include <memory>
#include <random>
#include <tuple>
#include <vector>

class SynapticElementsAdapter {
public:
    constexpr static double min_grown_elements = 0.0;
    constexpr static double max_grown_elements = 10.0;

    static double get_random_synaptic_element_count(std::mt19937& mt) {
        return RandomAdapter::get_random_double(min_grown_elements, std::nextafter(max_grown_elements, max_grown_elements * 2.0), mt);
    }

    static unsigned int get_random_synaptic_element_connected_count(unsigned int maximum, std::mt19937& mt) {
        return RandomAdapter::get_random_integer<unsigned int>(0, maximum, mt);
    }

    static std::tuple<SynapticElements, std::vector<double>, std::vector<unsigned int>, std::vector<SignalType>>
    create_random_synaptic_elements(size_t number_elements, ElementType element_type, double min_calcium_to_grow, std::mt19937& mt,
        double growth_factor = SynapticElements::default_nu, double retract_ratio = SynapticElements::default_vacant_retract_ratio,
        double lb_free_elements = SynapticElements::default_vacant_elements_initially_lower_bound, double ub_free_elements = SynapticElements::default_vacant_elements_initially_upper_bound) {

        SynapticElements se(element_type, min_calcium_to_grow, growth_factor, retract_ratio, lb_free_elements, ub_free_elements);
        se.init(number_elements);

        std::vector<double> grown_elements(number_elements);
        std::vector<unsigned int> connected_elements(number_elements);
        std::vector<SignalType> signal_types(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_synaptic_element_count(mt);
            const auto number_connected_elements = get_random_synaptic_element_connected_count(static_cast<unsigned int>(number_grown_elements), mt);
            const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

            se.update_grown_elements(neuron_id, number_grown_elements);
            se.update_connected_elements(neuron_id, number_connected_elements);
            se.set_signal_type(neuron_id, signal_type);

            const auto i = neuron_id.get_neuron_id();

            grown_elements[i] = number_grown_elements;
            connected_elements[i] = number_connected_elements;
            signal_types[i] = signal_type;
        }

        return std::make_tuple<SynapticElements, std::vector<double>, std::vector<unsigned int>, std::vector<SignalType>>(std::move(se), std::move(grown_elements), std::move(connected_elements), std::move(signal_types));
    }

    static std::shared_ptr<SynapticElements> create_axons(size_t number_elements, double minimal_grown, double maximal_grown, std::mt19937& mt) {
        SynapticElements axons(ElementType::Axon, CalciumCalculator::default_C_target);
        axons.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = RandomAdapter::get_random_double(minimal_grown, maximal_grown, mt);
            const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

            axons.update_grown_elements(neuron_id, number_grown_elements);
            axons.update_connected_elements(neuron_id, 0);
            axons.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(axons));
    }

    static std::shared_ptr<SynapticElements> create_dendrites(size_t number_elements, SignalType signal_type, double minimal_grown, double maximal_grown, std::mt19937& mt) {
        SynapticElements dendrites(ElementType::Axon, CalciumCalculator::default_C_target);
        dendrites.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = RandomAdapter::get_random_double(minimal_grown, maximal_grown, mt);

            dendrites.update_grown_elements(neuron_id, number_grown_elements);
            dendrites.update_connected_elements(neuron_id, 0);
            dendrites.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(dendrites));
    }

    static std::shared_ptr<SynapticElements> create_axons(size_t number_elements, std::mt19937& mt) {
        SynapticElements axons(ElementType::Axon, CalciumCalculator::default_C_target);
        axons.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_synaptic_element_count(mt);
            const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

            axons.update_grown_elements(neuron_id, number_grown_elements);
            axons.update_connected_elements(neuron_id, 0);
            axons.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(axons));
    }

    static std::shared_ptr<SynapticElements> create_dendrites(size_t number_elements, SignalType signal_type, std::mt19937& mt) {
        SynapticElements dendrites(ElementType::Axon, CalciumCalculator::default_C_target);
        dendrites.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_synaptic_element_count(mt);

            dendrites.update_grown_elements(neuron_id, number_grown_elements);
            dendrites.update_connected_elements(neuron_id, 0);
            dendrites.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(dendrites));
    }
};
