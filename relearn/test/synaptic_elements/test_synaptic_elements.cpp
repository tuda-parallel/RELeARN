/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synaptic_elements.h"

#include "adapter/random/RandomAdapter.h"

#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/SynapticElements.h"
#include "util/Random.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <numeric>
#include <range/v3/range/conversion.hpp>
#include <sstream>

#include <range/v3/functional/arithmetic.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/algorithm/sort.hpp>

TEST_F(SynapticElementsTest, testGaussianGrowthCurve) {
    constexpr auto number_values = 100;

    const auto intersection_1 = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
    const auto intersection_2 = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);

    const auto left_intersection = std::min(intersection_1, intersection_2);
    const auto right_intersection = std::max(intersection_1, intersection_2);

    const auto middle = (left_intersection + right_intersection) / 2;

    const auto growth_factor = RandomAdapter::get_random_double<double>(1e-6, 100.0, mt);

    std::stringstream ss{};
    ss << "Left intersection: " << left_intersection << '\n'
       << "Right intersection: " << right_intersection << '\n'
       << "Maximum: " << growth_factor << '\n';

    ASSERT_NEAR(gaussian_growth_curve(left_intersection, left_intersection, right_intersection, growth_factor), 0.0, eps) << ss.str();
    ASSERT_NEAR(gaussian_growth_curve(right_intersection, left_intersection, right_intersection, growth_factor), 0.0, eps) << ss.str();
    ASSERT_NEAR(gaussian_growth_curve(middle, left_intersection, right_intersection, growth_factor), growth_factor, eps) << ss.str();

    std::vector<double> smaller_negatives(number_values);
    std::vector<double> smaller_positives(number_values);
    std::vector<double> larger_positives(number_values);
    std::vector<double> larger_negatives(number_values);

    for (const auto i : ranges::views::indices(number_values)) {
        smaller_negatives[i] = RandomAdapter::get_random_double<double>(-100000.0, left_intersection, mt);
        smaller_positives[i] = RandomAdapter::get_random_double<double>(left_intersection, middle, mt);
        larger_positives[i] = RandomAdapter::get_random_double<double>(middle, right_intersection, mt);
        larger_negatives[i] = RandomAdapter::get_random_double<double>(right_intersection, 100000.0, mt);
    }

    ranges::sort(smaller_negatives);
    ranges::sort(smaller_positives);
    ranges::sort(larger_positives);
    ranges::sort(larger_negatives);

    auto last_value = -100000.0;
    auto last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (const auto i : ranges::views::indices(number_values)) {
        const auto current_value = smaller_negatives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_LE(last_change, current_change) << ss.str() << ss_loop.str();
        ASSERT_LE(current_change, 0.0) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }

    last_value = left_intersection;
    last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (const auto i : ranges::views::indices(number_values)) {
        const auto current_value = smaller_positives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_LE(last_change, current_change) << ss.str() << ss_loop.str();
        ASSERT_GE(current_change, 0.0) << ss.str() << ss_loop.str();
        ASSERT_GE(growth_factor, current_change) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }

    last_value = middle;
    last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (const auto i : ranges::views::indices(number_values)) {
        const auto current_value = larger_positives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_GE(last_change, current_change) << ss.str() << ss_loop.str();
        ASSERT_GE(current_change, 0.0) << ss.str() << ss_loop.str();
        ASSERT_LE(current_change, growth_factor) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }

    last_value = right_intersection;
    last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (const auto i : ranges::views::indices(number_values)) {
        const auto current_value = larger_negatives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_LE(current_change, last_change) << ss.str() << ss_loop.str();
        ASSERT_LE(current_change, 0.0) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }
}

TEST_F(SynapticElementsTest, testGaussianGrowthCurveSameIntersections) {
    constexpr auto number_values = 100;

    const auto intersection = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
    const auto growth_factor = RandomAdapter::get_random_double<double>(1e-6, 100.0, mt);

    ASSERT_EQ(gaussian_growth_curve(intersection, intersection, intersection, growth_factor), 0.0);

    std::vector<double> left_values(number_values);
    std::vector<double> right_values(number_values);

    for (const auto i : ranges::views::indices(number_values)) {
        left_values[i] = RandomAdapter::get_random_double<double>(-100000.0, intersection, mt);
        right_values[i] = RandomAdapter::get_random_double<double>(intersection, 100000.0, mt);
    }

    for (const auto i : ranges::views::indices(number_values)) {
        const auto current_val_left = left_values[i];
        const auto current_change = gaussian_growth_curve(current_val_left, intersection, intersection, growth_factor);

        ASSERT_EQ(gaussian_growth_curve(current_val_left, intersection, intersection, growth_factor), -growth_factor);
    }

    for (const auto i : ranges::views::indices(number_values)) {
        const auto current_val_right = right_values[i];
        const auto current_change = gaussian_growth_curve(current_val_right, intersection, intersection, growth_factor);

        ASSERT_EQ(gaussian_growth_curve(current_val_right, intersection, intersection, growth_factor), -growth_factor);
    }
}

TEST_F(SynapticElementsTest, testConstructor) {
    const auto& calcium_to_grow = RandomAdapter::get_random_double<double>(SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow, mt);
    const auto& nu = RandomAdapter::get_random_double<double>(SynapticElements::min_nu, SynapticElements::max_nu, mt);
    const auto& retract_ratio = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio, mt);
    const auto& vacant_elements_lb = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially, mt);
    const auto& vacant_elements_ub = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially, mt);

    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << element_type << ' ';
    ss << calcium_to_grow << ' ';
    ss << nu << ' ';
    ss << retract_ratio << ' ';
    ss << vacant_elements_lb << ' ';
    ss << vacant_elements_ub << '\n';

    SynapticElements synaptic_elements(element_type, calcium_to_grow, nu, retract_ratio, vacant_elements_lb, vacant_elements_ub);

    const auto& parameters = synaptic_elements.get_parameter();

    Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
    Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
    Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

    ASSERT_EQ(element_type, synaptic_elements.get_element_type()) << ss.str();

    ASSERT_EQ(param_min_C.value(), calcium_to_grow) << ss.str();
    ASSERT_EQ(param_nu.value(), nu) << ss.str();
    ASSERT_EQ(param_vacant.value(), retract_ratio) << ss.str();
    ASSERT_EQ(param_lower_bound.value(), vacant_elements_lb) << ss.str();
    ASSERT_EQ(param_upper_bound.value(), vacant_elements_ub) << ss.str();
}

TEST_F(SynapticElementsTest, testParameters) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& C = RandomAdapter::get_random_percentage<double>(mt);

    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << C << '\n';

    SynapticElements synaptic_elements(element_type, C);
    synaptic_elements.init(number_neurons);

    const auto& parameters = synaptic_elements.get_parameter();

    ASSERT_EQ(parameters.size(), 5) << ss.str();

    Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
    Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
    Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

    ASSERT_EQ(param_min_C.min(), SynapticElements::min_min_C_level_to_grow) << ss.str();
    ASSERT_EQ(param_min_C.value(), C) << ss.str();
    ASSERT_EQ(param_min_C.max(), SynapticElements::max_min_C_level_to_grow) << ss.str();

    ASSERT_EQ(param_nu.min(), SynapticElements::min_nu) << ss.str();
    ASSERT_EQ(param_nu.value(), SynapticElements::default_nu) << ss.str();
    ASSERT_EQ(param_nu.max(), SynapticElements::max_nu) << ss.str();

    ASSERT_EQ(param_vacant.min(), SynapticElements::min_vacant_retract_ratio) << ss.str();
    ASSERT_EQ(param_vacant.value(), SynapticElements::default_vacant_retract_ratio) << ss.str();
    ASSERT_EQ(param_vacant.max(), SynapticElements::max_vacant_retract_ratio) << ss.str();

    ASSERT_EQ(param_lower_bound.min(), SynapticElements::min_vacant_elements_initially) << ss.str();
    ASSERT_EQ(param_lower_bound.value(), SynapticElements::default_vacant_elements_initially_lower_bound) << ss.str();
    ASSERT_EQ(param_lower_bound.max(), SynapticElements::max_vacant_elements_initially) << ss.str();

    ASSERT_EQ(param_upper_bound.min(), SynapticElements::min_vacant_elements_initially) << ss.str();
    ASSERT_EQ(param_upper_bound.value(), SynapticElements::default_vacant_elements_initially_upper_bound) << ss.str();
    ASSERT_EQ(param_upper_bound.max(), SynapticElements::max_vacant_elements_initially) << ss.str();

    const auto& d1 = RandomAdapter::get_random_double<double>(SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow, mt);
    const auto& d2 = RandomAdapter::get_random_double<double>(SynapticElements::min_nu, SynapticElements::max_nu, mt);
    const auto& d3 = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio, mt);
    const auto& d4 = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially, mt);
    const auto& d5 = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially, mt);

    param_min_C.set_value(d1);
    param_nu.set_value(d2);
    param_vacant.set_value(d3);
    param_lower_bound.set_value(d4);
    param_upper_bound.set_value(d5);

    ASSERT_EQ(param_min_C.value(), d1) << ss.str();
    ASSERT_EQ(param_nu.value(), d2) << ss.str();
    ASSERT_EQ(param_vacant.value(), d3) << ss.str();
    ASSERT_EQ(param_lower_bound.value(), d4) << ss.str();
    ASSERT_EQ(param_upper_bound.value(), d5) << ss.str();
}

TEST_F(SynapticElementsTest, testClone) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& C = RandomAdapter::get_random_percentage<double>(mt);

    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << C << '\n';

    SynapticElements synaptic_elements(element_type, C);
    synaptic_elements.init(number_neurons);

    const auto& parameters = synaptic_elements.get_parameter();

    Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
    Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
    Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

    const auto& d1 = RandomAdapter::get_random_double<double>(SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow, mt);
    const auto& d2 = RandomAdapter::get_random_double<double>(SynapticElements::min_nu, SynapticElements::max_nu, mt);
    const auto& d3 = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio, mt);
    const auto& d4 = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially, mt);
    const auto& d5 = RandomAdapter::get_random_double<double>(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially, mt);

    param_min_C.set_value(d1);
    param_nu.set_value(d2);
    param_vacant.set_value(d3);
    param_lower_bound.set_value(d4);
    param_upper_bound.set_value(d5);

    auto cloned_synaptic_elements_ptr = synaptic_elements.clone();
    auto& cloned_synaptic_elements = *cloned_synaptic_elements_ptr;

    const auto& cloned_parameters = cloned_synaptic_elements.get_parameter();

    Parameter<double> cloned_param_min_C = std::get<Parameter<double>>(cloned_parameters[0]);
    Parameter<double> cloned_param_nu = std::get<Parameter<double>>(cloned_parameters[1]);
    Parameter<double> cloned_param_vacant = std::get<Parameter<double>>(cloned_parameters[2]);
    Parameter<double> cloned_param_lower_bound = std::get<Parameter<double>>(cloned_parameters[3]);
    Parameter<double> cloned_param_upper_bound = std::get<Parameter<double>>(cloned_parameters[4]);

    ASSERT_EQ(param_min_C.value(), d1) << ss.str();
    ASSERT_EQ(param_nu.value(), d2) << ss.str();
    ASSERT_EQ(param_vacant.value(), d3) << ss.str();
    ASSERT_EQ(param_lower_bound.value(), d4) << ss.str();
    ASSERT_EQ(param_upper_bound.value(), d5) << ss.str();

    ASSERT_EQ(cloned_param_min_C.value(), d1) << ss.str();
    ASSERT_EQ(cloned_param_nu.value(), d2) << ss.str();
    ASSERT_EQ(cloned_param_vacant.value(), d3) << ss.str();
    ASSERT_EQ(cloned_param_lower_bound.value(), d4) << ss.str();
    ASSERT_EQ(cloned_param_upper_bound.value(), d5) << ss.str();
}

TEST_F(SynapticElementsTest, testInitialize) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    ASSERT_EQ(synaptic_elements.get_size(), number_neurons);

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto& grown_element = synaptic_elements.get_grown_elements(neuron_id);
        const auto& connected_grown_element = synaptic_elements.get_connected_elements(neuron_id);
        const auto& delta_grown_element = synaptic_elements.get_delta(neuron_id);

        ASSERT_EQ(grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(connected_grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(delta_grown_element, 0.0) << ss.str() << neuron_id;
    }

    for (const auto& iteration : ranges::views::indices(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, number_neurons, mt);

        ASSERT_THROW(auto ret = synaptic_elements.get_grown_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_connected_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_delta(neuron_id), RelearnException) << ss.str() << neuron_id;
    }

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_grown_elements();
    const auto& delta_grown_elements = synaptic_elements.get_grown_elements();
    const auto& signal_types = synaptic_elements.get_signal_types();

    ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

    for (const auto& grown_element : grown_elements) {
        ASSERT_EQ(grown_element, 0.0) << ss.str();
    }

    for (const auto& connected_grown_element : connected_grown_elements) {
        ASSERT_EQ(connected_grown_element, 0.0) << ss.str();
    }

    for (const auto& delta_grown_element : delta_grown_elements) {
        ASSERT_EQ(delta_grown_element, 0.0) << ss.str();
    }
}

TEST_F(SynapticElementsTest, testCreateNeurons) {
    const auto& number_neurons_initially = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& number_neurons_added = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons_initially);
    synaptic_elements.create_neurons(number_neurons_added);

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    ASSERT_EQ(synaptic_elements.get_size(), number_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& grown_element = synaptic_elements.get_grown_elements(neuron_id);
        const auto& connected_grown_element = synaptic_elements.get_connected_elements(neuron_id);
        const auto& delta_grown_element = synaptic_elements.get_delta(neuron_id);

        ASSERT_EQ(grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(connected_grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(delta_grown_element, 0.0) << ss.str() << neuron_id;
    }

    for (const auto& iteration : ranges::views::indices(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, number_neurons, mt);

        ASSERT_THROW(auto ret = synaptic_elements.get_grown_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_connected_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_delta(neuron_id), RelearnException) << ss.str() << neuron_id;
    }

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_grown_elements();
    const auto& delta_grown_elements = synaptic_elements.get_grown_elements();
    const auto& signal_types = synaptic_elements.get_signal_types();

    ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

    for (const auto& grown_element : grown_elements) {
        ASSERT_EQ(grown_element, 0.0) << ss.str();
    }

    for (const auto& connected_grown_element : connected_grown_elements) {
        ASSERT_EQ(connected_grown_element, 0.0) << ss.str();
    }

    for (const auto& delta_grown_element : delta_grown_elements) {
        ASSERT_EQ(delta_grown_element, 0.0) << ss.str();
    }
}

TEST_F(SynapticElementsTest, testInitialElementsConstant) {
    const auto& number_neurons_initially = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& number_neurons_added = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = RandomAdapter::get_random_percentage<double>(mt);
    const auto& nu = RandomAdapter::get_random_percentage<double>(mt);
    const auto& retract_ratio = RandomAdapter::get_random_percentage<double>(mt);

    const auto& bound = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << bound << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, bound, bound);

    synaptic_elements.init(number_neurons_initially);
    synaptic_elements.create_neurons(number_neurons_added);

    const auto& counts = synaptic_elements.get_grown_elements();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(NeuronID{ neuron_id });
        const auto grown_elements_2 = counts[neuron_id];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << neuron_id;
        ASSERT_EQ(grown_elements_1, bound) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testInitialElements) {
    const auto& number_neurons_initially = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& number_neurons_added = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = RandomAdapter::get_random_percentage<double>(mt);
    const auto& nu = RandomAdapter::get_random_percentage<double>(mt);
    const auto& retract_ratio = RandomAdapter::get_random_percentage<double>(mt);

    const auto& bound_1 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);
    const auto& bound_2 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);

    const auto& lower_bound = std::min(bound_1, bound_2);
    const auto& upper_bound = std::max(bound_1, bound_2) + 1.0;

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << lower_bound << ' ' << upper_bound << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, lower_bound, upper_bound);

    synaptic_elements.init(number_neurons_initially);
    synaptic_elements.create_neurons(number_neurons_added);

    const auto& counts = synaptic_elements.get_grown_elements();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(NeuronID{ neuron_id });
        const auto grown_elements_2 = counts[neuron_id];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << neuron_id;

        ASSERT_TRUE(lower_bound <= grown_elements_1) << ss.str() << neuron_id;
        ASSERT_TRUE(grown_elements_1 <= upper_bound) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testInitialElementsException) {
    const auto& number_neurons_initially = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& number_neurons_added = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = RandomAdapter::get_random_percentage<double>(mt);
    const auto& nu = RandomAdapter::get_random_percentage<double>(mt);
    const auto& retract_ratio = RandomAdapter::get_random_percentage<double>(mt);

    const auto& bound_1 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);
    const auto& bound_2 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);

    const auto& lower_bound = std::min(bound_1, bound_2);
    const auto& upper_bound = std::max(bound_1, bound_2) + 1.0;

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << lower_bound << ' ' << upper_bound << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, upper_bound, lower_bound);

    ASSERT_THROW(synaptic_elements.init(number_neurons_initially), RelearnException) << ss.str();
    ASSERT_THROW(synaptic_elements.create_neurons(number_neurons_added), RelearnException) << ss.str();
}

TEST_F(SynapticElementsTest, testInitialElementsMultipleBounds) {
    const auto& number_neurons_initially = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& number_neurons_added = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = RandomAdapter::get_random_percentage<double>(mt);
    const auto& nu = RandomAdapter::get_random_percentage<double>(mt);
    const auto& retract_ratio = RandomAdapter::get_random_percentage<double>(mt);

    const auto& bound_1 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);
    const auto& bound_2 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);

    const auto& lower_bound_1 = std::min(bound_1, bound_2);
    const auto& upper_bound_1 = std::max(bound_1, bound_2) + 1.0;

    const auto& bound_3 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);
    const auto& bound_4 = RandomAdapter::get_random_double<double>(0.0, 10.0, mt);

    const auto& lower_bound_2 = std::min(bound_3, bound_4);
    const auto& upper_bound_2 = std::max(bound_3, bound_4) + 1.0;

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << lower_bound_1 << ' ' << upper_bound_1 << ' ' << lower_bound_2 << ' ' << upper_bound_2 << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, lower_bound_1, upper_bound_1);
    synaptic_elements.init(number_neurons_initially);

    auto parameters = synaptic_elements.get_parameter();

    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

    param_lower_bound.set_value(lower_bound_2);
    param_upper_bound.set_value(upper_bound_2);

    synaptic_elements.create_neurons(number_neurons_added);

    const auto& counts = synaptic_elements.get_grown_elements();

    for (auto neuron_id : NeuronID::range(number_neurons_initially)) {
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(neuron_id);
        const auto grown_elements_2 = counts[neuron_id.get_neuron_id()];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << neuron_id;

        ASSERT_TRUE(lower_bound_1 <= grown_elements_1) << ss.str() << neuron_id;
        ASSERT_TRUE(grown_elements_1 <= upper_bound_1) << ss.str() << neuron_id;
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons_initially, number_neurons)) {
        const auto id = NeuronID{ neuron_id };
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(id);
        const auto grown_elements_2 = counts[neuron_id];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << id;

        ASSERT_TRUE(lower_bound_2 <= grown_elements_1) << ss.str() << id;
        ASSERT_TRUE(grown_elements_1 <= upper_bound_2) << ss.str() << id;
    }
}

TEST_F(SynapticElementsTest, testSignalTypes) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    std::vector<SignalType> signal_types(number_neurons);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

        signal_types[neuron_id] = signal_type;
        golden_signal_types[neuron_id] = signal_type;
    }

    synaptic_elements.set_signal_types(std::move(signal_types));

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements.get_signal_type(neuron_id), golden_signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSingleUpdate) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_connected_elements();
    const auto& delta_grown_elements = synaptic_elements.get_deltas();
    const auto& signal_types = synaptic_elements.get_signal_types();

    ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& a1 = golden_counts[neuron_id.get_neuron_id()];
        const auto& a2 = synaptic_elements.get_grown_elements(neuron_id);
        const auto& a3 = grown_elements[neuron_id.get_neuron_id()];

        const auto& a_is_correct = a1 == a2 && a1 == a3;
        ASSERT_TRUE(a_is_correct) << ss.str() << neuron_id;

        const auto& b1 = golden_connected_counts[neuron_id.get_neuron_id()];
        const auto& b2 = synaptic_elements.get_connected_elements(neuron_id);
        const auto& b3 = connected_grown_elements[neuron_id.get_neuron_id()];

        const auto& b_is_correct = b1 == b2 && b1 == b3;
        ASSERT_TRUE(b_is_correct) << ss.str() << neuron_id;

        const auto& c1 = 0.0;
        const auto& c2 = synaptic_elements.get_delta(neuron_id);
        const auto& c3 = delta_grown_elements[neuron_id.get_neuron_id()];

        const auto& c_is_correct = c1 == c2 && c1 == c3;
        ASSERT_TRUE(c_is_correct) << ss.str() << neuron_id;

        const auto& d1 = golden_signal_types[neuron_id.get_neuron_id()];
        const auto& d2 = synaptic_elements.get_signal_type(neuron_id);
        const auto& d3 = signal_types[neuron_id.get_neuron_id()];

        const auto& d_is_correct = d1 == d2 && d1 == d3;
        ASSERT_TRUE(d_is_correct) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testHistogram) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);

    const auto& histogram = synaptic_elements.get_histogram();

    const auto golden_histogram = ranges::views::zip(
                                      golden_connected_counts,
                                      golden_counts | ranges::views::transform(ranges::convert_to<unsigned int>{}))
        | ranges::to_vector;

    for (const auto& [pair, count] : histogram) {
        const auto golden_count = ranges::count(golden_histogram, pair);

        ASSERT_EQ(count, golden_count) << ss.str() << ' ' << pair.first << ' ' << pair.second;
    }
}

TEST_F(SynapticElementsTest, testUpdateException) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    for (const auto iteration : ranges::views::indices(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, number_neurons, mt);

        ASSERT_THROW(auto ret = synaptic_elements.get_grown_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_connected_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_delta(neuron_id), RelearnException) << ss.str() << neuron_id;
    }

    for (const auto iteration : ranges::views::indices(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, number_neurons, mt);

        const auto& grown_element = SynapticElementsAdapter::get_random_synaptic_element_count(mt);
        const auto& connected_grown_element = SynapticElementsAdapter::get_random_synaptic_element_connected_count(static_cast<unsigned int>(grown_element), mt);
        const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

        ASSERT_THROW(synaptic_elements.update_grown_elements(neuron_id, grown_element), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(synaptic_elements.update_connected_elements(neuron_id, static_cast<int>(connected_grown_element)), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(synaptic_elements.set_signal_type(neuron_id, signal_type), RelearnException) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testMultipleUpdate) {
    uniform_int_distribution<unsigned int> uid_connected(0, 10);

    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    std::vector<double> golden_counts(number_neurons, 0.0);
    std::vector<unsigned int> golden_connected_counts(number_neurons, 0);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto iteration = 0; iteration < 10; iteration++) {
        for (auto neuron_id : NeuronID::range(number_neurons)) {
            const auto& grown_element = SynapticElementsAdapter::get_random_synaptic_element_count(mt);
            const auto& connected_grown_element = SynapticElementsAdapter::get_random_synaptic_element_connected_count(static_cast<unsigned int>(grown_element), mt);
            const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

            golden_counts[neuron_id.get_neuron_id()] += grown_element;
            golden_connected_counts[neuron_id.get_neuron_id()] += connected_grown_element;
            golden_signal_types[neuron_id.get_neuron_id()] = signal_type;

            synaptic_elements.update_grown_elements(neuron_id, grown_element);
            synaptic_elements.update_connected_elements(neuron_id, connected_grown_element);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }
    }

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_connected_elements();
    const auto& delta_grown_elements = synaptic_elements.get_deltas();
    const auto& signal_types = synaptic_elements.get_signal_types();

    ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(golden_counts[neuron_id.get_neuron_id()], synaptic_elements.get_grown_elements(neuron_id)) << ss.str() << neuron_id;
        ASSERT_EQ(golden_counts[neuron_id.get_neuron_id()], grown_elements[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        ASSERT_EQ(golden_connected_counts[neuron_id.get_neuron_id()], synaptic_elements.get_connected_elements(neuron_id)) << ss.str() << neuron_id;
        ASSERT_EQ(golden_connected_counts[neuron_id.get_neuron_id()], connected_grown_elements[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        ASSERT_EQ(golden_signal_types[neuron_id.get_neuron_id()], synaptic_elements.get_signal_type(neuron_id)) << ss.str() << neuron_id;
        ASSERT_EQ(golden_signal_types[neuron_id.get_neuron_id()], signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testFreeElements) {
    uniform_int_distribution<unsigned int> uid_connected(0, 10);

    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    std::vector<double> golden_counts(number_neurons, 0.0);
    std::vector<unsigned int> golden_connected_counts(number_neurons, 0);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto iteration = 0; iteration < 10; iteration++) {
        for (auto neuron_id : NeuronID::range(number_neurons)) {
            const auto& grown_element = SynapticElementsAdapter::get_random_synaptic_element_count(mt);
            const auto& connected_grown_element = SynapticElementsAdapter::get_random_synaptic_element_connected_count(static_cast<unsigned int>(grown_element), mt);
            const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

            golden_counts[neuron_id.get_neuron_id()] += grown_element;
            golden_connected_counts[neuron_id.get_neuron_id()] += connected_grown_element;
            golden_signal_types[neuron_id.get_neuron_id()] = signal_type;

            synaptic_elements.update_grown_elements(neuron_id, grown_element);
            synaptic_elements.update_connected_elements(neuron_id, connected_grown_element);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto nid = neuron_id.get_neuron_id();
        const auto expected_number_free_elements = golden_counts[nid] - golden_connected_counts[nid];
        const auto expected_number_free_elements_cast = static_cast<unsigned int>(expected_number_free_elements);

        ASSERT_EQ(expected_number_free_elements_cast, synaptic_elements.get_free_elements(neuron_id)) << ss.str() << neuron_id;

        if (golden_signal_types[nid] == SignalType::Excitatory) {
            ASSERT_EQ(expected_number_free_elements_cast, synaptic_elements.get_free_elements(neuron_id, SignalType::Excitatory)) << ss.str() << neuron_id;
            ASSERT_EQ(0, synaptic_elements.get_free_elements(neuron_id, SignalType::Inhibitory)) << ss.str() << neuron_id;
        } else {
            ASSERT_EQ(expected_number_free_elements_cast, synaptic_elements.get_free_elements(neuron_id, SignalType::Inhibitory)) << ss.str() << neuron_id;
            ASSERT_EQ(0, synaptic_elements.get_free_elements(neuron_id, SignalType::Excitatory)) << ss.str() << neuron_id;
        }
    }
}

TEST_F(SynapticElementsTest, testDisable) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);

    std::vector<unsigned int> changes(number_neurons, 0);
    std::vector<NeuronID> disabled_neurons{};
    std::vector<bool> disabled(number_neurons, false);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto disable = RandomAdapter::get_random_bool(mt);
        if (disable) {
            disabled_neurons.emplace_back(neuron_id);
            disabled[neuron_id] = true;
        }
    }

    shuffle(disabled_neurons, mt);

    synaptic_elements.update_after_deletion(changes, disabled_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto is_disabled = disabled[neuron_id.get_neuron_id()];

        ASSERT_EQ(synaptic_elements.get_signal_type(neuron_id), golden_signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        if (is_disabled) {
            ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), 0) << ss.str() << neuron_id << " disabled";
            ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), 0.0) << ss.str() << neuron_id << " disabled";
            ASSERT_EQ(synaptic_elements.get_delta(neuron_id), 0.0) << ss.str() << neuron_id << " disabled";
        } else {
            ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), golden_connected_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id << " enabled";
            ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id << " enabled";
        }
    }
}

TEST_F(SynapticElementsTest, testDisableException) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);

    for (auto iteration = 0; iteration < 10; iteration++) {
        std::vector<unsigned int> changes(number_neurons, 0);
        std::vector<NeuronID> disabled_neurons{};

        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            const auto disable = RandomAdapter::get_random_bool(mt);
            if (disable) {
                disabled_neurons.emplace_back(neuron_id);
            }
        }

        const auto faulty_id = NeuronIdAdapter::get_random_number_neurons(mt) + number_neurons;
        disabled_neurons.emplace_back(faulty_id);

        shuffle(disabled_neurons, mt);

        ASSERT_THROW(synaptic_elements.update_after_deletion(changes, disabled_neurons), RelearnException) << ss.str() << ' ' << faulty_id;
    }
}

TEST_F(SynapticElementsTest, testDelete) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);

    std::vector<unsigned int> changes(number_neurons, 0);
    std::vector<NeuronID> disabled_neurons{};

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto change = RandomAdapter::get_random_integer<unsigned int>(0, golden_connected_counts[neuron_id], mt);
        changes[neuron_id] = change;
    }

    synaptic_elements.update_after_deletion(changes, disabled_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto new_connected_count = golden_connected_counts[neuron_id.get_neuron_id()] - changes[neuron_id.get_neuron_id()];

        ASSERT_EQ(synaptic_elements.get_signal_type(neuron_id), golden_signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), new_connected_count) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testDeleteException) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);

    for (auto iteration = 0; iteration < 10; iteration++) {
        auto wrong_number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
        if (wrong_number_neurons == number_neurons) {
            wrong_number_neurons++;
        }

        std::vector<NeuronID> disabled_neurons{};
        std::vector<unsigned int> wrong_changes(wrong_number_neurons, 0);

        ASSERT_THROW(synaptic_elements.update_after_deletion(wrong_changes, disabled_neurons), RelearnException) << ss.str() << ' ' << wrong_number_neurons;

        std::vector<unsigned int> changes(number_neurons, 0);

        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            const auto change = RandomAdapter::get_random_integer<unsigned int>(0, golden_connected_counts[neuron_id], mt) + 1;
            changes[neuron_id] = change;
        }

        const auto faulty_id = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt);
        changes[faulty_id.get_neuron_id()] += golden_connected_counts[faulty_id.get_neuron_id()];

        ASSERT_THROW(synaptic_elements.update_after_deletion(changes, disabled_neurons), RelearnException) << ss.str() << ' ' << faulty_id;
    }
}

TEST_F(SynapticElementsTest, testUpdateNumberElements) {
    const auto minimum_calcium_to_grow = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
    const auto growth_factor = RandomAdapter::get_random_double<double>(1e-6, 100.0, mt);

    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << minimum_calcium_to_grow << ' ' << growth_factor << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, minimum_calcium_to_grow, mt, growth_factor);

    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);
    synaptic_elements.set_extra_infos(extra_info);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        calcium[neuron_id] = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
        target_calcium[neuron_id] = RandomAdapter::get_random_double<double>(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0, mt);
        if (RandomAdapter::get_random_bool(mt)) {
            disable_flags[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector<NeuronID>{ NeuronID{ neuron_id } });
        }
    }

    synaptic_elements.update_number_elements_delta(calcium, target_calcium);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), golden_connected_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }

    const auto& actual_deltas = synaptic_elements.get_deltas();
    for (auto id : NeuronID::range(number_neurons)) {
        const auto actual_delta = synaptic_elements.get_delta(id);
        const auto computed_delta = gaussian_growth_curve(calcium[id.get_neuron_id()], minimum_calcium_to_grow, target_calcium[id.get_neuron_id()], growth_factor);

        if (disable_flags[id.get_neuron_id()] == UpdateStatus::Disabled) {
            ASSERT_NEAR(actual_delta, 0.0, eps) << ss.str() << id;
            ASSERT_NEAR(actual_deltas[id.get_neuron_id()], 0.0, eps) << ss.str() << id;
        } else {
            ASSERT_NEAR(actual_delta, computed_delta, eps) << ss.str() << id;
            ASSERT_NEAR(actual_deltas[id.get_neuron_id()], computed_delta, eps) << ss.str() << id;
        }
    }
}

TEST_F(SynapticElementsTest, testMultipleUpdateNumberElements) {
    const auto minimum_calcium_to_grow = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
    const auto growth_factor = RandomAdapter::get_random_double<double>(1e-6, 100.0, mt);

    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, minimum_calcium_to_grow, mt, growth_factor);

    std::vector<double> golden_delta_counts(number_neurons, 0.0);

    for (auto i = 0; i < 10; i++) {
        std::vector<double> calcium(number_neurons, 0.0);
        std::vector<double> target_calcium(number_neurons, 0.0);

        auto extra_info = std::make_shared<NeuronsExtraInfo>();
        extra_info->init(number_neurons);
        synaptic_elements.set_extra_infos(extra_info);

        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            calcium[neuron_id] = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
            target_calcium[neuron_id] = RandomAdapter::get_random_double<double>(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0, mt);
            if (RandomAdapter::get_random_bool(mt)) {
                const auto current_expected_delta = gaussian_growth_curve(calcium[neuron_id], minimum_calcium_to_grow, target_calcium[neuron_id], growth_factor);
                golden_delta_counts[neuron_id] += current_expected_delta;
            } else {
                extra_info->set_disabled_neurons(std::vector<NeuronID>{ NeuronID{ neuron_id } });
            }
        }

        synaptic_elements.update_number_elements_delta(calcium, target_calcium);
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), golden_connected_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }

    const auto& actual_deltas = synaptic_elements.get_deltas();
    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto actual_delta = synaptic_elements.get_delta(neuron_id);
        const auto expected_delta = golden_delta_counts[neuron_id.get_neuron_id()];

        ASSERT_NEAR(actual_delta, expected_delta, eps) << ss.str() << neuron_id;
        ASSERT_NEAR(actual_deltas[neuron_id.get_neuron_id()], expected_delta, eps) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testUpdateNumberElementsException) {
    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto random_elements = SynapticElementsAdapter::create_random_synaptic_elements(
        number_neurons, element_type, 0.0, mt);

    auto random_synaptic_elements
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, 0.0, mt);
    auto& synaptic_elements = std::get<0>(random_elements);
    auto& grown_elements = std::get<1>(random_elements);
    auto& connected_elements = std::get<2>(random_elements);
    auto& signal_types = std::get<3>(random_elements);

    const auto number_too_small_calcium = RandomAdapter::get_random_integer<size_t>(1, number_neurons - 1, mt);
    const auto number_too_large_calcium = RandomAdapter::get_random_integer<size_t>(number_neurons + 1, number_neurons * 2 + 1, mt);

    const auto number_too_small_target_calcium = RandomAdapter::get_random_integer<size_t>(1, number_neurons - 1, mt);
    const auto number_too_large_target_calcium = RandomAdapter::get_random_integer<size_t>(number_neurons + 1, number_neurons * 2 + 1, mt);

    const auto number_too_small_disable_flags = RandomAdapter::get_random_integer<size_t>(1, number_neurons - 1, mt);
    const auto number_too_large_disable_flags = RandomAdapter::get_random_integer<size_t>(number_neurons + 1, number_neurons * 2 + 1, mt);

    std::vector<double> calcium_too_small(number_too_small_calcium, 0.0);
    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> calcium_too_large(number_too_large_calcium, 0.0);

    std::vector<double> target_calcium_too_small(number_too_small_target_calcium, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<double> target_calcium_too_large(number_too_large_target_calcium, 0.0);

    auto extra_info_too_small = std::make_shared<NeuronsExtraInfo>();
    extra_info_too_small->init(number_too_small_disable_flags);

    auto extra_info_correct = std::make_shared<NeuronsExtraInfo>();
    extra_info_correct->init(number_neurons);

    auto extra_info_too_large = std::make_shared<NeuronsExtraInfo>();
    extra_info_too_large->init(number_too_large_disable_flags);

    auto lambda = [&ss, &synaptic_elements](auto calcium, auto target_calcium, auto extra_info) {
        synaptic_elements.set_extra_infos(extra_info);
        ASSERT_THROW(synaptic_elements.update_number_elements_delta(calcium, target_calcium), RelearnException) << ss.str()
                                                                                                                << calcium.size() << ' '
                                                                                                                << target_calcium.size() << ' '
                                                                                                                << extra_info->get_size();
    };

    lambda(calcium_too_small, target_calcium_too_small, extra_info_too_small);
    lambda(calcium_too_small, target_calcium, extra_info_too_small);
    lambda(calcium_too_small, target_calcium_too_large, extra_info_too_small);

    lambda(calcium_too_small, target_calcium_too_small, extra_info_correct);
    lambda(calcium_too_small, target_calcium, extra_info_correct);
    lambda(calcium_too_small, target_calcium_too_large, extra_info_correct);

    lambda(calcium_too_small, target_calcium_too_small, extra_info_too_large);
    lambda(calcium_too_small, target_calcium, extra_info_too_large);
    lambda(calcium_too_small, target_calcium_too_large, extra_info_too_large);

    lambda(calcium, target_calcium_too_small, extra_info_too_small);
    lambda(calcium, target_calcium, extra_info_too_small);
    lambda(calcium, target_calcium_too_large, extra_info_too_small);

    lambda(calcium, target_calcium_too_small, extra_info_correct);
    lambda(calcium, target_calcium_too_large, extra_info_correct);

    lambda(calcium, target_calcium_too_small, extra_info_too_large);
    lambda(calcium, target_calcium, extra_info_too_large);
    lambda(calcium, target_calcium_too_large, extra_info_too_large);

    lambda(calcium_too_large, target_calcium_too_small, extra_info_too_small);
    lambda(calcium_too_large, target_calcium, extra_info_too_small);
    lambda(calcium_too_large, target_calcium_too_large, extra_info_too_small);

    lambda(calcium_too_large, target_calcium_too_small, extra_info_correct);
    lambda(calcium_too_large, target_calcium, extra_info_correct);
    lambda(calcium_too_large, target_calcium_too_large, extra_info_correct);

    lambda(calcium_too_large, target_calcium_too_small, extra_info_too_large);
    lambda(calcium_too_large, target_calcium, extra_info_too_large);
    lambda(calcium_too_large, target_calcium_too_large, extra_info_too_large);
}

TEST_F(SynapticElementsTest, testCommitUpdates) {
    const auto minimum_calcium_to_grow = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
    const auto growth_factor = RandomAdapter::get_random_double<double>(1e-6, 100.0, mt);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);

    const auto retract_ratio = RandomAdapter::get_random_double<double>(0.0, 1.0, mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << minimum_calcium_to_grow << ' ' << growth_factor << ' ' << retract_ratio << '\n';

    SynapticElements synaptic_elements(element_type, minimum_calcium_to_grow, growth_factor, retract_ratio);
    synaptic_elements.init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    auto extra_info_all_enabled = std::make_shared<NeuronsExtraInfo>();
    extra_info_all_enabled->init(number_neurons);

    std::vector<double> golden_counts(number_neurons);
    std::vector<unsigned int> golden_connected_counts(number_neurons);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& grown_element = RandomAdapter::get_random_double<double>(0, 10.0, mt);
        const auto& connected_grown_element = RandomAdapter::get_random_integer<unsigned int>(0, static_cast<unsigned int>(grown_element), mt);
        const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

        golden_counts[neuron_id.get_neuron_id()] = grown_element;
        golden_connected_counts[neuron_id.get_neuron_id()] = static_cast<unsigned int>(connected_grown_element);
        golden_signal_types[neuron_id.get_neuron_id()] = signal_type;

        synaptic_elements.update_grown_elements(neuron_id, grown_element);
        synaptic_elements.update_connected_elements(neuron_id, static_cast<int>(connected_grown_element));
        synaptic_elements.set_signal_type(neuron_id, signal_type);
    }

    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<UpdateStatus> enable_flags(number_neurons, UpdateStatus::Enabled);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        calcium[neuron_id] = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
        target_calcium[neuron_id] = RandomAdapter::get_random_double<double>(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0, mt);
        if (RandomAdapter::get_random_bool(mt)) {
            extra_info->set_disabled_neurons(std::vector<NeuronID>{ NeuronID{ neuron_id } });
            disable_flags[neuron_id] = UpdateStatus::Disabled;
        }
    }

    synaptic_elements.set_extra_infos(extra_info_all_enabled);
    synaptic_elements.update_number_elements_delta(calcium, target_calcium);
    synaptic_elements.set_extra_infos(extra_info);
    const auto& [number_deleted_elements, deleted_element_counts] = synaptic_elements.commit_updates();

    const auto& deltas = synaptic_elements.get_deltas();

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto computed_delta = gaussian_growth_curve(calcium[neuron_id.get_neuron_id()], minimum_calcium_to_grow, target_calcium[neuron_id.get_neuron_id()], growth_factor);
        const auto delta = synaptic_elements.get_delta(neuron_id);
        if (disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled) {
            ASSERT_NEAR(delta, computed_delta, eps) << ss.str() << neuron_id;
            ASSERT_NEAR(deltas[neuron_id.get_neuron_id()], computed_delta, eps) << ss.str() << neuron_id;
        } else {
            ASSERT_EQ(delta, 0.0) << ss.str() << neuron_id;
            ASSERT_EQ(deltas[neuron_id.get_neuron_id()], 0.0) << ss.str() << neuron_id;
        }
    }

    auto summed_number_deletions = 0;
    for (auto deleted_counts : deleted_element_counts) {
        summed_number_deletions += deleted_counts;
    }

    ASSERT_EQ(summed_number_deletions, number_deleted_elements) << ss.str() << summed_number_deletions << ' ' << number_deleted_elements;

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        if (disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled) {
            continue;
        }

        const auto previous_count = golden_counts[neuron_id.get_neuron_id()];
        const auto previous_connected = golden_connected_counts[neuron_id.get_neuron_id()];
        const auto previous_vacant = previous_count - previous_connected;

        const auto current_count = synaptic_elements.get_grown_elements(neuron_id);
        const auto current_connected = synaptic_elements.get_connected_elements(neuron_id);
        const auto current_delta = synaptic_elements.get_delta(neuron_id);

        const auto computed_delta = gaussian_growth_curve(calcium[neuron_id.get_neuron_id()], minimum_calcium_to_grow, target_calcium[neuron_id.get_neuron_id()], growth_factor);
        const auto new_vacant = previous_vacant + computed_delta;

        ASSERT_EQ(current_delta, 0.0) << ss.str() << neuron_id;

        if (new_vacant >= 0.0) {
            const auto retracted_count = (1 - retract_ratio) * new_vacant;
            const auto expected_count = retracted_count + previous_connected;

            ASSERT_NEAR(expected_count, current_count, eps) << ss.str() << neuron_id;
            ASSERT_EQ(previous_connected, current_connected) << ss.str() << neuron_id;

            continue;
        }

        const auto expected_deletions = static_cast<unsigned int>(std::ceil(std::abs(new_vacant)));

        if (expected_deletions > previous_connected) {
            ASSERT_EQ(current_count, 0.0) << ss.str() << neuron_id;
            ASSERT_EQ(current_connected, 0) << ss.str() << neuron_id;

            continue;
        }

        if (expected_deletions == previous_connected) {
            const auto expected_count = (1 - retract_ratio) * (previous_count + computed_delta);

            ASSERT_NEAR(current_count, expected_count, eps) << ss.str() << neuron_id;
            ASSERT_EQ(current_connected, 0) << ss.str() << neuron_id;

            continue;
        }

        const auto expected_connected = previous_connected - expected_deletions;

        ASSERT_EQ(expected_connected, current_connected) << ss.str() << neuron_id;
        ASSERT_EQ(expected_deletions, deleted_element_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        const auto expected_count = (1 - retract_ratio) * (previous_count + computed_delta - expected_connected) + expected_connected;
        ASSERT_NEAR(current_count, expected_count, eps) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testCommitUpdatesException) {
    const auto minimum_calcium_to_grow = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
    const auto growth_factor = RandomAdapter::get_random_double<double>(1e-6, 100.0, mt);

    const auto& number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& element_type = NeuronTypesAdapter::get_random_element_type(mt);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << minimum_calcium_to_grow << ' ' << growth_factor << '\n';

    auto [synaptic_elements, grown_elements, connected_elements, signal_types]
        = SynapticElementsAdapter::create_random_synaptic_elements(number_neurons, element_type, minimum_calcium_to_grow, mt, growth_factor);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);
    synaptic_elements.set_extra_infos(extra_info);

    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        calcium[neuron_id] = RandomAdapter::get_random_double<double>(-100.0, 100.0, mt);
        target_calcium[neuron_id] = RandomAdapter::get_random_double<double>(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0, mt);
        if (RandomAdapter::get_random_bool(mt)) {
            disable_flags[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector<NeuronID>{ NeuronID{ neuron_id } });
        }
    }

    synaptic_elements.update_number_elements_delta(calcium, target_calcium);

    const auto number_too_small_disable_flags = NeuronIdAdapter::get_random_neuron_id(number_neurons - 1, mt).get_neuron_id() + 1;
    const auto number_too_large_disable_flags = NeuronIdAdapter::get_random_neuron_id(number_neurons, mt).get_neuron_id() + number_neurons + 1;

    auto extra_info_too_small = std::make_shared<NeuronsExtraInfo>();
    extra_info_too_small->init(number_too_small_disable_flags);

    auto extra_info_too_large = std::make_shared<NeuronsExtraInfo>();
    extra_info_too_large->init(number_too_large_disable_flags);

    synaptic_elements.set_extra_infos(extra_info_too_small);
    ASSERT_THROW(auto ret = synaptic_elements.commit_updates(), RelearnException) << ss.str() << number_too_small_disable_flags;

    synaptic_elements.set_extra_infos(extra_info_too_large);
    ASSERT_THROW(auto ret = synaptic_elements.commit_updates(), RelearnException) << ss.str() << number_too_large_disable_flags;
}
