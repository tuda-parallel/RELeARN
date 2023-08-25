/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_background_activity.h"

#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/input/BackgroundActivityCalculator.h"
#include "neurons/input/BackgroundActivityCalculators.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>
#include <memory>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

void test_background_equality(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator) {
    const auto number_neurons = background_calculator->get_number_neurons();
    const auto inputs = background_calculator->get_background_activity();

    ASSERT_EQ(inputs.size(), number_neurons);

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto input = background_calculator->get_background_activity(neuron_id);
        ASSERT_EQ(inputs[neuron_id.get_neuron_id()], input);
    }
}

void test_background_exceptions(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator) {
    const auto number_neurons = background_calculator->get_number_neurons();

    for (const auto& neuron_id : NeuronID::range_id(number_neurons) | ranges::views::transform(plus(number_neurons))) {
        ASSERT_THROW(const auto input = background_calculator->get_background_activity(NeuronID{ neuron_id }), RelearnException);
    }
}

void test_init_create(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator, const size_t number_init_neurons, const size_t number_create_neurons, std::mt19937& mt, const bool check_input = true, const bool check_equality = true) {
    ASSERT_EQ(background_calculator->get_number_neurons(), 0);
    if (check_equality) {
        ASSERT_TRUE(background_calculator->get_background_activity().empty());
    }
    test_background_exceptions(background_calculator);

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    background_calculator->set_extra_infos(extra_infos);
    extra_infos->init(number_init_neurons);
    extra_infos->set_positions(SimulationAdapter::get_random_positions(mt, number_init_neurons));

    auto first_clone = background_calculator->clone();
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    if (check_equality) {
        ASSERT_TRUE(first_clone->get_background_activity().empty());
    }

    ASSERT_THROW(background_calculator->init(0), RelearnException);
    background_calculator->init(number_init_neurons);

    ASSERT_EQ(background_calculator->get_number_neurons(), number_init_neurons);
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    if (check_equality) {
        ASSERT_TRUE(first_clone->get_background_activity().empty());
        ASSERT_EQ(background_calculator->get_background_activity().size(), number_init_neurons);
    }

    if (check_equality) {
        test_background_equality(background_calculator);
    }
    test_background_exceptions(background_calculator);

    if (check_input) {
        for (const auto& neuron_id : NeuronID::range(number_init_neurons)) {
            const auto input = background_calculator->get_background_activity(neuron_id);
            ASSERT_EQ(input, 0.0);
        }
    }

    auto second_clone = background_calculator->clone();
    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    if (check_equality) {
        ASSERT_TRUE(second_clone->get_background_activity().empty());
    }
    ASSERT_THROW(background_calculator->create_neurons(0), RelearnException);
    background_calculator->create_neurons(number_create_neurons);
    extra_infos->create_neurons(number_create_neurons);

    ASSERT_EQ(background_calculator->get_number_neurons(), number_init_neurons + number_create_neurons);
    ASSERT_EQ(extra_infos->get_size(), background_calculator->get_number_neurons());

    if (check_equality) {
        ASSERT_EQ(background_calculator->get_background_activity().size(), number_init_neurons + number_create_neurons);

        test_background_equality(background_calculator);
    }
    test_background_exceptions(background_calculator);

    if (check_input) {
        for (const auto& neuron_id : NeuronID::range(number_init_neurons + number_create_neurons)) {
            const auto input = background_calculator->get_background_activity(neuron_id);
            ASSERT_EQ(input, 0.0);
        }
    }

    auto extra_infos2 = std::make_shared<NeuronsExtraInfo>();
    extra_infos2->init(1);
    first_clone->set_extra_infos(extra_infos2);
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    if (check_equality) {
        ASSERT_TRUE(first_clone->get_background_activity().empty());
    }

    auto extra_infos3 = std::make_shared<NeuronsExtraInfo>();
    extra_infos3->init(1);
    second_clone->set_extra_infos(extra_infos3);
    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    if (check_equality) {
        ASSERT_TRUE(second_clone->get_background_activity().empty());
    }
}

TEST_F(BackgroundActivityTest, testNullBackgroundActivityConstruct) {
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NullBackgroundActivityCalculator>();

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create, mt);
}

TEST_F(BackgroundActivityTest, testConstantBackgroundActivityConstruct) {
    const auto constant_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<ConstantBackgroundActivityCalculator>(constant_background);

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create, mt);

    const auto& parameters = background_calculator->get_parameter();
    ASSERT_EQ(parameters.size(), 1);

    ModelParameter mp = parameters[0];
    Parameter<double> param1 = std::get<Parameter<double>>(mp);

    ASSERT_EQ(param1.min(), BackgroundActivityCalculator::min_base_background_activity);
    ASSERT_EQ(param1.max(), BackgroundActivityCalculator::max_base_background_activity);
    ASSERT_EQ(param1.value(), constant_background);
}

TEST_F(BackgroundActivityTest, testNormalBackgroundActivityConstruct) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NormalBackgroundActivityCalculator>(mean_background, stddev_background);

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create, mt);

    const auto& parameters = background_calculator->get_parameter();
    ASSERT_EQ(parameters.size(), 2);

    ModelParameter mp1 = parameters[0];
    Parameter<double> param1 = std::get<Parameter<double>>(mp1);

    ASSERT_EQ(param1.min(), BackgroundActivityCalculator::min_background_activity_mean);
    ASSERT_EQ(param1.max(), BackgroundActivityCalculator::max_background_activity_mean);
    ASSERT_EQ(param1.value(), mean_background);

    ModelParameter mp2 = parameters[1];
    Parameter<double> param2 = std::get<Parameter<double>>(mp2);

    ASSERT_EQ(param2.min(), BackgroundActivityCalculator::min_background_activity_stddev);
    ASSERT_EQ(param2.max(), BackgroundActivityCalculator::max_background_activity_stddev);
    ASSERT_EQ(param2.value(), stddev_background);
}

TEST_F(BackgroundActivityTest, testFastNormalBackgroundActivityConstruct) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<FastNormalBackgroundActivityCalculator>(mean_background, stddev_background, 5);

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create, mt, false, false);

    const auto& parameters = background_calculator->get_parameter();
    ASSERT_EQ(parameters.size(), 2);

    ModelParameter mp1 = parameters[0];
    Parameter<double> param1 = std::get<Parameter<double>>(mp1);

    ASSERT_EQ(param1.min(), BackgroundActivityCalculator::min_background_activity_mean);
    ASSERT_EQ(param1.max(), BackgroundActivityCalculator::max_background_activity_mean);
    ASSERT_EQ(param1.value(), mean_background);

    ModelParameter mp2 = parameters[1];
    Parameter<double> param2 = std::get<Parameter<double>>(mp2);

    ASSERT_EQ(param2.min(), BackgroundActivityCalculator::min_background_activity_stddev);
    ASSERT_EQ(param2.max(), BackgroundActivityCalculator::max_background_activity_stddev);
    ASSERT_EQ(param2.value(), stddev_background);
}

TEST_F(BackgroundActivityTest, testNullBackgroundActivityUpdate) {
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NullBackgroundActivityCalculator>();

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(background_input[neuron_id], 0.0);
    }
}

TEST_F(BackgroundActivityTest, testConstantBackgroundActivityUpdate) {
    const auto constant_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity, mt);
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<ConstantBackgroundActivityCalculator>(constant_background);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(background_input[neuron_id], 0.0);
        } else {
            ASSERT_EQ(background_input[neuron_id], constant_background);
        }
    }
}

TEST_F(BackgroundActivityTest, testNormalBackgroundActivityUpdate) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NormalBackgroundActivityCalculator>(mean_background, stddev_background);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    std::vector<double> background_values{};
    background_values.reserve(number_neurons);

    auto number_enabled_neurons = 0;
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        auto bg = background_calculator->get_background_activity(NeuronID{ neuron_id });
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(bg, 0.0);
        } else {
            background_values.emplace_back(bg - mean_background);
            number_enabled_neurons++;
        }
    }

    const auto summed_background = ranges::accumulate(background_values, 0.0);

    if (std::abs(summed_background) >= eps * number_enabled_neurons) {
        std::cerr << "The total variance was: " << std::abs(summed_background) << '\n';
        std::cerr << "That's more than " << eps << " * " << number_enabled_neurons << '\n';
        // TODO(future): Insert some statistical test here
    }
}

TEST_F(BackgroundActivityTest, testFastNormalBackgroundActivityUpdate) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<FastNormalBackgroundActivityCalculator>(mean_background, stddev_background, 5);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    const auto background_values = NeuronID::range_id(number_neurons)
        | ranges::views::filter(not_equal_to(UpdateStatus::Disabled), lookup(update_status))
        | ranges::views::transform([&background_calculator, mean_background](const auto neuron_id) { return background_calculator->get_background_activity(NeuronID(neuron_id)) - mean_background; })
        | ranges::to_vector;

    const auto number_enabled_neurons = ranges::size(background_values);
    const auto summed_background
        = ranges::accumulate(background_values, 0.0);

    if (std::abs(summed_background) >= eps * number_enabled_neurons) {
        std::cerr << "The total variance was: " << std::abs(summed_background) << '\n';
        std::cerr << "That's more than " << eps << " * " << number_enabled_neurons << '\n';
        // TODO(future): Insert some statistical test here
    }
}

std::pair<std::string, std::vector<std::pair<std::string, double>>> get_random_calculator(std::mt19937& mt) {
    const auto p = RandomAdapter::get_random_integer(0, 2, mt);
    if (p == 0) {
        return std::make_pair("null", std::vector<std::pair<std::string, double>>{});
    }
    if (p == 1) {
        const auto level = RandomAdapter::get_random_double(0.1, 20.0, mt);

        std::vector<std::pair<std::string, double>> params{};
        params.emplace_back("Base background activity", level);
        return std::make_pair(fmt::format("constant:{}", level), params);
    }
    if (p == 2) {
        const auto mean = RandomAdapter::get_random_double(-10.0, 10.0, mt);
        const auto std = RandomAdapter::get_random_double(0.1, 20.0, mt);

        std::vector<std::pair<std::string, double>> params{};
        params.emplace_back("Mean background activity", mean);
        params.emplace_back("Stddev background activity", std);

        return std::make_pair(fmt::format("normal:{},{}", mean, std), params);
    }
    RelearnException::fail("Invalid calculator");
}

TEST_F(BackgroundActivityTest, testFlexible) {
    const auto num_steps = RandomAdapter::get_random_integer(100, 1000, mt);
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 500;
    const MPIRank my_rank{ 0 };
    auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(num_neurons, mt);

    std::filesystem::path file_path{ "./background_activity.tmp" };

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();
    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    const auto num_changes = RandomAdapter::get_random_integer(10, 50, mt);

    std::vector<std::tuple<RelearnTypes::step_type, std::vector<std::pair<std::string, double>>, std::vector<NeuronID>>> gold;

    std::vector<size_t> steps{};
    auto step = -1;
    for (auto i = 0; i < num_changes; i++) {
        do {
            step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, num_steps, mt);
        } while (step == -1 || std::find(steps.begin(), steps.end(), step) != steps.end());

        steps.push_back(step);
        const auto& [p, params] = get_random_calculator(mt);
        const auto num_selected_neurons = RandomAdapter::get_random_integer<size_t>(1, num_neurons, mt);
        const auto& neurons = NeuronIdAdapter::get_random_rank_neuron_ids(num_neurons, 1, num_selected_neurons, mt);

        auto local_neurons = neurons | ranges::view::filter([my_rank](const auto& rni) { return rni.get_rank() == my_rank; }) | ranges::view::transform([](const auto& rni) { return rni.get_neuron_id(); })
            | ranges::to_vector;

        of << step << " " << p << " ";

        for (const auto& neuron_id : neurons) {
            of << neuron_id.get_rank().get_rank() << ":" << neuron_id.get_neuron_id().get_neuron_id() + 1 << " ";
        }
        of << "\n";
        std::sort(local_neurons.begin(), local_neurons.end());

        gold.emplace_back(step, params, local_neurons);
    }
    of.close();

    std::sort(gold.begin(), gold.end(), [](const auto& t1, const auto& t2) {
        return std::get<0>(t1) < std::get<0>(t2);
    });

    auto flexible_calculator = std::make_unique<FlexibleBackgroundActivityCalculator>(file_path, my_rank, local_area_translator);
    flexible_calculator->init(num_neurons);
    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(num_neurons);
    flexible_calculator->set_extra_infos(extra_info);

    int cur_index = 0;

    std::vector<std::vector<std::pair<std::string, double>>> neurons_to_params(num_neurons);

    for (auto step = 0; step < num_steps; step++) {
        flexible_calculator->update_input(step);

        while (cur_index < gold.size() && std::get<0>(gold[cur_index]) <= step) {
            const auto& [_step, params, neurons] = gold[cur_index];
            if (_step <= step) {
                for (const auto& neuron_id : neurons) {
                    neurons_to_params[neuron_id.get_neuron_id()] = params;
                }
                cur_index++;
            }
        }

        for (auto neuron_id : NeuronID::range(num_neurons)) {
            const auto calculator_params = flexible_calculator->get_background_activity_calculator_for_neuron(neuron_id)->get_parameter();
            const auto gold_params = neurons_to_params[neuron_id.get_neuron_id()];
            for (auto i = 0; i < calculator_params.size(); i++) {
                ASSERT_NEAR(std::get<Parameter<double>>(calculator_params[i]).value(), gold_params[i].second, eps);
            }
        }
    }
}

TEST_F(BackgroundActivityTest, testFlexibleMultiplePerStep) {
    const auto num_steps = RandomAdapter::get_random_integer(100, 1000, mt);
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const MPIRank my_rank{ 0 };
    auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(num_neurons, mt);

    std::filesystem::path file_path{ "./background_activity.tmp" };

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    const auto num_changes = RandomAdapter::get_random_integer(10, 20, mt);

    std::vector<std::tuple<RelearnTypes::step_type, std::vector<std::pair<std::string, double>>, std::vector<NeuronID>>> gold;
    std::vector<NeuronID> flattened_neurons;
    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, num_steps, mt);
    const auto num_selected_neurons = RandomAdapter::get_random_integer<size_t>(1, num_neurons, mt);
    const auto& neurons = NeuronIdAdapter::get_random_rank_neuron_ids(num_neurons, 1, num_selected_neurons, mt) | ranges::to_vector;

    auto local_neurons = neurons | ranges::view::filter([my_rank](const auto& rni) { return rni.get_rank() == my_rank; }) | ranges::view::transform([](const auto& rni) { return rni.get_neuron_id(); })
        | ranges::to_vector;
    const auto width = local_neurons.size() / num_changes;
    for (size_t i = 0; i < num_changes; i++) {
        const auto& [p, params] = get_random_calculator(mt);
        of << step << " " << p << " ";

        std::vector<NeuronID> my_ids;
        for (auto j = i * width; j < (i + 1) * width; j++) {
            const auto& neuron_id = local_neurons[j];
            of << "0"
               << ":" << neuron_id.get_neuron_id() + 1 << " ";
            my_ids.emplace_back(neuron_id);
        }
        of << "\n";

        gold.emplace_back(step, params, my_ids);
    }
    of.close();

    auto flexible_calculator = std::make_unique<FlexibleBackgroundActivityCalculator>(file_path, my_rank, local_area_translator);
    flexible_calculator->init(num_neurons);
    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(num_neurons);
    flexible_calculator->set_extra_infos(extra_info);

    int cur_index = 0;
    std::vector<std::vector<std::pair<std::string, double>>> neurons_to_params(num_neurons);

    for (const auto neuron_id : NeuronID::range(num_neurons)) {
        ASSERT_EQ(0, flexible_calculator->get_background_activity(neuron_id));
    }
    flexible_calculator->update_input(step);
    for (const auto& [_1, params, my_neurons] : gold) {
        for (auto neuron_id : my_neurons) {
            const auto calculator_params = flexible_calculator->get_background_activity_calculator_for_neuron(neuron_id)->get_parameter();
            for (auto i = 0; i < calculator_params.size(); i++) {
                ASSERT_NEAR(std::get<Parameter<double>>(calculator_params[i]).value(), params[i].second, eps);
            }
        }
    }

    for (const auto neuron_id : NeuronID::range(num_neurons)) {
        if (std::find(local_neurons.begin(), local_neurons.end(), neuron_id) == local_neurons.end()) {
            ASSERT_EQ(0, flexible_calculator->get_background_activity(neuron_id));
        }
    }
}

TEST_F(BackgroundActivityTest, testFlexibleMultiplePerStepFail) {
    const auto num_steps = RandomAdapter::get_random_integer(100, 1000, mt);
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 100;
    const MPIRank my_rank{ 0 };
    auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(num_neurons, mt);

    std::filesystem::path file_path{ "./background_activity.tmp" };

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    const auto num_changes = RandomAdapter::get_random_integer(10, 20, mt);

    std::vector<std::tuple<RelearnTypes::step_type, std::vector<std::pair<std::string, double>>, std::vector<NeuronID>>> gold;
    std::vector<NeuronID> flattened_neurons;
    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, num_steps, mt);
    const auto num_selected_neurons = RandomAdapter::get_random_integer<size_t>(1, num_neurons, mt);
    const auto& neurons = NeuronIdAdapter::get_random_rank_neuron_ids(num_neurons, 1, num_selected_neurons, mt) | ranges::to_vector;

    auto local_neurons = neurons | ranges::view::filter([my_rank](const auto& rni) { return rni.get_rank() == my_rank; }) | ranges::view::transform([](const auto& rni) { return rni.get_neuron_id(); })
        | ranges::to_vector;
    const auto width = local_neurons.size() / num_changes;

    for (size_t i = 0; i < num_changes; i++) {
        const auto& [p, params] = get_random_calculator(mt);
        of << step << " " << p << " ";

        for (auto j = i * width; j < (i + 1) * width; j++) {
            const auto& neuron_id = local_neurons[j];
            flattened_neurons.push_back(neuron_id);
            of << "0"
               << ":" << neuron_id.get_neuron_id() + 1 << " ";
        }
        of << "\n";
    }
    const auto& [p, params] = get_random_calculator(mt);
    of << step << " " << p << " "
       << "0:" << flattened_neurons[0].get_neuron_id() + 1 << "\n";
    of.close();

    ASSERT_THROW(std::make_unique<FlexibleBackgroundActivityCalculator>(file_path, my_rank, local_area_translator), RelearnException);
}

TEST_F(BackgroundActivityTest, testFlexibleSameCalculator) {
    const auto num_steps = RandomAdapter::get_random_integer(100, 1000, mt);
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const MPIRank my_rank{ 0 };
    auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(num_neurons, mt);

    std::filesystem::path file_path{ "./background_activity.tmp" };

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    const auto num_changes = RandomAdapter::get_random_integer(10, 50, mt);

    std::vector<std::tuple<RelearnTypes::step_type, std::vector<std::pair<std::string, double>>, std::vector<NeuronID>>> gold;

    const auto& [p, params] = get_random_calculator(mt);

    std::vector<size_t> steps{};
    auto step = -1;
    for (auto i = 0; i < num_changes; i++) {
        do {
            step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, num_steps, mt);
        } while (step == -1 || std::find(steps.begin(), steps.end(), step) != steps.end());
        steps.emplace_back(step);

        const auto num_selected_neurons = RandomAdapter::get_random_integer<size_t>(1, num_neurons, mt);
        const auto& neurons = NeuronIdAdapter::get_random_rank_neuron_ids(num_neurons, 1, num_selected_neurons, mt);

        auto local_neurons = neurons | ranges::view::filter([my_rank](const auto& rni) { return rni.get_rank() == my_rank; }) | ranges::view::transform([](const auto& rni) { return rni.get_neuron_id(); })
            | ranges::to_vector;

        of << step << " " << p << " ";

        for (const auto& neuron_id : neurons) {
            of << neuron_id.get_rank().get_rank() << ":" << neuron_id.get_neuron_id().get_neuron_id() + 1 << " ";
        }
        of << "\n";
        std::sort(local_neurons.begin(), local_neurons.end());

        gold.emplace_back(step, params, local_neurons);
    }
    of.close();

    std::sort(gold.begin(), gold.end(), [](const auto& t1, const auto& t2) {
        return std::get<0>(t1) < std::get<0>(t2);
    });

    auto flexible_calculator = std::make_unique<FlexibleBackgroundActivityCalculator>(file_path, my_rank, local_area_translator);
    flexible_calculator->init(num_neurons);
    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(num_neurons);
    flexible_calculator->set_extra_infos(extra_info);

    int cur_index = 0;

    std::set<NeuronID> affected_neurons{};

    std::shared_ptr<BackgroundActivityCalculator> last_calculator{};

    for (auto step = 0; step < num_steps; step++) {
        flexible_calculator->update_input(step);

        while (cur_index < gold.size() && std::get<0>(gold[cur_index]) <= step) {
            const auto& [_step, params, neurons] = gold[cur_index];
            if (_step <= step) {
                for (const auto& neuron_id : neurons) {
                    affected_neurons.insert(neuron_id);
                }
                cur_index++;
            }
        }

        for (auto neuron_id : affected_neurons) {
            const auto calculator_params = flexible_calculator->get_background_activity_calculator_for_neuron(neuron_id)->get_parameter();
            for (auto i = 0; i < calculator_params.size(); i++) {
                ASSERT_NEAR(std::get<Parameter<double>>(calculator_params[i]).value(), params[i].second, eps);
            }
            if (last_calculator != nullptr) {
                ASSERT_EQ(flexible_calculator->get_background_activity_calculator_for_neuron(neuron_id), last_calculator);
            } else {
                last_calculator = flexible_calculator->get_background_activity_calculator_for_neuron(neuron_id);
            }
        }
    }
}