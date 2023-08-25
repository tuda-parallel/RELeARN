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

#include "neurons/input/BackgroundActivityCalculator.h"

#include "io/BackgroundActivityIO.h"
#include "io/InteractiveNeuronIO.h"
#include "util/Random.h"
#include "util/Timers.h"
#include "util/ranges/Functional.hpp"

#include <boost/lexical_cast.hpp>
#include <filesystem>
#include <functional>
#include <optional>
#include <range/v3/algorithm/generate.hpp>
#include <range/v3/view/transform.hpp>
#include <utility>

/**
 * This class provides no input whatsoever.
 */
class NullBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object of type NullBackgroundActivityCalculator
     * @param first_step The first step in which background activity is applied
     * @param last_step The last step in which background activity is applied
     */
    NullBackgroundActivityCalculator()
        : BackgroundActivityCalculator() { }

    virtual ~NullBackgroundActivityCalculator() = default;

    /**
     * @brief This activity calculator does not provide any input
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<NullBackgroundActivityCalculator>();
    }
};

/**
 * This class provides a constant input
 */
class ConstantBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given constant input
     * @brief first_step The first step in which background activity is applied
     * @brief last_step The last step in which background activity is applied
     * @brief input The base input
     */
    ConstantBackgroundActivityCalculator(const double input) noexcept
        : BackgroundActivityCalculator()
        , base_input(input) {
    }

    virtual ~ConstantBackgroundActivityCalculator() = default;

    /**
     * @brief Updates the input, providing constant or 0 input depending on the disable_flags
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto& disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
            "ConstantBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
        for (number_neurons_type neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
            const auto input = !extra_infos->does_update_plasticity(NeuronID{ neuron_id }) ? 0.0 : base_input;
            set_background_activity(step, neuron_id, input);
        }
        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<ConstantBackgroundActivityCalculator>(base_input);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Base background activity", base_input, BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity));

        return parameters;
    }

private:
    double base_input{ default_base_background_activity };
};

/**
 * This class provides a normally distributed input, i.e., according to some N(expected, standard deviation)
 */
class NormalBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given normal input, i.e.,
     *      from N(mean, stddev)
     * @param first_step The first step in which background activity is applied
     * @param last_step The last step in which background activity is applied
     * @param mean The mean input
     * @param stddev The standard deviation, must be > 0.0
     * @exception Throws a RelearnException if stddev <= 0.0
     */
    NormalBackgroundActivityCalculator(const double mean, const double stddev)
        : BackgroundActivityCalculator()
        , mean_input(mean)
        , stddev_input(stddev) {
        RelearnException::check(stddev > 0.0, "NormalBackgroundActivityCalculator::NormalBackgroundActivityCalculator: stddev was: {}", stddev);
    }

    virtual ~NormalBackgroundActivityCalculator() = default;

    /**
     * @brief Updates the input, providing normal or 0 input depending on the status of the neuron in the extra infos
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto& disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
            "NormalBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
        for (number_neurons_type neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
            const auto input = !extra_infos->does_update_plasticity(NeuronID{ neuron_id }) ? 0.0 : RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input);
            set_background_activity(step, neuron_id, input);
        }
        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<NormalBackgroundActivityCalculator>(mean_input, stddev_input);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Mean background activity", mean_input, BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean));
        parameters.emplace_back(Parameter<double>("Stddev background activity", stddev_input, BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev));

        return parameters;
    }

private:
    double mean_input{ default_background_activity_mean };
    double stddev_input{ default_background_activity_stddev };
};

/**
 * This class provides a normally distributed input, i.e., according to some N(expected, standard deviation).
 * However, it draws all input at the initialization phase and only returns pointers into that memory;
 * this speeds the update up enormously, but ignores if a neuron is disabled.
 */
class FastNormalBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given normal input, i.e., from N(mean, stddev).
     *      It draws number_neurons*multiplier inputs initially and than just returns pointers.
     * @param mean The mean input
     * @param stddev The standard deviation, must be > 0.0
     * @param first_step The first step in which background activity is applied
     * @param last_step The last step in which background activity is applied
     * @param multiplier The factor how many more values should be drawn
     * @exception Throws a RelearnException if stddev <= 0.0
     */
    FastNormalBackgroundActivityCalculator(const double mean, const double stddev, const size_t multiplier)
        : BackgroundActivityCalculator()
        , mean_input(mean)
        , stddev_input(stddev)
        , multiplier(multiplier) {
        RelearnException::check(stddev > 0.0, "FastNormalBackgroundActivityCalculator::FastNormalBackgroundActivityCalculator: stddev was: {}", stddev);
        RelearnException::check(multiplier > 0, "FastNormalBackgroundActivityCalculator::FastNormalBackgroundActivityCalculator: multiplier was: 0", stddev);
    }

    virtual ~FastNormalBackgroundActivityCalculator() = default;

    /**
     * @brief Updates the input, providing constant to all neurons to speed up the calculations.
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto number_neurons = get_number_neurons();
        const auto& disable_flags = extra_infos->get_disable_flags();
        RelearnException::check(disable_flags.size() == number_neurons,
            "FastNormalBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

        const auto min_offset = 0;
        const auto max_offset = pre_drawn_values.size() - number_neurons;

        const auto new_offset = RandomHolder::get_random_uniform_integer<size_t>(RandomHolderKey::BackgroundActivity, min_offset, max_offset);
        offset = new_offset;
        cur_step = step;

        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    void init(const number_neurons_type number_neurons) override {
        BackgroundActivityCalculator::init(number_neurons);

        pre_drawn_values.resize(number_neurons * multiplier);
        ranges::generate(pre_drawn_values, [this]() { return RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input); });
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type number_neurons) override {
        const auto previous_number_neurons = get_number_neurons();

        BackgroundActivityCalculator::create_neurons(number_neurons);

        const auto now_number_neurons = get_number_neurons();
        pre_drawn_values.resize(now_number_neurons * multiplier);
        ranges::generate(pre_drawn_values, [this]() { return RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input); });
    }

    /**
     * @brief Returns the calculated background activity for the given neuron. Changes after calls to update_input(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The background activity for the given neuron
     */
    double get_background_activity(const NeuronID neuron_id) const override {
        const auto number_neurons = get_number_neurons();
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_neurons, "FastNormalBackgroundActivityCalculator::get_background_activity: id is too large: {}", neuron_id);
        if (extra_infos->does_update_plasticity(NeuronID{ neuron_id })) {
            return pre_drawn_values[offset + local_neuron_id];
        }
        return 0.0;
    }

    /**
     * @brief Returns the calculated background activity for all. Changes after calls to update_input(...)
     * @return The background activity for all neurons
     */
    [[nodiscard]] std::span<const double> get_background_activity() const noexcept override {
        RelearnException::fail("FastNormalBackgroundActivityCalculator::get_background_activity: Not supported method with this calculator");
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<FastNormalBackgroundActivityCalculator>(mean_input, stddev_input, multiplier);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Mean background activity", mean_input, BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean));
        parameters.emplace_back(Parameter<double>("Stddev background activity", stddev_input, BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev));

        return parameters;
    }

private:
    double mean_input{ default_background_activity_mean };
    double stddev_input{ default_background_activity_stddev };
    size_t multiplier{ 1 };
    size_t offset{ 0 };

    std::vector<double> pre_drawn_values{};
    step_type cur_step;
};

/**
 * This class provides a normally distributed input, i.e., according to some N(expected, standard deviation).
 * However, it draws all input at the initialization phase and only returns pointers into that memory;
 * this speeds the update up enormously, but ignores if a neuron is disabled.
 */
class FlexibleBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given normal input, i.e., from N(mean, stddev).
     *      It draws number_neurons*multiplier inputs initially and than just returns pointers.
     * @param mean The mean input
     * @param stddev The standard deviation, must be > 0.0
     * @param first_step The first step in which background activity is applied
     * @param last_step The last step in which background activity is applied
     * @param multiplier The factor how many more values should be drawn
     * @exception Throws a RelearnException if stddev <= 0.0
     */
    FlexibleBackgroundActivityCalculator(const std::filesystem::path& file_path, const MPIRank& my_rank, const std::shared_ptr<LocalAreaTranslator>& local_area_translator)
        : file_path(file_path)
        , my_rank(my_rank)
        , local_area_translator(local_area_translator)
        , BackgroundActivityCalculator() {

        activities_sorted_for_begin = BackgroundActivityIO::load_background_activity(file_path, my_rank, local_area_translator);

        // Check for neurons assigned to different calculators in the same step
        std::unordered_map<size_t, std::vector<size_t>> step_to_indices{};

        auto last_step = -1;
        for (auto i = 0; i < activities_sorted_for_begin.size(); i++) {
            const auto& [step, _0, _1] = activities_sorted_for_begin[i];
            if (!step_to_indices.contains(step)) {
                step_to_indices[step] = std::vector<size_t>{};
            }
            step_to_indices[step].push_back(i);
        }

        for (const auto& [step, indices] : step_to_indices) {
            if (indices.size() > 1) {
                std::vector<NeuronID> flattened{};
                for (const auto index : indices) {
                    const auto& neuron_ids = std::get<2>(activities_sorted_for_begin[index]);
                    flattened.insert(flattened.end(), neuron_ids.begin(), neuron_ids.end());
                }
                auto flattened_set = flattened;
                std::sort(flattened_set.begin(), flattened_set.end());
                flattened_set.erase(std::unique(flattened_set.begin(), flattened_set.end()), flattened_set.end());
                RelearnException::check(flattened_set.size() == flattened.size(), "FlexibleBackgroundActivityCalculator::FlexibleBackgroundActivityCalculator: Multiple background activity calculators specified for the same neuron in the same step {}", step);
            }
        }
    }

    virtual ~FlexibleBackgroundActivityCalculator() = default;

    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_infos) override {
        BackgroundActivityCalculator::set_extra_infos(new_extra_infos);
        for (const auto& [_, calculator] : background_activity_calculators) {
            calculator->set_extra_infos(extra_infos);
        }
    }

    /**
     * @brief Updates the input, providing constant to all neurons to speed up the calculations.
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto number_neurons = get_number_neurons();
        const auto& disable_flags = extra_infos->get_disable_flags();
        RelearnException::check(disable_flags.size() == number_neurons,
            "FastNormalBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

        while (cur_index < activities_sorted_for_begin.size()) {
            const auto& [begin, type, neurons] = activities_sorted_for_begin[cur_index];
            if (begin <= step) {
                const auto& calculator = parse_calculator_type(type);
                for (const auto& neuron_id : neurons) {
                    neuron_id_to_background_activity_calculator[neuron_id.get_neuron_id()] = calculator;
                }
                cur_index++;
            } else {
                break;
            }
        }

        for (const auto& [_, calculator] : background_activity_calculators) {
            calculator->update_input(step);
        }

        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    std::shared_ptr<BackgroundActivityCalculator> get_background_activity_calculator_for_neuron(const NeuronID& neuron_id) {
        return neuron_id_to_background_activity_calculator[neuron_id.get_neuron_id()];
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    void init(const number_neurons_type number_neurons) override {
        BackgroundActivityCalculator::init(number_neurons);

        const auto null_calculator = parse_calculator_type("null");
        neuron_id_to_background_activity_calculator.resize(number_neurons, null_calculator);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type number_neurons) override {
        BackgroundActivityCalculator::create_neurons(number_neurons);
        const auto now_number_neurons = get_number_neurons();

        const auto null_calculator = parse_calculator_type("null");
        neuron_id_to_background_activity_calculator.resize(now_number_neurons, null_calculator);

        for (const auto& [_, calculator] : background_activity_calculators) {
            calculator->create_neurons(number_neurons);
        }
    }

    /**
     * @brief Returns the calculated background activity for the given neuron. Changes after calls to update_input(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The background activity for the given neuron
     */
    double get_background_activity(const NeuronID neuron_id) const override {
        const auto number_neurons = get_number_neurons();
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_neurons, "FastNormalBackgroundActivityCalculator::get_background_activity: id is too large: {}", neuron_id);
        return neuron_id_to_background_activity_calculator[local_neuron_id]->get_background_activity(neuron_id);
    }

    /**
     * @brief Returns the calculated background activity for all. Changes after calls to update_input(...)
     * @return The background activity for all neurons
     */
    [[nodiscard]] std::span<const double> get_background_activity() const noexcept override {
        RelearnException::fail("FastNormalBackgroundActivityCalculator::get_background_activity: Not supported method with this calculator");
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<FlexibleBackgroundActivityCalculator>(file_path, my_rank, local_area_translator);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        return parameters;
    }

private:
    std::shared_ptr<BackgroundActivityCalculator> parse_calculator_type(std::string type) {
        StringUtil::to_lower(type);
        const auto i1 = type.find(':');
        if (type.starts_with("normal")) {
            // normal background
            RelearnException::check(i1 > 0, "FlexibleBackgroundActivityCalculator::parse_calculator_type: No parameters for activity calculator {}", type);
            const auto i2 = type.find(',', i1);
            RelearnException::check(i2 > 0, "FlexibleBackgroundActivityCalculator::parse_calculator_type: No second parameter for activity calculator {}", type);
            const auto mean = boost::lexical_cast<double>(type.substr(i1 + 1, i2 - i1 - 1));
            const auto std = boost::lexical_cast<double>(type.substr(i2 + 1));

            const auto key = fmt::format("normal:{},{}", mean, std);
            if (background_activity_calculators.contains(key)) {
                return background_activity_calculators[key];
            }

            // Create
            const auto new_calculator = std::make_shared<NormalBackgroundActivityCalculator>(mean, std);
            new_calculator->init(get_number_neurons());
            if (extra_infos.operator bool()) {
                new_calculator->set_extra_infos(extra_infos);
            }
            background_activity_calculators[key] = new_calculator;
            return new_calculator;
        } else if (type.starts_with("constant")) {
            // constant
            RelearnException::check(i1 > 0, "FlexibleBackgroundActivityCalculator::parse_calculator_type: No parameters for activity calculator {}", type);
            const auto value = std::stod(type.substr(i1 + 1));

            const auto key = fmt::format("constant:{}", value);
            if (background_activity_calculators.contains(key)) {
                return background_activity_calculators[key];
            }

            // Create
            const auto new_calculator = std::make_shared<ConstantBackgroundActivityCalculator>(value);
            new_calculator->init(get_number_neurons());
            if (extra_infos.operator bool()) {
                new_calculator->set_extra_infos(extra_infos);
            }
            background_activity_calculators[key] = new_calculator;
            return new_calculator;
        } else if (type == "null") {
            const auto key = "null";

            if (background_activity_calculators.contains(key)) {
                return background_activity_calculators[key];
            }

            // Create
            const auto new_calculator = std::make_shared<NullBackgroundActivityCalculator>();
            new_calculator->init(get_number_neurons());
            if (extra_infos.operator bool()) {
                new_calculator->set_extra_infos(extra_infos);
            }
            background_activity_calculators[key] = new_calculator;
            return new_calculator;
        } else {
            RelearnException::fail("FlexibleBackgroundActivityCalculator::parse_calculator_type: Unknown backgroundActivityCalculator type {}", type);
        }
    }

    std::filesystem::path file_path;
    MPIRank my_rank;
    std::shared_ptr<LocalAreaTranslator> local_area_translator;

    size_t cur_index = 0;

    std::vector<std::tuple<RelearnTypes::step_type, std::string, std::vector<NeuronID>>> activities_sorted_for_begin;

    std::vector<std::shared_ptr<BackgroundActivityCalculator>> neuron_id_to_background_activity_calculator{};

    std::unordered_map<std::string, std::shared_ptr<BackgroundActivityCalculator>> background_activity_calculators{};
};