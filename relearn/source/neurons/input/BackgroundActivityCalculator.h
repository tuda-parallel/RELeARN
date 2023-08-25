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
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/ModelParameter.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"

#include <memory>
#include <vector>

class NeuronMonitor;

enum class BackgroundActivityCalculatorType : char {
    Null,
    Constant,
    Normal,
    FastNormal,
    Flexible
};

/**
 * @brief Pretty-prints the background activity calculator type to the chosen stream
 * @param out The stream to which to print the background activity
 * @param element_type The background activity to print
 * @return The argument out, now altered with the background activity
 */
inline std::ostream& operator<<(std::ostream& out, const BackgroundActivityCalculatorType& calculator_type) {
    if (calculator_type == BackgroundActivityCalculatorType::Null) {
        return out << "Null";
    }

    if (calculator_type == BackgroundActivityCalculatorType::Constant) {
        return out << "Constant";
    }

    if (calculator_type == BackgroundActivityCalculatorType::Normal) {
        return out << "Normal";
    }

    return out;
}
template <>
struct fmt::formatter<BackgroundActivityCalculatorType> : ostream_formatter { };

/**
 * This class provides an interface to calculate the background activity that neurons receive.
 * It also provides some default/min/max values via public static constexpr members,
 * because the child classes share them.
 */
class BackgroundActivityCalculator {
    friend class NeuronMonitor;

public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new instance of type BackgroundActivityCalculator with 0 neurons.
     * @param first_step The first step in which background activity is applied
     * @param last_step The last step in which background activity is applied
     */
    explicit BackgroundActivityCalculator() { }

    virtual ~BackgroundActivityCalculator() = default;

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] virtual std::unique_ptr<BackgroundActivityCalculator> clone() const = 0;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    virtual void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info.operator bool();
        RelearnException::check(is_filled, "BackgroundActivityCalculator::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    virtual void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_local_neurons == 0, "BackgroundActivityCalculator::init: Was already initialized");
        RelearnException::check(number_neurons > 0, "BackgroundActivityCalculator::init: number_neurons was 0");

        number_local_neurons = number_neurons;
        background_activity.resize(number_neurons, 0.0);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    virtual void create_neurons(const number_neurons_type creation_count) {
        RelearnException::check(number_local_neurons > 0, "BackgroundActivityCalculator::create_neurons: number_local_neurons was 0");
        RelearnException::check(creation_count > 0, "BackgroundActivityCalculator::create_neurons: creation_count was 0");

        const auto current_size = number_local_neurons;
        const auto new_size = current_size + creation_count;

        number_local_neurons = new_size;
        background_activity.resize(new_size, 0.0);
    }

    /**
     * @brief Updates the background activity based on which neurons to update
     * @param step The current update step
     * @param disable_flags Which neurons are disabled
     * @exception Throws a RelearnException if the number of local neurons didn't match the sizes of the arguments
     */
    virtual void update_input([[maybe_unused]] const step_type step) = 0;

    /**
     * @brief Returns the calculated background activity for the given neuron. Changes after calls to update_input(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The background activity for the given neuron
     */
    [[nodiscard]] virtual double get_background_activity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "BackgroundActivityCalculator::get_background_activity: id is too large: {}", neuron_id);
        return background_activity[local_neuron_id];
    }

    /**
     * @brief Returns the calculated background activity for all. Changes after calls to update_input(...)
     * @return The background activity for all neurons
     */
    [[nodiscard]] virtual std::span<const double> get_background_activity() const noexcept {
        return background_activity;
    }

    /**
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] virtual std::vector<ModelParameter> get_parameter() {
        return {};
    }

    static constexpr double default_base_background_activity{ 0.0 };
    static constexpr double default_background_activity_mean{ 0.0 };
    static constexpr double default_background_activity_stddev{ 0.0 };

    static constexpr double min_base_background_activity{ -10000.0 };
    static constexpr double min_background_activity_mean{ -10000.0 };
    static constexpr double min_background_activity_stddev{ 0.0 };

    static constexpr double max_base_background_activity{ 10000.0 };
    static constexpr double max_background_activity_mean{ 10000.0 };
    static constexpr double max_background_activity_stddev{ 10000.0 };

protected:
    /**
     * @brief Sets the background activity for the given neuron
     * @param step The current step
     * @param neuron_id The local neuron
     * @param value The new background activity
     * @exception Throws a RelearnException if the neuron_id is to large
     */
    void set_background_activity(const RelearnTypes::step_type step, const number_neurons_type neuron_id, const double value) {
        RelearnException::check(neuron_id < number_local_neurons, "SynapticInputCalculator::set_background_activity: neuron_id was too large: {} vs {}", neuron_id, number_local_neurons);
        background_activity[neuron_id] = value;
    }

    std::shared_ptr<NeuronsExtraInfo> extra_infos{};

private:
    number_neurons_type number_local_neurons{};

    std::vector<double> background_activity{};
};