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
#include "neurons/CalciumCalculator.h"
#include "neurons/Neurons.h"
#include "neurons/models/NeuronModels.h"

#include <memory>
#include <vector>

/**
 * An object of type NeuronInformation functions as a snapshot of one neuron at one point in the simulation.
 * It stores all necessary superficial information as a plain-old-data class.
 */
class NeuronInformation {
public:
    /**
     * @brief Constructs a NeuronInformation that holds the arguments in one class
     * @param c The current calcium concentration
     * @param tc The current calcium target
     * @param x The current membrane potential
     * @param f The current fire status
     * @param ff The fraction of spikes in during the last period
     * @param s The current secondary variable of the model
     * @param i The current synaptic input
     * @param ex_input number of fired excitatory input synapses
     * @param inh_input number of fired inhibitory input synapses
     * @param b The current background activity
     * @param ax The current number of axonal elements
     * @param ax_c The current number of connected axonal elements
     * @param de The current number of excitatory dendritic elements
     * @param de_c The current number of connected excitatory dendritic elements
     * @param di The current number of inhibitory dendritic elements
     * @param di_c The current number of connected inhibitory dendritic elements
     */
    NeuronInformation(const RelearnTypes::step_type step, const double c, const double tc, const double x, const bool f, const double ff, const double s,
        const double i, const double ex_input, const double inh_input, const double b, const double stim, const double ax, const double ax_c, const double de, const double de_c, const double di, const double di_c) noexcept
        : current_step(step)
        , calcium(c)
        , target_calcium(tc)
        , x(x)
        , fired(f)
        , fired_fraction(ff)
        , secondary(s)
        , synaptic_input(i)
        , ex_input(ex_input)
        , inh_input(inh_input)
        , background_activity(b)
        , stimulation(stim)
        , axons_grown(ax)
        , axons_connected(ax_c)
        , excitatory_dendrites_grown(de)
        , excitatory_dendrites_connected(de_c)
        , inhibitory_dendrites_grown(di)
        , inhibitory_dendrites_connected(di_c) {
    }

    /**
     * @brief Returns the step at which the data was recorded
     * @return The step
     */
    [[nodiscard]] RelearnTypes::step_type get_step() const noexcept {
        return current_step;
    }

    /**
     * @brief Returns the stored calcium concentration
     * @return The stored calcium concentration
     */
    [[nodiscard]] double get_calcium() const noexcept {
        return calcium;
    }

    /**
     * @brief Returns the stored calcium target
     * @return The stored calcium target
     */
    [[nodiscard]] double get_target_calcium() const noexcept {
        return target_calcium;
    }

    /**
     * @brief Returns the stored membrane potential
     * @return The stored membrane potential
     */
    [[nodiscard]] double get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns the stored fire status
     * @return The stored fire status
     */
    [[nodiscard]] bool get_fired() const noexcept {
        return fired;
    }

    /**
     * @brief Returns the fraction of spikes during the last recording period
     * @return The fraction of spikes
     */
    [[nodiscard]] double get_fraction_fired() const noexcept {
        return fired_fraction;
    }

    /**
     * @brief Returns the stored secondary variable of the model
     * @return The stored secondary variable of the model
     */
    [[nodiscard]] double get_secondary() const noexcept {
        return secondary;
    }

    /**
     * @brief Returns the stored synaptic input
     * @return The stored synaptic input
     */
    [[nodiscard]] double get_synaptic_input() const noexcept {
        return synaptic_input;
    }

    /**
     * @brief Returns the number of fired excitatory input synapses
     * @return Number of fired excitatory input synapses
     */
    [[nodiscard]] double get_ex_input() const noexcept {
        return ex_input;
    }

    /**
     * @brief Returns the number of fired inhibitory input synapses
     * @return Number of fired inhibitory input synapses
     */
    [[nodiscard]] double get_inh_input() const noexcept {
        return inh_input;
    }

    /**
     * @brief Returns the stored synaptic input
     * @return The stored synaptic input
     */
    [[nodiscard]] double get_background_activity() const noexcept {
        return background_activity;
    }

    /**
     * @brief Returns the stored number of axonal elements
     * @return The stored number of axonal elements
     */
    [[nodiscard]] double get_axons() const noexcept {
        return axons_grown;
    }

    /**
     * @brief Returns the stored number of connected axonal elements
     * @return The stored number of connected axonal elements
     */
    [[nodiscard]] double get_axons_connected() const noexcept {
        return axons_connected;
    }

    /**
     * @brief Returns the stored number of excitatory dendritic elements
     * @return The stored number of excitatory dendritic elements
     */
    [[nodiscard]] double get_excitatory_dendrites_grown() const noexcept {
        return excitatory_dendrites_grown;
    }

    [[nodiscard]] double get_stimulation() const noexcept {
        return stimulation;
    }

    /**
     * @brief Returns the stored number of connected excitatory dendritic elements
     * @return The stored number of connected excitatory dendritic elements
     */
    [[nodiscard]] double get_excitatory_dendrites_connected() const noexcept {
        return excitatory_dendrites_connected;
    }

    /**
     * @brief Returns the stored number of inhibitory dendritic elements
     * @return The stored number of inhibitory dendritic elements
     */
    [[nodiscard]] double get_inhibitory_dendrites_grown() const noexcept {
        return inhibitory_dendrites_grown;
    }

    /**
     * @brief Returns the stored number of connected inhibitory dendritic elements
     * @return The stored number of connected inhibitory dendritic elements
     */
    [[nodiscard]] double get_inhibitory_dendrites_connected() const noexcept {
        return inhibitory_dendrites_connected;
    }

private:
    RelearnTypes::step_type current_step{};

    double calcium{};
    double target_calcium{};
    double x{};
    bool fired{};
    double fired_fraction{};
    double secondary{};
    double synaptic_input{};
    double ex_input{};
    double inh_input{};
    double background_activity{};
    double stimulation{};

    double axons_grown{};
    double axons_connected{};
    double excitatory_dendrites_grown{};
    double excitatory_dendrites_connected{};
    double inhibitory_dendrites_grown{};
    double inhibitory_dendrites_connected{};
};

/**
 * An object of type NeuronMonitor monitors a specified neuron throughout the simulation.
 * It automatically gathers all necessary data, however, it only does so if there is space left.
 *
 * Offers the following static member:
 * neurons_to_monitor - an std::shared_ptr to the neurons to monitor. Has to be set before a call to record_data()
 */
class NeuronMonitor {
public:
    static inline std::shared_ptr<Neurons> neurons_to_monitor{};
    static inline RelearnTypes::step_type log_frequency{ Config::neuron_monitor_log_step };

    /**
     * @brief Constructs a NeuronMonitor that monitors the specified neuron
     * @param neuron_id The local neuron id for the object to monitor
     */
    explicit NeuronMonitor(const NeuronID neuron_id) noexcept
        : target_neuron_id(neuron_id) {
    }

    ~NeuronMonitor() = default;

    NeuronMonitor(const NeuronMonitor& other) noexcept = delete;
    NeuronMonitor& operator=(const NeuronMonitor& other) noexcept = delete;

    NeuronMonitor(NeuronMonitor&& other) noexcept = default;
    NeuronMonitor& operator=(NeuronMonitor&& other) noexcept = default;

    /**
     * @brief Returns the local neuron id which is monitored
     * @return The neuron id
     */
    [[nodiscard]] const NeuronID get_target_id() const noexcept {
        return target_neuron_id;
    }

    /**
     * @brief Captures the current state of the monitored neuron
     * @param current_step The current step of the recording
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons or if the std::shared_ptr is empty
     */
    void record_data(const RelearnTypes::step_type current_step) {
        RelearnException::check(neurons_to_monitor.operator bool(), "NeuronMonitor::record_data: The shared pointer is empty");

        const auto local_neuron_id = target_neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < neurons_to_monitor->number_neurons, "NeuronMonitor::record_data: The target id is too large for the neurons class");

        const auto calcium = neurons_to_monitor->calcium_calculator->calcium[local_neuron_id];
        const auto target_calcium = neurons_to_monitor->calcium_calculator->target_calcium[local_neuron_id];
        const auto x = neurons_to_monitor->neuron_model->x[local_neuron_id];
        const auto fired = neurons_to_monitor->neuron_model->fired[local_neuron_id] == FiredStatus::Fired;
        const auto fired_fraction = static_cast<double>(neurons_to_monitor->neuron_model->fired_recorder[NeuronModel::FireRecorderPeriod::NeuronMonitor][local_neuron_id]) / static_cast<double>(log_frequency);
        const auto secondary = neurons_to_monitor->neuron_model->get_secondary_variable(target_neuron_id);
        const auto synaptic_input = neurons_to_monitor->neuron_model->input_calculator->get_synaptic_input(target_neuron_id);
        const auto background_activity = neurons_to_monitor->neuron_model->background_calculator->get_background_activity(target_neuron_id);
        const auto stimulation = neurons_to_monitor->neuron_model->stimulus_calculator->get_stimulus(target_neuron_id);
        const auto fired_ex_inputs = neurons_to_monitor->neuron_model->input_calculator->raw_ex_input[local_neuron_id];
        const auto fired_inh_input = neurons_to_monitor->neuron_model->input_calculator->raw_inh_input[local_neuron_id];

        const auto axons = neurons_to_monitor->axons->grown_elements[local_neuron_id];
        const auto axons_connected = neurons_to_monitor->axons->connected_elements[local_neuron_id];
        const auto excitatory_dendrites_grown = neurons_to_monitor->dendrites_exc->grown_elements[local_neuron_id];
        const auto excitatory_dendrites_connected = neurons_to_monitor->dendrites_exc->connected_elements[local_neuron_id];
        const auto inhibitory_dendrites_grown = neurons_to_monitor->dendrites_inh->grown_elements[local_neuron_id];
        const auto inhibitory_dendrites_connected = neurons_to_monitor->dendrites_inh->connected_elements[local_neuron_id];

        information.emplace_back(current_step, calcium, target_calcium, x, fired, fired_fraction, secondary,
            synaptic_input, fired_ex_inputs, fired_inh_input, background_activity, stimulation, axons, axons_connected, excitatory_dendrites_grown,
            excitatory_dendrites_connected, inhibitory_dendrites_grown, inhibitory_dendrites_connected);
    }

    /**
     * @brief Increases the capacity for stored NeuronInformation by reserving
     * @param neuron_id The amount by which the storage should be increased
     */
    void increase_monitoring_capacity(const size_t increase_by) noexcept {
        information.reserve(information.size() + increase_by);
    }

    /**
     * @brief Clears the recorded data
     */
    void clear() noexcept {
        information.clear();
    }

    /**
     * @brief Returns the stored information
     * @return An std::vector of NeuronInformation
     */
    [[nodiscard]] const std::vector<NeuronInformation>& get_information() const noexcept {
        return information;
    }

    void init_print_file();

    void flush_current_contents();

private:
    NeuronID target_neuron_id{ NeuronID::uninitialized_id() };

    std::vector<NeuronInformation> information{};
};
