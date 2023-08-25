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
#include "mpi/MPIWrapper.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/input/FiredStatusCommunicator.h"
#include "neurons/models/ModelParameter.h"
#include "neurons/NetworkGraph.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"

#include "fmt/ostream.h"

#include <memory>
#include <vector>

class NetworkGraph;
class NeuronMonitor;

/**
 * This enums lists all types of synaptic input calculators
 */
enum class SynapticInputCalculatorType : char {
    Linear
};

/**
 * @brief Pretty-prints the synaptic input calculator type to the chosen stream
 * @param out The stream to which to print the synaptic input calculator
 * @param element_type The synaptic input calculator to print
 * @return The argument out, now altered with the synaptic input calculator
 */
inline std::ostream& operator<<(std::ostream& out, const SynapticInputCalculatorType& calculator_type) {
    if (calculator_type == SynapticInputCalculatorType::Linear) {
        return out << "Linear";
    }

    return out;
}

template <>
struct fmt::formatter<SynapticInputCalculatorType> : ostream_formatter { };

/**
 * This class provides an interface to calculate the background activity and synaptic input
 * that neurons receive. Performs communication with MPI to synchronize with different ranks.
 */
class SynapticInputCalculator {
    friend class AreaMonitor;
    friend class NeuronMonitor;

public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param synapse_conductance The factor by which the input of a neighboring spiking neuron is weighted
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticInputCalculator(const double synapse_conductance, std::unique_ptr<FiredStatusCommunicator>&& communicator)
        : synapse_conductance(synapse_conductance)
        , fired_status_comm(std::move(communicator)) {
        RelearnException::check(fired_status_comm.operator bool(), "SynapticInputCalculator::SynapticInputCalculator: communicator was empty.");
    }

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info.operator bool();
        RelearnException::check(is_filled, "SynapticInputCalculator::set_extra_infos: new_extra_info is empty");

        extra_infos = new_extra_info;
        fired_status_comm->set_extra_infos(std::move(new_extra_info));
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> new_network_graph) {
        const auto is_filled = new_network_graph.operator bool();
        RelearnException::check(is_filled, "SynapticInputCalculator::set_network_graph: new_network_graph is empty");

        network_graph = new_network_graph;
        fired_status_comm->set_network_graph(std::move(new_network_graph));
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] virtual std::unique_ptr<SynapticInputCalculator> clone() const = 0;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Updates the synaptic input and the background activity based on the current network graph, whether the local neurons spikes, and which neuron to update
     * @param step The current update step
     * @param fired Which local neuron fired
     * @exception Throws a RelearnException if the number of local neurons didn't match the sizes of the arguments
     */
    void update_input(const step_type step, const std::span<const FiredStatus> fired) {

        const auto& disable_flags = extra_infos->get_disable_flags();

        RelearnException::check(number_local_neurons > 0, "SynapticInputCalculator::update_input: There were no local neurons.");
        RelearnException::check(fired.size() == number_local_neurons, "SynapticInputCalculator::update_input: Size of fired did not match number of local neurons: {} vs {}", fired.size(), number_local_neurons);
        RelearnException::check(disable_flags.size() == number_local_neurons, "SynapticInputCalculator::update_input: Size of disable_flags did not match number of local neurons: {} vs {}", disable_flags.size(), number_local_neurons);

        std::fill(raw_ex_input.begin(), raw_ex_input.end(), 0.0);
        std::fill(raw_inh_input.begin(), raw_inh_input.end(), 0.0);

        fired_status_comm->set_local_fired_status(step, fired);
        fired_status_comm->exchange_fired_status(step);

        update_synaptic_input(fired);
    }

    /**
     * @brief Notifies this class and the input calculators that the plasticity has changed.
     *      Some might cache values, which than can be recalculated
     * @param step The current simulation step
     */
    void notify_of_plasticity_change(const step_type step) {
        fired_status_comm->notify_of_plasticity_change(step);
    }

    /**
     * @brief Returns the calculated synaptic input for the given neuron. Changes after calls to update_input(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The synaptic input for the given neuron
     */
    [[nodiscard]] double get_synaptic_input(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return synaptic_input[local_neuron_id];
    }

    /**
     * @brief Returns the calculated synaptic input for all. Changes after calls to update_input(...)
     * @return The synaptic input for all neurons
     */
    [[nodiscard]] std::span<const double> get_synaptic_input() const noexcept {
        return synaptic_input;
    }

    /**
     * @brief Returns the communicator which is used for synchronizing the fired status across multiple MPI ranks
     * @return The fired status communicator
     */
    [[nodiscard]] const std::unique_ptr<FiredStatusCommunicator>& get_fired_status_communicator() const {
        return fired_status_comm;
    }

    /**
     * @brief Returns the synapse conductance (The factor by which the input of a neighboring spiking neuron is weighted)
     * @return The synapse conductance
     */
    [[nodiscard]] double get_synapse_conductance() const noexcept {
        return synapse_conductance;
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
        return {
            Parameter<double>{ "synapse_conductance", synapse_conductance, min_conductance, max_conductance },
        };
    }

    static constexpr double default_conductance{ 0.03 };

    static constexpr double min_conductance{ 0.0 };
    static constexpr double max_conductance{ 10.0 };

protected:
    /**
     * @brief This hook needs to update this->synaptic_input for every neuron. Can make use of this->fired_status_comm
     * @param fired If the local neurons fired
     */
    virtual void update_synaptic_input(std::span<const FiredStatus> fired) = 0;

    /**
     * @brief Sets the synaptic input for the given neuron
     * @param neuron_id The local neuron
     * @param value The new synaptic input
     * @exception Throws a RelearnException if the neuron_id is to large
     */
    void set_synaptic_input(const number_neurons_type neuron_id, const double value) {
        RelearnException::check(neuron_id < number_local_neurons, "SynapticInputCalculator::set_synaptic_input: neuron_id was too large: {} vs {}", neuron_id, number_local_neurons);
        synaptic_input[neuron_id] = value;
    }

    /**
     * @brief Sets the synaptic input for all neurons
     * @param value The new background activity
     */
    void set_synaptic_input(double value) noexcept;

    [[nodiscard]] double get_local_and_distant_synaptic_input(const std::span<const FiredStatus> fired, const NeuronID& neuron_id);

protected:
    std::shared_ptr<NeuronsExtraInfo> extra_infos{};
    std::shared_ptr<NetworkGraph> network_graph{};

private:
    number_neurons_type number_local_neurons{};

    double synapse_conductance{ default_conductance }; // Proportionality factor for synapses in Hz

    std::vector<double> synaptic_input{};

    std::vector<double> raw_ex_input{};
    std::vector<double> raw_inh_input{};

    std::unique_ptr<FiredStatusCommunicator> fired_status_comm{};
};