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
#include "neurons/enums/FiredStatus.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"

#include "fmt/ostream.h"

#include <vector>

class NetworkGraph;

/**
 * This enums lists all types of fired status communicator
 */
enum class FiredStatusCommunicatorType : char {
    Map,
};

/**
 * @brief Pretty-prints the fired status communicator type to the chosen stream
 * @param out The stream to which to print the fired status communicator
 * @param element_type The fired status communicator to print
 * @return The argument out, now altered with the fired status communicator
 */
inline std::ostream& operator<<(std::ostream& out, const FiredStatusCommunicatorType& calculator_type) {
    if (calculator_type == FiredStatusCommunicatorType::Map) {
        return out << "Map";
    }

    return out;
}

template <>
struct fmt::formatter<FiredStatusCommunicatorType> : ostream_formatter { };

/**
 * This class provides a virtual interface for exchanging the NeuronID of those that fired in the simulation step.
 */
class FiredStatusCommunicator {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new object with the given number of ranks and local neurons (mainly used for pre-allocating memory)
     * @param number_ranks The number of MPI ranks
     * @exception Throws a RelearnException if number_ranks <= 0
     */
    FiredStatusCommunicator(const size_t number_ranks)
        : number_ranks(number_ranks) {
        RelearnException::check(number_ranks > 0, "FiredStatusCommunicator::FiredStatusCommunicator: number_ranks is too small: {}", number_ranks);
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    virtual void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_local_neurons == 0, "FiredStatusCommunicator::init: Was already initialized");
        RelearnException::check(number_neurons > 0, "FiredStatusCommunicator::init: Cannot initialize with 0 neurons");

        number_local_neurons = number_neurons;
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    virtual void create_neurons(const number_neurons_type creation_count) {
        RelearnException::check(number_local_neurons > 0, "FiredStatusCommunicator::create_neurons: Was not previously initialized");
        RelearnException::check(creation_count > 0, "FiredStatusCommunicator::create_neurons: Cannot create 0 neurons");

        const auto old_size = number_local_neurons;
        const auto new_size = old_size + creation_count;

        number_local_neurons = new_size;
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] virtual std::unique_ptr<FiredStatusCommunicator> clone() const = 0;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info.operator bool();
        RelearnException::check(is_filled, "FiredStatusCommunicator::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> new_network_graph) {
        const auto is_filled = new_network_graph.operator bool();
        RelearnException::check(is_filled, "FiredStatusCommunicator::set_network_graph: new_network_graph is empty");
        network_graph = std::move(new_network_graph);
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     * @param step The current update step
     * @param fired_status The current fired status of the neurons
     * @exception Can throw a RelearnException
     */
    virtual void set_local_fired_status(step_type step, std::span<const FiredStatus> fired_status) = 0;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    virtual void exchange_fired_status(step_type step) = 0;

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Can throw a RelearnException
     */
    [[nodiscard]] virtual bool contains(MPIRank rank, NeuronID neuron_id) const = 0;

    /**
     * @brief Notifies this class and the input calculators that the plasticity has changed.
     *      Some might cache values, which than can be recalculated
     * @param step The current simulation step
     */
    virtual void notify_of_plasticity_change(const step_type step) {
    }

    /**
     * @brief Returns the number of MPI ranks
     * @return The number of MPI ranks
     */
    [[nodiscard]] size_t get_number_ranks() const noexcept {
        return number_ranks;
    }

    /**
     * @brief Returns the number of local neurons
     * @return The number of local neurons
     */
    [[nodiscard]] number_neurons_type get_number_local_neurons() const noexcept {
        return number_local_neurons;
    }

    virtual ~FiredStatusCommunicator() = default;

protected:
    std::shared_ptr<NeuronsExtraInfo> extra_infos{};
    std::shared_ptr<NetworkGraph> network_graph{};

private:
    size_t number_ranks{ 0 };
    number_neurons_type number_local_neurons{ 0 };
};
