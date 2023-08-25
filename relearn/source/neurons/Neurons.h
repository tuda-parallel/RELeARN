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

#include "Config.h"
#include "Types.h"
#include "algorithm/Algorithm.h"
#include "models/NeuronModels.h"
#include "models/SynapticElements.h"
#include "mpi/CommunicationMap.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/enums/ElementType.h"
#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SignalType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"
#include "util/StatisticalMeasures.h"
#include "util/ranges/Functional.hpp"

#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <range/v3/functional/arithmetic.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

class AreaMonitor;
class Essentials;
class LocalAreaTranslator;
class NetworkGraph;
class NeuronMonitor;
class Octree;
class Partition;

/**
 * This class gathers all information for the neurons and provides the primary interface for the simulation
 */
class Neurons {
    friend class NeuronMonitor;
    friend class AreaMonitor;

public:
    using step_type = RelearnTypes::step_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    using Axons = SynapticElements;
    using DendritesExcitatory = SynapticElements;
    using DendritesInhibitory = SynapticElements;

    /**
     * @brief Creates a new object with the passed Partition, NeuronModel, Axons, DendritesExc, and DendritesInh
     * @param partition The partition, is only used for printing, must not be empty
     * @param model_ptr The electrical model for the neurons, must not be empty
     * @param calculator_ptr The calcium calculator, must not be empty
     * @param network The network graph for the connections, must not be empty
     * @param axons_ptr The model for the axons, must not be empty
     * @param dendrites_ex_ptr The model for the excitatory dendrites, must not be empty
     * @param dendrites_in_ptr The model for the inhibitory dendrites, must not be empty
     * @exception Throws a RelearnException if any of the pointers is empty
     */
    Neurons(std::shared_ptr<Partition> partition,
        std::unique_ptr<NeuronModel> model_ptr,
        std::unique_ptr<CalciumCalculator> calculator_ptr,
        std::shared_ptr<NetworkGraph> network_graph,
        std::shared_ptr<Axons> axons_ptr,
        std::shared_ptr<DendritesExcitatory> dendrites_ex_ptr,
        std::shared_ptr<DendritesInhibitory> dendrites_in_ptr,
        std::unique_ptr<SynapseDeletionFinder> synapse_del_ptr)
        : partition(std::move(partition))
        , neuron_model(std::move(model_ptr))
        , calcium_calculator(std::move(calculator_ptr))
        , network_graph(std::move(network_graph))
        , axons(std::move(axons_ptr))
        , dendrites_exc(std::move(dendrites_ex_ptr))
        , dendrites_inh(std::move(dendrites_in_ptr))
        , synapse_deletion_finder(std::move(synapse_del_ptr)) {

        const bool all_filled = this->partition && this->network_graph && neuron_model && calcium_calculator && axons && dendrites_exc && dendrites_inh && synapse_deletion_finder;
        RelearnException::check(all_filled, "Neurons::Neurons: Neurons was constructed with some null arguments");
    }

    ~Neurons() = default;

    Neurons(const Neurons& other) = delete;
    Neurons(Neurons&& other) = default;

    Neurons& operator=(const Neurons& other) = delete;
    Neurons& operator=(Neurons&& other) = default;

    /**
     * @brief Initializes this class and all models with number_neurons, i.e.,
     *      (a) Initializes the electrical model
     *      (b) Initializes the extra infos
     *      (c) Initializes the synaptic models
     *      (d) Enables all neurons
     *      (e) Calculates if the neurons fired once to initialize the calcium values to beta or 0.0
     * @param number_neurons The number of local neurons
     * @param target_calcium_values The target calcium values for the local neurons
     * @param initial_calcium_values The initial calcium values for the local neurons
     * @exception Throws a RelearnException if something unexpected happened
     */
    void init(number_neurons_type number_neurons);

    /**
     * Returns the algorithm that calculates to which neuron a neuron connects during the plasticity update
     * @return The algorithm
     */
    [[nodiscard]] const std::shared_ptr<Algorithm>& get_algorithm() const {
        return algorithm;
    }

    /**
     * @brief Sets the octree in which the neurons are stored
     * @param octree The octree
     */
    void set_octree(std::shared_ptr<Octree> octree) noexcept {
        global_tree = std::move(octree);
    }

    /**
     * @brief Sets the algorithm that calculates to which neuron a neuron connects during the plasticity update
     * @param algorithm_ptr The pointer to the algorithm
     */
    void set_algorithm(std::shared_ptr<Algorithm> algorithm_ptr) noexcept {
        algorithm = std::move(algorithm_ptr);
    }

    /**
     * @brief Sets the area translator that translates between the local area id on the current mpi rank and its area name
     * @param local_area_translator the local area translator for this mpi rank
     */
    void set_local_area_translator(std::shared_ptr<LocalAreaTranslator> local_area_translator) {
        this->local_area_translator = std::move(local_area_translator);
    }

    /**
     * @brief Neurons that are static are only allowed to have static connections. Plastic connections cannot be added during the simulation. This method marks the given neurons as static
     * @param static_neurons List of neuron ids that will be marked as static
     * @throws RelearnException When a static neuron is loaded with a plastic connection
     */
    void set_static_neurons(const std::span<const NeuronID> static_neurons) {
        extra_info->set_static_neurons(static_neurons);

        for (const auto neuron_id : static_neurons) {
            const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(neuron_id);
            RelearnException::check(distant_out_edges.empty(), "Plastic connection from a static neuron is forbidden. {} (static)  -> ?", neuron_id);

            const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(neuron_id);
            RelearnException::check(local_out_edges.empty(), "Plastic connection from a static neuron is forbidden. {} (static)  -> ?", neuron_id);

            const auto& [distant_in_edges, _3] = network_graph->get_distant_in_edges(neuron_id);
            RelearnException::check(distant_in_edges.empty(), "Plastic connection from a static neuron is forbidden. ? -> {} (static)", neuron_id);

            const auto& [local_in_edges, _4] = network_graph->get_local_in_edges(neuron_id);
            RelearnException::check(local_in_edges.empty(), "Plastic connection from a static neuron is forbidden. ? -> {} (static)", neuron_id);
        }
    }

    /**
     * @brief Returns the model parameters for the specified synaptic elements
     * @param element_type The element type
     * @param signal_type The signal type, only relevant if element_type == dendrites
     * @return The model parameters for the specified synaptic elements
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter(const ElementType element_type, const SignalType signal_type) {
        if (element_type == ElementType::Axon) {
            return axons->get_parameter();
        }

        if (signal_type == SignalType::Excitatory) {
            return dendrites_exc->get_parameter();
        }

        return dendrites_inh->get_parameter();
    }

    /**
     * @brief Returns the number of neurons in this object
     * @return The number of neurons in this object
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        return number_neurons;
    }

    /**
     * @brief Sets the positions in the extra infos
     * @param names The positions
     * @exception Throws the same RelearnException as NeuronsExtraInfo::set_positions
     */
    void set_positions(std::vector<NeuronsExtraInfo::position_type> pos) {
        extra_info->set_positions(std::move(pos));
    }

    /**
     * @brief Returns the area translate that translates between the local area id on the current mpi rank and its area name
     * @return the local area translator
     */
    [[nodiscard]] const std::shared_ptr<LocalAreaTranslator> get_local_area_translator() const {
        return local_area_translator;
    }

    /**
     * @brief Returns a constant reference to the extra information
     * @return The extra information for the neurons
     */
    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_extra_info() const noexcept {
        return extra_info;
    }

    /**
     * @brief Returns a constant reference to the neuron model
     * @return The neuron model for the neurons
     */
    [[nodiscard]] const std::shared_ptr<NeuronModel>& get_neuron_model() const noexcept {
        return neuron_model;
    }

    /**
     * @brief Returns the current calcium value of the neuron
     * @param neuron_id Local neuron id
     * @return Calcium of the neuron
     */
    [[nodiscard]] double get_calcium(const NeuronID neuron_id) const {
        return calcium_calculator->get_calcium()[neuron_id.get_neuron_id()];
    }

    /**
     * @brief Sets the signal types in the extra infos
     * @param names The signal types
     * @exception Throws the same RelearnException as NeuronsExtraInfo::set_signal_types
     */
    void set_signal_types(std::vector<SignalType> signal_types) {
        axons->set_signal_types(std::move(signal_types));
    }

    /**
     * @brief Manually sets the fired status of the neurons
     * @param fired The fired status of the neurons
     * @exception Throws a RelearnException if fired.size() is not equal to the number of local neurons
     */
    void set_fired(const std::span<const FiredStatus> fired) {
        RelearnException::check(fired.size() == number_neurons, "Neurons::set_fired: The sizes didn't match: {} vs {}", fired.size(), number_neurons);

        for (const auto neuron_id : NeuronID::range(number_neurons)) {
            neuron_model->set_fired(neuron_id, fired[neuron_id.get_neuron_id()]);
        }
    }

    /**
     * @brief Returns a constant reference to the axon model
     *      The reference is never invalidated
     * @return A constant reference to the axon model
     */
    [[nodiscard]] const Axons& get_axons() const noexcept {
        return *axons;
    }

    /**
     * @brief Returns a constant reference to the excitatory dendrites model
     *      The reference is never invalidated
     * @return A constant reference to the excitatory dendrites model
     */
    [[nodiscard]] const DendritesExcitatory& get_dendrites_exc() const noexcept {
        return *dendrites_exc;
    }

    /**
     * @brief Returns a constant reference to the inhibitory dendrites model
     *      The reference is never invalidated
     * @return A constant reference to the inhibitory dendrites model
     */
    [[nodiscard]] const DendritesInhibitory& get_dendrites_inh() const noexcept {
        return *dendrites_inh;
    }

    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] const std::span<const UpdateStatus> get_disable_flags() const noexcept {
        return extra_info->get_disable_flags();
    }

    /**
     * @brief Initializes the synaptic elements with respect to the network graph, i.e.,
     *      adds the synapses from the network graph as connected counts to the synaptic elements models
     */
    void init_synaptic_elements(const PlasticLocalSynapses& local_synapses_plastic, const PlasticDistantInSynapses& in_synapses_plastic, const PlasticDistantOutSynapses& out_synapses_plastic);

    /**
     * @brief Disables all neurons with specified ids
     *      If a neuron is already disabled, nothing happens for that one
     *      Otherwise, also deletes all synapses from the disabled neurons
     *      Returns a CommunicationMap containing the mpi requests for deleting distant connections on other ranks to the disabled neurons on this rank.
     * @param step The current simulation step
     * @exception Throws RelearnExceptions if something unexpected happens
     * @return Pair of number of local synapse deletion and requests for deletions on other ranks
     */
    std::pair<size_t, CommunicationMap<SynapseDeletionRequest>> disable_neurons(step_type step, std::span<const NeuronID> local_neuron_ids, int num_ranks);

    /**
     * @brief Enables all neurons with specified ids
     *      If a neuron is already enabled, nothing happens for that one
     * @exception Throws RelearnExceptions if something unexpected happens
     */
    void enable_neurons(const std::span<const NeuronID> neuron_ids) {
        extra_info->set_enabled_neurons(neuron_ids);
    }

    /**
     * @brief Creates creation_count many new neurons with default values
     *      (a) Creates neurons in the electrical model
     *      (b) Creates neurons in the extra infos
     *      (c) Creates neurons in the synaptic models
     *      (d) Enables all created neurons
     *      (e) Calculates if the neurons fired once to initialize the calcium values to beta or 0.0
     *      (f) Inserts the newly created neurons into the octree
     * @param creation_count The number of newly created neurons
     * @exception Throws a RelearnException if something unexpected happens
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Calls update_electrical_activity from the electrical model with the stored network graph,
     *      and updates the calcium values afterwards
     * @exception Throws a RelearnException if something unexpected happens
     */
    void update_electrical_activity(step_type step);

    /**
     * @brief Updates the delta of the synaptic elements for (1) axons, (2) excitatory dendrites, (3) inhibitory dendrites
     * @exception Throws a RelearnException if something unexpected happens
     */
    void update_number_synaptic_elements_delta();

    /**
     * @brief Updates the plasticity by
     *      (1) Deleting superfluous synapses
     *      (2) Creating new synapses with the stored algorithm
     * @param step The current simulation step
     * @exception Throws a RelearnException if the network graph, the octree, or the algorithm is empty,
     *      or something unexpected happens
     * @return Returns a tuple with (1) the number of deleted synapses, and (2) the number of created synapses
     */
    [[nodiscard]] std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> update_connectivity(step_type step);

    /**
     * @brief Calculates the number vacant axons and dendrites (excitatory, inhibitory) and prints them to LogFiles::EventType::Sums
     *      Performs communication with MPI
     * @param step The current simulation step
     * @param sum_synapses_deleted The number of deleted synapses (locally)
     * @param sum_synapses_created The number of created synapses (locally)
     */
    void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(step_type step, std::uint64_t sum_axon_deleted, std::uint64_t sum_dendrites_deleted, std::uint64_t sum_synapses_created);

    /**
     * @brief Prints the overview of the neurons to LogFiles::EventType::NeuronsOverview
     *      Performs communication with MPI
     * @param step The current simulation step
     */
    void print_neurons_overview_to_log_file_on_rank_0(step_type step) const;

    /**
     * @brief Inserts the calcium statistics in the essentials
     *      Performs communication with MPI
     * @param essentials The essentials
     */
    void print_calcium_statistics_to_essentials(const std::unique_ptr<Essentials>& essentials);

    /**
     * @brief Inserts the calcium statistics in the essentials
     *      Performs communication with MPI
     * @param essentials The essentials
     */
    void print_synaptic_changes_to_essentials(const std::unique_ptr<Essentials>& essentials);

    /**
     * @brief Prints the network graph to LogFiles::EventType::Network. Stores current step in file name and log
     * @param step The current simulation step
     * @param with_prefix If the file name should contain the current step as prefix
     */
    void print_network_graph_to_log_file(step_type step, bool with_prefix) const;

    /**
     * @brief Prints the neuron positions to LogFiles::EventType::Positions
     */
    void print_positions_to_log_file();

    void print_area_mapping_to_log_file();

    /**
     * @brief Prints some overview to LogFiles::EventType::Cout
     */
    void print();

    /**
     * @brief Prints some algorithm overview to LogFiles::EventType::Cout
     */
    void print_info_for_algorithm();

    /**
     * @brief Prints the histogram of in edges for the local neurons at the current simulation step
     * @param current_step The current simulation step
     */
    void print_local_network_histogram(step_type current_step);

    /**
     * @brief Prints the calcium values for the local neurons at the current simulation step
     * @param current_step The current simulation step
     */
    void print_calcium_values_to_file(step_type current_step);

    void print_fire_rate_to_file(step_type current_step);

    /**
     * @brief Prints the synaptic inputs for the local neurons at the current simulation step
     * @param current_step The current simulation step
     */
    void print_synaptic_inputs_to_file(step_type current_step);

    /**
     * @brief Performs debug checks on the synaptic element models if Config::do_debug_checks
     * @exception Throws a RelearnException if a check fails
     */
    void debug_check_counts();

    /**
     * @brief Returns a statistical measure for the selected attribute, considering all MPI ranks.
     *      Performs communication across MPI processes
     * @param attribute The selected attribute of the neurons
     * @return The statistical measure across all MPI processes. All MPI processes have the same return value
     */
    [[nodiscard]] StatisticalMeasures get_statistics(NeuronAttribute attribute) const;

    /**
     * @brief Checks if the weights of the out-going connections match their signal type
     * @param network_graph Network graph with all connections of the current mpi rank
     * @param signal_types Vector of SignalTypes. Neuron i has signal_type[i]
     * @throws RelearnException If signal_type does not match weight
     */
    static void check_signal_types(const std::shared_ptr<NetworkGraph> network_graph, std::span<const SignalType> signal_types, MPIRank my_rank);

    /**
     * Processes the requests of other mpi ranks to delete distant synapses on this rank to disabled remote neurons
     * @param list The communication map
     * @param my_rank Current mpi rank
     * @return Number of deletions
     */
    [[nodiscard]] size_t delete_disabled_distant_synapses(const CommunicationMap<SynapseDeletionRequest>& list, MPIRank my_rank);

private:
    [[nodiscard]] StatisticalMeasures global_statistics(std::span<const double> local_values, MPIRank root) const;

    template <typename T>
    [[nodiscard]] StatisticalMeasures global_statistics_integral(const std::span<const T> local_values, const MPIRank root) const {
        auto values = local_values
            | ranges::views::transform(ranges::convert_to<double>{})
            | ranges::to_vector;
        return global_statistics(std::move(values), root);
    }

    [[nodiscard]] std::uint64_t create_synapses();

    number_neurons_type number_neurons = 0;

    PlasticLocalSynapses last_created_local_synapses{};
    PlasticDistantInSynapses last_created_in_synapses{};
    PlasticDistantOutSynapses last_created_out_synapses{};

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<LocalAreaTranslator> local_area_translator{};

    std::shared_ptr<Octree> global_tree{};
    std::shared_ptr<Algorithm> algorithm{};

    std::shared_ptr<NetworkGraph> network_graph{};

    std::shared_ptr<NeuronModel> neuron_model{};
    std::unique_ptr<CalciumCalculator> calcium_calculator{};

    std::shared_ptr<Axons> axons{};
    std::shared_ptr<DendritesExcitatory> dendrites_exc{};
    std::shared_ptr<DendritesInhibitory> dendrites_inh{};

    std::unique_ptr<SynapseDeletionFinder> synapse_deletion_finder{};

    std::shared_ptr<NeuronsExtraInfo> extra_info{ std::make_shared<NeuronsExtraInfo>() };
};
