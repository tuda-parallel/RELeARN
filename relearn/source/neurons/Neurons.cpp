/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Neurons.h"

#include "io/Event.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/NetworkGraph.h"
#include "neurons/enums/UpdateStatus.h"
#include "sim/Essentials.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "io/NeuronIO.h"
#include "util/NeuronID.h"
#include "util/Random.h"
#include "util/Timers.h"
#include "util/Utility.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

void Neurons::init(const number_neurons_type number_neurons) {
    RelearnException::check(this->number_neurons == 0, "Neurons::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "Neurons::init: number_neurons was 0");

    this->number_neurons = number_neurons;

    network_graph->init(number_neurons);

    neuron_model->set_network_graph(network_graph);
    neuron_model->set_extra_infos(extra_info);

    neuron_model->init(number_neurons);
    extra_info->init(number_neurons);

    axons->init(number_neurons);
    dendrites_exc->init(number_neurons);
    dendrites_inh->init(number_neurons);

    for (const auto& id : NeuronID::range(number_neurons)) {
        dendrites_exc->set_signal_type(id, SignalType::Excitatory);
        dendrites_inh->set_signal_type(id, SignalType::Inhibitory);
    }

    calcium_calculator->set_extra_infos(extra_info);
    calcium_calculator->init(number_neurons);

    axons->set_extra_infos(extra_info);
    dendrites_exc->set_extra_infos(extra_info);
    dendrites_inh->set_extra_infos(extra_info);

    synapse_deletion_finder->set_extra_infos(extra_info);
    synapse_deletion_finder->set_network_graph(network_graph);
}

void Neurons::init_synaptic_elements(const PlasticLocalSynapses& local_synapses_plastic, const PlasticDistantInSynapses& in_synapses_plastic, const PlasticDistantOutSynapses& out_synapses_plastic) {
    last_created_local_synapses = local_synapses_plastic;
    last_created_in_synapses = in_synapses_plastic;
    last_created_out_synapses = out_synapses_plastic;

    const auto& axons_counts = axons->get_grown_elements();
    const auto& dendrites_inh_counts = dendrites_inh->get_grown_elements();
    const auto& dendrites_exc_counts = dendrites_exc->get_grown_elements();

    for (const auto& id : NeuronID::range(number_neurons)) {
        const auto [axon_connections, _1] = network_graph->get_number_out_edges(id);
        const auto [dendrites_ex_connections, _2] = network_graph->get_number_excitatory_in_edges(id);
        const auto [dendrites_in_connections, _3] = network_graph->get_number_inhibitory_in_edges(id);

        axons->update_grown_elements(id, static_cast<double>(axon_connections));
        dendrites_exc->update_grown_elements(id, static_cast<double>(dendrites_ex_connections));
        dendrites_inh->update_grown_elements(id, static_cast<double>(dendrites_in_connections));

        axons->update_connected_elements(id, static_cast<int>(axon_connections));
        dendrites_exc->update_connected_elements(id, static_cast<int>(dendrites_ex_connections));
        dendrites_inh->update_connected_elements(id, static_cast<int>(dendrites_in_connections));

        const auto local_id = id.get_neuron_id();

        RelearnException::check(axons_counts[local_id] >= axons->get_connected_elements()[local_id],
            "Error is with: %d", local_id);
        RelearnException::check(dendrites_inh_counts[local_id] >= dendrites_inh->get_connected_elements()[local_id],
            "Error is with: %d", local_id);
        RelearnException::check(dendrites_exc_counts[local_id] >= dendrites_exc->get_connected_elements()[local_id],
            "Error is with: %d", local_id);
    }

    check_signal_types(network_graph, axons->get_signal_types(), MPIWrapper::get_my_rank());
}

void Neurons::check_signal_types(const std::shared_ptr<NetworkGraph> network_graph,
    const std::span<const SignalType> signal_types, const MPIRank my_rank) {
    for (const auto& neuron_id : NeuronID::range(signal_types.size())) {
        const auto& signal_type = signal_types[neuron_id.get_neuron_id()];

        const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(neuron_id);
        for (const auto& [tgt_rni, weight] : distant_out_edges) {
            RelearnException::check(SignalType::Excitatory == signal_type && weight > 0 || SignalType::Inhibitory == signal_type && weight < 0,
                "Neuron has outgoing connections not matching its signal type. {} {} -> {} {} {}",
                my_rank, neuron_id, tgt_rni, signal_type, weight);
        }

        const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(neuron_id);
        for (const auto& [tgt_rni, weight] : local_out_edges) {
            RelearnException::check(SignalType::Excitatory == signal_type && weight > 0 || SignalType::Inhibitory == signal_type && weight < 0,
                "Neuron has outgoing connections not matching its signal type. {} {} -> {} {} {}",
                my_rank, neuron_id, tgt_rni, signal_type, weight);
        }
    }
}

std::pair<size_t, CommunicationMap<SynapseDeletionRequest>> Neurons::disable_neurons(const step_type step, const std::span<const NeuronID> local_neuron_ids, const int num_ranks) {
    extra_info->set_disabled_neurons(local_neuron_ids);

    neuron_model->disable_neurons(local_neuron_ids);

    std::vector<unsigned int> deleted_axon_connections(number_neurons, 0);
    std::vector<unsigned int> deleted_dend_ex_connections(number_neurons, 0);
    std::vector<unsigned int> deleted_dend_in_connections(number_neurons, 0);

    size_t number_deleted_out_inh_edges_within = 0;
    size_t number_deleted_out_exc_edges_within = 0;

    size_t number_deleted_out_inh_edges_to_outside = 0;
    size_t number_deleted_out_exc_edges_to_outside = 0;

    size_t number_deleted_distant_out_axons = 0;
    size_t number_deleted_distant_in_exc = 0;
    size_t number_deleted_distant_in_inh = 0;

    const auto size_hint = std::min(number_neurons, number_neurons_type(num_ranks));
    CommunicationMap<SynapseDeletionRequest> synapse_deletion_requests_outgoing(num_ranks, size_hint);

    for (const auto& neuron_id : local_neuron_ids) {
        RelearnException::check(neuron_id.get_neuron_id() < number_neurons,
            "Neurons::disable_neurons: There was a too large id: {} vs {}", neuron_id,
            number_neurons);

        const auto [local_out_edges_ref, _1] = network_graph->get_local_out_edges(neuron_id);
        const auto [distant_out_edges_ref, _2] = network_graph->get_distant_out_edges(neuron_id);

        auto local_out_edges = local_out_edges_ref;
        auto distant_out_edges = distant_out_edges_ref;

        for (const auto& [target_neuron_id, weight] : local_out_edges) {
            network_graph->add_synapse(PlasticLocalSynapse(target_neuron_id, neuron_id, -weight));

            // Shall target_neuron_id also be disabled? Important: Do not remove synapse twice in this case
            const bool is_within = std::ranges::binary_search(local_neuron_ids, target_neuron_id);
            const auto local_target_neuron_id = target_neuron_id.get_neuron_id();

            if (weight > 0) {
                if (is_within) {
                    number_deleted_out_exc_edges_within++;
                    deleted_axon_connections[neuron_id.get_neuron_id()]++;
                    deleted_dend_ex_connections[local_target_neuron_id]++;

                } else {
                    deleted_dend_ex_connections[local_target_neuron_id]++;
                    number_deleted_out_exc_edges_to_outside++;
                }
            } else {
                if (is_within) {
                    number_deleted_out_inh_edges_within++;
                    deleted_axon_connections[neuron_id.get_neuron_id()]++;
                    deleted_dend_in_connections[local_target_neuron_id]++;

                } else {
                    deleted_dend_in_connections[local_target_neuron_id]++;
                    number_deleted_out_inh_edges_to_outside++;
                }
            }
        }

        for (const auto& [target_neuron_id, weight] : distant_out_edges) {
            network_graph->add_synapse(PlasticDistantOutSynapse(target_neuron_id, neuron_id, -weight));
            deleted_axon_connections[neuron_id.get_neuron_id()]++;
            number_deleted_distant_out_axons++;
            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            synapse_deletion_requests_outgoing.append(target_neuron_id.get_rank(), { neuron_id, target_neuron_id.get_neuron_id(), ElementType::Axon, signal_type });
        }
    }

    size_t number_deleted_in_edges_from_outside = 0;

    for (const auto& neuron_id : local_neuron_ids) {
        const auto [local_in_edges_ref, _1] = network_graph->get_local_in_edges(neuron_id);
        const auto [distant_in_edges_ref, _2] = network_graph->get_distant_in_edges(neuron_id);

        auto local_in_edges = local_in_edges_ref;
        auto distant_in_edges = distant_in_edges_ref;

        for (const auto& [source_neuron_id, weight] : local_in_edges) {
            network_graph->add_synapse(PlasticLocalSynapse(neuron_id, source_neuron_id, -weight));

            deleted_axon_connections[source_neuron_id.get_neuron_id()]++;

            const bool is_within = std::ranges::binary_search(local_neuron_ids, source_neuron_id);

            if (is_within) {
                RelearnException::fail(
                    "Neurons::disable_neurons: While disabling neurons, found a within-in-edge that has not been deleted");
            } else {
                number_deleted_in_edges_from_outside++;
            }
        }

        for (const auto& [source_neuron_id, weight] : distant_in_edges) {
            network_graph->add_synapse(PlasticDistantInSynapse(neuron_id, source_neuron_id, -weight));

            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            synapse_deletion_requests_outgoing.append(source_neuron_id.get_rank(), { neuron_id, source_neuron_id.get_neuron_id(), ElementType::Dendrite, signal_type });

            if (weight > 0) {
                deleted_dend_ex_connections[neuron_id.get_neuron_id()]++;
                number_deleted_distant_in_exc++;
            } else {
                deleted_dend_in_connections[neuron_id.get_neuron_id()]++;
                number_deleted_distant_in_inh++;
            }
        }
    }

    const auto number_deleted_edges_within = number_deleted_out_inh_edges_within + number_deleted_out_exc_edges_within;

    axons->update_after_deletion(deleted_axon_connections, local_neuron_ids);
    dendrites_exc->update_after_deletion(deleted_dend_ex_connections, local_neuron_ids);
    dendrites_inh->update_after_deletion(deleted_dend_in_connections, local_neuron_ids);

    neuron_model->notify_of_plasticity_change(step);

    LogFiles::print_message_rank(MPIRank::root_rank(),
        "Deleted {} in-edges with and ({}, {}) out-edges (exc., inh.) within the deleted portion",
        number_deleted_edges_within,
        number_deleted_out_exc_edges_within,
        number_deleted_out_inh_edges_within);

    LogFiles::print_message_rank(MPIRank::root_rank(),
        "Deleted {} in-edges and ({}, {}) out-edges  (exc., inh.) connecting to the outside",
        number_deleted_in_edges_from_outside,
        number_deleted_out_exc_edges_to_outside,
        number_deleted_out_inh_edges_to_outside);

    LogFiles::print_message_rank(MPIRank::root_rank(),
        "Deleted ({},{}) in-edges (exc., inh.) and {} out-edges connecting to the other ranks",
        number_deleted_distant_in_exc, number_deleted_distant_in_inh,
        number_deleted_distant_out_axons);

    LogFiles::print_message_rank(MPIRank::root_rank(),
        "Deleted {} in-edges and ({}, {}) out-edges (exc., inh.) altogether",
        number_deleted_edges_within + number_deleted_in_edges_from_outside,
        number_deleted_out_exc_edges_within + number_deleted_out_exc_edges_to_outside,
        number_deleted_out_inh_edges_within + number_deleted_out_inh_edges_to_outside);

    const auto deleted_connections = number_deleted_distant_out_axons + number_deleted_distant_in_inh + number_deleted_distant_in_exc
        + number_deleted_in_edges_from_outside + number_deleted_out_inh_edges_to_outside + number_deleted_out_exc_edges_to_outside
        + number_deleted_out_exc_edges_within + number_deleted_out_inh_edges_within;

    return std::make_pair(deleted_connections, synapse_deletion_requests_outgoing);
}

void Neurons::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(number_neurons > 0, "Neurons::create_neurons: Was not initialized");
    RelearnException::check(creation_count > 0, "Neurons::create_neurons: Cannot create 0 neurons");

    const auto current_size = number_neurons;
    const auto new_size = current_size + creation_count;

    local_area_translator->create_neurons(creation_count);
    neuron_model->create_neurons(creation_count);
    calcium_calculator->create_neurons(creation_count);
    extra_info->create_neurons(creation_count);

    network_graph->create_neurons(creation_count);

    axons->create_neurons(creation_count);
    dendrites_exc->create_neurons(creation_count);
    dendrites_inh->create_neurons(creation_count);

    for (const auto& neuron_id : NeuronID::range(current_size, new_size)) {
        dendrites_exc->set_signal_type(neuron_id, SignalType::Excitatory);
        dendrites_inh->set_signal_type(neuron_id, SignalType::Inhibitory);

        const auto& pos = extra_info->get_position(neuron_id);
        global_tree->insert(pos, neuron_id);
    }

    global_tree->initializes_leaf_nodes(new_size);

    number_neurons = new_size;
}

void Neurons::update_electrical_activity(const step_type step) {
    neuron_model->update_electrical_activity(step);

    const auto& fired = neuron_model->get_fired();
    calcium_calculator->update_calcium(step, fired);

    Timers::start(TimerRegion::CALC_CALCIUM_EXTREME_VALUES);
    const auto& calcium_values = calcium_calculator->get_calcium();
    const auto& current_min_id = calcium_calculator->get_current_minimum().get_neuron_id();
    const auto& current_max_id = calcium_calculator->get_current_maximum().get_neuron_id();

    LogFiles::write_to_file(LogFiles::EventType::ExtremeCalciumValues, false, "{};{:.6f};{};{:.6f}",
        current_min_id, calcium_values[current_min_id], current_max_id, calcium_values[current_max_id]);
    Timers::stop_and_add(TimerRegion::CALC_CALCIUM_EXTREME_VALUES);
}

void Neurons::update_number_synaptic_elements_delta() {
    const auto& calcium = calcium_calculator->get_calcium();
    const auto& target_calcium = calcium_calculator->get_target_calcium();

    axons->update_number_elements_delta(calcium, target_calcium);
    dendrites_exc->update_number_elements_delta(calcium, target_calcium);
    dendrites_inh->update_number_elements_delta(calcium, target_calcium);
}

StatisticalMeasures Neurons::global_statistics(const std::span<const double> local_values, const MPIRank root) const {
    const auto disable_flags = extra_info->get_disable_flags();
    const auto [d_my_min, d_my_max, d_my_acc, d_num_values] = Util::min_max_acc(local_values, extra_info);
    const double my_avg = d_my_acc / static_cast<double>(d_num_values);

    const double d_min = MPIWrapper::reduce(d_my_min, MPIWrapper::ReduceFunction::Min, root);
    const double d_max = MPIWrapper::reduce(d_my_max, MPIWrapper::ReduceFunction::Max, root);

    const auto num_values = static_cast<double>(MPIWrapper::all_reduce_uint64(d_num_values,
        MPIWrapper::ReduceFunction::Sum));

    // Get global avg at all ranks (needed for variance)
    const double avg = MPIWrapper::all_reduce_double(my_avg, MPIWrapper::ReduceFunction::Sum) / MPIWrapper::get_num_ranks();

    /**
     * Calc variance
     */
    const auto my_var = ranges::accumulate(NeuronID::range_id(number_neurons)
                                | ranges::views::filter(not_equal_to(UpdateStatus::Disabled), lookup(disable_flags))
                                | ranges::views::transform([&local_values, avg](const auto& neuron_id) {
                                      const auto val = local_values[neuron_id] - avg;
                                      return val * val;
                                  }),
                            0.0)
        / num_values;

    // Get global variance at rank "root"
    const double var = MPIWrapper::reduce(my_var, MPIWrapper::ReduceFunction::Sum, root);

    // Calc standard deviation
    const double std = std::sqrt(var);

    return { d_min, d_max, avg, var, std };
}

std::uint64_t Neurons::create_synapses() {
    const auto my_rank = MPIWrapper::get_my_rank();

    Event::create_and_print_duration_begin_event("Neurons::update_octree", { EventCategory::mpi, EventCategory::calculation }, {}, true);
    // Lock local RMA memory for local stores and make them visible afterwards
    MPIWrapper::lock_window(MPIWindow::Window::Octree, my_rank, MPI_Locktype::Exclusive);
    algorithm->update_octree();
    Event::create_and_print_duration_end_event(true);
    MPIWrapper::unlock_window(MPIWindow::Window::Octree, my_rank);

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    MPIWrapper::barrier();
    // MPIWrapper::sync_window(MPIWindow::Octree);

    MPIWrapper::start_measuring_communication();
    // Delegate the creation of new synapses to the algorithm
    Event::create_and_print_duration_begin_event("Neurons::update_connectivity", { EventCategory::mpi, EventCategory::calculation }, {}, true);
    MPIWrapper::lock_window_all(MPIWindow::Octree);
    const auto& [local_synapses, distant_in_synapses, distant_out_synapses]
        = algorithm->update_connectivity(number_neurons);
    Event::create_and_print_duration_end_event(true);

    MPIWrapper::unlock_window_all(MPIWindow::Octree);
    // MPIWrapper::sync_window(MPIWindow::Octree);

    MPIWrapper::stop_measureing_communication();

    // Update the network graph all at once
    Timers::start(TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);
    Event::create_and_print_duration_begin_event("Neurons::add_edges", { EventCategory::mpi, EventCategory::calculation }, {}, true);
    network_graph->add_edges(local_synapses, distant_in_synapses, distant_out_synapses);
    Event::create_and_print_duration_end_event(true);
    Timers::stop_and_add(TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);

    // The distant_out_synapses are counted on the ranks where they are in
    const auto num_synapses_created = local_synapses.size() + distant_in_synapses.size();

    last_created_local_synapses = std::move(local_synapses);
    last_created_in_synapses = std::move(distant_in_synapses);
    last_created_out_synapses = std::move(distant_out_synapses);

    return num_synapses_created;
}

void Neurons::debug_check_counts() {
    if (!Config::do_debug_checks) {
        return;
    }

    RelearnException::check(network_graph != nullptr,
        "Neurons::debug_check_counts: network_graph is nullptr");

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& grown_axons = axons->get_grown_elements();
    const auto& connected_axons = axons->get_connected_elements();
    const auto& grown_excitatory_dendrites = dendrites_exc->get_grown_elements();
    const auto& connected_excitatory_dendrites = dendrites_exc->get_connected_elements();
    const auto& grown_inhibitory_dendrites = dendrites_inh->get_grown_elements();
    const auto& connected_inhibitory_dendrites = dendrites_inh->get_connected_elements();

    for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        const auto vacant_axons = grown_axons[neuron_id] - connected_axons[neuron_id];
        const auto vacant_excitatory_dendrites = grown_excitatory_dendrites[neuron_id] - connected_excitatory_dendrites[neuron_id];
        const auto vacant_inhibitory_dendrites = grown_inhibitory_dendrites[neuron_id] - connected_inhibitory_dendrites[neuron_id];

        RelearnException::check(vacant_axons >= 0.0,
            "Neurons::debug_check_counts: {} has a weird number of vacant axons: {}", neuron_id,
            vacant_axons);
        RelearnException::check(vacant_excitatory_dendrites >= 0.0,
            "Neurons::debug_check_counts: {} has a weird number of vacant excitatory dendrites: {}",
            neuron_id, vacant_excitatory_dendrites);
        RelearnException::check(vacant_inhibitory_dendrites >= 0.0,
            "Neurons::debug_check_counts: {} has a weird number of vacant inhibitory dendrites: {}",
            neuron_id, vacant_inhibitory_dendrites);
    }

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        const auto connected_axons_neuron = connected_axons[local_neuron_id];
        const auto connected_excitatory_dendrites_neuron = connected_excitatory_dendrites[local_neuron_id];
        const auto connected_inhibitory_dendrites_neuron = connected_inhibitory_dendrites[local_neuron_id];

        const auto [number_out_edges, _1] = network_graph->get_number_out_edges(neuron_id);
        const auto [number_excitatory_in_edges, _2] = network_graph->get_number_excitatory_in_edges(neuron_id);
        const auto [number_inhibitory_in_edges, _3] = network_graph->get_number_inhibitory_in_edges(neuron_id);

        RelearnException::check(connected_axons_neuron == number_out_edges,
            "Neurons::debug_check_counts: Neuron {} has {} axons but {} out edges (rank {})",
            neuron_id, connected_axons_neuron, number_out_edges, my_rank);

        RelearnException::check(connected_excitatory_dendrites_neuron == number_excitatory_in_edges,
            "Neurons::debug_check_counts: Neuron {} has {} excitatory dendrites but {} excitatory in edges (rank {})",
            neuron_id, connected_excitatory_dendrites_neuron, number_excitatory_in_edges, my_rank);

        RelearnException::check(connected_inhibitory_dendrites_neuron == number_inhibitory_in_edges,
            "Neurons::debug_check_counts: Neuron {} has {} inhibitory dendrites but {} inhibitory in edges (rank {})",
            neuron_id, connected_inhibitory_dendrites_neuron, number_inhibitory_in_edges, my_rank);
    }
}

StatisticalMeasures Neurons::get_statistics(const NeuronAttribute attribute) const {
    switch (attribute) {
    case NeuronAttribute::Calcium:
        return global_statistics(calcium_calculator->get_calcium(), MPIRank::root_rank());

    case NeuronAttribute::X:
        return global_statistics(neuron_model->get_x(), MPIRank::root_rank());

    case NeuronAttribute::Fired:
        return global_statistics_integral(neuron_model->get_fired(), MPIRank::root_rank());

    case NeuronAttribute::SynapticInput:
        return global_statistics(neuron_model->get_synaptic_input(), MPIRank::root_rank());

    case NeuronAttribute::BackgroundActivity:
        return global_statistics(neuron_model->get_background_activity(), MPIRank::root_rank());

    case NeuronAttribute::Axons:
        return global_statistics(axons->get_grown_elements(), MPIRank::root_rank());

    case NeuronAttribute::AxonsConnected:
        return global_statistics_integral(axons->get_connected_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesExcitatory:
        return global_statistics(dendrites_exc->get_grown_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesExcitatoryConnected:
        return global_statistics_integral(dendrites_exc->get_connected_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesInhibitory:
        return global_statistics(dendrites_inh->get_grown_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesInhibitoryConnected:
        return global_statistics_integral(dendrites_inh->get_connected_elements(), MPIRank::root_rank());
    }

    RelearnException::fail("Neurons::get_statistics: Got an unsupported attribute: {}", static_cast<int>(attribute));

    return {};
}

std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> Neurons::update_connectivity(const step_type step) {
    RelearnException::check(network_graph != nullptr, "Network graph is nullptr");
    RelearnException::check(global_tree != nullptr, "Global octree is nullptr");
    RelearnException::check(algorithm != nullptr, "Algorithm is nullptr");

    extra_info->publish_fire_history();

    debug_check_counts();
    network_graph->debug_check();
    const auto& [num_axons_deleted, num_dendrites_deleted] = synapse_deletion_finder->delete_synapses();
    debug_check_counts();
    network_graph->debug_check();
    size_t num_synapses_created = create_synapses();
    debug_check_counts();
    network_graph->debug_check();

    neuron_model->notify_of_plasticity_change(step);

    return { num_axons_deleted, num_dendrites_deleted, num_synapses_created };
}

size_t Neurons::delete_disabled_distant_synapses(const CommunicationMap<SynapseDeletionRequest>& list, const MPIRank my_rank) {
    size_t num_synapses_deleted = 0;

    const auto& disable_flags = extra_info->get_disable_flags();

    for (const auto& [other_rank, requests] : list) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            if (disable_flags[my_neuron_id.get_neuron_id()] != UpdateStatus::Enabled) {
                continue;
            }

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                RelearnException::fail("Local synapse deletion is not allowed via mpi");
            }

            if (ElementType::Dendrite == element_type) {
                const auto& [out_edges, _1] = network_graph->get_distant_out_edges(my_neuron_id);
                RelearnTypes::plastic_synapse_weight weight = 0;
                for (const auto& [target, edge_weight] : out_edges) {
                    if (target.get_rank() == other_rank && target.get_neuron_id() == other_neuron_id) {
                        weight = edge_weight;
                        break;
                    }
                }
                RelearnException::check(weight != 0, "Couldnot find the weight of the connection");
                network_graph->add_synapse(
                    PlasticDistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, -weight));
            } else {
                const auto& [in_edges, _2] = network_graph->get_distant_in_edges(my_neuron_id);
                RelearnTypes::plastic_synapse_weight weight = 0;
                for (const auto& [source, edge_weight] : in_edges) {
                    if (source.get_rank() == other_rank && source.get_neuron_id() == other_neuron_id) {
                        weight = edge_weight;
                        break;
                    }
                }
                network_graph->add_synapse(
                    PlasticDistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), -weight));
            }

            if (ElementType::Dendrite == element_type) {
                axons->update_connected_elements(my_neuron_id, -1);
                continue;
            }

            if (SignalType::Excitatory == signal_type) {
                dendrites_exc->update_connected_elements(my_neuron_id, -1);
            } else {
                dendrites_inh->update_connected_elements(my_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(const step_type step,
    const std::uint64_t sum_axon_deleted,
    const std::uint64_t sum_dendrites_deleted,
    const std::uint64_t sum_synapses_created) {
    int64_t sum_axons_excitatory_counts = 0;
    int64_t sum_axons_excitatory_connected_counts = 0;
    int64_t sum_axons_inhibitory_counts = 0;
    int64_t sum_axons_inhibitory_connected_counts = 0;

    const auto& axon_counts = axons->get_grown_elements();
    const auto& axons_connected_counts = axons->get_connected_elements();
    const auto& axons_signal_types = axons->get_signal_types();

    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (SignalType::Excitatory == axons_signal_types[neuron_id]) {
            sum_axons_excitatory_counts += static_cast<int64_t>(axon_counts[neuron_id]);
            sum_axons_excitatory_connected_counts += static_cast<int64_t>(axons_connected_counts[neuron_id]);
        } else {
            sum_axons_inhibitory_counts += static_cast<int64_t>(axon_counts[neuron_id]);
            sum_axons_inhibitory_connected_counts += static_cast<int64_t>(axons_connected_counts[neuron_id]);
        }
    }

    int64_t sum_dendrites_excitatory_counts = 0;
    int64_t sum_dendrites_excitatory_connected_counts = 0;
    const auto& excitatory_dendrites_counts = dendrites_exc->get_grown_elements();
    const auto& excitatory_dendrites_connected_counts = dendrites_exc->get_connected_elements();
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        sum_dendrites_excitatory_counts += static_cast<int64_t>(excitatory_dendrites_counts[neuron_id]);
        sum_dendrites_excitatory_connected_counts += static_cast<int64_t>(excitatory_dendrites_connected_counts[neuron_id]);
    }

    int64_t sum_dendrites_inhibitory_counts = 0;
    int64_t sum_dendrites_inhibitory_connected_counts = 0;
    const auto& inhibitory_dendrites_counts = dendrites_inh->get_grown_elements();
    const auto& inhibitory_dendrites_connected_counts = dendrites_inh->get_connected_elements();
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        sum_dendrites_inhibitory_counts += static_cast<int64_t>(inhibitory_dendrites_counts[neuron_id]);
        sum_dendrites_inhibitory_connected_counts += static_cast<int64_t>(inhibitory_dendrites_connected_counts[neuron_id]);
    }

    int64_t sum_dends_exc_vacant = sum_dendrites_excitatory_counts - sum_dendrites_excitatory_connected_counts;
    int64_t sum_dends_inh_vacant = sum_dendrites_inhibitory_counts - sum_dendrites_inhibitory_connected_counts;

    int64_t sum_axons_exc_vacant = sum_axons_excitatory_counts - sum_axons_excitatory_connected_counts;
    int64_t sum_axons_inh_vacant = sum_axons_inhibitory_counts - sum_axons_inhibitory_connected_counts;

    // Get global sums at rank 0
    std::array<int64_t, 7> sums_local = { sum_axons_exc_vacant,
        sum_axons_inh_vacant,
        sum_dends_exc_vacant,
        sum_dends_inh_vacant,
        static_cast<int64_t>(sum_axon_deleted),
        static_cast<int64_t>(sum_dendrites_deleted),
        static_cast<int64_t>(sum_synapses_created) };

    std::array<int64_t, 7> sums_global = MPIWrapper::reduce(sums_local, MPIWrapper::ReduceFunction::Sum,
        MPIRank::root_rank());

    // Output data
    if (MPIRank::root_rank() == MPIWrapper::get_my_rank()) {
        const int cwidth = 20; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            LogFiles::write_to_file(LogFiles::EventType::Sums, false,
                "# SUMS OVER ALL NEURONS\n{1:{0}}{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}{7:{0}}{8:{0}}",
                cwidth,
                "# step",
                "Axons exc. (vacant)",
                "Axons inh. (vacant)",
                "Dends exc. (vacant)",
                "Dends inh. (vacant)",
                "Synapses (axons) deleted",
                "Synapses (dendrites) deleted",
                "Synapses created");
        }

        LogFiles::write_to_file(LogFiles::EventType::Sums, false,
            "{2:<{0}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}{9:<{0}}",
            cwidth,
            Constants::print_precision,
            step,
            sums_global[0],
            sums_global[1],
            sums_global[2],
            sums_global[3],
            sums_global[4] / 2,
            sums_global[5] / 2,
            sums_global[6] / 2);
    }
}

void Neurons::print_neurons_overview_to_log_file_on_rank_0(const step_type step) const {
    const StatisticalMeasures& calcium_statistics = get_statistics(NeuronAttribute::Calcium);
    const StatisticalMeasures& axons_statistics = get_statistics(NeuronAttribute::Axons);
    const StatisticalMeasures& axons_connected_statistics = get_statistics(NeuronAttribute::AxonsConnected);
    const StatisticalMeasures& dendrites_excitatory_statistics = get_statistics(NeuronAttribute::DendritesExcitatory);
    const StatisticalMeasures& dendrites_excitatory_connected_statistics = get_statistics(
        NeuronAttribute::DendritesExcitatoryConnected);

    if (MPIRank::root_rank() != MPIWrapper::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    const int cwidth = 20; // Column width

    // Write headers to file if not already done so
    if (0 == step) {
        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, false,
            "# ALL NEURONS\n{1:{0}}"
            "{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}"
            "{7:{0}}{8:{0}}{9:{0}}{10:{0}}{11:{0}}"
            "{12:{0}}{13:{0}}{14:{0}}{15:{0}}{16:{0}}"
            "{17:{0}}{18:{0}}{19:{0}}{20:{0}}{21:{0}}"
            "{22:{0}}{23:{0}}{24:{0}}{25:{0}}{26:{0}}",
            cwidth,
            "# step",
            "C (avg)",
            "C (min)",
            "C (max)",
            "C (var)",
            "C (std_dev)",
            "axons (avg)",
            "axons (min)",
            "axons (max)",
            "axons (var)",
            "axons (std_dev)",
            "axons.c (avg)",
            "axons.c (min)",
            "axons.c (max)",
            "axons.c (var)",
            "axons.c (std_dev)",
            "den.ex (avg)",
            "den.ex (min)",
            "den.ex (max)",
            "den.ex (var)",
            "den.ex (std_dev)",
            "den.ex.c (avg)",
            "den.ex.c (min)",
            "den.ex.c (max)",
            "den.ex.c (var)",
            "den.ex.c (std_dev)");

        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverviewCSV, false,
            "# step",
            "C (avg)",
            "C (min)",
            "C (max)",
            "C (var)",
            "C (std_dev)",
            "axons (avg)",
            "axons (min)",
            "axons (max)",
            "axons (var)",
            "axons (std_dev)",
            "axons.c (avg)",
            "axons.c (min)",
            "axons.c (max)",
            "axons.c (var)",
            "axons.c (std_dev)",
            "den.ex (avg)",
            "den.ex (min)",
            "den.ex (max)",
            "den.ex (var)",
            "den.ex (std_dev)",
            "den.ex.c (avg)",
            "den.ex.c (min)",
            "den.ex.c (max)",
            "den.ex.c (var)",
            "den.ex.c (std_dev)");
    }

    // Write data at step "step"
    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, false,
        "{2:<{0}}"
        "{3:<{0}.{1}f}{4:<{0}.{1}f}{5:<{0}.{1}f}{6:<{0}.{1}f}{7:<{0}.{1}f}"
        "{8:<{0}.{1}f}{9:<{0}.{1}f}{10:<{0}.{1}f}{11:<{0}.{1}f}{12:<{0}.{1}f}"
        "{13:<{0}.{1}f}{14:<{0}.{1}f}{15:<{0}.{1}f}{16:<{0}.{1}f}{17:<{0}.{1}f}"
        "{18:<{0}.{1}f}{19:<{0}.{1}f}{20:<{0}.{1}f}{21:<{0}.{1}f}{22:<{0}.{1}f}"
        "{23:<{0}.{1}f}{24:<{0}.{1}f}{25:<{0}.{1}f}{26:<{0}.{1}f}{27:<{0}.{1}f}",
        cwidth,
        Constants::print_precision,
        step,
        calcium_statistics.avg,
        calcium_statistics.min,
        calcium_statistics.max,
        calcium_statistics.var,
        calcium_statistics.std,
        axons_statistics.avg,
        axons_statistics.min,
        axons_statistics.max,
        axons_statistics.var,
        axons_statistics.std,
        axons_connected_statistics.avg,
        axons_connected_statistics.min,
        axons_connected_statistics.max,
        axons_connected_statistics.var,
        axons_connected_statistics.std,
        dendrites_excitatory_statistics.avg,
        dendrites_excitatory_statistics.min,
        dendrites_excitatory_statistics.max,
        dendrites_excitatory_statistics.var,
        dendrites_excitatory_statistics.std,
        dendrites_excitatory_connected_statistics.avg,
        dendrites_excitatory_connected_statistics.min,
        dendrites_excitatory_connected_statistics.max,
        dendrites_excitatory_connected_statistics.var,
        dendrites_excitatory_connected_statistics.std);

    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverviewCSV, false,
        "{};"
        "{};{};{};{};{};"
        "{};{};{};{};{};"
        "{};{};{};{};{};"
        "{};{};{};{};{};"
        "{};{};{};{};{}",
        step,
        calcium_statistics.avg,
        calcium_statistics.min,
        calcium_statistics.max,
        calcium_statistics.var,
        calcium_statistics.std,
        axons_statistics.avg,
        axons_statistics.min,
        axons_statistics.max,
        axons_statistics.var,
        axons_statistics.std,
        axons_connected_statistics.avg,
        axons_connected_statistics.min,
        axons_connected_statistics.max,
        axons_connected_statistics.var,
        axons_connected_statistics.std,
        dendrites_excitatory_statistics.avg,
        dendrites_excitatory_statistics.min,
        dendrites_excitatory_statistics.max,
        dendrites_excitatory_statistics.var,
        dendrites_excitatory_statistics.std,
        dendrites_excitatory_connected_statistics.avg,
        dendrites_excitatory_connected_statistics.min,
        dendrites_excitatory_connected_statistics.max,
        dendrites_excitatory_connected_statistics.var,
        dendrites_excitatory_connected_statistics.std);
}

void Neurons::print_calcium_statistics_to_essentials(const std::unique_ptr<Essentials>& essentials) {
    const auto& calcium = calcium_calculator->get_calcium();
    const StatisticalMeasures& calcium_statistics = global_statistics(calcium, MPIRank::root_rank());

    if (MPIRank::root_rank() != MPIWrapper::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    essentials->insert("Calcium-Minimum", calcium_statistics.min);
    essentials->insert("Calcium-Average", calcium_statistics.avg);
    essentials->insert("Calcium-Maximum", calcium_statistics.max);
}

void Neurons::print_synaptic_changes_to_essentials(const std::unique_ptr<Essentials>& essentials) {
    auto helper = [this, &essentials](const auto& synaptic_elements, std::string message) {
        const auto local_adds = synaptic_elements.get_total_additions();
        const auto local_dels = synaptic_elements.get_total_deletions();

        const auto global_adds = MPIWrapper::reduce(local_adds, MPIWrapper::ReduceFunction::Sum, MPIRank::root_rank());
        const auto global_dels = MPIWrapper::reduce(local_dels, MPIWrapper::ReduceFunction::Sum, MPIRank::root_rank());

        if (MPIRank::root_rank() == MPIWrapper::get_my_rank()) {
            essentials->insert(message + "Additions", global_adds);
            essentials->insert(message + "Deletions", global_dels);
        }
    };

    helper(*axons, "Axons-");
    helper(*dendrites_exc, "Dendrites-Excitatory-");
    helper(*dendrites_inh, "Dendrites-Inhibitory-");
}

void Neurons::print_network_graph_to_log_file(const step_type step, const bool with_prefix) const {
    std::string prefix = "";
    if (with_prefix) {
        prefix = "step_" + std::to_string(step) + "_";
    }
    LogFiles::save_and_open_new(LogFiles::EventType::InNetwork, prefix + "in_network", "network/");
    LogFiles::save_and_open_new(LogFiles::EventType::OutNetwork, prefix + "out_network", "network/");

    std::stringstream ss_in_network{};
    std::stringstream ss_out_network{};

    const auto& [plastic_distant_out, static_distant_out] = network_graph->get_all_distant_out_edges();
    const auto& [plastic_local_out, static_local_out] = network_graph->get_all_local_out_edges();

    const auto& [plastic_distant_in, static_distant_in] = network_graph->get_all_distant_in_edges();
    const auto& [plastic_local_in, static_local_in] = network_graph->get_all_local_in_edges();

    NeuronIO::write_out_synapses(static_local_out, static_distant_out, plastic_local_out, plastic_distant_out, MPIWrapper::get_my_rank(),
        partition->get_number_mpi_ranks(), partition->get_number_local_neurons(),
        partition->get_total_number_neurons(), ss_out_network, step);

    NeuronIO::write_in_synapses(static_local_in, static_distant_in, plastic_local_in, plastic_distant_in, MPIWrapper::get_my_rank(),
        partition->get_number_mpi_ranks(), partition->get_number_local_neurons(),
        partition->get_total_number_neurons(), ss_in_network, step);

    LogFiles::write_to_file(LogFiles::EventType::InNetwork, false, ss_in_network.str());
    LogFiles::write_to_file(LogFiles::EventType::OutNetwork, false, ss_out_network.str());
}

void Neurons::print_positions_to_log_file() {
    std::stringstream ss;
    NeuronIO::write_neurons_componentwise(NeuronID::range(number_neurons) | ranges::to_vector, extra_info->get_positions(), local_area_translator,
        axons->get_signal_types(), ss, partition->get_total_number_neurons(), partition->get_simulation_box_size(), partition->get_all_local_subdomain_boundaries());
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, ss.str());
    LogFiles::flush_file(LogFiles::EventType::Positions);
}

void Neurons::print_area_mapping_to_log_file() {
    std::stringstream ss;
    NeuronIO::write_area_names(ss, local_area_translator);
    LogFiles::write_to_file(LogFiles::EventType::AreaMapping, false, ss.str());
    LogFiles::flush_file(LogFiles::EventType::AreaMapping);
}

void Neurons::print() {
    const auto& calcium = calcium_calculator->get_calcium();

    // Column widths
    constexpr int cwidth_left = 6;
    constexpr int cwidth = 20;

    std::stringstream ss{};

    // Heading
    LogFiles::write_to_file(LogFiles::EventType::Cout, true,
        "{2:<{1}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}{9:<{0}}", cwidth, cwidth_left,
        "gid", "x", "AP", "refractory_time", "C", "A", "D_ex", "D_in");

    // Values
    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        LogFiles::write_to_file(LogFiles::EventType::Cout, true,
            "{3:<{1}}{4:<{0}.{2}f}{5:<{0}}{6:<{0}.{2}f}{7:<{0}.{2}f}{8:<{0}.{2}f}{9:<{0}.{2}f}{10:<{0}.{2}f}",
            cwidth, cwidth_left, Constants::print_precision, local_neuron_id,
            neuron_model->get_x(neuron_id), neuron_model->get_fired(neuron_id),
            neuron_model->get_secondary_variable(neuron_id), calcium[local_neuron_id],
            axons->get_grown_elements(neuron_id),
            dendrites_exc->get_grown_elements(neuron_id),
            dendrites_inh->get_grown_elements(neuron_id));
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_info_for_algorithm() {
    const auto& axons_counts = axons->get_grown_elements();
    const auto& dendrites_exc_counts = dendrites_exc->get_grown_elements();
    const auto& dendrites_inh_counts = dendrites_inh->get_grown_elements();

    const auto& axons_connected_counts = axons->get_connected_elements();
    const auto& dendrites_exc_connected_counts = dendrites_exc->get_connected_elements();
    const auto& dendrites_inh_connected_counts = dendrites_inh->get_connected_elements();

    // Column widths
    const int cwidth_small = 8;
    const int cwidth_medium = 16;
    const int cwidth_big = 27;

    std::stringstream ss{};
    std::string my_string{};

    // Heading
    ss << std::left << std::setw(cwidth_small) << "gid" << std::setw(cwidth_small) << "region"
       << std::setw(cwidth_medium) << "position";
    ss << std::setw(cwidth_big) << "axon (exist|connected)" << std::setw(cwidth_big) << "exc_den (exist|connected)";
    ss << std::setw(cwidth_big) << "inh_den (exist|connected)\n";

    // Values
    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        ss << std::left << std::setw(cwidth_small) << neuron_id;

        const auto [x, y, z] = extra_info->get_position(neuron_id);

        my_string = "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
        ss << std::setw(cwidth_medium) << my_string;

        my_string = std::to_string(axons_counts[local_neuron_id]) + "|" + std::to_string(axons_connected_counts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_exc_counts[local_neuron_id]) + "|" + std::to_string(dendrites_exc_connected_counts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_inh_counts[local_neuron_id]) + "|" + std::to_string(dendrites_inh_connected_counts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        ss << '\n';
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_local_network_histogram(const step_type current_step) {
    const auto& out_histogram = axons->get_histogram();
    const auto& in_inhibitory_histogram = dendrites_inh->get_histogram();
    const auto& in_excitatory_histogram = dendrites_exc->get_histogram();

    const auto print_histogram = [current_step](
                                     const std::map<std::pair<unsigned int, unsigned int>, uint64_t>& hist) -> std::string {
        std::stringstream ss{};
        ss << '#' << current_step;
        for (const auto& [val, occurrences] : hist) {
            const auto& [connected, grown] = val;
            ss << ";(" << connected << ',' << grown << "):" << occurrences;
        }

        return ss.str();
    };

    LogFiles::write_to_file(LogFiles::EventType::NetworkOutHistogramLocal, false, print_histogram(out_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkInInhibitoryHistogramLocal, false,
        print_histogram(in_inhibitory_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkInExcitatoryHistogramLocal, false,
        print_histogram(in_excitatory_histogram));
}

void Neurons::print_calcium_values_to_file(const step_type current_step) {
    const auto& calcium = calcium_calculator->get_calcium();

    std::stringstream ss{};

    ss << '#' << current_step;
    for (const auto val : calcium) {
        ss << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::CalciumValues, false, ss.str());
}

void Neurons::print_fire_rate_to_file(const step_type current_step) {

    const auto& fire_recorder = neuron_model->get_fired_recorder(NeuronModel::FireRecorderPeriod::NeuronMonitor);

    std::stringstream ss{};

    ss << '#' << current_step;
    for (const auto val : fire_recorder) {
        ss << ';' << val / static_cast<double>(Config::neuron_monitor_log_step);
    }

    LogFiles::write_to_file(LogFiles::EventType::FireRates, false, ss.str());
}

void Neurons::print_synaptic_inputs_to_file(const step_type current_step) {
    const auto& synaptic_input = neuron_model->get_synaptic_input();

    std::stringstream ss{};

    ss << '#' << current_step;
    for (const auto val : synaptic_input) {
        ss << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::SynapticInput, false, ss.str());
}
