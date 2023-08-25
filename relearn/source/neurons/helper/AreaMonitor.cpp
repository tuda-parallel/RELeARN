/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AreaMonitor.h"

#include "mpi/MPIWrapper.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/NetworkGraph.h"
#include "neurons/Neurons.h"
#include "neurons/helper/RankNeuronId.h"
#include "sim/Simulation.h"
#include "util/ranges/Functional.hpp"

#include <fstream>
#include <set>
#include <tuple>
#include <utility>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/transform.hpp>

AreaMonitor::AreaMonitor(std::shared_ptr<Neurons> neurons, std::shared_ptr<GlobalAreaMapper> global_area_mapper, RelearnTypes::area_id area_id, RelearnTypes::area_name area_name,
    int my_rank, const std::filesystem::path& path, bool monitor_connectivity)
    : neurons(neurons)
    , area_id(area_id)
    , area_name(std::move(area_name))
    , global_area_mapper(global_area_mapper)
    , my_rank(my_rank)
    , path(std::move(path))
    , flag_monitor_connectivity(monitor_connectivity) {
    write_header();
}

void AreaMonitor::monitor_connectivity() {
    if (!flag_monitor_connectivity) {
        return;
    }
    Timers::start(TimerRegion::AREA_MONITORS_LOCAL_EDGES);
    for (const auto& synapse : neurons->last_created_local_synapses) {
        if (neurons->local_area_translator->get_area_id_for_neuron_id(synapse.get_target().get_neuron_id()) != area_id) {
            continue;
        }
        const auto other_area_id = neurons->get_local_area_translator()->get_area_id_for_neuron_id(
            synapse.get_source().get_neuron_id());

        const auto signal_type = synapse.get_weight() > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
        add_ingoing_connection({ my_rank, other_area_id, synapse.get_target(), signal_type }, synapse.get_weight());
    }

    Timers::stop_and_add(TimerRegion::AREA_MONITORS_LOCAL_EDGES);
    Timers::start(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
    for (const auto& synapse : neurons->last_created_in_synapses) {
        if (neurons->local_area_translator->get_area_id_for_neuron_id(synapse.get_target().get_neuron_id()) != area_id) {
            continue;
        }

        const auto& rank_neuron_id = synapse.get_source();
        const auto& other_rank = rank_neuron_id.get_rank().get_rank();
        const auto other_area_id = global_area_mapper->get_area_id(rank_neuron_id);

        const auto signal_type = synapse.get_weight() > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
        add_ingoing_connection({ other_rank, other_area_id, synapse.get_target(), signal_type }, synapse.get_weight());
    }
    Timers::stop_and_add(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
}

void AreaMonitor::record_data(NeuronID neuron_id) {
    Timers::start(TimerRegion::AREA_MONITORS_DELETIONS);

    // Deletions
    if (flag_monitor_connectivity) {
        const auto& deletions_in_step = neurons->extra_info->get_deletions_log(neuron_id);
        for (const auto& [other_neuron_id, weight] : deletions_in_step) {
            const auto other_area_id = global_area_mapper->get_area_id(other_neuron_id);
            const auto& other_rank = other_neuron_id.get_rank().get_rank();

            auto pair = std::make_pair(other_rank, other_area_id);
            deletions[pair]++;
            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            remove_ingoing_connection(AreaConnection(other_rank, other_area_id, neuron_id, signal_type), weight);
        }
    }

    Timers::stop_and_add(TimerRegion::AREA_MONITORS_DELETIONS);
    Timers::start(TimerRegion::AREA_MONITORS_STATISTICS);

    // Store statistics
    internal_statistics.axons_grown += neurons->get_axons().get_grown_elements(neuron_id);
    internal_statistics.den_ex_grown += neurons->get_dendrites_exc().get_grown_elements(neuron_id);
    internal_statistics.den_inh_grown += neurons->get_dendrites_inh().get_grown_elements(neuron_id);

    internal_statistics.axons_conn += neurons->get_axons().get_connected_elements(neuron_id);
    internal_statistics.den_ex_conn += neurons->get_dendrites_exc().get_connected_elements(neuron_id);
    internal_statistics.den_inh_conn += neurons->get_dendrites_inh().get_connected_elements(neuron_id);

    internal_statistics.background += neurons->neuron_model->get_background_activity(neuron_id);
    internal_statistics.syn_input_total += neurons->neuron_model->get_synaptic_input(neuron_id);
    internal_statistics.syn_input_ex_raw += neurons->neuron_model->input_calculator->raw_ex_input[neuron_id.get_neuron_id()];
    internal_statistics.syn_input_inh_raw += neurons->neuron_model->input_calculator->raw_inh_input[neuron_id.get_neuron_id()];

    internal_statistics.calcium += neurons->get_calcium(neuron_id);
    internal_statistics.fired_fraction += static_cast<double>(neurons->get_neuron_model()->fired_recorder[NeuronModel::FireRecorderPeriod::AreaMonitor][neuron_id.get_neuron_id()]) / static_cast<double>(Config::plasticity_update_step);
    internal_statistics.num_enabled_neurons++;

    Timers::stop_and_add(TimerRegion::AREA_MONITORS_STATISTICS);
}

void AreaMonitor::prepare_recording() {
    deletions = EnsembleDeletions{};
    internal_statistics = InternalStatistics{};
}

void AreaMonitor::finish_recording() {
    debug_checks();
    data.emplace_back(connections, deletions, internal_statistics);
}

void AreaMonitor::write_header() {
    std::ofstream out(path, std::ios_base::app);

    // Header
    out << "# Connections from ensemble " << area_name << " (" << my_rank << ":" << area_id << ") to ..."
        << "\n";
    out << "# Rank: " << my_rank << "\n";
    out << "# Area id: " << area_id << "\n";
    out << "# Area name: " << area_name << "\n";
    out.close();
}

void AreaMonitor::write_data_to_file() {
    std::ofstream out(path, std::ios_base::app);

    auto unique_area_ids = data
        | ranges::views::for_each(element<0>)
        | ranges::views::keys
        | ranges::to<std::set>;

    const auto unique_area_ids2 = data
        | ranges::views::for_each(element<1>)
        | ranges::views::keys
        | ranges::to<std::set>;

    unique_area_ids.insert(unique_area_ids2.begin(), unique_area_ids2.end());

    std::vector<std::pair<int, RelearnTypes::area_id>> unique_area_ids_list;
    std::copy(unique_area_ids.begin(), unique_area_ids.end(), std::back_inserter(unique_area_ids_list));
    std::sort(unique_area_ids_list.begin(), unique_area_ids_list.end());
    // Header
    out << "# Step;";
    for (const auto& [rank, area_id] : unique_area_ids_list) {
        out << rank << ":" << area_id << "ex;"
            << rank << ":" << area_id << "in;"
            << rank << ":" << area_id << "del;";
    }
    out << "Axons grown;Axons conn;Den ex grown;Den ex conn;Den inh grown;Den inh conn;Background;Syn input total;Syn input ex;Syn input inh;Calcium;Fire rate;Enabled neurons;";
    out << "\n";

    // Data
    for (const auto& single_record : data) {
        out << step << ";";
        auto connection_data = std::get<0>(single_record);
        auto deletion_data = std::get<1>(single_record);
        auto internal_statistics_data = std::get<2>(single_record);
        for (const auto& rank_area_id : unique_area_ids_list) {
            const auto& connections = connection_data[rank_area_id];
            out << std::to_string(connections.den_ex) << ";";
            out << std::to_string(connections.den_inh) << ";";
            const auto& deletions_in_step = deletion_data[rank_area_id];
            out << std::to_string(deletions_in_step) << ";";
        }
        out << internal_statistics_data.axons_grown << ";";
        out << internal_statistics_data.axons_conn << ";";
        out << internal_statistics_data.den_ex_grown << ";";
        out << internal_statistics_data.den_ex_conn << ";";
        out << internal_statistics_data.den_inh_grown << ";";
        out << internal_statistics_data.den_inh_conn << ";";
        out << internal_statistics_data.background << ";";
        out << internal_statistics_data.syn_input_total << ";";
        out << internal_statistics_data.syn_input_ex_raw << ";";
        out << internal_statistics_data.syn_input_inh_raw << ";";
        out << internal_statistics_data.calcium << ";";
        out << internal_statistics_data.fired_fraction << ";";
        out << internal_statistics_data.num_enabled_neurons << ";";

        out << "\n";
        step += Config::neuron_monitor_log_step;
    }
    out.close();
    data.clear();
}

void AreaMonitor::request_data() const {
    if (!flag_monitor_connectivity) {
        return;
    }
    Timers::start(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
    for (const auto& synapse : neurons->last_created_in_synapses) {
        // Other area is on different mpi rank. Save connection for communication over mpi
        const auto& rank_neuron_id = synapse.get_source();
        global_area_mapper->request_area_id(rank_neuron_id);
    }

    for (const auto& neuron_id : NeuronID::range(neurons->get_extra_info()->get_size())) {
        const auto& deletions_in_step = neurons->get_extra_info()->get_deletions_log(neuron_id);
        for (const auto& [other_neuron_id, weight] : deletions_in_step) {
            if (other_neuron_id.get_rank().get_rank() == my_rank) {
                continue;
            }
            global_area_mapper->request_area_id(other_neuron_id);
        }
    }
    Timers::stop_and_add(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
}

void AreaMonitor::add_ingoing_connection(const AreaMonitor::AreaConnection& connection, const RelearnTypes::plastic_synapse_weight weight) {
    auto pair = std::make_pair(connection.from_rank, connection.from_area);
    auto& conn = connections[pair];
    if (connection.signal_type == SignalType::Excitatory) {
        conn.den_ex += std::abs(weight);
    } else {
        conn.den_inh += std::abs(weight);
    }
}

void AreaMonitor::remove_ingoing_connection(const AreaMonitor::AreaConnection& connection, const RelearnTypes::plastic_synapse_weight weight) {
    auto pair = std::make_pair(connection.from_rank, connection.from_area);
    auto& conn = connections[pair];
    if (connection.signal_type == SignalType::Excitatory) {
        conn.den_ex -= std::abs(weight);
    } else {
        conn.den_inh -= std::abs(weight);
    }
    RelearnException::check(conn.den_ex >= 0, "AreaMonitor::remove_ingoing_connection: Excitatory connections smaller 0");
    RelearnException::check(conn.den_inh >= 0, "AreaMonitor::remove_ingoing_connection: Inhibitory connections smaller 0");
}

void AreaMonitor::debug_checks() {
    if (!Config::do_debug_checks) {
        return;
    }

    std::unordered_map<std::pair<size_t, size_t>, size_t, boost::hash<std::pair<size_t, size_t>>> in_connectivity_ex{};
    std::unordered_map<std::pair<size_t, size_t>, size_t, boost::hash<std::pair<size_t, size_t>>> in_connectivity_inh{};

    auto add_to_map = [](std::unordered_map<std::pair<size_t, size_t>, size_t, boost::hash<std::pair<size_t, size_t>>>& map, const std::pair<size_t, size_t>& key, const size_t weight) {
        if (!map.contains(key)) {
            map.insert(std::make_pair(key, 0));
        }
        map[key] += weight;
    };

    for (const auto& neuron_id : NeuronID::range(neurons->extra_info->get_size())) {
        if (neurons->local_area_translator->get_area_id_for_neuron_id(neuron_id.get_neuron_id()) != area_id) {
            continue;
        }

        const auto& [distant_in, _0] = neurons->network_graph->get_distant_in_edges(neuron_id);
        const auto& [local_in, _1] = neurons->network_graph->get_local_in_edges(neuron_id);

        for (const auto& [source_rni, weight] : distant_in) {
            const auto& other_area_id = global_area_mapper->get_area_id(source_rni);
            const auto p = std::make_pair(source_rni.get_rank().get_rank(), other_area_id);
            if (weight > 0) {
                add_to_map(in_connectivity_ex, p, weight);
            } else {
                add_to_map(in_connectivity_inh, p, std::abs(weight));
            }
        }

        for (const auto& [source_neuron_id, weight] : local_in) {
            const auto& other_area_id = neurons->local_area_translator->get_area_id_for_neuron_id(source_neuron_id.get_neuron_id());
            const auto p = std::make_pair(my_rank, other_area_id);
            if (weight > 0) {
                add_to_map(in_connectivity_ex, p, weight);
            } else {
                add_to_map(in_connectivity_inh, p, std::abs(weight));
            }
        }
    }

    for (const auto& [key, connectivity] : connections) {
        const auto& [source_rank, source_area_id] = key;

        RelearnException::check(connectivity.den_ex == in_connectivity_ex[key], "AreaMonitor::debug_checks: Unequal number of excitatory connections from ({},{}) to ({},{}): {} != {}", source_rank, source_area_id, my_rank, area_id, connectivity.den_ex, in_connectivity_ex[key]);
        RelearnException::check(connectivity.den_inh == in_connectivity_inh[key], "AreaMonitor::debug_checks: Unequal number of inhibitory connections from ({},{}) to ({},{}): {} != {}", source_rank, source_area_id, my_rank, area_id, connectivity.den_inh, in_connectivity_inh[key]);
    }
}