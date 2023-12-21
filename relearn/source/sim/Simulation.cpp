/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Simulation.h"

#include "algorithm/Algorithms.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/StringUtil.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "neurons/Neurons.h"
#include "neurons/helper/AreaMonitor.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/models/NeuronModels.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"
#include "util/ranges/Functional.hpp"

#include <bitset>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>

#include <range/v3/action/insert.hpp>
#include <range/v3/action/transform.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/repeat_n.hpp>

Simulation::Simulation(std::unique_ptr<Essentials> essentials, std::shared_ptr<Partition> partition)
    : essentials(std::move(essentials))
    , partition(std::move(partition)) {

    monitors = std::make_shared<std::vector<NeuronMonitor>>();
    area_monitors = std::make_shared<std::unordered_map<RelearnTypes::area_id, AreaMonitor>>();
}

void Simulation::register_neuron_monitor(const NeuronID& neuron_id) {
    monitors->emplace_back(neuron_id);
    monitors->at(monitors->size() - 1).init_print_file();
}

void Simulation::set_acceptance_criterion_for_barnes_hut(const double value) {
    // Needed to avoid creating autapses
    RelearnException::check(value <= Constants::bh_max_theta,
        "Simulation::set_acceptance_criterion_for_barnes_hut: Acceptance criterion must be smaller or equal to {} but was {}", Constants::bh_max_theta, value);
    RelearnException::check(value > 0.0, "Simulation::set_acceptance_criterion_for_barnes_hut: Acceptance criterion must larger than 0.0, but it was {}", value);

    accept_criterion = value;
}

void Simulation::set_neuron_model(std::unique_ptr<NeuronModel>&& nm) noexcept {
    neuron_models = std::move(nm);
}

void Simulation::set_calcium_calculator(std::unique_ptr<CalciumCalculator>&& calculator) noexcept {
    calcium_calculator = std::move(calculator);
}

void Simulation::set_axons(std::shared_ptr<SynapticElements>&& se) noexcept {
    axons = std::move(se);
}

void Simulation::set_dendrites_ex(std::shared_ptr<SynapticElements>&& se) noexcept {
    dendrites_ex = std::move(se);
}

void Simulation::set_dendrites_in(std::shared_ptr<SynapticElements>&& se) noexcept {
    dendrites_in = std::move(se);
}

void Simulation::set_synapse_deletion_finder(std::unique_ptr<SynapseDeletionFinder>&& sdf) noexcept {
    synapse_deletion_finder = std::move(sdf);
}

namespace {
constexpr auto sort_ids = [](auto pair) {
    ranges::sort(pair.second);
    return pair;
};
} // namespace

void Simulation::set_enable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts) {
    enable_interrupts = std::move(interrupts) | ranges::actions::transform(sort_ids);
}

void Simulation::set_disable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts) {
    disable_interrupts = std::move(interrupts) | ranges::actions::transform(sort_ids);
}

void Simulation::set_creation_interrupts(std::vector<std::pair<step_type, number_neurons_type>> interrupts) noexcept {
    creation_interrupts = std::move(interrupts);
}

void Simulation::set_algorithm(const AlgorithmEnum algorithm) noexcept {
    algorithm_enum = algorithm;
}

void Simulation::set_percentage_initial_fired_neurons(double percentage) {
    RelearnException::check(percentage >= 0.0, "Simulation::set_percentage_initial_fired_neurons: percentage is too low: {}", percentage);
    RelearnException::check(percentage <= 1.0, "Simulation::set_percentage_initial_fired_neurons: percentage is too high: {}", percentage);
    percentage_initially_fired = percentage;
}

void Simulation::set_subdomain_assignment(std::unique_ptr<NeuronToSubdomainAssignment>&& subdomain_assignment) noexcept {
    neuron_to_subdomain_assignment = std::move(subdomain_assignment);
}

void Simulation::initialize() {
    RelearnException::check(neuron_models != nullptr, "Simulation::initialize: neuron_models is nullptr");
    RelearnException::check(calcium_calculator != nullptr, "Simulation::initialize: calcium_calculator is nullptr");
    RelearnException::check(axons != nullptr, "Simulation::initialize: axons is nullptr");
    RelearnException::check(dendrites_ex != nullptr, "Simulation::initialize: dendrites_ex is nullptr");
    RelearnException::check(dendrites_in != nullptr, "Simulation::initialize: dendrites_in is nullptr");
    RelearnException::check(neuron_to_subdomain_assignment != nullptr, "Simulation::initialize: neuron_to_subdomain_assignment is nullptr");

    neuron_to_subdomain_assignment->initialize();
    const auto number_total_neurons = neuron_to_subdomain_assignment->get_total_number_placed_neurons();

    partition->set_total_number_neurons(number_total_neurons);
    const auto number_local_neurons = partition->get_number_local_neurons();

    //    auto check = [](RelearnTypes::number_neurons_type value) -> bool {
    //        const auto min = MPIWrapper::reduce(value, MPIWrapper::ReduceFunction::Min, MPIRank::root_rank());
    //        const auto max = MPIWrapper::reduce(value, MPIWrapper::ReduceFunction::Max, MPIRank::root_rank());
    //        return min == max;
    //    };
    // RelearnException::check(check(number_local_neurons), "Simulation::initialize: Different number of local neurons on ranks. Mine: {}", number_local_neurons);

    const auto my_rank = MPIWrapper::get_my_rank();
    RelearnException::check(number_local_neurons > 0, "I have 0 neurons at rank {}", my_rank.get_rank());

    synapse_deletion_finder->set_axons(axons);
    synapse_deletion_finder->set_dendrites_ex(dendrites_ex);
    synapse_deletion_finder->set_dendrites_in(dendrites_in);

    network_graph = std::make_shared<NetworkGraph>(my_rank);

    neurons = std::make_shared<Neurons>(partition, std::move(neuron_models), std::move(calcium_calculator), network_graph, axons, dendrites_ex, dendrites_in, std::move(synapse_deletion_finder));
    neurons->init(number_local_neurons);
    NeuronMonitor::neurons_to_monitor = neurons;

    auto number_local_neurons_ntsa = neuron_to_subdomain_assignment->get_number_neurons_in_subdomains();

    RelearnException::check(number_local_neurons_ntsa == number_local_neurons,
        "Simulation::initialize: The partition and the NTSA had a disagreement about the number of local neurons");

    auto neuron_positions = neuron_to_subdomain_assignment->get_neuron_positions_in_subdomains();
    auto local_area_translator = neuron_to_subdomain_assignment->get_local_area_translator();
    auto signal_types = neuron_to_subdomain_assignment->get_neuron_types_in_subdomains();

    global_area_mapper = std::make_shared<GlobalAreaMapper>(local_area_translator, MPIWrapper::get_num_ranks(), my_rank);

    RelearnException::check(neuron_positions.size() == number_local_neurons, "Simulation::initialize: neuron_positions had the wrong size");
    RelearnException::check(local_area_translator->get_number_neurons_in_total() == number_local_neurons, "Simulation::initialize: neuron_id_vs_area_id had the wrong size {} != {}", local_area_translator->get_number_neurons_in_total(), number_local_neurons);
    RelearnException::check(signal_types.size() == number_local_neurons, "Simulation::initialize: signal_types had the wrong size");

    partition->print_my_subdomains_info_rank();

    LogFiles::print_message_rank(MPIRank::root_rank(), "Neurons created");

    const auto& [simulation_box_min, simulation_box_max] = partition->get_simulation_box_size();
    const auto level_of_branch_nodes = partition->get_level_of_subdomain_trees();

    if (algorithm_enum == AlgorithmEnum::BarnesHut) {
        global_tree = std::make_shared<OctreeImplementation<BarnesHutCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else if (algorithm_enum == AlgorithmEnum::BarnesHutInverted) {
        global_tree = std::make_shared<OctreeImplementation<BarnesHutInvertedCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else if (algorithm_enum == AlgorithmEnum::Naive) {
        global_tree = std::make_shared<OctreeImplementation<NaiveCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else {
        RelearnException::fail("Simulation::initialize: Cannot construct the octree for an unknown algorithm.");
    }

    LogFiles::print_message_rank(MPIRank::root_rank(), "Level of branch nodes is: {}", global_tree->get_level_of_branch_nodes());

    for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
        const auto& position = neuron_positions[neuron_id.get_neuron_id()];
        global_tree->insert(position, neuron_id);
    }

    global_tree->initializes_leaf_nodes(number_local_neurons);

    LogFiles::print_message_rank(MPIRank::root_rank(), "Inserted a total of {} neurons", number_total_neurons);

    if (algorithm_enum == AlgorithmEnum::BarnesHut) {
        auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(global_tree);
        auto algorithm_barnes_hut = std::make_shared<BarnesHut>(std::move(cast));
        algorithm_barnes_hut->set_acceptance_criterion(accept_criterion);
        algorithm = std::move(algorithm_barnes_hut);
    } else if (algorithm_enum == AlgorithmEnum::BarnesHutInverted) {
        auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutInvertedCell>>(global_tree);
        auto algorithm_barnes_hut_inverted = std::make_shared<BarnesHutInverted>(std::move(cast));
        algorithm_barnes_hut_inverted->set_acceptance_criterion(accept_criterion);
        algorithm = std::move(algorithm_barnes_hut_inverted);
    } else {
        RelearnException::fail("Simulation::initialize: AlgorithmEnum {} not yet implemented!", static_cast<int>(algorithm_enum));
    }

    const auto& extra_infos = neurons->get_extra_info();
    algorithm->set_neuron_extra_infos(extra_infos);
    algorithm->set_synaptic_elements(axons, dendrites_ex, dendrites_in);

    neurons->set_local_area_translator(local_area_translator);
    neurons->set_signal_types(std::move(signal_types));
    neurons->set_positions(std::move(neuron_positions));
    neurons->set_octree(global_tree);
    neurons->set_algorithm(algorithm);

    if (area_monitor_enabled) {
        for (size_t area_id = 0; area_id < local_area_translator->get_number_of_areas(); area_id++) {
            const auto& area_name = local_area_translator->get_area_name_for_area_id(area_id);
            const std::filesystem::path dir = LogFiles::get_output_path() / "area_monitors";
            if (!std::filesystem::exists(dir)) {
                std::filesystem::create_directories(dir);
            }
            auto path = dir / (MPIWrapper::get_my_rank_str() + "_area_" + std::to_string(area_id) + ".csv");
            area_monitors->insert(
                std::make_pair(area_id, AreaMonitor(neurons, global_area_mapper, area_id, area_name, my_rank.get_rank(), path, area_monitor_connectivity)));
        }
    }

    auto synapse_loader = neuron_to_subdomain_assignment->get_synapse_loader();

    const auto& [synapses_static, synapses_plastic] = synapse_loader->load_synapses(essentials);
    const auto& [local_synapses_static, in_synapses_static, out_synapses_static] = synapses_static;
    const auto& [local_synapses_plastic, in_synapses_plastic, out_synapses_plastic] = synapses_plastic;

    Timers::start(TimerRegion::INITIALIZE_NETWORK_GRAPH);
    network_graph->add_edges(local_synapses_plastic, in_synapses_plastic, out_synapses_plastic);
    network_graph->add_edges(local_synapses_static, in_synapses_static, out_synapses_static);
    neurons->set_static_neurons(static_neurons);
    Timers::stop_and_add(TimerRegion::INITIALIZE_NETWORK_GRAPH);

    LogFiles::print_message_rank(MPIRank::root_rank(), "Network graph created");
    LogFiles::print_message_rank(MPIRank::root_rank(), "Synaptic elements initialized");

    neurons->init_synaptic_elements(local_synapses_plastic, in_synapses_plastic, out_synapses_plastic);

    if (area_monitor_enabled) {
        // Update area monitor
        Timers::start(TimerRegion::CAPTURE_AREA_MONITORS);

        Timers::start(TimerRegion::AREA_MONITORS_PREPARE);
        for (auto& [_, area_monitor] : *area_monitors) {
            area_monitor.prepare_recording();
        }

        Timers::stop_and_add(TimerRegion::AREA_MONITORS_PREPARE);
        Timers::start(TimerRegion::AREA_MONITORS_REQUEST);

        for (auto& [_, area_monitor] : *area_monitors) {
            area_monitor.request_data();
        }

        Timers::stop_and_add(TimerRegion::AREA_MONITORS_REQUEST);
        Timers::start(TimerRegion::AREA_MONITORS_EXCHANGE);

        global_area_mapper->exchange_requests();

        Timers::stop_and_add(TimerRegion::AREA_MONITORS_EXCHANGE);
        Timers::start(TimerRegion::AREA_MONITORS_RECORD_DATA);

        for (auto& [_, area_monitor] : *area_monitors) {
            area_monitor.monitor_connectivity();
        }

        Timers::stop_and_add(TimerRegion::AREA_MONITORS_RECORD_DATA);
        Timers::start(TimerRegion::AREA_MONITORS_FINISH);

        for (auto& [_, area_monitor] : *area_monitors) {
            area_monitor.finish_recording();
        }
        Timers::stop_and_add(TimerRegion::AREA_MONITORS_FINISH);
        Timers::stop_and_add(TimerRegion::CAPTURE_AREA_MONITORS);
    }

    const auto fired_neurons = static_cast<size_t>(number_local_neurons * percentage_initially_fired);

    const auto initial_fired = ranges::views::concat(
                                   ranges::views::repeat_n(FiredStatus::Fired, fired_neurons),
                                   ranges::views::repeat_n(FiredStatus::Inactive, number_local_neurons - fired_neurons))
        | ranges::to_vector
        | RandomHolder::shuffleAction(RandomHolderKey::BackgroundActivity);

    neurons->set_fired(std::move(initial_fired));

    MPIWrapper::create_rma_window<std::bitset<NeuronsExtraInfo::fire_history_length>>(MPIWindow::FireHistory, number_local_neurons, MPIWrapper::get_num_ranks());

    neurons->debug_check_counts();
    neurons->print_neurons_overview_to_log_file_on_rank_0(0);
    neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(0, 0, 0, 0);
    neurons->print_area_mapping_to_log_file();
}

void Simulation::simulate(const step_type number_steps) {
    RelearnException::check(number_steps > 0, "Simulation::simulate: number_steps must be greater than 0");
    const auto my_rank = MPIWrapper::get_my_rank();

    Timers::start(TimerRegion::SIMULATION_LOOP);

    const auto previous_synapse_creations = total_synapse_creations;
    const auto previous_synapse_deletions = total_synapse_deletions;

    /**
     * Simulation loop
     */
    const auto final_step_count = step + number_steps;
    for (; step <= final_step_count; ++step) { // NOLINT(altera-id-dependent-backward-branch)
        for (const auto& [disable_step, disable_ids] : disable_interrupts | ranges::views::filter(equal_to(step), element<0>)) {
            LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Disabling {} neurons in step {}", disable_ids.size(), disable_step);
            const auto& [num_deleted_synapses, number_deleted_distant_out_axons, number_deleted_distant_in, number_deleted_in_edges_from_outside, number_deleted_out_edges_to_outside, number_deleted_out_edges_within, synapse_deletion_requests_outgoing] = neurons->disable_neurons(step, disable_ids, MPIWrapper::get_num_ranks());
            total_synapse_deletions += static_cast<int64_t>(num_deleted_synapses);
            const auto& synapse_deletion_requests_ingoing = MPIWrapper::exchange_requests(synapse_deletion_requests_outgoing);
            total_synapse_deletions += neurons->delete_disabled_distant_synapses(synapse_deletion_requests_ingoing, my_rank);
        }

        for (const auto& [enable_step, enable_ids] : enable_interrupts | ranges::views::filter(equal_to(step), element<0>)) {
            LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Enabling {} neurons in step {}", enable_ids.size(), enable_step);
            neurons->enable_neurons(enable_ids);
        }

        for (const auto& [creation_step, creation_count] : creation_interrupts | ranges::views::filter(equal_to(step), element<1>)) {
            LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Creating {} neurons in step {}", creation_count, creation_step);
            neurons->create_neurons(creation_count);
        }

        if (interval_neuron_monitor.hits_step(step)) {
            const auto number_neurons = neurons->get_number_neurons();

            Timers::start(TimerRegion::CAPTURE_MONITORS);
            for (auto& mn : *monitors) {
                if (mn.get_target_id().get_neuron_id() < number_neurons) {
                    mn.record_data(step);
                }
            }

            neurons->print_fire_rate_to_file(step);

            neurons->get_neuron_model()->reset_fired_recorder(NeuronModel::FireRecorderPeriod::NeuronMonitor);
            Timers::stop_and_add(TimerRegion::CAPTURE_MONITORS);
        }

        if (interval_update_electrical_activity.hits_step(step)) {
            Timers::start(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
            neurons->update_electrical_activity(step);
            Timers::stop_and_add(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
        }

        if (interval_update_synaptic_elements.hits_step(step)) {
            Timers::start(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
            neurons->update_number_synaptic_elements_delta();
            Timers::stop_and_add(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
        }

        if (interval_update_plasticity.hits_step(step)) {
            Timers::start(TimerRegion::UPDATE_CONNECTIVITY);

            const auto& [num_axons_deleted, num_dendrites_deleted, num_synapses_created] = neurons->update_connectivity(step);

            Timers::stop_and_add(TimerRegion::UPDATE_CONNECTIVITY);

            // Get total number of synapses deleted and created
            const std::array<int64_t, 3> local_counts = { static_cast<int64_t>(num_axons_deleted), static_cast<int64_t>(num_dendrites_deleted), static_cast<int64_t>(num_synapses_created) };
            const std::array<int64_t, 3> global_counts = MPIWrapper::reduce(local_counts, MPIWrapper::ReduceFunction::Sum, MPIRank::root_rank());

            const auto local_deletions = local_counts[0] + local_counts[1];
            const auto local_creations = local_counts[2];

            const auto global_deletions = global_counts[0] + global_counts[1];
            const auto global_creations = global_counts[2];

            if (MPIRank::root_rank() == my_rank) {
                total_synapse_deletions += global_deletions;
                total_synapse_creations += global_creations;
            }

            neurons->get_neuron_model()->reset_fired_recorder(NeuronModel::Plasticity);

            Timers::start(TimerRegion::PRINT_IO);

            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "{}: {} {} {}", step, global_creations, global_deletions, global_creations - global_deletions);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "{};{};{};{}", step, global_creations, global_deletions, global_creations - global_deletions);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "{}: {} {} {}", step, local_creations, local_deletions, local_creations - local_deletions);

            neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(step, num_axons_deleted, num_dendrites_deleted, num_synapses_created);

            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_neuron_monitor.hits_step(step)) {
            if (area_monitor_enabled) {
                // Update area monitor
                Timers::start(TimerRegion::CAPTURE_AREA_MONITORS);

                Timers::start(TimerRegion::AREA_MONITORS_PREPARE);
                ranges::for_each(*area_monitors | ranges::views::values, &AreaMonitor::prepare_recording);

                Timers::stop_and_add(TimerRegion::AREA_MONITORS_PREPARE);
                Timers::start(TimerRegion::AREA_MONITORS_REQUEST);

                global_area_mapper->check_cache();

                ranges::for_each(*area_monitors | ranges::views::values, &AreaMonitor::request_data);

                Timers::stop_and_add(TimerRegion::AREA_MONITORS_REQUEST);
                Timers::start(TimerRegion::AREA_MONITORS_EXCHANGE);

                global_area_mapper->exchange_requests();

                Timers::stop_and_add(TimerRegion::AREA_MONITORS_EXCHANGE);
                Timers::start(TimerRegion::AREA_MONITORS_RECORD_DATA);

                ranges::for_each(*area_monitors | ranges::views::values, &AreaMonitor::monitor_connectivity);

                for (NeuronID neuron_id :
                    NeuronID::range(neurons->get_number_neurons())) {
                    if (neurons->get_disable_flags()[neuron_id.get_neuron_id()] == UpdateStatus::Disabled) {
                        continue;
                    }
                    const auto& area_id = neurons->get_local_area_translator()->get_area_id_for_neuron_id(neuron_id.get_neuron_id());

                    auto& area_monitor = area_monitors->at(area_id);
                    area_monitor.record_data(neuron_id);
                }

                Timers::stop_and_add(TimerRegion::AREA_MONITORS_RECORD_DATA);
                Timers::start(TimerRegion::AREA_MONITORS_FINISH);

                ranges::for_each(*area_monitors | ranges::views::values, &AreaMonitor::finish_recording);

                Timers::stop_and_add(TimerRegion::AREA_MONITORS_FINISH);

                neurons->get_neuron_model()->reset_fired_recorder(NeuronModel::FireRecorderPeriod::AreaMonitor);
                neurons->get_extra_info()->reset_deletion_log();

                Timers::stop_and_add(TimerRegion::CAPTURE_AREA_MONITORS);
            }
            network_graph->debug_check();
        }

        if (interval_histogram_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_local_network_histogram(step);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_calcium_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_calcium_values_to_file(step);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_synaptic_input_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_synaptic_inputs_to_file(step);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_network_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_network_graph_to_log_file(step, true);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_statistics_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_neurons_overview_to_log_file_on_rank_0(step);
            Timers::stop_and_add(TimerRegion::PRINT_IO);

            for (auto& [attribute, vector] : statistics) {
                vector.emplace_back(neurons->get_statistics(attribute));
            }
        }

        if (step % Config::flush_monitor_step == 0) {
            Timers::start(TimerRegion::PRINT_IO);
            ranges::for_each(*monitors, &NeuronMonitor::flush_current_contents);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (step % Config::flush_area_monitor_step == 0) {
            ranges::for_each(*area_monitors | ranges::views::values, &AreaMonitor::write_data_to_file);
        }

        if (step % Config::console_update_step == 0) {
            if (my_rank != MPIRank::root_rank()) {
                continue;
            }

            const auto net_creations = total_synapse_creations - total_synapse_deletions;

            LogFiles::write_to_file(LogFiles::EventType::Cout, true,
                "[Step: {}\t] Total up to now     (creations, deletions, net):\t{}\t\t{}\t\t{}",
                step, total_synapse_creations, total_synapse_deletions, net_creations);
        }
    }

    delta_synapse_creations = total_synapse_creations - previous_synapse_creations;
    delta_synapse_deletions = total_synapse_deletions - previous_synapse_deletions;

    // Stop timing simulation loop
    Timers::stop_and_add(TimerRegion::SIMULATION_LOOP);

    LogFiles::print_message_rank(MPIRank::root_rank(), "Final flush of neuron monitors");
    ranges::for_each(*monitors, &NeuronMonitor::flush_current_contents);

    LogFiles::print_message_rank(MPIRank::root_rank(), "Final flush of area monitors");
    ranges::for_each(*area_monitors | ranges::views::values, &AreaMonitor::write_data_to_file);

    LogFiles::print_message_rank(MPIRank::root_rank(), "Print positions");
    neurons->print_positions_to_log_file();
}

void Simulation::finalize() const {
    Timers::print(essentials);

    const auto net_creations = total_synapse_creations - total_synapse_deletions;
    const auto previous_net_creations = delta_synapse_creations - delta_synapse_deletions;

    LogFiles::print_message_rank(MPIRank::root_rank(),
        "Total up to now     (creations, deletions, net): {}\t{}\t{}\nDiff. from previous (creations, deletions, net): {}\t{}\t{}\nEND: {}",
        total_synapse_creations, total_synapse_deletions, net_creations,
        delta_synapse_creations, delta_synapse_deletions, previous_net_creations,
        Timers::wall_clock_time());

    neurons->print_calcium_statistics_to_essentials(essentials);
    neurons->print_synaptic_changes_to_essentials(essentials);

    essentials->insert("Created-Synapses", total_synapse_creations);
    essentials->insert("Deleted-Synapses", total_synapse_deletions);
    essentials->insert("net-Synapses", net_creations);

    std::stringstream ss{};
    essentials->print(ss);

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, ss.str());

    // Print final network graph
    neurons->print_network_graph_to_log_file(step, false);
}

std::vector<std::unique_ptr<NeuronModel>> Simulation::get_models() {
    return NeuronModel::get_models();
}

void Simulation::increase_monitoring_capacity(const size_t size) {
    for (auto& mon : *monitors) {
        mon.increase_monitoring_capacity(size);
    }
}

void Simulation::snapshot_monitors() {
    if (!monitors->empty()) {
        // record data at step 0
        Timers::start(TimerRegion::CAPTURE_MONITORS);
        for (auto& m : *monitors) {
            m.record_data(0);
        }
        Timers::stop_and_add(TimerRegion::CAPTURE_MONITORS);

        neurons->get_neuron_model()->reset_fired_recorder(NeuronModel::FireRecorderPeriod::NeuronMonitor);
    }
}

void Simulation::set_static_neurons(std::vector<NeuronID> static_neurons) {
    this->static_neurons = std::move(static_neurons);
}
