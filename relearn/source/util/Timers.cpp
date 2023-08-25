/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Timers.h"

#include "Config.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "sim/Essentials.h"

#include <array>
#include <ctime>
#include <iomanip>
#include <sstream>

std::string Timers::wall_clock_time() {
    // The time is printed with 24 interesting characters followed by '\n'
    constexpr auto size_of_date_string = 24;

#ifdef __linux__
    time_t raw_time = 0;
    time(&raw_time);
    // NOLINTNEXTLINE
    struct tm* time_info = localtime(&raw_time);
    // NOLINTNEXTLINE
    char* string = asctime(time_info);

    // Avoid '\n'
    return std::string(string, size_of_date_string);
#else
    time_t raw_time = 0;
    struct tm time_info;

    // Need some more space for '\n' and other checks
    char char_buff[size_of_date_string + 3];

    time(&raw_time);
    localtime_s(&time_info, &raw_time);
    asctime_s(char_buff, &time_info);

    // Avoid '\n'
    return std::string(char_buff, size_of_date_string);
#endif
}

void Timers::print(const std::unique_ptr<Essentials>& essentials) {
    /**
     * Print timers and memory usage
     */
    constexpr auto expected_num_timers = size_t(3) * NUMBER_TIMERS;

    std::array<double, expected_num_timers> timers_local{};

    std::stringstream local_timer_output{};

    for (auto i = 0U; i < NUMBER_TIMERS; ++i) {
        const auto timer = static_cast<TimerRegion>(i);
        const auto elapsed = get_elapsed(timer);

        local_timer_output << elapsed.count() << '\n';

        for (auto j = 0U; j < 3; ++j) {
            const auto idx = 3 * i + j;
            const auto counted = elapsed.count();
            const auto seconds = static_cast<double>(counted) * 1e-9;

            // NOLINTNEXTLINE
            timers_local[idx] = seconds;
        }
    }

    LogFiles::write_to_file(LogFiles::EventType::TimersLocal, false, local_timer_output.str());

    auto timers_global = MPIWrapper::reduce(timers_local, MPIWrapper::ReduceFunction::MinSumMax, MPIRank::root_rank());
    if (MPIRank::root_rank() != MPIWrapper::get_my_rank()) {
        return;
    }

    std::stringstream sstring{};

    auto print_timer = [&timers_global, &sstring](auto message, const TimerRegion timer) {
        const auto timer_index = get_timer_index(timer);
        const auto min_time = timers_global[3 * timer_index];
        const auto avg_time = timers_global[3 * timer_index + 1];
        const auto max_time = timers_global[3 * timer_index + 2];

        sstring << message
                << std::setw(Constants::print_width) << std::fixed << std::setprecision(Constants::print_precision) << min_time << " | "
                << std::setw(Constants::print_width) << std::fixed << std::setprecision(Constants::print_precision) << avg_time << " | "
                << std::setw(Constants::print_width) << std::fixed << std::setprecision(Constants::print_precision) << max_time << '\n';
    };

    // Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
    // so that sum becomes average
    for (auto i = index_type(0); i < NUMBER_TIMERS; i++) {
        const index_type idx = 3 * i + 1U;
        // NOLINTNEXTLINE
        timers_global[idx] /= MPIWrapper::get_num_ranks();
    }

    // Set precision for aligned double output
    sstring.precision(Constants::print_precision);

    sstring << "\n======== TIMERS GLOBAL OVER ALL RANKS ========\n";
    sstring << "                                                ("
            << std::setw(Constants::print_width) << " min"
            << " | "
            << std::setw(Constants::print_width) << " avg"
            << " | "
            << std::setw(Constants::print_width) << " max"
            << ") sec.\n";
    sstring << "TIMERS: main()\n";

    print_timer("  Initialization                               : ", TimerRegion::INITIALIZATION);
    print_timer("    Load Synapses                              : ", TimerRegion::LOAD_SYNAPSES);
    print_timer("    Initialize Network Graph                   : ", TimerRegion::INITIALIZE_NETWORK_GRAPH);
    print_timer("  Simulation loop                              : ", TimerRegion::SIMULATION_LOOP);
    print_timer("    Update electrical activity                 : ", TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
    print_timer("      Neuron Model Update electrical activity  : ", TimerRegion::NEURON_MODEL_UPDATE_ELECTRICAL_ACTIVITY);
    print_timer("        Prepare sending spikes                 : ", TimerRegion::PREPARE_SENDING_SPIKES);
    print_timer("        Exchange neuron ids                    : ", TimerRegion::EXCHANGE_NEURON_IDS);
    print_timer("        Calculate synaptic background          : ", TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    print_timer("        Calculate synaptic input               : ", TimerRegion::CALC_SYNAPTIC_INPUT);
    print_timer("        Calculate stimulus input               : ", TimerRegion::CALC_STIMULUS);
    print_timer("        Calculate activity                     : ", TimerRegion::CALC_ACTIVITY);
    print_timer("        Calculate calcium                      : ", TimerRegion::UPDATE_CALCIUM);
    print_timer("        Calculate target calcium               : ", TimerRegion::UPDATE_TARGET_CALCIUM);
    print_timer("      Update Calcium extreme values            : ", TimerRegion::CALC_CALCIUM_EXTREME_VALUES);
    print_timer("    Update #synaptic elements delta            : ", TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
    print_timer("    Connectivity update                        : ", TimerRegion::UPDATE_CONNECTIVITY);
    print_timer("      Update fire history                      : ", TimerRegion::UPDATE_FIRE_HISTORY);
    print_timer("      Delete synapses                          : ", TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);
    print_timer("        Commit #synaptic elements              : ", TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    print_timer("        Find synapses to delete                : ", TimerRegion::FIND_SYNAPSES_TO_DELETE);
    print_timer("        Exchange deletions (w/ all to all)     : ", TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
    print_timer("        Process deletion requests              : ", TimerRegion::PROCESS_DELETE_REQUESTS);
    print_timer("      Update leaf nodes                        : ", TimerRegion::UPDATE_LEAF_NODES);
    print_timer("      Update local trees                       : ", TimerRegion::UPDATE_LOCAL_TREES);
    print_timer("      Exchange branch nodes (w/ All-gather)    : ", TimerRegion::EXCHANGE_BRANCH_NODES);
    print_timer("      Insert branch nodes into global tree     : ", TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
    print_timer("      Update global tree                       : ", TimerRegion::UPDATE_GLOBAL_TREE);
    print_timer("      Create synapses                          : ", TimerRegion::CREATE_SYNAPSES);
    print_timer("        Find target neurons (w/ RMA)           : ", TimerRegion::FIND_TARGET_NEURONS);
    print_timer("        Create synapses Exchange Requests      : ", TimerRegion::EXCHANGE_CREATION_REQUESTS);
    print_timer("        Create synapses Process Requests       : ", TimerRegion::PROCESS_CREATION_REQUESTS);
    print_timer("        Create synapses Exchange Responses     : ", TimerRegion::CREATE_CREATION_RESPONSES);
    print_timer("        Create synapses Process Responses      : ", TimerRegion::PROCESS_CREATION_RESPONSES);
    print_timer("      Add synapses in local network graphs     : ", TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);
    print_timer("      Empty remote nodes cache                 : ", TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    print_timer("    Capture neuron monitors                    : ", TimerRegion::CAPTURE_MONITORS);
    print_timer("    Capture area monitors                      : ", TimerRegion::CAPTURE_AREA_MONITORS);
    print_timer("      Prepare                                  : ", TimerRegion::AREA_MONITORS_PREPARE);
    print_timer("      Request                                  : ", TimerRegion::AREA_MONITORS_REQUEST);
    print_timer("      Exchange                                 : ", TimerRegion::AREA_MONITORS_EXCHANGE);
    print_timer("      Record                                   : ", TimerRegion::AREA_MONITORS_RECORD_DATA);
    print_timer("        Local edges                            : ", TimerRegion::AREA_MONITORS_LOCAL_EDGES);
    print_timer("        Distant edges                          : ", TimerRegion::AREA_MONITORS_DISTANT_EDGES);
    print_timer("        Deletions                              : ", TimerRegion::AREA_MONITORS_DELETIONS);
    print_timer("        Statistics                             : ", TimerRegion::AREA_MONITORS_STATISTICS);
    print_timer("      Finish                                   : ", TimerRegion::AREA_MONITORS_FINISH);

    print_timer("    Print IO                                   : ", TimerRegion::PRINT_IO);

    sstring << "\n\n";

    LogFiles::write_to_file(LogFiles::EventType::Timers, true, sstring.str());

    const auto average_simulation_time = timers_global[3 * get_timer_index(TimerRegion::SIMULATION_LOOP) + 1];
    essentials->insert("Simulation-Time-Seconds", average_simulation_time);
}
