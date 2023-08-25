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

#include "util/RelearnException.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

class Essentials;

/**
 * This type allows type-safe specification of a specific timer
 */
enum class TimerRegion : unsigned int {
    INITIALIZATION = 0,
    LOAD_SYNAPSES,
    INITIALIZE_NETWORK_GRAPH,

    SIMULATION_LOOP,
    UPDATE_ELECTRICAL_ACTIVITY,
    NEURON_MODEL_UPDATE_ELECTRICAL_ACTIVITY,
    PREPARE_SENDING_SPIKES,
    EXCHANGE_NEURON_IDS,
    CALC_SYNAPTIC_BACKGROUND,
    CALC_SYNAPTIC_INPUT,
    CALC_STIMULUS,
    CALC_ACTIVITY,
    CALC_CALCIUM_EXTREME_VALUES,

    UPDATE_CALCIUM,
    UPDATE_TARGET_CALCIUM,

    UPDATE_SYNAPTIC_ELEMENTS_DELTA,

    UPDATE_CONNECTIVITY,

    UPDATE_FIRE_HISTORY,

    UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES,
    COMMIT_NUM_SYNAPTIC_ELEMENTS,
    FIND_SYNAPSES_TO_DELETE,
    DELETE_SYNAPSES_ALL_TO_ALL,
    PROCESS_DELETE_REQUESTS,

    UPDATE_LEAF_NODES,
    UPDATE_LOCAL_TREES,
    EXCHANGE_BRANCH_NODES,
    INSERT_BRANCH_NODES_INTO_GLOBAL_TREE,
    UPDATE_GLOBAL_TREE,

    CREATE_SYNAPSES,
    FIND_TARGET_NEURONS,
    EXCHANGE_CREATION_REQUESTS,
    PROCESS_CREATION_REQUESTS,
    CREATE_CREATION_RESPONSES,
    PROCESS_CREATION_RESPONSES,

    ADD_SYNAPSES_TO_NETWORK_GRAPH,

    EMPTY_REMOTE_NODES_CACHE,

    CAPTURE_MONITORS,

    CAPTURE_AREA_MONITORS,
    AREA_MONITORS_PREPARE,
    AREA_MONITORS_REQUEST,
    AREA_MONITORS_EXCHANGE,
    AREA_MONITORS_RECORD_DATA,
    AREA_MONITORS_LOCAL_EDGES,
    AREA_MONITORS_DISTANT_EDGES,
    AREA_MONITORS_DELETIONS,
    AREA_MONITORS_STATISTICS,
    AREA_MONITORS_FINISH,

    PRINT_IO,
};

/**
 * This number is used as a shortcut to count the number of values valid for TimerRegion
 */
constexpr size_t NUMBER_TIMERS = 48;

/**
 * This class is used to collect all sorts of different timers (see TimerRegion).
 * It provides an interface to start, stop, and print the timers
 */
class Timers {
    using time_point = std::chrono::high_resolution_clock::time_point;
    using index_type = std::vector<time_point>::size_type;

public:
    /**
     * @brief Starts the respective timer
     * @param timer The timer to start
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void start(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::start: timer_id was {}", timer_id);
        time_start[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void stop(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::stop: timer_id was: {}", timer_id);
        time_stop[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer and adds the elapsed time
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void stop_and_add(const TimerRegion timer) {
        stop(timer);
        add_start_stop_diff_to_elapsed(timer);
    }

    /**
     * @brief Adds the difference between the current start and stop time points to the elapsed time
     * @param timer The timer for which to add the difference
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void add_start_stop_diff_to_elapsed(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::add_start_stop_diff_to_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] += (time_stop[timer_id] - time_start[timer_id]);
    }

    /**
     * @brief Resets the elapsed time for the timer
     * @param timer The timer for which to reset the elapsed time
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void reset_elapsed(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::reset_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] = std::chrono::nanoseconds(0);
    }

    /**
     * @brief Returns the elapsed time for the respective timer
     * @param timer The timer for which to return the elapsed time
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     * @return The elapsed time
     */
    [[nodiscard]] static std::chrono::nanoseconds get_elapsed(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::get_elapsed: timer_id was: {}", timer_id);
        return time_elapsed[timer_id];
    }

    /**
     * @brief Prints all timers with min, max, and sum across all MPI ranks to LogFiles::EventType::Timers.
     *      Performs MPI communication.
     * @param essentials The essentials
     */
    static void print(const std::unique_ptr<Essentials>& essentials);

    /**
     * @brief Returns the current time as a string
     * @return The current time as a string
     */
    [[nodiscard]] static std::string wall_clock_time();

private:
    /**
     * @brief Casts the value of timer to an index for the vectors
     * @param timer The timer as an enum value
     * @result The timer as an index
     */
    [[nodiscard]] static index_type get_timer_index(const TimerRegion timer) noexcept {
        const auto timer_id = static_cast<index_type>(timer);
        return timer_id;
    }

    // NOLINTNEXTLINE
    static inline std::vector<time_point> time_start{ NUMBER_TIMERS };
    // NOLINTNEXTLINE
    static inline std::vector<time_point> time_stop{ NUMBER_TIMERS };

    // NOLINTNEXTLINE
    static inline std::vector<std::chrono::nanoseconds> time_elapsed{ NUMBER_TIMERS };
};