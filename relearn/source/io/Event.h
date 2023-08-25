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

#include "fmt/ostream.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <set>
#include <utility>
#include <vector>

/**
 * Specifies categories an event can have.
 * These are custom an can be extended
 */
enum class EventCategory {
    async,
    calculation,
    mpi,
    sync,
};

/**
 * @brief Pretty-prints the event category to the chosen stream
 * @param out The stream to which to print the event category
 * @param event_category The event category to print
 * @return The argument out, now altered with the event category
 */
inline std::ostream& operator<<(std::ostream& out, const EventCategory event_category) {
    if (event_category == EventCategory::mpi) {
        return out << "mpi";
    }

    if (event_category == EventCategory::calculation) {
        return out << "calculation";
    }

    if (event_category == EventCategory::async) {
        return out << "async";
    }

    if (event_category == EventCategory::sync) {
        return out << "sync";
    }

    return out << "cat-unkown";
}

template <>
struct fmt::formatter<EventCategory> : ostream_formatter { };

/**
 * Specifies the phase of an event. These are not custom and should not change. See:
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
enum class EventPhase {
    DurationBegin,
    DurationEnd,
    Complete,
    Instant,
    Counter,
    AsyncNestableStart,
    AsyncNestableInstant,
    AsyncNestableEnd,
    FlowStart,
    FlowStep,
    FlowEnd,
    ObjectCreated,
    ObjectSnapshot,
    ObjectDestroyed,
    Metadata,
    MemoryDumpGlobal,
    MemoryDumpProcess,
    Mark,
    ClockSync,
    ContextBegin,
    ContextEnd
};

/**
 * @brief Pretty-prints the event phase to the chosen stream
 * @param out The stream to which to print the event phase
 * @param event_phase The event phase to print
 * @return The argument out, now altered with the event phase
 */
inline std::ostream& operator<<(std::ostream& out, const EventPhase event_phase) {
    if (event_phase == EventPhase::DurationBegin) {
        return out << 'B';
    }

    if (event_phase == EventPhase::DurationEnd) {
        return out << 'E';
    }

    if (event_phase == EventPhase::Complete) {
        return out << 'X';
    }

    if (event_phase == EventPhase::Instant) {
        return out << 'i';
    }

    if (event_phase == EventPhase::Counter) {
        return out << 'C';
    }

    if (event_phase == EventPhase::AsyncNestableStart) {
        return out << 'b';
    }

    if (event_phase == EventPhase::AsyncNestableInstant) {
        return out << 'n';
    }

    if (event_phase == EventPhase::AsyncNestableEnd) {
        return out << 'e';
    }

    if (event_phase == EventPhase::FlowStart) {
        return out << 's';
    }

    if (event_phase == EventPhase::FlowStep) {
        return out << 't';
    }

    if (event_phase == EventPhase::FlowEnd) {
        return out << 'f';
    }

    if (event_phase == EventPhase::ObjectCreated) {
        return out << 'N';
    }

    if (event_phase == EventPhase::ObjectSnapshot) {
        return out << 'O';
    }

    if (event_phase == EventPhase::ObjectDestroyed) {
        return out << 'D';
    }

    if (event_phase == EventPhase::Metadata) {
        return out << 'M';
    }

    if (event_phase == EventPhase::MemoryDumpGlobal) {
        return out << 'V';
    }

    if (event_phase == EventPhase::MemoryDumpProcess) {
        return out << 'v';
    }

    if (event_phase == EventPhase::Mark) {
        return out << 'R';
    }

    if (event_phase == EventPhase::ClockSync) {
        return out << 'c';
    }

    if (event_phase == EventPhase::ContextBegin) {
        return out << '(';
    }

    if (event_phase == EventPhase::ContextEnd) {
        return out << ')';
    }

    return out << '?';
}

template <>
struct fmt::formatter<EventPhase> : ostream_formatter { };

/**
 * An instant event can be of global, process, and thread level.
 * There is a forth level (default), which collapses to thread.
 */
enum class InstantEventScope {
    Global,
    Process,
    Thread
};

/**
 * @brief Pretty-prints the instant event scope to the chosen stream
 * @param out The stream to which to print the instant event scope
 * @param event_scope The instant event scope to print
 * @return The argument out, now altered with the instant event scope
 */
inline std::ostream& operator<<(std::ostream& out, const InstantEventScope event_scope) {
    if (event_scope == InstantEventScope::Global) {
        return out << 'g';
    }

    if (event_scope == InstantEventScope::Process) {
        return out << 'p';
    }

    if (event_scope == InstantEventScope::Thread) {
        return out << 't';
    }

    return out << '?';
}

template <>
struct fmt::formatter<InstantEventScope> : ostream_formatter { };

/**
 * Provides the possibility to create events in the style of the Google Trace Event Format:
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
class Event {
public:
    /**
     * @brief Creates an event that signals the begin of some duration
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param tracing_clock The clock at the start of the event
     * @param process_id The id of the process to which the event belongs
     * @param thread_id The id of the thread to which the event belongs
     * @param args The arguments for the event, can be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories,
        const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
        return Event(std::move(name), std::move(categories), EventPhase::DurationBegin, {}, tracing_clock, process_id, thread_id, std::move(args), {});
    }

    /**
     * @brief Creates an event that signals the begin of some duration with default arguments for process-id, thread-id, and tracing-clock
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param args The arguments for the event, can be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args);

    /**
     * @brief Creates an event that signals the begin of some duration with default arguments for process-id, thread-id, and tracing-clock.
     *      Prints the event directly to the file (if event tracing is enabled)
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param args The arguments for the event, can be empty
     * @param flush True if the file should be flushed
     */
    static void create_and_print_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

    /**
     * @brief Creates an event that signals the end of some duration. Always ends the latest begun event
     * @param tracing_clock The clock at the start of the event
     * @param process_id The id of the process to which the event belongs
     * @param thread_id The id of the thread to which the event belongs
     * @return The created object that can be printed using operator<<
     */
    static Event create_duration_end_event(const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id) {
        return Event({}, {}, EventPhase::DurationEnd, {}, tracing_clock, process_id, thread_id, {}, {});
    }

    /**
     * @brief Creates an event that signals the end of some duration with default arguments for process-id, thread-id, and tracing-clock. Always ends the latest begun event
     * @return The created object that can be printed using operator<<
     */
    static Event create_duration_end_event();

    /**
     * @brief Creates an event that signals the end of some duration with default arguments for process-id, thread-id, and tracing-clock. Always ends the latest begun event.
     *      Prints the event directly to the file (if event tracing is enabled)
     * @param flush True if the file should be flushed
     */
    static void create_and_print_duration_end_event(bool flush = false);

    /**
     * @brief Creates an event that signals the completion of some event (not a duration)
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param duration The duration of the event
     * @param tracing_clock The clock at the start of the event
     * @param process_id The id of the process to which the event belongs
     * @param thread_id The id of the thread to which the event belongs
     * @param args The arguments for the event, can be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_complete_event(std::string&& name, std::set<EventCategory>&& categories, const double duration,
        const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
        return Event(std::move(name), std::move(categories), EventPhase::Complete, {}, tracing_clock, process_id, thread_id, std::move(args), duration);
    }

    /**
     * @brief Creates an event that signals the completion of some event (not a duration event) with default arguments for process-id, thread-id, and tracing-clock.
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param duration The duration of the event
     * @param args The arguments for the event, can be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_complete_event(std::string&& name, std::set<EventCategory>&& categories, double duration, std::vector<std::pair<std::string, std::string>>&& args);

    /**
     * @brief Creates an event that signals the completion of some event (not a duration event) with default arguments for process-id, thread-id, and tracing-clock.
     *      Prints the event directly to the file (if event tracing is enabled)
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param duration The duration of the event
     * @param args The arguments for the event, can be empty
     * @param flush True if the file should be flushed
     */
    static void create_and_print_complete_event(std::string&& name, std::set<EventCategory>&& categories, double duration, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

    /**
     * @brief Creates an event that signals same instant
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param scope The scope of the event, can be global, process, or thread
     * @param tracing_clock The clock at the start of the event
     * @param process_id The id of the process to which the event belongs
     * @param thread_id The id of the thread to which the event belongs
     * @param args The arguments for the event, can be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_instant_event(std::string&& name, std::set<EventCategory>&& categories, const InstantEventScope scope,
        const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
        return Event(std::move(name), std::move(categories), EventPhase::Instant, scope, tracing_clock, process_id, thread_id, std::move(args), {});
    }

    /**
     * @brief Creates an event that signals same instant with default arguments for process-id, thread-id, and tracing-clock.
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param scope The scope of the event, can be global, process, or thread
     * @param args The arguments for the event, can be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_instant_event(std::string&& name, std::set<EventCategory>&& categories, InstantEventScope scope, std::vector<std::pair<std::string, std::string>>&& args);

    /**
     * @brief Creates an event that signals same instant with default arguments for process-id, thread-id, and tracing-clock.
     *      Prints the event directly to the file (if event tracing is enabled)
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param scope The scope of the event, can be global, process, or thread
     * @param args The arguments for the event, can be empty
     * @param flush True if the file should be flushed
     */
    static void create_and_print_instant_event(std::string&& name, std::set<EventCategory>&& categories, InstantEventScope scope, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

    /**
     * @brief Creates an event that signals the change of some counter (values specified in args)
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param tracing_clock The clock at the start of the event
     * @param process_id The id of the process to which the event belongs
     * @param thread_id The id of the thread to which the event belongs
     * @param args The arguments for the event, should not be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_counter_event(std::string&& name, std::set<EventCategory>&& categories,
        const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
        return Event(std::move(name), std::move(categories), EventPhase::Counter, {}, tracing_clock, process_id, thread_id, std::move(args), {});
    }

    /**
     * @brief Creates an event that signals the change of some counter (values specified in args) with default arguments for process-id, thread-id, and tracing-clock.
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param args The arguments for the event, should not be empty
     * @return The created object that can be printed using operator<<
     */
    static Event create_counter_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args);

    /**
     * @brief Creates an event that signals the change of some counter (values specified in args) with default arguments for process-id, thread-id, and tracing-clock.
     *      Prints the event directly to the file (if event tracing is enabled)
     * @param name The name of the event
     * @param categories The categories for the event, can be empty
     * @param args The arguments for the event, should not be empty
     * @param flush True if the file should be flushed
     */
    static void create_and_print_counter_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

private:
    Event(std::optional<std::string>&& _name, std::set<EventCategory>&& _categories, const EventPhase _phase,
        std::optional<InstantEventScope>&& _scope, const double _tracing_clock, const std::uint64_t _process_id,
        const std::uint64_t _thread_id, std::vector<std::pair<std::string, std::string>>&& _arguments, std::optional<double>&& _duration)
        : name(std::move(_name))
        , categories(std::move(_categories))
        , phase(_phase)
        , scope(std::move(_scope))
        , tracing_clock(_tracing_clock)
        , process_id(_process_id)
        , thread_id(_thread_id)
        , arguments(std::move(_arguments))
        , duration(std::move(_duration)) { }

    std::optional<std::string> name{};
    std::set<EventCategory> categories{};
    EventPhase phase{};
    std::optional<InstantEventScope> scope{};
    double tracing_clock{};
    std::uint64_t process_id{};
    std::uint64_t thread_id{};
    std::vector<std::pair<std::string, std::string>> arguments{};
    std::optional<double> duration{};

    friend std::ostream& operator<<(std::ostream& out, const Event& event);
};

/**
 * @brief Prints the event in JSON format as one line to the stream. Does not add a line break
 * @param out The stream to which to print the instant event scope
 * @param event The event to print
 * @return The argument out, now altered with the event
 */
inline std::ostream& operator<<(std::ostream& out, const Event& event) {
    out << '{';

    if (event.name.has_value()) {
        out << "\"name\": \"" << event.name.value() << "\", ";
    }

    out << fmt::format("\"ph\": \"{}\", \"pid\": {}, \"tid\": {}, \"ts\": {}",
        event.phase, event.process_id, event.thread_id, event.tracing_clock);

    if (event.duration.has_value()) {
        out << fmt::format(", \"dur\": {}", event.duration.value());
    }

    if (event.scope.has_value()) {
        out << fmt::format(", \"s\": \"{}\"", event.scope.value());
    }

    if (!event.categories.empty()) {
        out << fmt::format(", \"cat\": \"{}", *event.categories.begin());
        for (auto i = 1; i < event.categories.size(); i++) {
            auto curr = event.categories.begin();
            std::advance(curr, i);
            out << ',' << *curr;
        }
        out << "\"";
    }

    if (!event.arguments.empty()) {
        out << ", \"args\": {";

        const auto& [first_name, first_value] = event.arguments[0];
        out << fmt::format("\"{}\": {}", first_name, first_value);

        for (auto i = 1; i < event.arguments.size(); i++) {
            const auto& [name, value] = event.arguments[i];
            out << fmt::format(", \"{}\": {}", name, value);
        }

        out << '}';
    }

    out << '}';

    return out;
}

template <>
struct fmt::formatter<Event> : ostream_formatter { };
