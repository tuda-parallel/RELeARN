/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_event.h"

#include "io/Event.h"

#include <sstream>

TEST_F(EventTest, testEventPhasePrint) {
    auto test = [](const EventPhase ep, const char expected) {
        std::stringstream ss{};
        ss << ep;

        const auto str = ss.str();

        ASSERT_EQ(str.size(), 1);
        ASSERT_EQ(str[0], expected);
    };

    test(EventPhase::DurationBegin, 'B');
    test(EventPhase::DurationEnd, 'E');
    test(EventPhase::Complete, 'X');
    test(EventPhase::Counter, 'C');
    test(EventPhase::AsyncNestableStart, 'b');
    test(EventPhase::AsyncNestableInstant, 'n');
    test(EventPhase::AsyncNestableEnd, 'e');
    test(EventPhase::FlowStart, 's');
    test(EventPhase::FlowStep, 't');
    test(EventPhase::FlowEnd, 'f');
    test(EventPhase::ObjectCreated, 'N');
    test(EventPhase::ObjectSnapshot, 'O');
    test(EventPhase::ObjectDestroyed, 'D');
    test(EventPhase::Metadata, 'M');
    test(EventPhase::MemoryDumpGlobal, 'V');
    test(EventPhase::MemoryDumpProcess, 'v');
    test(EventPhase::Mark, 'R');
    test(EventPhase::ClockSync, 'c');
    test(EventPhase::ContextBegin, '(');
    test(EventPhase::ContextEnd, ')');
}

TEST_F(EventTest, testInstantEventScopePrint) {
    auto test = [](const InstantEventScope es, const char expected) {
        std::stringstream ss{};
        ss << es;

        const auto str = ss.str();

        ASSERT_EQ(str.size(), 1);
        ASSERT_EQ(str[0], expected);
    };

    test(InstantEventScope::Global, 'g');
    test(InstantEventScope::Process, 'p');
    test(InstantEventScope::Thread, 't');
}

TEST_F(EventTest, testDurationBeginPrint) {
    constexpr static auto expected_output = "{\"name\": \"this-name-123\", \"ph\": \"B\", \"pid\": 65, \"tid\": 135, \"ts\": 1236.05}";

    const auto event_trace = Event::create_duration_begin_event("this-name-123", {}, 1236.05, 65, 135, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testDurationBeginArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"this-name-123\", \"ph\": \"B\", \"pid\": 65, \"tid\": 135, \"ts\": 1236.05, \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_duration_begin_event("this-name-123", {}, 1236.05, 65, 135, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testDurationBeginCategoriesPrint) {
    constexpr static auto expected_output = "{\"name\": \"this-name-123\", \"ph\": \"B\", \"pid\": 65, \"tid\": 135, \"ts\": 1236.05, \"cat\": \"async,sync\"}";

    const auto event_trace = Event::create_duration_begin_event("this-name-123", { EventCategory::async, EventCategory::sync }, 1236.05, 65, 135, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testDurationBeginCategoriesArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"this-name-123\", \"ph\": \"B\", \"pid\": 65, \"tid\": 135, \"ts\": 1236.05, \"cat\": \"async,sync\", \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_duration_begin_event("this-name-123", { EventCategory::sync, EventCategory::async }, 1236.05, 65, 135, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testDurationEndPrint) {
    constexpr static auto expected_output = "{\"ph\": \"E\", \"pid\": 159, \"tid\": 951, \"ts\": 65423365}";

    const auto event_trace = Event::create_duration_end_event(65423365.0, 159, 951);

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testCounterPrint) {
    constexpr static auto expected_output = "{\"name\": \"that-name-123\", \"ph\": \"C\", \"pid\": 12, \"tid\": 0, \"ts\": 165}";

    const auto event_trace = Event::create_counter_event("that-name-123", {}, 165, 12, 0, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testCounterCategoriesPrint) {
    constexpr static auto expected_output = "{\"name\": \"that-name-123\", \"ph\": \"C\", \"pid\": 12, \"tid\": 0, \"ts\": 165, \"cat\": \"mpi\"}";

    const auto event_trace = Event::create_counter_event("that-name-123", { EventCategory::mpi }, 165, 12, 0, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testCounterArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"that-name-123\", \"ph\": \"C\", \"pid\": 12, \"tid\": 0, \"ts\": 165, \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_counter_event("that-name-123", {}, 165, 12, 0, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testCounterCategoriesArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"that-name-123\", \"ph\": \"C\", \"pid\": 12, \"tid\": 0, \"ts\": 165, \"cat\": \"async,calculation,mpi\", \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_counter_event("that-name-123", { EventCategory::async, EventCategory::calculation, EventCategory::mpi }, 165, 12, 0, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testInstantGlobalPrint) {
    constexpr static auto expected_output = "{\"name\": \"instant-753-name\", \"ph\": \"i\", \"pid\": 111, \"tid\": 56654, \"ts\": 999, \"s\": \"g\"}";

    const auto event_trace = Event::create_instant_event("instant-753-name", {}, InstantEventScope::Global, 999, 111, 56654, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testInstantProcessPrint) {
    constexpr static auto expected_output = "{\"name\": \"instant-7532-name\", \"ph\": \"i\", \"pid\": 1112, \"tid\": 562654, \"ts\": 9992, \"s\": \"p\"}";

    const auto event_trace = Event::create_instant_event("instant-7532-name", {}, InstantEventScope::Process, 9992, 1112, 562654, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testInstantThreadPrint) {
    constexpr static auto expected_output = "{\"name\": \"instant-7853-name\", \"ph\": \"i\", \"pid\": 8111, \"tid\": 856654, \"ts\": 8999, \"s\": \"t\"}";

    const auto event_trace = Event::create_instant_event("instant-7853-name", {}, InstantEventScope::Thread, 8999, 8111, 856654, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testInstantGlobalArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"instant-753-name\", \"ph\": \"i\", \"pid\": 111, \"tid\": 56654, \"ts\": 999, \"s\": \"g\", \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_instant_event("instant-753-name", {}, InstantEventScope::Global, 999, 111, 56654, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testInstantProcessArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"instant-7532-name\", \"ph\": \"i\", \"pid\": 1112, \"tid\": 562654, \"ts\": 9992, \"s\": \"p\", \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_instant_event("instant-7532-name", {}, InstantEventScope::Process, 9992, 1112, 562654, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testInstantThreadArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"instant-7853-name\", \"ph\": \"i\", \"pid\": 8111, \"tid\": 856654, \"ts\": 8999, \"s\": \"t\", \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_instant_event("instant-7853-name", {}, InstantEventScope::Thread, 8999, 8111, 856654, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testCompletionPrint) {
    constexpr static auto expected_output = "{\"name\": \"9119-completion-name\", \"ph\": \"X\", \"pid\": 333, \"tid\": 555, \"ts\": 777.777, \"dur\": 963.369}";

    const auto event_trace = Event::create_complete_event("9119-completion-name", {}, 963.369, 777.777, 333, 555, {});

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}

TEST_F(EventTest, testCompletionArgumentsPrint) {
    constexpr static auto expected_output = "{\"name\": \"9119-completion-name\", \"ph\": \"X\", \"pid\": 3333, \"tid\": 5555, \"ts\": 7777.7777, \"dur\": 9630.0369, \"args\": {\"arg1\": val1, \"arg3\": val3}}";

    const auto event_trace = Event::create_complete_event("9119-completion-name", {}, 9630.0369, 7777.7777, 3333, 5555, { { "arg1", "val1" }, { "arg3", "val3" } });

    std::stringstream ss{};
    ss << event_trace;

    const auto str = ss.str();

    EXPECT_EQ(str, expected_output);
}
