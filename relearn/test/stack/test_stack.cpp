/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_stack.h"

#include "util/Stack.h"

#include <utility>

TEST_F(StackTest, testConstructor) {
    Stack<double> stack_1{};
    ASSERT_TRUE(stack_1.empty());
    ASSERT_EQ(stack_1.size(), 0);

    Stack<bool> stack_2{};
    ASSERT_TRUE(stack_2.empty());
    ASSERT_EQ(stack_2.size(), 0);

    Stack<int> stack_3{ 100 };
    ASSERT_TRUE(stack_3.empty());
    ASSERT_EQ(stack_3.size(), 0);
    ASSERT_GE(stack_3.capacity(), 100);

    Stack<double> stack_4{ 56464161 };
    ASSERT_TRUE(stack_4.empty());
    ASSERT_EQ(stack_4.size(), 0);
    ASSERT_GE(stack_4.capacity(), 56464161);
}

TEST_F(StackTest, testReserve) {
    Stack<double> stack_1{ 1000 };
    stack_1.reserve(5000);
    ASSERT_GE(stack_1.capacity(), 5000);
    ASSERT_TRUE(stack_1.empty());
    ASSERT_EQ(stack_1.size(), 0);

    Stack<double> stack_2{ 1000 };
    stack_2.reserve(1000);
    ASSERT_GE(stack_2.capacity(), 1000);
    ASSERT_TRUE(stack_2.empty());
    ASSERT_EQ(stack_2.size(), 0);

    Stack<double> stack_3{ 1000 };
    stack_3.reserve(20);
    ASSERT_GE(stack_3.capacity(), 1000);
    ASSERT_TRUE(stack_3.empty());
    ASSERT_EQ(stack_3.size(), 0);
}

TEST_F(StackTest, testEmplaceBackAndTop) {
    using type = std::pair<double, double>;
    Stack<type> stack{};

    stack.emplace_back(0.1, 0.2);
    ASSERT_FALSE(stack.empty());
    ASSERT_EQ(stack.size(), 1);
    ASSERT_GE(stack.capacity(), 1);

    ASSERT_EQ(stack.top(), type(0.1, 0.2));

    const auto current_capacity = stack.capacity();

    for (auto i = 1; i < current_capacity + 500; i++) {
        stack.emplace_back(i + 0.1, i + 0.2);
        ASSERT_FALSE(stack.empty());
        ASSERT_EQ(stack.size(), i + 1);
        ASSERT_EQ(stack.top(), type(i + 0.1, i + 0.2));
    }
}

TEST_F(StackTest, testPop) {
    Stack<int> stack{};

    for (auto i = 1000; i >= 0; i--) {
        stack.emplace_back(i);
    }

    for (auto i = 0; i < 1001; i++) {
        auto& top = stack.top();
        ASSERT_EQ(top, i);
        stack.pop();
    }

    ASSERT_TRUE(stack.empty());
}

TEST_F(StackTest, testPopBack) {
    Stack<int> stack{};

    for (auto i = 1000; i >= 0; i--) {
        stack.emplace_back(i);
    }

    for (auto i = 0; i < 1001; i++) {
        auto top = stack.pop_back();
        ASSERT_EQ(top, i);
    }

    ASSERT_TRUE(stack.empty());
}

TEST_F(StackTest, testTop) {
    Stack<int> stack{};

    for (auto i = 1000; i >= 0; i--) {
        stack.emplace_back(i);
        stack.top() *= 2;
    }

    for (auto i = 0; i < 1001; i++) {
        auto top = stack.pop_back();
        ASSERT_EQ(top, i * 2);
    }
}

TEST_F(StackTest, testEmplaceBack) {
    Stack<int> stack{};

    for (auto i = 1000; i >= 0; i--) {
        stack.emplace_back(i) *= 2;
    }

    for (auto i = 0; i < 1001; i++) {
        auto top = stack.pop_back();
        ASSERT_EQ(top, i * 2);
    }
}
