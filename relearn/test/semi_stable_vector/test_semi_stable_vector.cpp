/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_semi_stable_vector.h"

#include <structure/BaseCell.h>
#include "structure/Cell.h"
#include "structure/OctreeNode.h"
#include "util/SemiStableVector.h"

#include <iterator>

template <typename T>
void testSize(SemiStableVector<T>& ssv, size_t expected_size) {
    const auto begin = ssv.begin();
    const auto end = ssv.end();

    ASSERT_EQ(std::distance(begin, end), expected_size);

    const auto cbegin = ssv.cbegin();
    const auto cend = ssv.cend();

    ASSERT_EQ(std::distance(begin, end), expected_size);

    ASSERT_EQ(expected_size, ssv.size());

    if (expected_size == 0) {
        ASSERT_TRUE(ssv.empty());
    } else {
        ASSERT_FALSE(ssv.empty());
    }
}

template <typename T>
void testSizeConst(const SemiStableVector<T>& ssv, size_t expected_size) {
    const auto begin = ssv.begin();
    const auto end = ssv.end();

    ASSERT_EQ(std::distance(begin, end), expected_size);

    const auto cbegin = ssv.cbegin();
    const auto cend = ssv.cend();

    ASSERT_EQ(std::distance(begin, end), expected_size);

    ASSERT_EQ(expected_size, ssv.size());

    if (expected_size == 0) {
        ASSERT_TRUE(ssv.empty());
    } else {
        ASSERT_FALSE(ssv.empty());
    }
}

TEST_F(SemiStableVectorTest, testConstructor) {
    SemiStableVector<double> ssv_1{};
    testSize(ssv_1, 0);
    testSizeConst(ssv_1, 0);

    SemiStableVector<int> ssv_2{};
    testSize(ssv_2, 0);
    testSizeConst(ssv_2, 0);

    SemiStableVector<unsigned int> ssv_3{};
    testSize(ssv_3, 0);
    testSizeConst(ssv_3, 0);

    SemiStableVector<OctreeNode<Cell<BaseCell<true, false, false, true>>>> ssv_4{};
    testSize(ssv_4, 0);
    testSizeConst(ssv_4, 0);
}

TEST_F(SemiStableVectorTest, testPushReference) {
    SemiStableVector<std::pair<int, int>> ssv{};

    auto check = [](const SemiStableVector<std::pair<int, int>>& ssv, int i) {
        const auto& [a1, a2] = ssv[i];
        ASSERT_EQ(a1, i);
        ASSERT_EQ(a2, i * 10);

        const auto& [b1, b2] = ssv.at(i);
        ASSERT_EQ(b1, i);
        ASSERT_EQ(b2, i * 10);
    };

    auto val_0 = std::pair(0, 00);
    auto val_1 = std::pair(1, 10);
    auto val_2 = std::pair(2, 20);
    auto val_3 = std::pair(3, 30);
    auto val_4 = std::pair(4, 40);
    auto val_5 = std::pair(5, 50);
    auto val_6 = std::pair(6, 60);

    ssv.push_back(val_3);
    ssv.push_back(val_4);
    ssv.push_front(val_2);
    ssv.push_back(val_5);
    ssv.push_front(val_1);
    ssv.push_front(val_0);
    ssv.push_back(val_6);

    testSize(ssv, 7);
    testSizeConst(ssv, 7);

    for (auto i = 0; i < 7; i++) {
        const auto& [a1, a2] = ssv[i];
        ASSERT_EQ(a1, i);
        ASSERT_EQ(a2, i * 10);

        const auto& [b1, b2] = ssv.at(i);
        ASSERT_EQ(b1, i);
        ASSERT_EQ(b2, i * 10);

        check(ssv, i);
    }
}

TEST_F(SemiStableVectorTest, testPushMove) {
    SemiStableVector<std::pair<int, int>> ssv{};

    auto check = [](const SemiStableVector<std::pair<int, int>>& ssv, int i) {
        const auto& [a1, a2] = ssv[i];
        ASSERT_EQ(a1, i);
        ASSERT_EQ(a2, i * 10);

        const auto& [b1, b2] = ssv.at(i);
        ASSERT_EQ(b1, i);
        ASSERT_EQ(b2, i * 10);
    };

    ssv.push_back({ 3, 30 });
    ssv.push_back({ 4, 40 });
    ssv.push_front({ 2, 20 });
    ssv.push_back({ 5, 50 });
    ssv.push_front({ 1, 10 });
    ssv.push_front({ 0, 0 });
    ssv.push_back({ 6, 60 });

    testSize(ssv, 7);
    testSizeConst(ssv, 7);

    for (auto i = 0; i < 7; i++) {
        const auto& [a1, a2] = ssv[i];
        ASSERT_EQ(a1, i);
        ASSERT_EQ(a2, i * 10);

        const auto& [b1, b2] = ssv.at(i);
        ASSERT_EQ(b1, i);
        ASSERT_EQ(b2, i * 10);

        check(ssv, i);
    }
}

TEST_F(SemiStableVectorTest, testEmplace) {
    SemiStableVector<std::pair<int, int>> ssv{};

    auto check = [](const SemiStableVector<std::pair<int, int>>& ssv, int i) {
        const auto& [a1, a2] = ssv[i];
        ASSERT_EQ(a1, i * 10);
        ASSERT_EQ(a2, i);

        const auto& [b1, b2] = ssv.at(i);
        ASSERT_EQ(b1, i * 10);
        ASSERT_EQ(b2, i);
    };

    auto& ref_3 = ssv.emplace_back(30, 3);
    auto& ref_4 = ssv.emplace_back(40, 4);
    auto& ref_2 = ssv.emplace_front(20, 2);
    auto& ref_5 = ssv.emplace_back(50, 5);
    auto& ref_1 = ssv.emplace_front(10, 1);
    auto& ref_0 = ssv.emplace_front(0, 0);
    auto& ref_6 = ssv.emplace_back(60, 6);

    testSize(ssv, 7);
    testSizeConst(ssv, 7);

    for (auto i = 0; i < 7; i++) {
        const auto& [a1, a2] = ssv[i];
        ASSERT_EQ(a1, i * 10);
        ASSERT_EQ(a2, i);

        const auto& [b1, b2] = ssv.at(i);
        ASSERT_EQ(b1, i * 10);
        ASSERT_EQ(b2, i);

        check(ssv, i);
    }

    ASSERT_EQ(ref_0, ssv[0]);
    ASSERT_EQ(ref_1, ssv[1]);
    ASSERT_EQ(ref_2, ssv[2]);
    ASSERT_EQ(ref_3, ssv[3]);
    ASSERT_EQ(ref_4, ssv[4]);
    ASSERT_EQ(ref_5, ssv[5]);
    ASSERT_EQ(ref_6, ssv[6]);
}

TEST_F(SemiStableVectorTest, testResize) {
    SemiStableVector<double> ssv{};

    ssv.resize(20);
    testSize(ssv, 20);
    testSizeConst(ssv, 20);

    ssv.resize(0);
    testSize(ssv, 0);
    testSizeConst(ssv, 0);

    ssv.resize(20, 4.0);
    testSize(ssv, 20);
    testSizeConst(ssv, 20);

    for (auto i = 0; i < 20; i++) {
        ASSERT_EQ(ssv[i], 4.0);
    }

    ssv.clear();
    testSize(ssv, 0);
    testSizeConst(ssv, 0);

    ssv.resize(4, 3.0);
    testSize(ssv, 4);
    testSizeConst(ssv, 4);

    ssv.resize(8, 1.0);
    testSize(ssv, 8);
    testSizeConst(ssv, 8);

    ASSERT_EQ(ssv[0], 3.0);
    ASSERT_EQ(ssv[1], 3.0);
    ASSERT_EQ(ssv[2], 3.0);
    ASSERT_EQ(ssv[3], 3.0);
    ASSERT_EQ(ssv[4], 1.0);
    ASSERT_EQ(ssv[5], 1.0);
    ASSERT_EQ(ssv[6], 1.0);
    ASSERT_EQ(ssv[7], 1.0);
}

TEST_F(SemiStableVectorTest, testIterators) {
    SemiStableVector<float> ssv{};

    for (auto i = 0; i < iterations; i++) {
        ssv.push_back(float(i));
    }

    for (auto i = 0; i < iterations; i++) {
        ASSERT_EQ(ssv[i], float(i));
    }

    for (auto it = ssv.begin(); it != ssv.end(); ++it) {
        ASSERT_EQ(*it, float(std::distance(ssv.begin(), it)));
    }

    for (auto it = ssv.cbegin(); it != ssv.cend(); ++it) {
        ASSERT_EQ(*it, float(std::distance(ssv.cbegin(), it)));
    }

    const auto& ref = ssv;

    for (auto it = ref.begin(); it != ref.end(); ++it) {
        ASSERT_EQ(*it, float(std::distance(ref.begin(), it)));
    }

    for (auto it = ref.cbegin(); it != ref.cend(); ++it) {
        ASSERT_EQ(*it, float(std::distance(ref.cbegin(), it)));
    }
}

TEST_F(SemiStableVectorTest, testPointerStability) {
    using value_type = OctreeNode<Cell<BaseCell<true, false, false, true>>>;
    constexpr std::size_t bound = 1024 * 1024;

    SemiStableVector<value_type> ssv{};
    std::vector<value_type*> ptrs{};
    ptrs.reserve(bound);

    for (std::size_t i = 0; i < bound; i++) {
        auto& ref = ssv.emplace_back();
        ptrs.emplace_back(&ref);
    }

    for (std::size_t i = 0; i < bound; i++) {
        auto& ref = ssv[i];
        ASSERT_EQ(&ref, ptrs[i]);
    }
}

TEST_F(SemiStableVectorTest, testReferences) {
    SemiStableVector<double> ssv{};
    for (auto i = 0; i < 1024; i++) {
        auto& ref = ssv.emplace_back(i);
    }

    for (auto i = 0; i < 1024; i++) {
        ssv[i] = double(i) * 2.0;
    }
    testSize(ssv, 1024);
    testSizeConst(ssv, 1024);

    for (auto i = 0; i < 1024; i++) {
        ASSERT_EQ(ssv[i], double(i) * 2.0);
        ASSERT_EQ(ssv.at(i), double(i) * 2.0);
    }
    testSize(ssv, 1024);
    testSizeConst(ssv, 1024);

    for (auto i = 0; i < 1024; i++) {
        ssv.at(i) = double(i) * 5.0;
    }

    for (auto i = 0; i < 1024; i++) {
        ASSERT_EQ(ssv[i], double(i) * 5.0);
        ASSERT_EQ(ssv.at(i), double(i) * 5.0);
    }
    testSize(ssv, 1024);
    testSizeConst(ssv, 1024);
}
