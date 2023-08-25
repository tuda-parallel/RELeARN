/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_id.h"

#include "adapter/random/RandomAdapter.h"

#include <compare>
#include <cstdint>
#include <functional>
#include <type_traits>

TEST_F(NeuronIDTest, testNeuronIDUninitialized) { // NOLINT
    const auto id = NeuronID::uninitialized_id();

    ASSERT_FALSE(id.is_initialized());
    ASSERT_FALSE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TEST_F(NeuronIDTest, testNeuronIDVirtual) { // NOLINT
    const auto id = NeuronID::virtual_id();

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_TRUE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TEST_F(NeuronIDTest, testNeuronIDConstructorDefault) { // NOLINT
    NeuronID id{};

    ASSERT_FALSE(id.is_initialized());
    ASSERT_FALSE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TEST_F(NeuronIDTest, testNeuronIDConstructorOnlyID) { // NOLINT
    const auto id_val = RandomAdapter::template get_random_integer(NeuronID::limits::min, NeuronID::limits::max, this->mt);

    const NeuronID id{ id_val };

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_TRUE(id.is_local());

    ASSERT_EQ(id.get_neuron_id(), id_val);
    ASSERT_EQ(static_cast<std::uint64_t>(id), id_val);
}

TEST_F(NeuronIDTest, testNeuronIDConstructorLocal) {
    const auto id_val = RandomAdapter::template get_random_integer(NeuronID::limits::min, NeuronID::limits::max, this->mt);

    const NeuronID id{ false, id_val };

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_TRUE(id.is_local());

    ASSERT_EQ(id.get_neuron_id(), id_val);
    ASSERT_EQ(static_cast<std::uint64_t>(id), id_val);
}

TEST_F(NeuronIDTest, testNeuronIDConstructorVirtual) {
    const auto id_val = RandomAdapter::template get_random_integer(NeuronID::limits::min, NeuronID::limits::max, this->mt);

    const NeuronID id{ true, id_val };

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_TRUE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TEST_F(NeuronIDTest, testNeuronIDComparisons1) { // NOLINT
    constexpr static auto min = NeuronID::limits::min;
    constexpr static auto max = NeuronID::limits::max;

    const auto get_random_id = [this]() { return NeuronID{ RandomAdapter::template get_random_integer(min, max, this->mt) }; };

    const auto id1 = get_random_id();
    const auto id2 = get_random_id();

    ASSERT_EQ(id1 <=> id2, id1.get_neuron_id() <=> id2.get_neuron_id());
    ASSERT_EQ(NeuronID{}, NeuronID{});
}

TEST_F(NeuronIDTest, testNeuronIDComparisons2) { // NOLINT
    constexpr static auto min = NeuronID::limits::min;
    constexpr static auto max = NeuronID::limits::max;

    const auto get_random_id = [this]() {
        auto res = NeuronID{
            RandomAdapter::get_random_bool(this->mt),
            RandomAdapter::template get_random_integer(min, max, this->mt)
        };

        // res.is_initialized() = this->RandomAdapter::get_random_bool(this->mt);
        return res;
    };

    const auto id1 = get_random_id();
    const auto id2 = get_random_id();

    const auto failure_message = fmt::format("ID 1: {}\n ID 2: {}\n", id1, id2);

    // members are compared in order of declaration
    // -> if any compares not equal,
    // then the result of that first comparison that is not equal
    // is the result of the comparison
    const auto comp = id1 <=> id2;

    if (const auto initialized_comparison = id1.is_initialized() <=> id2.is_initialized();
        std::is_neq(initialized_comparison)) {
        EXPECT_EQ(comp, initialized_comparison) << failure_message;
        return;
    }

    if (const auto virtual_comparison = id1.is_virtual() <=> id2.is_virtual();
        std::is_neq(virtual_comparison)) {
        EXPECT_EQ(comp, virtual_comparison) << failure_message;
        return;
    }

    // id's are only valid if they are initialized and virtual
    if (id1.is_initialized() && !id1.is_virtual()) {
        if (const auto id_comparison = id1.get_neuron_id() <=> id2.get_neuron_id();
            std::is_neq(id_comparison)) {
            EXPECT_EQ(comp, id_comparison) << failure_message;
            return;
        }
    }
}
