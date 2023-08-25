/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_misc.h"

#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/Utility.h"
#include "util/shuffle/shuffle.h"

#include <range/v3/view/concat.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>

TEST_F(MiscTest, testNumberDigitsInt) {
    using integer_type = int;

    for (const auto val : ranges::views::iota(integer_type{ 0 }, integer_type{ 10 })) {
        ASSERT_EQ(1, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(10, 99, mt);
        ASSERT_EQ(2, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(100, 999, mt);
        ASSERT_EQ(3, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(1000, 9999, mt);
        ASSERT_EQ(4, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(10000, 99999, mt);
        ASSERT_EQ(5, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(100000, 999999, mt);
        ASSERT_EQ(6, Util::num_digits(val));
    }
}

TEST_F(MiscTest, testNumberDigitsUnsignedInt) {
    using integer_type = unsigned int;

    for (const auto val : ranges::views::iota(integer_type{ 0 }, integer_type{ 10 })) {
        ASSERT_EQ(1, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(10, 99, mt);
        ASSERT_EQ(2, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(100, 999, mt);
        ASSERT_EQ(3, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(1000, 9999, mt);
        ASSERT_EQ(4, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(10000, 99999, mt);
        ASSERT_EQ(5, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = RandomAdapter::get_random_integer<integer_type>(100000, 999999, mt);
        ASSERT_EQ(6, Util::num_digits(val));
    }
}

TEST_F(MiscTest, testFactorial) {
    constexpr auto fac0 = Util::factorial(0ULL);
    constexpr auto fac1 = Util::factorial(1ULL);
    constexpr auto fac2 = Util::factorial(2ULL);
    constexpr auto fac3 = Util::factorial(3ULL);
    constexpr auto fac4 = Util::factorial(4ULL);
    constexpr auto fac5 = Util::factorial(5ULL);
    constexpr auto fac6 = Util::factorial(6ULL);
    constexpr auto fac7 = Util::factorial(7ULL);
    constexpr auto fac8 = Util::factorial(8ULL);
    constexpr auto fac9 = Util::factorial(9ULL);
    constexpr auto fac10 = Util::factorial(10ULL);
    constexpr auto fac11 = Util::factorial(11ULL);
    constexpr auto fac12 = Util::factorial(12ULL);
    constexpr auto fac13 = Util::factorial(13ULL);
    constexpr auto fac14 = Util::factorial(14ULL);
    constexpr auto fac15 = Util::factorial(15ULL);

    ASSERT_EQ(fac0, 1ULL);
    ASSERT_EQ(fac1, 1ULL);
    ASSERT_EQ(fac2, 2ULL);
    ASSERT_EQ(fac3, 6ULL);
    ASSERT_EQ(fac4, 24ULL);
    ASSERT_EQ(fac5, 120ULL);
    ASSERT_EQ(fac6, 720ULL);
    ASSERT_EQ(fac7, 5040ULL);
    ASSERT_EQ(fac8, 40320ULL);
    ASSERT_EQ(fac9, 362880ULL);
    ASSERT_EQ(fac10, 3628800ULL);
    ASSERT_EQ(fac11, 39916800ULL);
    ASSERT_EQ(fac12, 479001600ULL);
    ASSERT_EQ(fac13, 6227020800ULL);
    ASSERT_EQ(fac14, 87178291200ULL);
    ASSERT_EQ(fac15, 1307674368000ULL);
}

TEST_F(MiscTest, testMinMaxAccEmpty) {
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const double>{}, {}), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const float>{}, {}), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const int>{}, {}), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const size_t>{}, {}), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeMismatch) {
    const auto num_neurons = 3;
    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(num_neurons);

    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const double>{ { 4.0, 1.2 } }, extra_infos), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const float>{ { 0.8f } }, extra_infos), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const int>{ { 5, -4, 8, -6, 9 } }, extra_infos), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const size_t>{ { 10, 422, 5223, 554315 } }, extra_infos), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeAllDisabled) {
    const auto num_neurons = 3;
    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(num_neurons);

    const auto disabled_neurons = NeuronID::range(num_neurons) | ranges::to_vector;
    extra_infos->set_disabled_neurons(disabled_neurons);

    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const double>{ { 4.0, 1.2, 5.2 } }, extra_infos), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const float>{ { 0.8f, -1.6f, 65423.8f } }, extra_infos), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const int>{ { 5, -4, 8 } }, extra_infos), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::span<const size_t>{ { 10, 422, 5223 } }, extra_infos), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccDouble) {
    const auto number_enabled = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_disabled = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_static = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto number_values = number_enabled + number_disabled + number_static;

    const auto update_status = ranges::views::concat(
                                   ranges::views::repeat_n(UpdateStatus::Enabled, number_enabled),
                                   ranges::views::repeat_n(UpdateStatus::Disabled, number_disabled),
                                   ranges::views::repeat_n(UpdateStatus::Static,
                                       number_values - (number_disabled + number_enabled)))
        | ranges::to_vector | actions::shuffle(mt);

    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_values);

    std::vector<NeuronID> disabled_neurons{};
    std::vector<NeuronID> static_neurons{};
    for (const auto& neuron_id : NeuronID::range(number_values)) {
        const auto& us = update_status[neuron_id.get_neuron_id()];
        if (us == UpdateStatus::Static) {
            static_neurons.push_back(neuron_id);
        } else if (us == UpdateStatus::Disabled) {
            disabled_neurons.push_back(neuron_id);
        }
    }

    extra_infos->set_disabled_neurons(disabled_neurons);
    extra_infos->set_static_neurons(static_neurons);

    std::vector<double> values{};
    values.reserve(number_values);

    auto min = std::numeric_limits<double>::max();
    auto max = -std::numeric_limits<double>::max();
    auto sum = 0.0;

    for (auto i : ranges::views::indices(number_values)) {
        const auto random_value = RandomAdapter::get_random_double<double>(-100000.0, 100000.0, mt);

        if (update_status[i] == UpdateStatus::Enabled) {
            min = std::min(min, random_value);
            max = std::max(max, random_value);
            sum += random_value;
        }

        values.emplace_back(random_value);
    }

    const auto [minimum, maximum, accumulated, num] = Util::min_max_acc(std::span<const double>{ values }, extra_infos);

    ASSERT_EQ(minimum, min);
    ASSERT_EQ(maximum, max);
    ASSERT_NEAR(sum, accumulated, eps);
    ASSERT_EQ(number_enabled, num);
}

TEST_F(MiscTest, testMinMaxAccSizet) {
    const auto number_enabled = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_disabled = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_static = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto number_values = number_enabled + number_disabled + number_static;

    const auto update_status = ranges::views::concat(
                                   ranges::views::repeat_n(UpdateStatus::Enabled, number_enabled),
                                   ranges::views::repeat_n(UpdateStatus::Disabled, number_disabled),
                                   ranges::views::repeat_n(UpdateStatus::Static,
                                       number_values - (number_disabled + number_enabled)))
        | ranges::to_vector | actions::shuffle(mt);

    std::vector<size_t> values{};
    values.reserve(number_values);

    auto min = std::numeric_limits<size_t>::max();
    auto max = std::numeric_limits<size_t>::min();
    auto sum = size_t(0);

    for (const auto i : ranges::views::indices(number_values)) {
        const auto random_value = RandomAdapter::get_random_integer<size_t>(std::numeric_limits<size_t>::min(), std::numeric_limits<size_t>::max(), mt);

        if (update_status[i] == UpdateStatus::Enabled) {
            min = std::min(min, random_value);
            max = std::max(max, random_value);
            sum += random_value;
        }

        values.emplace_back(random_value);
    }

    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_values);

    std::vector<NeuronID> disabled_neurons{};
    std::vector<NeuronID> static_neurons{};
    for (const auto& neuron_id : NeuronID::range(number_values)) {
        const auto& us = update_status[neuron_id.get_neuron_id()];
        if (us == UpdateStatus::Static) {
            static_neurons.push_back(neuron_id);
        } else if (us == UpdateStatus::Disabled) {
            disabled_neurons.push_back(neuron_id);
        }
    }

    extra_infos->set_disabled_neurons(disabled_neurons);
    extra_infos->set_static_neurons(static_neurons);

    const auto [minimum, maximum, accumulated, num] = Util::min_max_acc(std::span<const size_t>{ values }, extra_infos);

    ASSERT_EQ(minimum, min);
    ASSERT_EQ(maximum, max);
    ASSERT_EQ(sum, accumulated);
    ASSERT_EQ(number_enabled, num);
}
