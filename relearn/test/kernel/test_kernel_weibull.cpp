/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_kernel.h"

#include "adapter/kernel/KernelAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/kernel/KernelAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/Cells.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

TEST_F(ProbabilityKernelTest, testWeibullSetterGetter) {
    WeibullDistributionKernel::set_k(WeibullDistributionKernel::default_k);
    WeibullDistributionKernel::set_b(WeibullDistributionKernel::default_b);

    ASSERT_EQ(WeibullDistributionKernel::get_k(), WeibullDistributionKernel::default_k);
    ASSERT_EQ(WeibullDistributionKernel::get_b(), WeibullDistributionKernel::default_b);

    const auto k = KernelAdapter::get_random_weibull_k(mt);
    const auto b = KernelAdapter::get_random_weibull_b(mt);

    WeibullDistributionKernel::set_k(k);
    WeibullDistributionKernel::set_b(b);

    ASSERT_EQ(WeibullDistributionKernel::get_k(), k);
    ASSERT_EQ(WeibullDistributionKernel::get_b(), b);
}

TEST_F(ProbabilityKernelTest, testWeibullSetterGetterException) {
    const auto k = KernelAdapter::get_random_weibull_k(mt);
    const auto b = KernelAdapter::get_random_weibull_b(mt);

    WeibullDistributionKernel::set_k(k);
    WeibullDistributionKernel::set_b(b);

    ASSERT_THROW(WeibullDistributionKernel::set_k(0.0), RelearnException);
    ASSERT_THROW(WeibullDistributionKernel::set_k(-k), RelearnException);
    ASSERT_THROW(WeibullDistributionKernel::set_b(0.0), RelearnException);
    ASSERT_THROW(WeibullDistributionKernel::set_b(-b), RelearnException);

    ASSERT_EQ(WeibullDistributionKernel::get_k(), k);
    ASSERT_EQ(WeibullDistributionKernel::get_b(), b);
}

TEST_F(ProbabilityKernelTest, testWeibullNoFreeElements) {
    WeibullDistributionKernel::set_k(WeibullDistributionKernel::default_k);
    WeibullDistributionKernel::set_b(WeibullDistributionKernel::default_b);

    const auto k = KernelAdapter::get_random_weibull_k(mt);
    const auto b = KernelAdapter::get_random_weibull_b(mt);

    WeibullDistributionKernel::set_k(k);
    WeibullDistributionKernel::set_b(b);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    const auto attractiveness = WeibullDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(ProbabilityKernelTest, testWeibullLinearElements) {
    WeibullDistributionKernel::set_k(WeibullDistributionKernel::default_k);
    WeibullDistributionKernel::set_b(WeibullDistributionKernel::default_b);

    const auto k = KernelAdapter::get_random_weibull_k(mt);
    const auto b = KernelAdapter::get_random_weibull_b(mt);

    WeibullDistributionKernel::set_k(k);
    WeibullDistributionKernel::set_b(b);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    const auto attractiveness_one = WeibullDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_free_elements = 0U; number_free_elements < 10000U; number_free_elements++) {
        const auto attractiveness = WeibullDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_free_elements);

        const auto expected_attractiveness = attractiveness_one * number_free_elements;
        ASSERT_NEAR(attractiveness, expected_attractiveness, eps);
    }
}

TEST_F(ProbabilityKernelTest, testWeibullPrecalculatedValues) {
    WeibullDistributionKernel::set_k(WeibullDistributionKernel::default_k);
    WeibullDistributionKernel::set_b(WeibullDistributionKernel::default_b);

    std::array<std::tuple<double, double, double, double>, 5> precalculated_values{
        {
            { 2.0, 0.0001, 10.0, 0.0019801 },
            { 5.0, 3.05176E-10, 100.0, 0.00721371 },
            { 3.0, 0.037037037, 4.0, 0.166126 },
            { 1.0, 1.0, 11.5, 1.01301E-05 },
            { 5.8, 0.000593534, 1.4, 0.0172375 },
        }
    };

    const auto sqrt3 = std::sqrt(3);

    for (const auto& [k, b, position_difference, expected] : precalculated_values) {
        const auto& source_position = SimulationAdapter::get_random_position(mt);
        const auto& target_position = source_position + (position_difference / sqrt3);

        WeibullDistributionKernel::set_k(k);
        WeibullDistributionKernel::set_b(b);

        const auto attractiveness = WeibullDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);
        ASSERT_NEAR(attractiveness, expected, eps);
    }
}
