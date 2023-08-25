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

#include "gtest/gtest.h"

#include <array>
#include <iostream>
#include <tuple>

TEST_F(ProbabilityKernelTest, testGammaSetterGetter) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    ASSERT_EQ(GammaDistributionKernel::get_k(), GammaDistributionKernel::default_k);
    ASSERT_EQ(GammaDistributionKernel::get_theta(), GammaDistributionKernel::default_theta);

    const auto k = KernelAdapter::get_random_gamma_k(mt);
    const auto theta = KernelAdapter::get_random_gamma_theta(mt);

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    ASSERT_EQ(GammaDistributionKernel::get_k(), k);
    ASSERT_EQ(GammaDistributionKernel::get_theta(), theta);
}

TEST_F(ProbabilityKernelTest, testGammaSetterGetterException) {
    const auto k = KernelAdapter::get_random_gamma_k(mt);
    const auto theta = KernelAdapter::get_random_gamma_theta(mt);

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    ASSERT_THROW(GammaDistributionKernel::set_k(0.0), RelearnException);
    ASSERT_THROW(GammaDistributionKernel::set_k(-k), RelearnException);
    ASSERT_THROW(GammaDistributionKernel::set_theta(0.0), RelearnException);
    ASSERT_THROW(GammaDistributionKernel::set_theta(-theta), RelearnException);

    ASSERT_EQ(GammaDistributionKernel::get_k(), k);
    ASSERT_EQ(GammaDistributionKernel::get_theta(), theta);
}

TEST_F(ProbabilityKernelTest, testGammaNoFreeElements) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    const auto k = KernelAdapter::get_random_gamma_k(mt);
    const auto theta = KernelAdapter::get_random_gamma_theta(mt);

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(ProbabilityKernelTest, testGammaLinearElements) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    const auto k = KernelAdapter::get_random_gamma_k(mt);
    const auto theta = KernelAdapter::get_random_gamma_theta(mt);

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    const auto attractiveness_one = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_free_elements = 0U; number_free_elements < 10000U; number_free_elements++) {
        const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_free_elements);

        const auto expected_attractiveness = attractiveness_one * number_free_elements;
        ASSERT_NEAR(attractiveness, expected_attractiveness, eps);
    }
}

TEST_F(ProbabilityKernelTest, testGammaPrecalculatedValues) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    std::array<std::tuple<double, double, double, double>, 5> precalculated_values{
        {
            { 100.0, 10.0, 10.0, 0.012511 },
            { 20.0, 1.0, 10.0, 0.0135335 },
            { 0.15, 1.0, 1.0, 0.860708 },
            { 7.0, 1.0, 1 / 1.4, 0.0000776322 },
            { 6.25, 3.0, 11.4, 0.00761931 },
        }
    };

    const auto sqrt3 = std::sqrt(3);

    for (const auto& [position_difference, k, theta, expected] : precalculated_values) {
        const auto& source_position = SimulationAdapter::get_random_position(mt);
        const auto& target_position = source_position + (position_difference / sqrt3);

        GammaDistributionKernel::set_k(k);
        GammaDistributionKernel::set_theta(theta);

        const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);
        ASSERT_NEAR(attractiveness, expected, eps);
    }
}