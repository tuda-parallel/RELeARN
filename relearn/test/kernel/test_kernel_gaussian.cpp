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

#include <gtest/gtest.h>
#include <range/v3/action/sort.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/sliding.hpp>

TEST_F(ProbabilityKernelTest, testGaussianGetterSetter) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);
    const auto mu = KernelAdapter::get_random_gaussian_mu(mt);

    ASSERT_EQ(GaussianDistributionKernel::get_mu(), GaussianDistributionKernel::default_mu);
    ASSERT_EQ(GaussianDistributionKernel::get_sigma(), GaussianDistributionKernel::default_sigma);

    ASSERT_NO_THROW(GaussianDistributionKernel::set_sigma(sigma));

    ASSERT_EQ(GaussianDistributionKernel::get_mu(), GaussianDistributionKernel::default_mu);
    ASSERT_EQ(GaussianDistributionKernel::get_sigma(), sigma);

    ASSERT_NO_THROW(GaussianDistributionKernel::set_mu(mu));

    ASSERT_EQ(GaussianDistributionKernel::get_mu(), mu);
    ASSERT_EQ(GaussianDistributionKernel::get_sigma(), sigma);
}

TEST_F(ProbabilityKernelTest, testGaussianGetterSetterExceptions) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = -KernelAdapter::get_random_gaussian_sigma(mt);

    ASSERT_THROW(GaussianDistributionKernel::set_sigma(0.0), RelearnException);

    ASSERT_EQ(GaussianDistributionKernel::get_mu(), GaussianDistributionKernel::default_mu);
    ASSERT_EQ(GaussianDistributionKernel::get_sigma(), GaussianDistributionKernel::default_sigma);

    ASSERT_THROW(GaussianDistributionKernel::set_sigma(sigma), RelearnException);

    ASSERT_EQ(GaussianDistributionKernel::get_mu(), GaussianDistributionKernel::default_mu);
    ASSERT_EQ(GaussianDistributionKernel::get_sigma(), GaussianDistributionKernel::default_sigma);
}

TEST_F(ProbabilityKernelTest, testGaussianNoFreeElements) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    GaussianDistributionKernel::set_sigma(sigma);
    const auto attractiveness = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(ProbabilityKernelTest, testGaussianLinearFreeElements) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    GaussianDistributionKernel::set_sigma(sigma);
    const auto attractiveness_one = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_free_elements = 0U; number_free_elements < 10000U; number_free_elements++) {
        const auto attractiveness = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_free_elements);

        const auto expected_attractiveness = attractiveness_one * number_free_elements;
        ASSERT_NEAR(attractiveness, expected_attractiveness, eps);
    }
}

TEST_F(ProbabilityKernelTest, testGaussianSamePosition) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);
    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);
    const auto converted_double = static_cast<double>(number_elements);

    const auto& position = SimulationAdapter::get_random_position(mt);

    GaussianDistributionKernel::set_sigma(sigma);
    const auto attractiveness = GaussianDistributionKernel::calculate_attractiveness_to_connect(position, position, number_elements);

    ASSERT_NEAR(attractiveness, converted_double, eps);
}

TEST_F(ProbabilityKernelTest, testGaussianVariableSigma) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    const auto sigmas = ranges::views::generate_n([this]() { return KernelAdapter::get_random_gaussian_sigma(mt); }, 100)
        | ranges::to_vector
        | ranges::actions::sort;

    const auto get_attractiveness = [&source_position, &target_position, &number_elements]() {
        return GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_elements);
    };

    const auto attractiveness_pairs = sigmas
        | ranges::views::transform([&get_attractiveness](const auto sigma) {
              GaussianDistributionKernel::set_sigma(sigma);
              return get_attractiveness();
          })
        | ranges::views::sliding(2);

    for (const auto [attractiveness_a, attractiveness_b] : attractiveness_pairs) {
        ASSERT_LE(attractiveness_a, attractiveness_b);
    }
}

TEST_F(ProbabilityKernelTest, testGaussianVariablePosition) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);
    GaussianDistributionKernel::set_sigma(sigma);

    const auto& source_position = SimulationAdapter::get_random_position(mt);

    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);

    const auto smaller_distance_to_source_position = [&source_position](const Vec3d& pos_a, const Vec3d& pos_b) {
        const auto dist_a = (pos_a - source_position).calculate_squared_2_norm();
        const auto dist_b = (pos_b - source_position).calculate_squared_2_norm();
        return dist_a < dist_b;
    };

    const auto positions = ranges::views::generate_n([this]() { return SimulationAdapter::get_random_position(mt); }, 100)
        | ranges::to_vector
        | ranges::actions::sort(smaller_distance_to_source_position);

    std::vector<double> attractivenesses{};
    for (auto i = 0; i < 100; i++) {
        const auto attr_a = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, positions[i], number_elements);
        const auto attr_b = GaussianDistributionKernel::calculate_attractiveness_to_connect(positions[i], source_position, number_elements);

        ASSERT_NEAR(attr_a, attr_b, eps);
        attractivenesses.emplace_back(attr_a);
    }

    for (auto i = 1; i < 100; i++) {
        const auto attractiveness_a = attractivenesses[i - 1];
        const auto attractiveness_b = attractivenesses[i];

        ASSERT_GE(attractiveness_a, attractiveness_b);
    }
}

TEST_F(ProbabilityKernelTest, testGaussianConstantDistance) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);
    GaussianDistributionKernel::set_sigma(sigma);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& [x, y, z] = source_position;

    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);

    const auto distance = SimulationAdapter::get_random_position_element(mt);

    const auto sqrt3 = std::sqrt(3);

    const Vec3d target_position_1{ x + distance, y + distance, z + distance };
    const Vec3d target_position_2{ x + (sqrt3 * distance), y, z };
    const Vec3d target_position_3{ x, y + (sqrt3 * distance), z };
    const Vec3d target_position_4{ x, y, z + (sqrt3 * distance) };

    const auto attr_1 = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position_1, number_elements);
    const auto attr_2 = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position_2, number_elements);
    const auto attr_3 = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position_3, number_elements);
    const auto attr_4 = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position_4, number_elements);

    ASSERT_NEAR(attr_1, attr_2, eps);
    ASSERT_NEAR(attr_1, attr_3, eps);
    ASSERT_NEAR(attr_1, attr_4, eps);
}

TEST_F(ProbabilityKernelTest, testGaussianShiftedMu) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    const auto sigma = KernelAdapter::get_random_gaussian_sigma(mt);
    GaussianDistributionKernel::set_sigma(sigma);

    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);

    const Vec3d source = SimulationAdapter::get_random_position(mt);

    for (auto i = 0; i < 100; i++) {
        const auto mu = KernelAdapter::get_random_gaussian_mu(mt);
        GaussianDistributionKernel::set_mu(mu);

        const auto attr_a = GaussianDistributionKernel::calculate_attractiveness_to_connect(source, source, number_elements);

        GaussianDistributionKernel::set_mu(0.0);
        const auto attr_b = GaussianDistributionKernel::calculate_attractiveness_to_connect(source, source - Vec3d{ mu, 0.0, 0.0 }, number_elements);
        const auto attr_c = GaussianDistributionKernel::calculate_attractiveness_to_connect(source, source - Vec3d{ 0.0, mu, 0.0 }, number_elements);
        const auto attr_d = GaussianDistributionKernel::calculate_attractiveness_to_connect(source, source - Vec3d{ 0.0, 0.0, mu }, number_elements);

        ASSERT_NEAR(attr_a, attr_b, eps);
        ASSERT_NEAR(attr_a, attr_c, eps);
        ASSERT_NEAR(attr_a, attr_d, eps);
    }
}

TEST_F(ProbabilityKernelTest, testGaussianPrecalculatedValues) {
    GaussianDistributionKernel::set_mu(GaussianDistributionKernel::default_mu);
    GaussianDistributionKernel::set_sigma(GaussianDistributionKernel::default_sigma);

    std::array<std::tuple<double, double, double>, 5> precalculated_values{
        {
            { 100.0, 250.0, 0.85214378896621133 },
            { 20.0, 100.0, 0.96078943915232320 },
            { 10.0, 0.3, 0.0 },
            { 10.0, 20.3, 0.784533945772685 },
            { 15.0, 175, 0.992679984005486 },
        }
    };

    const auto sqrt3 = std::sqrt(3);

    for (const auto& [position_difference, sigma, golden_attractiveness] : precalculated_values) {
        const auto& source_position = SimulationAdapter::get_random_position(mt);
        const auto& target_position = source_position + (position_difference / sqrt3);

        GaussianDistributionKernel::set_sigma(sigma);
        const auto attractiveness = GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);
        ASSERT_NEAR(attractiveness, golden_attractiveness, eps);
    }
}
