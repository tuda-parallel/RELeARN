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
#include <sstream>
#include <tuple>

TEST_F(ProbabilityKernelTest, testLinearGetterSetter) {
    LinearDistributionKernel::set_cutoff(LinearDistributionKernel::default_cutoff);

    const auto cutoff_point = KernelAdapter::get_random_linear_cutoff(mt);

    std::stringstream ss{};
    ss << "Cutoff Point: " << cutoff_point << '\n';

    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), LinearDistributionKernel::default_cutoff) << ss.str();
    ASSERT_NO_THROW(LinearDistributionKernel::set_cutoff(cutoff_point)) << ss.str();
    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), cutoff_point) << ss.str();
}

TEST_F(ProbabilityKernelTest, testLinearGetterSetterInf) {
    LinearDistributionKernel::set_cutoff(LinearDistributionKernel::default_cutoff);

    constexpr auto cutoff_point_inf = std::numeric_limits<double>::infinity();

    std::stringstream ss{};
    ss << "Cutoff Point: " << cutoff_point_inf << '\n';

    ASSERT_NO_THROW(LinearDistributionKernel::set_cutoff(cutoff_point_inf)) << ss.str();
    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), cutoff_point_inf) << ss.str();
}

TEST_F(ProbabilityKernelTest, testLinearGetterSetterException) {
    LinearDistributionKernel::set_cutoff(LinearDistributionKernel::default_cutoff);

    const auto cutoff_point = -KernelAdapter::get_random_linear_cutoff(mt);

    std::stringstream ss{};
    ss << "Cutoff Point: " << cutoff_point << '\n';

    ASSERT_THROW(LinearDistributionKernel::set_cutoff(cutoff_point), RelearnException) << ss.str();
    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), LinearDistributionKernel::default_cutoff) << ss.str();
}

TEST_F(ProbabilityKernelTest, testLinearNoFreeElements) {
    const auto cutoff_point = KernelAdapter::get_random_linear_cutoff(mt);
    LinearDistributionKernel::set_cutoff(cutoff_point);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    std::stringstream ss{};
    ss << "Cutoff Point: " << cutoff_point << '\n';
    ss << "Source Position: " << source_position << '\n';
    ss << "Target Position: " << target_position << '\n';

    const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0) << ss.str();
}

TEST_F(ProbabilityKernelTest, testLinearLinearFreeElements) {
    const auto cutoff_point = KernelAdapter::get_random_linear_cutoff(mt);
    LinearDistributionKernel::set_cutoff(cutoff_point);

    const auto& source_position = SimulationAdapter::get_random_position(mt);
    const auto& target_position = SimulationAdapter::get_random_position(mt);

    const auto attractiveness_one = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_elements = 0U; number_elements < 10000U; number_elements++) {
        const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_elements);

        const auto expected_attractiveness = attractiveness_one * number_elements;

        std::stringstream ss{};
        ss << "Cutoff Point: " << cutoff_point << '\n';
        ss << "Source Position: " << source_position << '\n';
        ss << "Target Position: " << target_position << '\n';
        ss << "Number Elements: " << number_elements << '\n';
        ss << "Attractiveness: " << attractiveness << '\n';
        ss << "Expected Attractiveness: " << expected_attractiveness << '\n';

        ASSERT_NEAR(attractiveness, expected_attractiveness, eps) << ss.str();
    }
}

TEST_F(ProbabilityKernelTest, testLinearSamePosition) {
    const auto cutoff_point = KernelAdapter::get_random_linear_cutoff(mt);
    LinearDistributionKernel::set_cutoff(cutoff_point);

    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);
    const auto converted_double = static_cast<double>(number_elements);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(position, position, number_elements);

    std::stringstream ss{};
    ss << "Cutoff Point: " << cutoff_point << '\n';
    ss << "Source Position: " << position << '\n';
    ss << "Target Position: " << position << '\n';
    ss << "Number Elements: " << number_elements << '\n';
    ss << "Attractiveness: " << attractiveness << '\n';
    ss << "Expected Attractiveness: " << converted_double << '\n';

    ASSERT_NEAR(attractiveness, converted_double, eps) << ss.str();
}

TEST_F(ProbabilityKernelTest, testLinearInf) {
    constexpr auto cutoff_point_inf = std::numeric_limits<double>::infinity();
    LinearDistributionKernel::set_cutoff(cutoff_point_inf);

    const auto& source = SimulationAdapter::get_random_position(mt);
    const auto& target = SimulationAdapter::get_random_position(mt);

    const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);

    const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(source, target, number_elements);

    std::stringstream ss{};
    ss << "Cutoff Point: " << cutoff_point_inf << '\n';
    ss << "Source Position: " << source << '\n';
    ss << "Target Position: " << target << '\n';
    ss << "Number Elements: " << number_elements << '\n';
    ss << "Attractiveness: " << attractiveness << '\n';
    ss << "Expected Attractiveness: " << static_cast<double>(number_elements) << '\n';

    ASSERT_EQ(attractiveness, static_cast<double>(number_elements)) << ss.str();
}

TEST_F(ProbabilityKernelTest, testLinearFinite) {
    const auto cutoff_point = KernelAdapter::get_random_linear_cutoff(mt);
    LinearDistributionKernel::set_cutoff(cutoff_point);

    for (auto i = 0; i < 100; i++) {
        const auto number_elements = RandomAdapter::get_random_integer<unsigned int>(0, 10000, mt);

        const auto& source = SimulationAdapter::get_random_position(mt);
        const auto& target = SimulationAdapter::get_random_position(mt);

        const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(source, target, number_elements);

        const auto difference = (source - target).calculate_2_norm();

        std::stringstream ss{};
        ss << "Cutoff Point: " << cutoff_point << '\n';
        ss << "Source Position: " << source << '\n';
        ss << "Target Position: " << target << '\n';
        ss << "Number Elements: " << number_elements << '\n';
        ss << "Attractiveness: " << attractiveness << '\n';
        ss << "Difference: " << difference << '\n';

        if (difference > cutoff_point) {
            ss << "Expected Attractiveness: 0.0\n";
            ASSERT_EQ(attractiveness, 0.0) << ss.str();
            continue;
        }

        const auto expected_attraction = number_elements * (cutoff_point - difference) / cutoff_point;

        ss << "Expected Attractiveness: " << expected_attraction << '\n';

        ASSERT_NEAR(attractiveness, expected_attraction, eps) << ss.str();
    }
}
