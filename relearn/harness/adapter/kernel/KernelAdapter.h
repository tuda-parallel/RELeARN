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

#include "adapter/random/RandomAdapter.h"

#include "algorithm/Kernel/Kernel.h"

#include <random>
#include <sstream>
#include <string>

class KernelAdapter {
public:
    static double get_random_gamma_k(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(0.001, 10.0, mt);
    }

    static double get_random_gamma_theta(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(0.001, 100.0, mt);
    }

    static double get_random_gaussian_mu(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(-10000.0, 10000.0, mt);
    }

    static double get_random_gaussian_sigma(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(0.001, 10000.0, mt);
    }

    static double get_random_linear_cutoff(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(0.001, 1000.0, mt);
    }

    static double get_random_weibull_k(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(0.001, 10.0, mt);
    }

    static double get_random_weibull_b(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_double(0.001, 10000.0, mt);
    }

    static KernelType get_random_kernel_type(std::mt19937& mt) noexcept {
        const auto choice = RandomAdapter::get_random_integer<int>(0, 3, mt);

        switch (choice) {
        case 0:
            return KernelType::Gamma;
        case 1:
            return KernelType::Gaussian;
        case 2:
            return KernelType::Linear;
        case 3:
            return KernelType::Weibull;
        }

        return KernelType::Gamma;
    }

    template <typename AdditionalCellAttributes>
    static std::string set_random_kernel(std::mt19937& mt) {
        const auto kernel_choice = get_random_kernel_type(mt);

        Kernel<AdditionalCellAttributes>::set_kernel_type(kernel_choice);

        std::stringstream ss{};

        ss << kernel_choice;

        if (kernel_choice == KernelType::Gamma) {
            const auto k = get_random_gamma_k(mt);
            const auto theta = get_random_gamma_theta(mt);

            ss << '\t' << k << '\t' << theta;

            GammaDistributionKernel::set_k(k);
            GammaDistributionKernel::set_theta(theta);
        }

        if (kernel_choice == KernelType::Gaussian) {
            const auto sigma = get_random_gaussian_sigma(mt);
            const auto mu = get_random_gaussian_mu(mt);

            ss << '\t' << sigma << '\t' << mu;

            GaussianDistributionKernel::set_sigma(sigma);
            GaussianDistributionKernel::set_mu(mu);
        }

        if (kernel_choice == KernelType::Linear) {
            const auto cutoff = get_random_linear_cutoff(mt);

            ss << '\t' << cutoff;

            LinearDistributionKernel::set_cutoff(cutoff);
        }

        if (kernel_choice == KernelType::Weibull) {
            const auto k = get_random_weibull_k(mt);
            const auto b = get_random_weibull_b(mt);

            ss << '\t' << k << '\t' << b;

            WeibullDistributionKernel::set_k(k);
            WeibullDistributionKernel::set_b(b);
        }

        return ss.str();
    }
};
