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

#include "Types.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cmath>
#include <numeric>

/**
 * Offers a static interface to calculate the attraction based on a gamma distribution, i.e.,
 * if x is the distance, the attraction is proportional to
 * 1/[Gamma(k)*theta^k] * x^(k-1) * exp(-x/theta)
 */
class GammaDistributionKernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    static constexpr double default_k = 1.0;
    static constexpr double default_theta = 1.0;

    /**
     * @brief Sets the shape parameter k, must be greater than 0.0
     * @param k The shape parameter, > 0.0
     * @exception Throws a RelearnException if k <= 0.0
     */
    static void set_k(const double k) {
        RelearnException::check(k > 0.0, "In GammaDistributionKernel::set_k, k was not greater than 0.0");
        GammaDistributionKernel::k = k;
        GammaDistributionKernel::gamma_divisor_inv = 1.0 / (std::tgamma(k) * std::pow(theta, k));
    }

    /**
     * @brief Returns the currently used shape parameter
     * @return The currently used shape parameter
     */
    [[nodiscard]] static double get_k() noexcept {
        return k;
    }

    /**
     * @brief Sets the scale parameter theta, must be greater than 0.0
     * @param theta The scaling parameter, > 0.0
     * @exception Throws a RelearnException if theta <= 0.0
     */
    static void set_theta(const double theta) {
        RelearnException::check(theta > 0.0, "In GammaDistributionKernel::set_theta, theta was not greater than 0.0");
        GammaDistributionKernel::theta = theta;
        GammaDistributionKernel::gamma_divisor_inv = 1.0 / (std::tgamma(k) * std::pow(theta, k));
        GammaDistributionKernel::theta_divisor = -1.0 / theta;
    }

    /**
     * @brief Returns the currently used scale parameter
     * @return The currently used scale parameter
     */
    [[nodiscard]] static double get_theta() noexcept {
        return theta;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the gamma distribution
     * @param source_position The source position s
     * @param target_position The target position t
     * @param number_free_elements The linear scaling factor k
     * @return The calculated attractiveness
     */
    [[nodiscard]] static double calculate_attractiveness_to_connect(const position_type& source_position, const position_type& target_position,
        const counter_type& number_free_elements) noexcept {
        if (number_free_elements == 0) {
            return 0.0;
        }

        const auto factor_1 = number_free_elements * gamma_divisor_inv;

        const auto x = (source_position - target_position).calculate_2_norm();

        const auto factor_2 = std::pow(x, k - 1);
        const auto factor_3 = std::exp(x * theta_divisor);

        const auto result = factor_1 * factor_2 * factor_3;

        return result;
    }

private:
    static inline double k{ default_k };
    static inline double theta{ default_theta };

    static inline double gamma_divisor_inv{ 1.0 / (std::tgamma(k) * std::pow(theta, k)) };
    static inline double theta_divisor{ -1.0 / theta };
};
