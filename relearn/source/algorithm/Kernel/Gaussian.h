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

#include <numeric>

/**
 * Offers a static interface to calculate the attraction based on a gaussian, i.e.,
 * if x is the distance, the attraction is proportional to
 * exp(-((x-mu)/sigma)^2)
 */
class GaussianDistributionKernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    static constexpr double default_mu = 0.0; // In Sebastian's work: 0.0
    static constexpr double default_sigma = 750.0; // In Sebastian's work: 750.0

    /**
     * @brief Sets the variance sigma, must be greater than 0.0
     * @param sigma The variance, > 0.0
     * @exception Throws a RelearnException if sigma <= 0.0
     */
    static void set_sigma(const double sigma) {
        RelearnException::check(sigma > 0.0, "In GaussianDistributionKernel::set_sigma, sigma was not greater than 0.0");
        GaussianDistributionKernel::sigma = sigma;
        GaussianDistributionKernel::squared_sigma_inv = 1.0 / (sigma * sigma);
    }

    /**
     * @brief Returns the currently used variance
     * @return The currently used variance
     */
    [[nodiscard]] static double get_sigma() noexcept {
        return sigma;
    }

    /**
     * @brief Sets the offset mu
     * @param mu The offset
     */
    static void set_mu(const double mu) noexcept {
        GaussianDistributionKernel::mu = mu;
    }

    /**
     * @brief Returns the currently used offset
     * @return The currently used offset
     */
    [[nodiscard]] static double get_mu() noexcept {
        return mu;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of
     *      k * exp(-||s - t||_2^2 / sigma^2)
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

        const auto position_diff = target_position - source_position;
        const auto x = position_diff.calculate_2_norm();
        const auto numerator = (x - mu) * (x - mu);

        const auto exponent = -numerator * squared_sigma_inv;

        // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
        const auto exp_val = std::exp(exponent);
        const auto ret_val = number_free_elements * exp_val;

        return ret_val;
    }

private:
    static inline double mu{ default_mu };
    static inline double sigma{ default_sigma };
    static inline double squared_sigma_inv{ 1.0 / (default_sigma * default_sigma) };
};
