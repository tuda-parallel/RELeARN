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

/**
 * Offers a static interface to calculate the attraction linearly with
 * a cut-off, i.e., the attraction is 0 if the distance is larger then the cut-off,
 * constant if the cut-off is infinite, and linear interpolated based on the
 * distance is neither is the case
 */
class LinearDistributionKernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    static constexpr double default_cutoff = std::numeric_limits<double>::infinity();

    /**
     * @brief Sets cut-off, must be greater than or equal to 0.0
     * @param cutoff_point The cut-off parameter, >= 0.0
     * @exception Throws a RelearnException if cutoff_point < 0.0
     */
    static void set_cutoff(const double cutoff_point) {
        RelearnException::check(cutoff_point >= 0.0, "In LinearDistributionKernel::set_sigma, sigma was less than 0.0");
        LinearDistributionKernel::cutoff_point = cutoff_point;
    }

    /**
     * @brief Returns the currently used cut-off parameter
     * @return The currently used cut-off parameter
     */
    [[nodiscard]] static double get_cutoff() noexcept {
        return cutoff_point;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of ||s - t||_2,
     *      i.e., if this is smaller than the cut-off point, the return value is k, otherwise it's 0.0
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

        const auto cast_number_elements = static_cast<double>(number_free_elements);

        if (std::isinf(cutoff_point)) {
            return cast_number_elements;
        }

        const auto x = (source_position - target_position).calculate_2_norm();
        if (x > cutoff_point) {
            return 0.0;
        }

        const auto factor = x / cutoff_point;

        return (1 - factor) * cast_number_elements;
    }

private:
    static inline double cutoff_point{ default_cutoff };
};
