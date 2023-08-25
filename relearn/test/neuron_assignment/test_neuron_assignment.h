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

#include "RelearnTest.hpp"

#include <cmath>

class NeuronAssignmentTest : public RelearnTest {
protected:
    static double calculate_box_length(const size_t number_neurons, const double um_per_neuron) noexcept {
        return std::ceil(std::pow(static_cast<double>(number_neurons), 1 / 3.)) * um_per_neuron;
    }
};
