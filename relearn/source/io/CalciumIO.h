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
#include "util/MPIRank.h"

#include <filesystem>
#include <functional>
#include <utility>

/**
 * Provides the functionality to load the target calcium values and the initial calcium values
 * on a per-neuron basis (currently, the MPI rank is ignored). The format is as follows (lines starting with # are ignored):
 * <neuron_id + 1> <initial_calcium_value> <target_calcium_value>
 * or once in the file:
 * 0 <default_initial_calcium_value> <target_calcium_value>
 */
class CalciumIO {
public:
    using initial_value_calculator = std::function<double(MPIRank, NeuronID::value_type)>;
    using target_value_calculator = std::function<double(MPIRank, NeuronID::value_type)>;

    /**
     * @brief Loads from a file a function that maps the current MPI rank and a neuron's id to its initial calcium value
     * @param path_to_file The file that contains the description
     * @return The function to calculate the initial calcium value
     */
    static initial_value_calculator load_initial_function(const std::filesystem::path& path_to_file);

    /**
     * @brief Loads from a file a function that maps the current MPI rank and a neuron's id to its target calcium value
     * @param path_to_file The file that contains the description
     * @return The function to calculate the target calcium value
     */
    static target_value_calculator load_target_function(const std::filesystem::path& path_to_file);

    /**
     * @brief Loads from a file the functions that map the current MPI rank and a neuron's id to its initial and target calcium value
     * @param path_to_file The file that contains the description
     * @return A pair of (a) the initial calcium calculator and (b) the target calcium calculator
     */
    static std::pair<initial_value_calculator, target_value_calculator> load_initial_and_target_function(const std::filesystem::path& path_to_file);
};
