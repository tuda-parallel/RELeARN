/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CalciumIO.h"

#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <fstream>
#include <optional>
#include <sstream>
#include <unordered_map>

CalciumIO::initial_value_calculator CalciumIO::load_initial_function(const std::filesystem::path& path_to_file) {
    const auto& [initial_calculator, _] = load_initial_and_target_function(path_to_file);
    return initial_calculator;
}

CalciumIO::target_value_calculator CalciumIO::load_target_function(const std::filesystem::path& path_to_file) {
    const auto& [_, target_calculator] = load_initial_and_target_function(path_to_file);
    return target_calculator;
}

std::pair<CalciumIO::initial_value_calculator, CalciumIO::target_value_calculator>
CalciumIO::load_initial_and_target_function(const std::filesystem::path& path_to_file) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_enable_interrupts: Opening the file was not successful");

    std::optional<double> default_initial_calcium{};
    std::optional<double> default_target_calcium{};

    std::unordered_map<NeuronID::value_type, double> id_to_initial{};
    std::unordered_map<NeuronID::value_type, double> id_to_target{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        NeuronID::value_type neuron_id{};
        double initial_calcium{};
        double target_calcium{};

        bool success = (sstream >> neuron_id) && (sstream >> initial_calcium) && (sstream >> target_calcium);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        if (neuron_id == 0) {
            RelearnException::check(!default_initial_calcium.has_value(),
                "CalciumIO: {} had more than one default neuron (with id 0)", path_to_file.string());

            default_initial_calcium = initial_calcium;
            default_target_calcium = target_calcium;

            continue;
        }

        // The IDs start at 0
        neuron_id--;

        const auto& found_initial = id_to_initial.find(neuron_id) != id_to_initial.end();
        const auto& found_target = id_to_target.find(neuron_id) != id_to_target.end();

        RelearnException::check(!found_initial, "CalciumIO: Found the neuron id {} twice", (neuron_id + 1));

        id_to_initial[neuron_id] = initial_calcium;
        id_to_target[neuron_id] = target_calcium;
    }

    auto initial_calculator = [lookup = std::move(id_to_initial), default_initial = default_initial_calcium]([[maybe_unused]] MPIRank mpi_rank, NeuronID::value_type neuron_id) {
        const auto& contains = lookup.find(neuron_id) != lookup.end();
        if (contains) {
            const double initial = lookup.at(neuron_id);
            return initial;
        }

        RelearnException::check(default_initial.has_value(), "Initial Calcium Calculator: Got id {} but I don't have a default value", neuron_id);
        return default_initial.value();
    };

    auto target_calculator = [lookup = std::move(id_to_target), default_target = default_target_calcium]([[maybe_unused]] MPIRank mpi_rank, NeuronID::value_type neuron_id) {
        const auto& contains = lookup.find(neuron_id) != lookup.end();
        if (contains) {
            const double target = lookup.at(neuron_id);
            return target;
        }

        RelearnException::check(default_target.has_value(), "Target Calcium Calculator: Got id {} but I don't have a default value", neuron_id);
        return default_target.value();
    };

    return std::make_pair(std::move(initial_calculator), std::move(target_calculator));
}
