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

#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"
#include "util/ranges/views/IO.hpp"

#include <range/v3/view/getlines.hpp>

#include <algorithm>
#include <filesystem>
#include <istream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class FileValidator {
public:
    /**
     * @brief Checks if the specified file contains only synapses between neurons with specified ids (only works locally).
     * @param path_synapses The path to the file in which the synapses are stored (with the ids starting at 1)
     * @param neuron_ids The neuron ids between which the synapses should be formed. Must be sorted ascendingly
     * @tparam synapse_weight The type of synapse weight
     * @exception Throws an exception if the allocation of memory fails
     * @return Returns true iff the file has the correct format and only ids in neuron_ids are present
     */
    template <typename synapse_weight>
    static bool check_edges_from_file(const std::filesystem::path& path_synapses, const std::vector<NeuronID::value_type>& neuron_ids) {
        std::ifstream file_synapses(path_synapses, std::ios::binary | std::ios::in);

        std::set<NeuronID::value_type> ids_in_file{};

        for (const auto& line : ranges::getlines(file_synapses) | views::filter_not_comment_not_empty_line) {
            NeuronID::value_type source_id = 0;
            NeuronID::value_type target_id = 0;
            synapse_weight weight = 0;

            std::stringstream sstream(line);
            const bool success = (sstream >> source_id) && (sstream >> target_id) && (sstream >> weight);

            if (!success) {
                return false;
            }

            // The neurons start with 1
            source_id--;
            target_id--;

            ids_in_file.insert(source_id);
            ids_in_file.insert(target_id);
        }

        return std::ranges::all_of(ids_in_file, [&neuron_ids](const NeuronID::value_type val) {
            return std::ranges::binary_search(neuron_ids, val);
        });
    }
};
