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
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <regex>

/**
 * Over all mpi ranks, neurons can be assigned to the same area. This area is identified by a name (string). Each mpi rank assigns this area an individual id.
 * Hence, the same area (same name) can have different ids on two mpi ranks.
 * This class helps to convert between the area id on this mpi rank and its unique name.
 */
class LocalAreaTranslator {
public:
    /**
     * Constructor
     * @param area_id_to_area_name Maps the id of an area to its name
     * @param neuron_id_to_area_id Map the neuron id to its assigned area id
     */
    LocalAreaTranslator(std::vector<RelearnTypes::area_name> area_id_to_area_name,
        std::vector<RelearnTypes::area_id> neuron_id_to_area_id)
        : area_id_to_area_name(std::move(area_id_to_area_name))
        , neuron_id_to_area_id(std::move(neuron_id_to_area_id)) {

        RelearnException::check(!this->area_id_to_area_name.empty(), "LocalAreaTranslator::Area id to area name is empty");
        RelearnException::check(!this->neuron_id_to_area_id.empty(), "LocalAreaTranslator:: Neuron id to area id is empty");

        RelearnException::check(this->area_id_to_area_name.size() <= this->neuron_id_to_area_id.size(), "LocalAreaTranslator:: More area ids than neurons");

        for (const uint64_t& area_id : this->neuron_id_to_area_id) {
            RelearnException::check(area_id >= 0 && area_id < this->area_id_to_area_name.size(), "LocalAreaTranslator:: Invalid area id {}. Must be between 0 and {}", area_id, this->area_id_to_area_name.size());
        }

        const auto num_area_names = this->area_id_to_area_name.size();
        const auto set = this->area_id_to_area_name | ranges::to<std::set>;
        RelearnException::check(num_area_names == set.size(), "LocalAreaTranslator::Area name must be unique on single mpi rank");
    }

    /**
     * Returns the area id in which the neuron lays
     * @param neuron_id Id of the neuron
     * @return Area id
     */
    [[nodiscard]] RelearnTypes::area_id get_area_id_for_neuron_id(RelearnTypes::neuron_id neuron_id) const {
        RelearnException::check(neuron_id < neuron_id_to_area_id.size(), "LocalAreaTranslator::get_area_id_for_neuron_id: Neuron id is too large. {} < {}", neuron_id, neuron_id_to_area_id.size());
        return neuron_id_to_area_id[neuron_id];
    }

    /**
     * Returns the area name of an area id
     * @param area_id The id of the area
     * @return Name of the area
     */
    [[nodiscard]] const RelearnTypes::area_name& get_area_name_for_area_id(RelearnTypes::area_id area_id) const {
        RelearnException::check(area_id < area_id_to_area_name.size(), "LocalAreaTranslator::get_area_name_for_area_id: Area id is too large. {} < {}", area_id, area_id_to_area_name.size());
        return area_id_to_area_name[area_id];
    }

    /**
     * Returns the name of an area in which the neuron lays
     * @param neuron_id Id of the neuron
     * @return Name of the area
     */
    [[nodiscard]] const RelearnTypes::area_name& get_area_name_for_neuron_id(RelearnTypes::neuron_id neuron_id) const {
        return get_area_name_for_area_id(get_area_id_for_neuron_id(neuron_id));
    }

    /**
     * Returns the area id on this mpi rank for the area name
     * @param area_name Name of the area
     * @return Id of the area
     */
    [[nodiscard]] RelearnTypes::area_id get_area_id_for_area_name(const RelearnTypes::area_name& area_name) const {
        const auto it = ranges::find(area_id_to_area_name, area_name);
        RelearnException::check(it != area_id_to_area_name.end(), "LocalAreaTranslator::get_area_id_for_area_name: Area name {} is unknown", area_name);
        return static_cast<RelearnTypes::area_id>(ranges::distance(area_id_to_area_name.begin(), it));
    }

    /**
     * Returns if the area name is known on this mpi rank
     * @param area_name Name of the area
     * @return True, if area name exists on this mpi rank
     */
    [[nodiscard]] bool knows_area_name(const RelearnTypes::area_name& area_name) const noexcept {
        return ranges::contains(area_id_to_area_name, area_name);
    }

    /**
     * Returns the number of neurons on this mpi rank (over all local areas)
     * @return Number of neurons on this mpi rank
     */
    [[nodiscard]] RelearnTypes::number_neurons_type get_number_neurons_in_total() const noexcept {
        return neuron_id_to_area_id.size();
    }

    /**
     * Vector with all used area names on this mpi rank
     * @return Vector of area names
     */
    [[nodiscard]] const std::vector<RelearnTypes::area_name>& get_all_area_names() const noexcept {
        return area_id_to_area_name;
    }

    /**
     * Returns number of areas available on this mpi rank
     * @return Number of areas
     */
    [[nodiscard]] size_t get_number_of_areas() const noexcept {
        return area_id_to_area_name.size();
    }

    /**
     * @brief Return number of neurons placed with a certain area name
     * @return Number of neurons currently stored under the given area name
     */
    [[nodiscard]] RelearnTypes::number_neurons_type get_number_neurons_in_area(const RelearnTypes::area_id& area_id) const {
        return static_cast<RelearnTypes::number_neurons_type>(ranges::count(neuron_id_to_area_id, area_id));
    }

    /**
     * @brief Returns all neuron ids in a specific area on this rank
     * @param neuron_id_vs_area_id
     * @param my_area_ids Vector of area ids in which the neuron ids must lay
     * @return Vector of neuron ids within the specified area in my_area_ids
     */
    [[nodiscard]] std::unordered_set<NeuronID> get_neuron_ids_in_area(RelearnTypes::area_id my_area_id) const {
        RelearnException::check(my_area_id < area_id_to_area_name.size(), "LocalAreaTranslator::get_neuron_ids_in_area: Area id {} is too large", my_area_id);

        return NeuronID::range(neuron_id_to_area_id.size())
            | ranges::views::filter(equal_to(my_area_id), lookup(neuron_id_to_area_id, &NeuronID::get_neuron_id))
            | ranges::to<std::unordered_set>;
    }

    /**
     * @brief Returns all neuron ids in a specific area on this rank
     * @param neuron_id_vs_area_id
     * @param my_area_ids Vector of area ids in which the neuron ids must lay
     * @return Vector of neuron ids within the specified area in my_area_ids
     */
    [[nodiscard]] std::vector<NeuronID> get_neuron_ids_in_areas(const std::vector<RelearnTypes::area_id>& my_area_ids) const {
        const auto is_id_in_my_ranks_areas = [my_area_ids, this](const NeuronID neuron_id) {
            return ranges::contains(my_area_ids, neuron_id_to_area_id[neuron_id.get_neuron_id()]);
        };

        return NeuronID::range(neuron_id_to_area_id.size())
            | ranges::views::filter(is_id_in_my_ranks_areas)
            | ranges::to_vector;
    }

    [[nodiscard]] std::unordered_set<NeuronID> get_neuron_ids_in_areas(const std::unordered_set<RelearnTypes::area_id>& my_area_ids) const {
        const auto is_id_in_my_ranks_areas = [my_area_ids, this](const NeuronID neuron_id) {
            return ranges::contains(my_area_ids, neuron_id_to_area_id[neuron_id.get_neuron_id()]);
        };

        std::unordered_set<NeuronID> result;
        for (const auto& id : NeuronID::range(neuron_id_to_area_id.size())) {
            if (is_id_in_my_ranks_areas(id)) {
                result.insert(id);
            }
        }
        return result;
    }

    [[nodiscard]] const std::vector<RelearnTypes::area_id>& get_neuron_ids_to_area_ids() const noexcept {
        return neuron_id_to_area_id;
    }

    [[nodiscard]] std::vector<RelearnTypes::area_id> translate_area_names_to_area_ids_ordered(const std::vector<std::string>& area_names) {
        std::vector<RelearnTypes::area_id> area_ids;
        area_ids.reserve(area_names.size());
        for (const auto& name : area_names) {
            const auto& id_ = get_area_id_for_area_name(name);
            area_ids.push_back(id_);
        }
        return area_ids;
    }

    [[nodiscard]] std::unordered_set<RelearnTypes::area_id> translate_area_names_to_area_ids(const std::unordered_set<std::string>& area_names) const {
        std::unordered_set<RelearnTypes::area_id> area_ids;
        area_ids.reserve(area_names.size());
        for (const auto& name : area_names) {
            const auto& id_ = get_area_id_for_area_name(name);
            area_ids.insert(id_);
        }
        return area_ids;
    }

    [[nodiscard]] std::unordered_set<RelearnTypes::area_name> get_matching_area_names(const std::string& area_regex) const {
        std::unordered_set<RelearnTypes::area_name> matching_area_names{};
        for (const auto& known_area_name : get_all_area_names()) {
            std::smatch match;
            const bool is_match = std::regex_match(known_area_name, match, std::regex(area_regex));
            if (is_match) {
                matching_area_names.insert(known_area_name);
            }
        }
        return matching_area_names;
    }

    [[nodiscard]] std::unordered_set<RelearnTypes::area_id> get_area_ids_for_matching_area_names(const std::string& area_regex) const {
        const auto& area_names = get_matching_area_names(area_regex);
        return translate_area_names_to_area_ids(area_names);
    }

    void create_neurons(RelearnTypes::number_neurons_type created_neurons) {
        RelearnException::check(!neuron_id_to_area_id.empty(), "LocalAreaTranslator::create_neurons: Was not initialized");
        RelearnException::check(created_neurons > 0, "LocalAreaTranslator::create_neurons: Cannot create 0 neurons");
        neuron_id_to_area_id.insert(neuron_id_to_area_id.end(), created_neurons, 0UL);
    }

private:
    std::vector<RelearnTypes::area_name> area_id_to_area_name;
    std::vector<RelearnTypes::area_id> neuron_id_to_area_id;
};
