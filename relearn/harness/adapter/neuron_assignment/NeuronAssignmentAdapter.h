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

#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/random/RandomAdapter.h"

#include "Types.h"
#include "neurons/enums/SignalType.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "structure/Partition.h"
#include "util/ranges/Functional.hpp"

#include <random>
#include <vector>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/iota.hpp>

class NeuronAssignmentAdapter {
public:
    static void generate_random_neurons(std::vector<RelearnTypes::position_type>& positions, std::vector<RelearnTypes::area_id>& neuron_id_to_area_ids,
        std::vector<RelearnTypes::area_name>& area_id_to_area_name, std::vector<SignalType>& types, std::mt19937& mt) {

        const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
        const auto fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
        const auto um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100.0;

        const auto part = std::make_shared<Partition>(1, MPIRank(0));
        part->set_total_number_neurons(number_neurons);
        SubdomainFromNeuronDensity sfnd{ number_neurons, fraction_excitatory_neurons, um_per_neuron, part };

        sfnd.initialize();

        positions = sfnd.get_neuron_positions_in_subdomains();
        neuron_id_to_area_ids = sfnd.get_local_area_translator()->get_neuron_ids_to_area_ids();
        area_id_to_area_name = sfnd.get_local_area_translator()->get_all_area_names();
        types = sfnd.get_neuron_types_in_subdomains();

        sfnd.write_neurons_to_file("neurons.tmp");
    }

    static std::vector<RelearnTypes::area_id> get_random_area_ids(size_t number_areas, size_t number_neurons, std::mt19937& mt) {
        return ranges::views::generate_n(
                   [number_areas, &mt]() {
                       return RandomAdapter::get_random_integer<RelearnTypes::area_id>(0, RelearnTypes::area_id(number_areas - 1), mt);
                   },
                   number_neurons)
            | ranges::to_vector;
    }

    static std::vector<RelearnTypes::area_name> get_random_area_names(size_t max_areas, std::mt19937& mt) {
        const auto number_areas = RandomAdapter::get_random_integer<size_t>(1, max_areas, mt);
        return get_random_area_names_specific(number_areas, mt);
    }

    static RelearnTypes::area_name get_random_area_name(std::mt19937& mt) {
        return RandomAdapter::get_random_string(10, mt);
    }

    static std::vector<RelearnTypes::area_name> get_random_area_names_specific(size_t number_areas, std::mt19937& mt) {
        std::vector<RelearnTypes::area_name> area_names{};
        area_names.reserve(number_areas);

        for (size_t area_id = 0; area_id < number_areas; area_id++) {
            RelearnTypes::area_name name{};
            do {
                name = get_random_area_name(mt);
            } while (name.empty() || ranges::contains(area_names, name));

            area_names.emplace_back(std::move(name));
        }

        return area_names;
    }

    static std::vector<RelearnTypes::area_name> get_neuron_id_vs_area_name(const std::vector<RelearnTypes::area_id>& neuron_id_vs_area_id,
        const std::vector<RelearnTypes::area_name>& area_id_vs_area_name) {
        return neuron_id_vs_area_id
            | ranges::views::transform(lookup(area_id_vs_area_name))
            | ranges::to_vector;
    }

    static std::shared_ptr<LocalAreaTranslator> get_randomized_area_translator(std::mt19937& mt) {
        const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
        const auto num_areas = std::max(size_t{ 1 }, num_neurons / 10);
        auto area_id_to_area_name = NeuronAssignmentAdapter::get_random_area_names(num_areas, mt);
        auto neuron_id_to_area_id = NeuronAssignmentAdapter::get_random_area_ids(area_id_to_area_name.size(), num_neurons, mt);

        return std::make_shared<LocalAreaTranslator>(area_id_to_area_name, neuron_id_to_area_id);
    }

    static std::shared_ptr<LocalAreaTranslator> get_randomized_area_translator(const RelearnTypes::number_neurons_type num_neurons, std::mt19937& mt) {
        const auto num_areas = std::max(size_t{ 1 }, num_neurons / 10);
        auto area_id_to_area_name = NeuronAssignmentAdapter::get_random_area_names(num_areas, mt);
        auto neuron_id_to_area_id = NeuronAssignmentAdapter::get_random_area_ids(area_id_to_area_name.size(), num_neurons, mt);

        return std::make_shared<LocalAreaTranslator>(area_id_to_area_name, neuron_id_to_area_id);
    }

    static std::string get_invalid_area_name(const std::vector<RelearnTypes::area_name>& area_id_to_area_name, std::mt19937& mt) {
        RelearnTypes::area_name area_name = "";
        do {
            area_name = get_random_area_name(mt);
        } while (ranges::contains(area_id_to_area_name, area_name));
        return area_name;
    }
};
