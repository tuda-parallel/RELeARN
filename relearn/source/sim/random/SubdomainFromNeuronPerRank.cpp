/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SubdomainFromNeuronPerRank.h"

#include "sim/Essentials.h"
#include "sim/random/RandomSynapseLoader.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"

#include <vector>

#include <range/v3/action/insert.hpp>

SubdomainFromNeuronPerRank::SubdomainFromNeuronPerRank(const SubdomainFromNeuronPerRank::number_neurons_type number_neurons_per_rank, const double fraction_excitatory_neurons, const double um_per_neuron, std::shared_ptr<Partition> partition)
    : BoxBasedRandomSubdomainAssignment(partition, fraction_excitatory_neurons, um_per_neuron)
    , number_neurons_per_rank(number_neurons_per_rank) {

    RelearnException::check(number_neurons_per_rank >= 1, "SubdomainFromNeuronPerRank::SubdomainFromNeuronPerRank: There must be at least one neuron per mpi rank!");

    const auto my_rank = static_cast<unsigned int>(partition->get_my_mpi_rank().get_rank());
    const auto number_ranks = partition->get_number_mpi_ranks();
    const auto number_local_subdomains = partition->get_number_local_subdomains();

    RandomHolder::seed(RandomHolderKey::Subdomain, my_rank);

    const auto number_neurons = number_ranks * number_neurons_per_rank;
    const auto preliminary_number_neurons_per_subdomain = number_neurons_per_rank / number_local_subdomains;
    const auto additional_neuron = (number_neurons_per_rank % number_local_subdomains == 0) ? 0 : 1;

    const auto number_neurons_per_subdomain = preliminary_number_neurons_per_subdomain + additional_neuron;

    // Calculate size of simulation box based on neuron density
    // number_neurons_per_subdomain^(1/3) == #neurons per dimension for one subdomain
    const auto number_boxes_per_subdomain_one_dimension = static_cast<number_neurons_type>(ceil(pow(static_cast<double>(number_neurons_per_subdomain), 1. / 3)));
    const auto number_boxes_one_dimension = partition->get_number_subdomains_per_dimension() * number_boxes_per_subdomain_one_dimension;

    const auto simulation_box_length_ = static_cast<double>(number_boxes_one_dimension) * um_per_neuron;

    partition->set_simulation_box_size({ 0, 0, 0 }, box_size_type(simulation_box_length_));

    set_requested_number_neurons(number_neurons);
    set_total_number_placed_neurons(number_neurons);
    set_area_id_to_area_name({ "random" });

    set_number_placed_neurons(0);
    set_ratio_placed_excitatory_neurons(0.0);

    synapse_loader = std::make_shared<RandomSynapseLoader>(std::move(partition));

    create_local_area_translator(number_neurons_per_rank);
}

void SubdomainFromNeuronPerRank::print_essentials(const std::unique_ptr<Essentials>& essentials) {
    essentials->insert("Neurons-Placed", get_total_number_placed_neurons());
    essentials->insert("Neurons-Placed-Per-Rank", number_neurons_per_rank);
}

void SubdomainFromNeuronPerRank::fill_all_subdomains() {
    RelearnException::check(!initialized, "SubdomainFromNeuronPerRank::fill_all_subdomains: The object is already initialized.");

    const auto number_local_subdomains = partition->get_number_local_subdomains();
    const auto preliminary_number_neurons_per_subdomain = number_neurons_per_rank / number_local_subdomains;

    number_neurons_type currently_placed_neurons = 0;
    number_neurons_type currently_placed_excitatory_neurons = 0;

    std::vector<LoadedNeuron> loaded_neurons{};
    loaded_neurons.reserve(number_neurons_per_rank);

    for (auto i = 0; i < number_local_subdomains; i++) {
        const auto additional_neuron = (i < number_neurons_per_rank % number_local_subdomains) ? 1 : 0;
        const auto number_neurons_per_subdomain = preliminary_number_neurons_per_subdomain + additional_neuron;

        const auto& [min, max] = partition->get_subdomain_boundaries(i);

        auto [nodes, placed_excitatory_neurons] = place_neurons_in_box(min, max, number_neurons_per_subdomain, currently_placed_neurons);

        currently_placed_neurons += number_neurons_per_subdomain;
        currently_placed_excitatory_neurons += placed_excitatory_neurons;

        ranges::insert(loaded_neurons, loaded_neurons.end(), nodes);
    }

    set_loaded_nodes(std::move(loaded_neurons));

    set_number_placed_neurons(currently_placed_neurons);

    const auto fraction_excitatory_neurons = static_cast<double>(currently_placed_excitatory_neurons) / static_cast<double>(currently_placed_neurons);
    set_ratio_placed_excitatory_neurons(fraction_excitatory_neurons);
}
