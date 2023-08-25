/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SubdomainFromNeuronDensity.h"

#include "sim/Essentials.h"
#include "sim/random/RandomSynapseLoader.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"

SubdomainFromNeuronDensity::SubdomainFromNeuronDensity(const SubdomainFromNeuronDensity::number_neurons_type number_neurons, const double fraction_excitatory_neurons, const double um_per_neuron, std::shared_ptr<Partition> partition)
    : BoxBasedRandomSubdomainAssignment(partition, fraction_excitatory_neurons, um_per_neuron) {

    RelearnException::check(partition->get_my_mpi_rank() == MPIRank::root_rank() && partition->get_number_mpi_ranks() == 1, "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: Can only be used for 1 MPI rank.");
    RelearnException::check(number_neurons > 0, "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: There must be more than 0 neurons.");

    RandomHolder::seed(RandomHolderKey::Subdomain, 0);

    // Calculate size of simulation box based on neuron density
    // number_neurons^(1/3) == #neurons per dimension
    const auto approx_number_of_neurons_per_dimension = ceil(pow(static_cast<double>(number_neurons), 1. / 3));
    const auto simulation_box_length_ = approx_number_of_neurons_per_dimension * um_per_neuron;

    partition->set_simulation_box_size({ 0, 0, 0 }, box_size_type(simulation_box_length_));

    set_requested_number_neurons(number_neurons);

    set_number_placed_neurons(0);
    set_ratio_placed_excitatory_neurons(0.0);

    synapse_loader = std::make_shared<RandomSynapseLoader>(std::move(partition));

    set_area_id_to_area_name({ "random" });
    create_local_area_translator(number_neurons);
}

void SubdomainFromNeuronDensity::print_essentials(const std::unique_ptr<Essentials>& essentials) {
    essentials->insert("Neurons-Placed", get_total_number_placed_neurons());
}

void SubdomainFromNeuronDensity::fill_all_subdomains() {
    RelearnException::check(!initialized, "SubdomainFromNeuronDensity::fill_all_subdomains: The object is already initialized.");

    const auto& [min, max] = partition->get_subdomain_boundaries(0);
    const auto requested_number_neurons = get_requested_number_neurons();
    auto [neurons, number_excitatory_neurons] = place_neurons_in_box(min, max, requested_number_neurons, 0);

    set_number_placed_neurons(requested_number_neurons);
    set_total_number_placed_neurons(requested_number_neurons);

    const auto fraction_excitatory_neurons = static_cast<double>(number_excitatory_neurons) / static_cast<double>(requested_number_neurons);
    set_ratio_placed_excitatory_neurons(fraction_excitatory_neurons);

    set_loaded_nodes(std::move(neurons));
}
