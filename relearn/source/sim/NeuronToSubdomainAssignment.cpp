/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronToSubdomainAssignment.h"

#include "io/NeuronIO.h"
#include "structure/Partition.h"

void NeuronToSubdomainAssignment::initialize() {
    partition->set_boundary_correction_function(get_subdomain_boundary_fix());
    partition->calculate_and_set_subdomain_boundaries();

    fill_all_subdomains();

    initialized = true;

    const auto number_local_neurons = get_number_neurons_in_subdomains();
    partition->set_number_local_neurons(number_local_neurons);
}

void NeuronToSubdomainAssignment::write_neurons_to_file(const std::filesystem::path& file_path) const {
    NeuronIO::write_neurons(loaded_neurons, file_path, local_area_translator, partition);
}
