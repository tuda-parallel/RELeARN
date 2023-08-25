/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticInputCalculator.h"

#include "mpi/MPIWrapper.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/NetworkGraph.h"
#include "neurons/input/FiredStatusCommunicationMap.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/filter.hpp>

void SynapticInputCalculator::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_local_neurons == 0, "SynapticInputCalculator::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "SynapticInputCalculator::init: number_neurons was 0");

    number_local_neurons = number_neurons;
    synaptic_input.resize(number_neurons, 0.0);
    raw_ex_input.resize(number_neurons, 0.0);
    raw_inh_input.resize(number_neurons, 0.0);

    fired_status_comm->init(number_neurons);
}

void SynapticInputCalculator::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(number_local_neurons > 0, "SynapticInputCalculator::create_neurons: number_local_neurons was 0");
    RelearnException::check(creation_count > 0, "SynapticInputCalculator::create_neurons: creation_count was 0");

    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;

    number_local_neurons = new_size;
    synaptic_input.resize(new_size, 0.0);
    raw_ex_input.resize(new_size, 0.0);
    raw_inh_input.resize(new_size, 0.0);

    fired_status_comm->create_neurons(creation_count);
}

void SynapticInputCalculator::set_synaptic_input(const double value) noexcept {
    std::ranges::fill(synaptic_input, value);
}

double SynapticInputCalculator::get_local_and_distant_synaptic_input(const std::span<const FiredStatus> fired, const NeuronID& neuron_id) {
    auto local_input = 0.0;

    // Walk through local in-edges
    const auto& [local_in_edges_plastic, local_in_edges_static] = network_graph->get_local_in_edges(neuron_id);
    //        register_local_in_edges(local_in_edges_plastic);
    //        register_local_in_edges(local_in_edges_static);
    for (const auto& [src_neuron_id, edge_val] : local_in_edges_static) {
        const auto spike = fired[src_neuron_id.get_neuron_id()];
        if (spike == FiredStatus::Fired) {
            local_input += synapse_conductance * edge_val;
            if (edge_val > 0.0) {
                raw_ex_input[neuron_id.get_neuron_id()] += edge_val;
            } else {
                raw_inh_input[neuron_id.get_neuron_id()] += std::abs(edge_val);
            }
        }
    }

    for (const auto& [src_neuron_id, edge_val] : local_in_edges_plastic) {
        const auto spike = fired[src_neuron_id.get_neuron_id()];
        if (spike == FiredStatus::Fired) {
            local_input += synapse_conductance * edge_val;
            if (edge_val > 0.0) {
                raw_ex_input[neuron_id.get_neuron_id()] += edge_val;
            } else {
                raw_inh_input[neuron_id.get_neuron_id()] += std::abs(edge_val);
            }
        }
    }

    // Walk through the distant in-edges of my neuron
    const auto& [distant_in_edges_plastic, distant_in_edges_static] = network_graph->get_distant_in_edges(neuron_id);
    //        register_distant_in_edges(distant_in_edges_plastic);
    //        register_distant_in_edges(distant_in_edges_static);
    for (const auto& [key, edge_val] : distant_in_edges_static) {
        const auto& rank = key.get_rank();
        const auto& initiator_neuron_id = key.get_neuron_id();

        const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
        if (contains_id) {
            local_input += synapse_conductance * edge_val;
            if (edge_val > 0.0) {
                raw_ex_input[neuron_id.get_neuron_id()] += edge_val;
            } else {
                raw_inh_input[neuron_id.get_neuron_id()] += std::abs(edge_val);
            }
        }
    }
    for (const auto& [key, edge_val] : distant_in_edges_plastic) {
        const auto& rank = key.get_rank();
        const auto& initiator_neuron_id = key.get_neuron_id();

        const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
        if (contains_id) {
            local_input += synapse_conductance * edge_val;
            if (edge_val > 0.0) {
                raw_ex_input[neuron_id.get_neuron_id()] += edge_val;
            } else {
                raw_inh_input[neuron_id.get_neuron_id()] += std::abs(edge_val);
            }
        }
    }

    return local_input;
}
