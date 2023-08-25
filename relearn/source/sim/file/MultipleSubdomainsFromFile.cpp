/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MultipleSubdomainsFromFile.h"

#include "Config.h"
#include "io/NeuronIO.h"
#include "mpi/MPIWrapper.h"
#include "sim/Essentials.h"
#include "sim/file/MultipleFilesSynapseLoader.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"
#include "util/Utility.h"

#include <string>

MultipleSubdomainsFromFile::MultipleSubdomainsFromFile(const std::filesystem::path& path_to_neurons,
    std::optional<std::filesystem::path> path_to_synapses, const std::shared_ptr<Partition>& partition)
    : NeuronToSubdomainAssignment(partition) {
    //    RelearnException::check(partition->get_number_mpi_ranks() > 1, "MultipleSubdomainsFromFile::MultipleSubdomainsFromFile: There was only one MPI rank.");
    const std::filesystem::path path_to_file = Util::find_file_for_rank(path_to_neurons, partition->get_my_mpi_rank().get_rank(), "rank_", "_positions.txt", 5U);

    read_neurons_from_file(path_to_file);
    synapse_loader = std::make_shared<MultipleFilesSynapseLoader>(partition, std::move(path_to_synapses));
}

void MultipleSubdomainsFromFile::print_essentials(const std::unique_ptr<Essentials>& essentials) {
    essentials->insert("Neurons-Placed", get_total_number_placed_neurons());
}

void MultipleSubdomainsFromFile::read_neurons_from_file(const std::filesystem::path& path_to_neurons) {

    auto [nodes, area_id_vs_area_name, additional_infos, additional_position_information] = NeuronIO::read_neurons(path_to_neurons);

    auto check = [](double value) -> bool {
        const auto min = MPIWrapper::reduce(value, MPIWrapper::ReduceFunction::Min, MPIRank::root_rank());
        const auto max = MPIWrapper::reduce(value, MPIWrapper::ReduceFunction::Max, MPIRank::root_rank());
        return min == max;
    };

    auto min_x = additional_position_information.sim_size.get_minimum().get_x();
    auto min_y = additional_position_information.sim_size.get_minimum().get_y();
    auto min_z = additional_position_information.sim_size.get_minimum().get_z();
    auto max_x = additional_position_information.sim_size.get_maximum().get_x();
    auto max_y = additional_position_information.sim_size.get_maximum().get_y();
    auto max_z = additional_position_information.sim_size.get_maximum().get_z();

    const auto all_same_min_x = check(min_x);
    const auto all_same_min_y = check(min_y);
    const auto all_same_min_z = check(min_z);
    const auto all_same_max_x = check(max_x);
    const auto all_same_max_y = check(max_y);
    const auto all_same_max_z = check(max_z);

    RelearnException::check(all_same_min_x, "MultipleSubdomainsFromFile::read_neurons_from_file: min_x is different across the ranks! Mine: {}", min_x);
    RelearnException::check(all_same_min_y, "MultipleSubdomainsFromFile::read_neurons_from_file: min_y is different across the ranks! Mine: {}", min_y);
    RelearnException::check(all_same_min_z, "MultipleSubdomainsFromFile::read_neurons_from_file: min_z is different across the ranks! Mine: {}", min_z);
    RelearnException::check(all_same_max_x, "MultipleSubdomainsFromFile::read_neurons_from_file: max_x is different across the ranks! Mine: {}", max_x);
    RelearnException::check(all_same_max_y, "MultipleSubdomainsFromFile::read_neurons_from_file: max_y is different across the ranks! Mine: {}", max_y);
    RelearnException::check(all_same_max_z, "MultipleSubdomainsFromFile::read_neurons_from_file: max_z is different across the ranks! Mine: {}", max_z);

    RelearnTypes::box_size_type minimum{ min_x, min_y, min_z };
    RelearnTypes::box_size_type maximum{ max_x, max_y, max_z };

    set_area_id_to_area_name(area_id_vs_area_name);
    const auto& [_1, _2, loaded_ex_neurons, loaded_in_neurons] = additional_infos;
    const auto total_number_neurons = loaded_ex_neurons + loaded_in_neurons;

    RelearnException::check(additional_position_information.local_neurons == total_number_neurons, "MultipleSubdomainsFromFile::read_neurons_from_file: Number of loaded neurons does not equals commented number {} vs {}", additional_position_information.local_neurons, total_number_neurons);

    partition->set_simulation_box_size(minimum, maximum);

    set_total_number_placed_neurons(total_number_neurons);
    set_requested_number_neurons(total_number_neurons);
    set_number_placed_neurons(total_number_neurons);

    const auto ratio_excitatory_neurons = static_cast<double>(loaded_ex_neurons) / static_cast<double>(total_number_neurons);

    set_requested_ratio_excitatory_neurons(ratio_excitatory_neurons);
    set_ratio_placed_excitatory_neurons(ratio_excitatory_neurons);

    partition->set_total_number_neurons(total_number_neurons);

    set_loaded_nodes(std::move(nodes));
    create_local_area_translator(total_number_neurons);

    this->additional_position_information = additional_position_information;
}
void MultipleSubdomainsFromFile::fill_all_subdomains() {
    RelearnException::check(additional_position_information.subdomain_sizes.size() == partition->get_number_local_subdomains(), "MultipleSubdomainsFromFile::read_neurons_from_file:Number of subdomains {} in positions file is not equal to the actual number {}", additional_position_information.subdomain_sizes.size(), partition->get_number_local_subdomains());

    auto subdomain_id_first = partition->get_local_subdomain_id_start();
    auto subdomain_id_last = partition->get_local_subdomain_id_end();
    auto num_subdomains = partition->get_number_local_subdomains();
    const auto sim_size = additional_position_information.sim_size;
    for (auto i = 0; i < num_subdomains; i++) {
        auto subdomain_bb = partition->get_subdomain_boundaries(i);
        RelearnException::check(subdomain_bb == additional_position_information.subdomain_sizes[i], "MultipleSubdomainsFromFile::read_neurons_from_file: Wrong subdomain boundaries for subdomain {} on rank {}. Expected: {}, found: {}", i, MPIWrapper::get_my_rank(), subdomain_bb, additional_position_information.subdomain_sizes[i]);
        RelearnException::check(subdomain_bb.get_minimum().check_in_box(sim_size.get_minimum(), sim_size.get_maximum())
                && subdomain_bb.get_maximum().check_in_box(sim_size.get_minimum(), sim_size.get_maximum()),
            "MultipleSubdomainsFromFile::read_neurons_from_file: Subdomain outside of simulation box");
    }

    for (const auto& node : loaded_neurons) {
        bool contains = false;
        for (auto i = 0; i < num_subdomains; i++) {
            auto subdomain_bb = additional_position_information.subdomain_sizes[i];
            if (node.pos.check_in_box(subdomain_bb.get_minimum(), subdomain_bb.get_maximum())) {
                contains = true;
                break;
            }
        }
        RelearnException::check(contains, "MultipleSubdomainsFromFile::read_neurons_from_file: Neuron {} outside of subdomains", node.id);
    }

    std::vector<Vec3d> positions{};
    positions.reserve(loaded_neurons.size());
    std::transform(loaded_neurons.begin(), loaded_neurons.end(), std::back_inserter(positions), [](const LoadedNeuron& node) { return node.pos; });
    const std::set<Vec3d> positions_set(positions.begin(), positions.end());
    RelearnException::check(positions.size() == positions_set.size(), "MultipleSubdomainsFromFile::read_neurons_from_file: Same position occurs multiple times");
}