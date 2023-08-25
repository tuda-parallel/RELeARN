/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronMonitor.h"

#include "Config.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/LocalAreaTranslator.h"

#include <fstream>
#include <string>

void NeuronMonitor::init_print_file() {
    const auto& path = LogFiles::get_output_path() / "neuron_monitors";
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }

    const auto& file_path = path / (MPIWrapper::get_my_rank_str() + '_' + std::to_string(target_neuron_id.get_neuron_id() + 1) + ".csv");
    std::ofstream outfile(file_path, std::ios_base::out | std::ios_base::trunc);

    constexpr auto description = "# Step;Fired;Fired Fraction;x;Secondary Variable;Calcium;Target Calcium;Synaptic Input;Excitatory input; Inhibitory input;Background Activity;Stimulation;Grown Axons;Connected Axons;Grown Excitatory Dendrites;Connected Excitatory Dendrites;Grown Inhibitory Dendrites;Connected Inhibitory Dendrites\n";

    outfile << std::setprecision(Constants::print_precision);
    outfile.imbue(std::locale());

    outfile << "# Rank: " << MPIWrapper::get_my_rank_str() << "\n";
    outfile << "# Neuron ID: " << target_neuron_id.get_neuron_id() + 1 << "\n";
    outfile << "# Area name: " << neurons_to_monitor->get_local_area_translator()->get_area_name_for_neuron_id(target_neuron_id.get_neuron_id()) << "\n";
    outfile << "# Area id: " << neurons_to_monitor->get_local_area_translator()->get_area_id_for_neuron_id(target_neuron_id.get_neuron_id()) << "\n";
    outfile << description;
}

void NeuronMonitor::flush_current_contents() {
    std::filesystem::path path = LogFiles::get_output_path() / "neuron_monitors";

    const auto& file_path = path / (MPIWrapper::get_my_rank_str() + '_' + std::to_string(target_neuron_id.get_neuron_id() + 1) + ".csv");
    std::ofstream outfile(file_path, std::ios_base::ate | std::ios_base::app);

    constexpr auto filler = ";";

    for (const auto& info : information) {
        outfile << info.get_step() << filler;
        outfile << info.get_fired() << filler;
        outfile << info.get_fraction_fired() << filler;
        outfile << info.get_x() << filler;
        outfile << info.get_secondary() << filler;
        outfile << info.get_calcium() << filler;
        outfile << info.get_target_calcium() << filler;
        outfile << info.get_synaptic_input() << filler;
        outfile << info.get_ex_input() << filler;
        outfile << info.get_inh_input() << filler;
        outfile << info.get_background_activity() << filler;
        outfile << info.get_stimulation() << filler;
        outfile << info.get_axons() << filler;
        outfile << info.get_axons_connected() << filler;
        outfile << info.get_excitatory_dendrites_grown() << filler;
        outfile << info.get_excitatory_dendrites_connected() << filler;
        outfile << info.get_inhibitory_dendrites_grown() << filler;
        outfile << info.get_inhibitory_dendrites_connected() << '\n';
    }

    information.clear();
}
