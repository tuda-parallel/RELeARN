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
#include "neurons/enums/SignalType.h"
#include "util/NeuronID.h"
#include "GlobalAreaMapper.h"

#include "neurons/Neurons.h"

#include <boost/functional/hash.hpp>

#include <filesystem>
#include <tuple>
#include <vector>

class Simulation;

/**
 * Monitors the number of connections between areas and more area statistics
 */
class AreaMonitor {
public:
    struct AreaConnection {
    public:
        AreaConnection() = default;

        AreaConnection(const int from_rank, const RelearnTypes::area_id from_area,
            const NeuronID to_local_neuron_id, const SignalType signal_type)
            : from_rank(from_rank)
            , from_area(from_area)
            , to_local_neuron_id(to_local_neuron_id)
            , signal_type(signal_type) { }

        int from_rank{ -1 };
        RelearnTypes::area_id from_area{};
        NeuronID to_local_neuron_id{};
        SignalType signal_type{};
    };

    /**
     * Construct an object for monitoring a specific area on this mpi rank
     * @param simulation Pointer to the simulation
     * @param area_id Id of the area that will be monitored
     * @param area_name Name of the area that will be monitored
     * @param my_rank The mpi rank of this process
     */
    AreaMonitor(std::shared_ptr<Neurons> neurons, std::shared_ptr<GlobalAreaMapper> global_area_mapper, RelearnTypes::area_id area_id, RelearnTypes::area_name area_name, const int my_rank, const std::filesystem::path& path, bool monitor_connectivity);

    void request_data() const;

    /**
     * Add an ingoing connection to the area. This method shall be called by other area monitors with ingoing connections to this area
     * @param connection Connection whose source is this area
     */
    void add_ingoing_connection(const AreaConnection& connection, const RelearnTypes::plastic_synapse_weight weight);

    void remove_ingoing_connection(const AreaMonitor::AreaConnection& connection, const RelearnTypes::plastic_synapse_weight weight);

    /**
     * Prepares the monitor for a new logging step. Call this method before each logging step.
     */
    void prepare_recording();

    /**
     * Add the data of a single neuron to the recording. The neuron must be part of the ensemble.
     * Call this method with each neuron of the ensemble in each logging step
     * @param neuron_id Neuron which is part of the ensemble
     */
    void record_data(NeuronID neuron_id);

    /**
     * Indicates end of a single logging step. Call this method after the data off each neuron was recorded.
     */
    void finish_recording();

    /**
     * Write all recorded data to a csv file
     * @param file_path Path to new csv file
     */
    void write_data_to_file();

    /**
     * Returns the name of the area that is monitored
     * @return Area name
     */
    [[nodiscard]] const RelearnTypes::area_name& get_area_name() const noexcept {
        return area_name;
    }

    void monitor_connectivity();

    /**
     * Returns the id of the area that is monitored
     * @return Area id
     */
    [[nodiscard]] const RelearnTypes::area_id& get_area_id() const noexcept {
        return area_id;
    }

    void debug_checks();

private:
    /**
     * Number of connections to another ensemble in a single step
     */
    struct ConnectionCount {
        int den_ex = 0;
        int den_inh = 0;
    };

    std::shared_ptr<Neurons> neurons;

    int my_rank;

    bool flag_monitor_connectivity{ true };

    size_t step = 0;

    std::filesystem::path path;

    RelearnTypes::area_name area_name;

    RelearnTypes::area_id area_id;

    struct InternalStatistics {
        double axons_grown = 0;
        double den_ex_grown = 0;
        double den_inh_grown = 0;
        int axons_conn = 0;
        int den_ex_conn = 0;
        int den_inh_conn = 0;
        double background = 0;
        double syn_input_total = 0;
        double syn_input_ex_raw = 0;
        double syn_input_inh_raw = 0;
        double calcium = 0;
        double fired_fraction = 0.0;
        RelearnTypes::number_neurons_type num_enabled_neurons = 0;
    };

    using EnsembleConnections = std::unordered_map<std::pair<int, RelearnTypes::area_id>, ConnectionCount,
        boost::hash<std::pair<int, RelearnTypes::area_id>>>;
    using EnsembleDeletions = std::unordered_map<std::pair<int, RelearnTypes::area_id>, long,
        boost::hash<std::pair<int, RelearnTypes::area_id>>>;

    /**
     * For current logging step: Maps for each ensemble the number of connections
     */
    EnsembleConnections connections;
    EnsembleDeletions deletions;
    InternalStatistics internal_statistics{};

    /**
     * Complete data of all earlier logging steps
     */
    std::vector<std::tuple<EnsembleConnections, EnsembleDeletions, InternalStatistics>> data;

    std::shared_ptr<GlobalAreaMapper> global_area_mapper{};
    void write_header();
};
