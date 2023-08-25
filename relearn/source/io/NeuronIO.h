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
#include "sim/file/AdditionalPositionInformation.h"
#include "sim/LoadedNeuron.h"
#include "structure/Partition.h"
#include "util/MPIRank.h"

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

class LocalAreaTranslator;

/**
 * This class provides a static interface to load/store neurons and synapses from/to files,
 * as well as other linked information
 */
class NeuronIO {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using position_type = RelearnTypes::position_type;

    using PlasticInSynapses = std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses>;
    using PlasticOutSynapses = std::tuple<PlasticLocalSynapses, PlasticDistantOutSynapses>;

    using StaticInSynapses = std::tuple<StaticLocalSynapses, StaticDistantInSynapses>;
    using StaticOutSynapses = std::tuple<StaticLocalSynapses, StaticDistantOutSynapses>;

    using InSynapses = std::tuple<StaticInSynapses, PlasticInSynapses>;
    using OutSynapses = std::tuple<StaticOutSynapses, PlasticOutSynapses>;

    [[nodiscard]] static bool is_valid_area_name(const RelearnTypes::area_name& area_name);

    /**
     * @brief Reads all comments from the beginning of the file and returns those.
     *      Comments start with '#'. It stops at the file end or the fist non-comment line
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if opening the file failed
     * @return Returns all comments at the beginning of the file
     */
    [[nodiscard]] static std::vector<std::string> read_comments(const std::filesystem::path& file_path);

    [[nodiscard]] static AdditionalPositionInformation parse_additional_position_information(const std::vector<std::string>& comments);

    /**
     * @brief Reads all neurons from the file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1). All positions must be non-negative
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if a position has a negative component or the ids are not sorted properly
     * @return Returns a tuple with (1) all loaded neurons and (2) a vector which assigns an area id to its area name and (3) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<LoadedNeuron>, std::vector<RelearnTypes::area_name>, LoadedNeuronsInfo, AdditionalPositionInformation> read_neurons(const std::filesystem::path& file_path);

    /**
     * @brief Reads all neurons from the file and returns those in their components.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if a position has a negative component or the ids are not sorted properly
     * @return Returns a tuple with
     *      (1) The IDs (which index (2)-(4))
     *      (2) The positions
     *      (3) Neuron id -> area id
     *      (4) Area id <-> area name
     *      (5) The signal types
     *      (6) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<RelearnTypes::area_id>, std::vector<RelearnTypes::area_name>, std::vector<SignalType>, LoadedNeuronsInfo>
    read_neurons_componentwise(const std::filesystem::path& file_path);

    /**
     * @brief Writes all neurons to the file
     * @param neurons The neurons
     * @param file_path The path to the file
     * @param local_area_translator Maps local area id to area map
     * @param partition Partition of the entire simulation
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::shared_ptr<Partition> partition);

    /**
     * @brief Writes all neurons to the file
     * @param neurons The neurons
     * @param file_path The path to the file
     * @param local_area_translator Maps local area id to area map
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<LocalAreaTranslator>& local_area_translator);

    /**
     * @brief Writes all neurons to the file
     * @param neurons The neurons
     * @param ss StringStream in which the content is written
     * @param local_area_translator Maps local area id to area map
     * @param partition The partition of the simulation
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, std::stringstream& ss, const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::shared_ptr<Partition>& partition);

    /**
     * @brief Writes the assignment of area ids to area names
     * @param ss Stringstream in which the area mapping is written
     * @param area_names Vector which assigns an area id to an area name
     * @param local_area_translator Maps local area id to area map
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_area_names(std::stringstream& ss, const std::shared_ptr<LocalAreaTranslator>& local_area_translator);

    /**
     * @brief Writes all neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param local_area_translator Maps local area id to area map
     * @param signal_types The signal types
     * @param ss Stringstream to which is written
     * @param total_number_neurons Number of all neurons in the simulation
     * @param simulation_box Bounding box of the entire simulation
     * @param local_subdomain_boundaries List of bounding boxes (as pair of bounding box min and bounding box max) of the local subdomains
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::span<const SignalType> signal_types, std::stringstream& ss,
        size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box, std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries);

    /**
     * @brief Writes all neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param local_area_translator Maps local area id to area map
     * @param signal_types The signal types
     * @param ss Stringstream to which is written
     * @param total_number_neurons Number of all neurons in the simulation
     * @param simulation_box Bounding box of the entire simulation
     * @param local_subdomain_boundaries List of bounding boxes (as pair of bounding box min and bounding box max) of the local subdomains
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::span<const SignalType> signal_types, const std::filesystem::path& path,
        size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box, std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries);

    /**
     * @brief Writes all neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param local_area_translator Maps local area id to area map
     * @param signal_types The signal types
     * @param file_path Path to the output file
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::span<const SignalType> signal_types, std::filesystem::path& file_path);

    /**
     * @brief Reads all neuron ids from a file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @return Empty if the file did not meet the sorting requirement, the ascending ids otherwise
     */
    [[nodiscard]] static std::optional<std::vector<NeuronID>> read_neuron_ids(const std::filesystem::path& file_path);

    /**
     * @brief Reads all in-synapses from a file and returns those.
     *      Checks that no target id is larger or equal to number_local_neurons and that no source rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synapse is 0
     *      (3) a target rank is not my_rank
     *      (4) a source rank is not from [0, number_mpi_ranks)
     *      (5) or a target id is not from [0, number_local_neurons)
     * @return All in-synapses as a tuple: { { (1) Static synapses: (1.1) The local ones and (1.2) the distant ones }, { (2) Plastic synapses: (2.1) The local ones and (2.2) the distant ones } }
     */
    static InSynapses read_in_synapses(const std::filesystem::path& file_path, number_neurons_type number_local_neurons, MPIRank my_rank, size_t number_mpi_ranks);

    /**
     * @brief Writes all in-synapses to the specified file
     * @param local_in_synapses_static The local in-synapses that are static
     * @param distant_in_synapses_static The distant in-synapses that are static
     * @param local_in_synapses_plastic The local in-synapses that are plastic
     * @param distant_in_synapses_plastic The distant in-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param num_neurons Number of local neurons on the current mpi rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if the source rank of a distant in-synapse is equal to my_rank
     */
    static void write_in_synapses(const StaticLocalSynapses& local_in_synapses_static, const StaticDistantInSynapses& distant_in_synapses_static, const PlasticLocalSynapses& local_in_synapses_plastic,
        const PlasticDistantInSynapses& distant_in_synapses_plastic, MPIRank my_rank, RelearnTypes::number_neurons_type num_neurons, const std::filesystem::path& file_path);

    /**
     * @brief Writes all in-synapses to the specified stream
     * @param local_in_edges_static The local in-synapses that are static
     * @param distant_in_edges_static The distant in-synapses that are static
     * @param local_in_edges_plastic The local in-synapses that are plastic
     * @param distant_in_edges_plastic The distant out-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param mpi_ranks Number of used mpi ranks
     * @param number_local_neurons Number of local neurons on the current mpi rank
     * @param number_total_neurons Number of neurons over all ranks
     * @param step The current step of the simulation
     * @param ss StringStream to which the output is written
     */
    static void write_in_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_in_edges_static,
        const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_in_edges_static,
        const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_in_edges_plastic,
        const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_in_edges_plastic,
        MPIRank my_rank, size_t mpi_ranks, RelearnTypes::number_neurons_type number_local_neurons, RelearnTypes::number_neurons_type number_total_neurons, std::stringstream& ss, size_t step);

    /**
     * @brief Reads all out-synapses from a file and returns those.
     *      Checks that no source id is larger or equal to number_local_neurons and that no target rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synapse is 0
     *      (3) a source rank is not my_rank
     *      (4) a target rank is not from [0, number_mpi_ranks)
     *      (5) or a source id is not from [0, number_local_neurons)
     * @return All out-synapses as a tuple: { { (1) Static synapses: (1.1) The local ones and (1.2) the distant ones }, { (2) Plastic synapses: (2.1) The local ones and (2.2) the distant ones } }
     */
    static OutSynapses read_out_synapses(const std::filesystem::path& file_path, number_neurons_type number_local_neurons, MPIRank my_rank, size_t number_mpi_ranks);

    /**
     * @brief Writes all out-synapses to the specified file
     * @param local_out_synapses_static The local out-synapses that are static
     * @param distant_out_synapses_static The distant out-synapses that are static
     * @param local_out_synapses_plastic The local out-synapses that are plastic
     * @param distant_out_synapses_plastic The distant out-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param num_neurons Number of local neurons on the current mpi rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if the target rank of a distant out-synapse is equal to my_rank
     */
    static void write_out_synapses(const StaticLocalSynapses& local_out_synapses_static, const StaticDistantOutSynapses& distant_out_synapses_static,
        const PlasticLocalSynapses& local_out_synapses_plastic, const PlasticDistantOutSynapses& distant_out_synapses_plastic,
        MPIRank my_rank, RelearnTypes::number_neurons_type num_neurons, const std::filesystem::path& file_path);

    /**
     * @brief Writes all out-synapses to the specified stream
     * @param local_out_edges_static The local out-synapses that are static
     * @param distant_out_edges_static The distant out-synapses that are static
     * @param local_out_edges_plastic The local out-synapses that are plastic
     * @param distant_out_edges_plastic The distant out-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param mpi_ranks Number of used mpi ranks
     * @param number_local_neurons Number of local neurons on the current mpi rank
     * @param number_total_neurons Number of neurons over all ranks
     * @param step The current step of the simulation
     * @param ss StringStream to which the output is written
     */
    static void write_out_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_out_edges_static,
        const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_out_edges_static,
        const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_out_edges_plastic,
        const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_out_edges_plastic,
        MPIRank my_rank, size_t mpi_ranks, RelearnTypes::number_neurons_type number_local_neurons, RelearnTypes::number_neurons_type number_total_neurons, std::stringstream& ss, size_t step);
};