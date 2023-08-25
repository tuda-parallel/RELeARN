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

#include "Config.h"
#include "Types.h"
#include "neurons/enums/SignalType.h"
#include "sim/LoadedNeuron.h"
#include "sim/SynapseLoader.h"
#include "util/RelearnException.h"
#include "neurons/LocalAreaTranslator.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class Essentials;
class Partition;

/**
 * This class provides an interface for every algorithm that is used to assign neurons to MPI processes
 */
class NeuronToSubdomainAssignment {
public:
    using position_type = RelearnTypes::position_type;
    using box_size_type = RelearnTypes::box_size_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object with the given partition
     * @param partition The partition to use
     */
    explicit NeuronToSubdomainAssignment(std::shared_ptr<Partition> partition)
        : partition(std::move(partition)) {
    }

    virtual ~NeuronToSubdomainAssignment() = default;

    NeuronToSubdomainAssignment(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment(NeuronToSubdomainAssignment&& other) = delete;

    NeuronToSubdomainAssignment& operator=(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment& operator=(NeuronToSubdomainAssignment&& other) = delete;

    /**
     * @brief Initializes the assignment class, i.e., loads all neurons for the subdomains
     * @exception Can throw a RelearnException
     */
    void initialize();

    /**
     * @brief Prints relevant metrics to the essentials
     * @param essentials The essentials
     */
    virtual void print_essentials(const std::unique_ptr<Essentials>& essentials) = 0;

    /**
     * @brief Returns the associated SynapseLoader (some type that inherits from SynapseLoader)
     * @exception Throws a RelearnException if synapse_loader is nullptr
     * @return The associated SynapseLoader
     */
    std::shared_ptr<SynapseLoader> get_synapse_loader() const {
        RelearnException::check(synapse_loader.operator bool(), "NeuronToSubdomainAssignment::get_synapse_loader: synapse_loader is empty");
        return synapse_loader;
    }

    /**
     * @brief Returns a function object that is used to fix calculated subdomain boundaries.
     *      This might be necessary if special boundaries must be considered
     * @return A function object that corrects subdomain boundaries
     */
    virtual std::function<box_size_type(box_size_type)> get_subdomain_boundary_fix() const {
        return [](Vec3d arg) { return arg; };
    }

    /**
     * @brief Returns the total number of neurons that should be placed
     * @return The total number of neurons that should be placed
     */
    [[nodiscard]] number_neurons_type get_requested_number_neurons() const noexcept {
        return requested_number_neurons;
    }

    /**
     * @brief Returns the current number of placed neurons on this MPI rank
     * @return The current number of placed neurons
     */
    [[nodiscard]] number_neurons_type get_number_placed_neurons() const noexcept {
        return number_placed_neurons;
    }

    /**
     * @brief Returns the total number of placed neurons across all MPI ranks
     * @return The total number of placed neurons across all MPI ranks
     */
    [[nodiscard]] number_neurons_type get_total_number_placed_neurons() const noexcept {
        return total_number_neurons;
    }

    /**
     * @brief Returns the total fraction of excitatory neurons that should be placed
     * @return The total fraction of excitatory neurons that should be placed
     */
    [[nodiscard]] double get_requested_ratio_excitatory_neurons() const noexcept {
        return requested_ratio_excitatory_neurons;
    }

    /**
     * @brief Returns the current fraction of placed excitatory neurons in the local subdomains
     * @return The total current fraction of placed excitatory neurons in the local subdomains
     */
    [[nodiscard]] double get_ratio_placed_excitatory_neurons() const noexcept {
        return ratio_placed_excitatory_neurons;
    }

    /**
     * @brief Returns the total number of neurons in the local subdomains
     * @return The total number of neurons in the local subdomains
     */
    [[nodiscard]] number_neurons_type get_number_neurons_in_subdomains() const noexcept {
        const auto total_number_neurons_in_subdomains = loaded_neurons.size();
        return total_number_neurons_in_subdomains;
    }

    /**
     * @brief Returns all positions of neurons in the local subdomains, indexed by the neuron id
     * @return The all position of neurons in the local subdomains
     */
    [[nodiscard]] std::vector<position_type> get_neuron_positions_in_subdomains() const {
        return loaded_neurons | ranges::views::transform(&LoadedNeuron::pos) | ranges::to_vector;
    }

    /**
     * @brief Returns all signal types of neurons in the local subdomains, indexed by the neuron id
     * @return The all position of neurons in the local subdomains
     */
    [[nodiscard]] std::vector<SignalType> get_neuron_types_in_subdomains() const {
        return loaded_neurons | ranges::views::transform(&LoadedNeuron::signal_type) | ranges::to_vector;
    }

    /**
     * @brief Returns the area translator that translates between the local area id on the current mpi rank and its area name
     * @return the local area translator for this mpi rank
     */
    [[nodiscard]] std::shared_ptr<LocalAreaTranslator> get_local_area_translator() const noexcept {
        return local_area_translator;
    }

    /**
     * @brief Writes all loaded neurons into the specified file.
     *      The format is
     *      # ID, Position (x y z),	Area, type
     * @param file_path The filepath where to write the neurons
     * @exception Might throw a RelearnException
     */
    virtual void write_neurons_to_file(const std::filesystem::path& file_path) const;

protected:
    virtual void fill_all_subdomains() = 0;

    void set_requested_ratio_excitatory_neurons(const double desired_frac_neurons_exc) noexcept {
        requested_ratio_excitatory_neurons = desired_frac_neurons_exc;
    }

    void set_requested_number_neurons(const size_t get_requested_number_neurons) noexcept {
        requested_number_neurons = get_requested_number_neurons;
    }

    void set_ratio_placed_excitatory_neurons(const double current_frac_neurons_exc) noexcept {
        ratio_placed_excitatory_neurons = current_frac_neurons_exc;
    }

    void set_number_placed_neurons(const size_t current_num_neurons) noexcept {
        number_placed_neurons = current_num_neurons;
    }

    void set_total_number_placed_neurons(const size_t total_number_placed_neurons) const {
        total_number_neurons = total_number_placed_neurons;
    }

    void set_loaded_nodes(std::vector<LoadedNeuron>&& neurons) {
        loaded_neurons = std::move(neurons);
    }

    void set_area_id_to_area_name(const std::vector<RelearnTypes::area_name>& area_id_vs_area_name) {
        RelearnException::check(!area_id_vs_area_name.empty(), "NeuronToSubdomainAssignment::set_area_id_to_area_name is empty");
        this->area_id_to_area_name = area_id_vs_area_name;
    }

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<SynapseLoader> synapse_loader{};

    std::shared_ptr<LocalAreaTranslator> local_area_translator{};

    bool initialized{ false };

    void create_local_area_translator(size_t num_local_neurons) {
        std::vector<RelearnTypes::area_id> neuron_id_to_area_id{};
        neuron_id_to_area_id.reserve(num_local_neurons);

        if (loaded_neurons.empty()) {
            neuron_id_to_area_id.resize(num_local_neurons, 0);
        } else {
            for (const auto& loaded_neuron : loaded_neurons) {
                neuron_id_to_area_id.push_back(loaded_neuron.area_id);
            }
        }
        local_area_translator = std::make_shared<LocalAreaTranslator>(area_id_to_area_name, neuron_id_to_area_id);
    }

protected:
    std::vector<LoadedNeuron> loaded_neurons{};

private:
    std::vector<RelearnTypes::area_name> area_id_to_area_name{};

    double requested_ratio_excitatory_neurons{ 0.0 };
    number_neurons_type requested_number_neurons{ 0 };

    double ratio_placed_excitatory_neurons{ 0.0 };
    number_neurons_type number_placed_neurons{ 0 };

    mutable number_neurons_type total_number_neurons{ Constants::uninitialized };
};
