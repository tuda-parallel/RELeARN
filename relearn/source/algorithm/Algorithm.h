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
#include "util/RelearnException.h"

#include <memory>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;
class SynapticElements;

/**
 * This is a virtual interface for all algorithms that can be used to create new synapses.
 * It provides Algorithm::update_connectivity and Algorithm::update_octree.
 */
class Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Registers the synaptic elements with the algorithm
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if one of the pointers is empty
     */
    void set_synaptic_elements(std::shared_ptr<SynapticElements> axons,
        std::shared_ptr<SynapticElements> excitatory_dendrites, std::shared_ptr<SynapticElements> inhibitory_dendrites) {
        const bool axons_full = axons.operator bool();
        const bool excitatory_dendrites_full = excitatory_dendrites.operator bool();
        const bool inhibitory_dendrites_full = inhibitory_dendrites.operator bool();

        RelearnException::check(axons_full, "Algorithm::set_synaptic_elements: axons was empty");
        RelearnException::check(excitatory_dendrites_full, "Algorithm::set_synaptic_elements: excitatory_dendrites was empty");
        RelearnException::check(inhibitory_dendrites_full, "Algorithm::set_synaptic_elements: inhibitory_dendrites was empty");

        this->axons = std::move(axons);
        this->excitatory_dendrites = std::move(excitatory_dendrites);
        this->inhibitory_dendrites = std::move(inhibitory_dendrites);
    }

    /**
     * @brief Sets the extra infos for the neurons. They hold the positions and update flags for the neurons.
     * @param infos The extra infos, not empty
     * @exception throws a RelearnException if infos is empty
     */
    void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) {
        RelearnException::check(infos.operator bool(), "Algorithm::set_neuron_extra_infos: infos is empty");
        this->extra_infos = std::move(infos);
    }

    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] virtual std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses> update_connectivity(number_neurons_type number_neurons) = 0;

    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      Performs communication via MPI
     * @exception Can throw a RelearnException
     */
    virtual void update_octree() = 0;

protected:
    std::shared_ptr<SynapticElements> axons{}; // NOLINT
    std::shared_ptr<SynapticElements> excitatory_dendrites{}; // NOLINT
    std::shared_ptr<SynapticElements> inhibitory_dendrites{}; // NOLINT

    std::shared_ptr<NeuronsExtraInfo> extra_infos{}; // NOLINT
};
