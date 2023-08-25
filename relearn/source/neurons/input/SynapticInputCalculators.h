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

#include "neurons/input/SynapticInputCalculator.h"

/**
 * This class returns the input from the synapses directly
 */
class LinearSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Constructs a new instance of type LinearSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param synapse_conductance The factor by which the input of a neighboring spiking neuron is weighted
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    LinearSynapticInputCalculator(const double synapse_conductance, std::unique_ptr<FiredStatusCommunicator>&& communicator)
        : SynapticInputCalculator(synapse_conductance, std::move(communicator)) { }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LinearSynapticInputCalculator>(get_synapse_conductance(), get_fired_status_communicator()->clone());
    }

protected:
    void update_synaptic_input(std::span<const FiredStatus> fired) override;
};