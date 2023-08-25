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

#include "RelearnTest.hpp"
#include "neurons/Neurons.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/input/FiredStatusCommunicationMap.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/input/BackgroundActivityCalculators.h"

#include <utility>

class NeuronsTest : public RelearnTest {
protected:
    static std::tuple<std::shared_ptr<Neurons>, std::shared_ptr<NetworkGraph>> create_neurons_object(std::shared_ptr<Partition>& partition, MPIRank rank) {
        auto model = std::make_unique<models::PoissonModel>(models::PoissonModel::default_h,
            std::make_unique<LinearSynapticInputCalculator>(SynapticInputCalculator::default_conductance, std::make_unique<FiredStatusCommunicationMap>(1)),
            std::make_unique<NullBackgroundActivityCalculator>(),
            std::make_unique<Stimulus>(),
            models::PoissonModel::default_x_0,
            models::PoissonModel::default_tau_x,
            models::PoissonModel::default_refractory_period);
        auto calcium = std::make_unique<CalciumCalculator>();
        calcium->set_initial_calcium_calculator([](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
        calcium->set_target_calcium_calculator([](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
        auto network_graph = std::make_shared<NetworkGraph>(rank);
        auto dends_ex = std::make_shared<SynapticElements>(ElementType::Dendrite, 0.2);
        auto dends_in = std::make_shared<SynapticElements>(ElementType::Dendrite, 0.2);
        auto axs = std::make_shared<SynapticElements>(ElementType::Axon, 0.2);

        auto sdf = std::make_unique<RandomSynapseDeletionFinder>();
        sdf->set_axons(axs);
        sdf->set_dendrites_ex(dends_ex);
        sdf->set_dendrites_in(dends_in);

        auto neurons = std::make_shared<Neurons>(partition, std::move(model), std::move(calcium), network_graph, std::move(axs), std::move(dends_ex), std::move(dends_in), std::move(sdf));
        return { neurons, network_graph };
    }
};
