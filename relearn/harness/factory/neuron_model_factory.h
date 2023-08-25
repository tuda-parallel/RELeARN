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

#include "neurons/models/NeuronModels.h"

#include <memory>

class NeuronModelFactory {
public:
    template <typename NeuronModelType>
    static std::unique_ptr<NeuronModelType> construct_model(const unsigned int h,
                                                            std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator, std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator) {
        if constexpr (std::is_same_v<NeuronModelType, models::PoissonModel>) {
            return construct_poisson_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator));
        } else if constexpr (std::is_same_v<NeuronModelType, models::IzhikevichModel>) {
            return construct_izhikevich_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator));
        } else if constexpr (std::is_same_v<NeuronModelType, models::FitzHughNagumoModel>) {
            return construct_fitzhughnaguma_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator));
        } else {
            return construct_aeif_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator));
        }
    }

    static std::unique_ptr<models::PoissonModel> construct_poisson_model(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
                                                                         std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator) {
        return std::make_unique<models::PoissonModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                                                      models::PoissonModel::default_x_0, models::PoissonModel::default_tau_x, models::PoissonModel::default_refractory_period);
    }

    static std::unique_ptr<models::IzhikevichModel> construct_izhikevich_model(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
                                                                               std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator) {
        return std::make_unique<models::IzhikevichModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                                                         models::IzhikevichModel::default_a, models::IzhikevichModel::default_b, models::IzhikevichModel::default_c, models::IzhikevichModel::default_d,
                                                         models::IzhikevichModel::default_V_spike, models::IzhikevichModel::default_k1, models::IzhikevichModel::default_k2, models::IzhikevichModel::default_k3);
    }

    static std::unique_ptr<models::FitzHughNagumoModel> construct_fitzhughnaguma_model(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
                                                                                       std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator) {
        return std::make_unique<models::FitzHughNagumoModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                                                             models::FitzHughNagumoModel::default_a, models::FitzHughNagumoModel::default_b, models::FitzHughNagumoModel::default_phi);
    }

    static std::unique_ptr<models::AEIFModel> construct_aeif_model(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
                                                                   std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator) {
        return std::make_unique<models::AEIFModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                                                   models::AEIFModel::default_C, models::AEIFModel::default_g_L, models::AEIFModel::default_E_L, models::AEIFModel::default_V_T,
                                                   models::AEIFModel::default_d_T, models::AEIFModel::default_tau_w, models::AEIFModel::default_a, models::AEIFModel::default_b, models::AEIFModel::default_V_spike);
    }
};
