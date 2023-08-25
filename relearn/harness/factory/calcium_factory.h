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

#include "neurons/CalciumCalculator.h"

#include <memory>

class CalciumFactory {
public:
    static std::unique_ptr<CalciumCalculator> construct_calcium_calculator_no_decay() {
        auto calc = std::make_unique<CalciumCalculator>(TargetCalciumDecay::None);
        calc->set_initial_calcium_calculator([](MPIRank, NeuronID::value_type) { return 0.4; });
        calc->set_target_calcium_calculator([](MPIRank, NeuronID::value_type) { return 0.8; });
        return calc;
    }

    static std::unique_ptr<CalciumCalculator> construct_calcium_calculator_relative_decay() {
        auto calc = std::make_unique<CalciumCalculator>(TargetCalciumDecay::Relative, 0.1, 1);
        calc->set_initial_calcium_calculator([](MPIRank, NeuronID::value_type) { return 0.4; });
        calc->set_target_calcium_calculator([](MPIRank, NeuronID::value_type) { return 0.8; });
        return calc;
    }

    static std::unique_ptr<CalciumCalculator> construct_calcium_calculator_absolute_decay() {
        auto calc = std::make_unique<CalciumCalculator>(TargetCalciumDecay::Absolute, 0.1, 1);
        calc->set_initial_calcium_calculator([](MPIRank, NeuronID::value_type) { return 0.4; });
        calc->set_target_calcium_calculator([](MPIRank, NeuronID::value_type) { return 0.8; });
        return calc;
    }
};
