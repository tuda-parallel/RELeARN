/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_calcium_calculator.h"

#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"

#include "neurons/CalciumCalculator.h"
#include "neurons/NeuronsExtraInfo.h"
#include <range/v3/range/conversion.hpp>

TEST_F(CalciumCalculatorTest, testCalciumCalculatorConstructorNone) {
    const auto decay_amount = RandomAdapter::get_random_double(-10000.0, 10000.0, mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_NO_THROW(CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0)) << 0.0 << ' ' << 0;
    ASSERT_NO_THROW(CalciumCalculator cc2(TargetCalciumDecay::None, 0.0, decay_step)) << 0.0 << ' ' << decay_step;
    ASSERT_NO_THROW(CalciumCalculator cc3(TargetCalciumDecay::None, decay_amount, 0)) << decay_amount << ' ' << 0;
    ASSERT_NO_THROW(CalciumCalculator cc4(TargetCalciumDecay::None, decay_amount, decay_step)) << decay_amount << ' ' << decay_step;
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorConstructorRelative) {
    const auto decay_amount = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1.0, mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_NO_THROW(CalciumCalculator cc1(TargetCalciumDecay::Relative, 0.0, 1000)) << 1.0 << ' ' << 1000;
    ASSERT_NO_THROW(CalciumCalculator cc2(TargetCalciumDecay::Relative, 0.0, decay_step)) << 1.0 << ' ' << decay_step;
    ASSERT_NO_THROW(CalciumCalculator cc3(TargetCalciumDecay::Relative, decay_amount, 1000)) << decay_amount << ' ' << 1000;
    ASSERT_NO_THROW(CalciumCalculator cc4(TargetCalciumDecay::Relative, decay_amount, decay_step)) << decay_amount << ' ' << decay_step;
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorConstructorAbsolute) {
    const auto decay_amount = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1.0, mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_NO_THROW(CalciumCalculator cc1(TargetCalciumDecay::Absolute, 1.0, 1000)) << 1.0 << ' ' << 1000;
    ASSERT_NO_THROW(CalciumCalculator cc2(TargetCalciumDecay::Absolute, 1.0, decay_step)) << 1.0 << ' ' << decay_step;
    ASSERT_NO_THROW(CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount, 1000)) << decay_amount << ' ' << 1000;
    ASSERT_NO_THROW(CalciumCalculator cc4(TargetCalciumDecay::Absolute, decay_amount, decay_step)) << decay_amount << ' ' << decay_step;
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorConstructorRelativeException) {
    const auto decay_amount_low = RandomAdapter::get_random_double(-1000.0, std::nextafter(0.0, -1.0), mt);
    const auto decay_amount_high = RandomAdapter::get_random_double(1.0, 1000.0, mt);

    ASSERT_THROW(CalciumCalculator cc1(TargetCalciumDecay::Relative, decay_amount_low, 1000), RelearnException) << decay_amount_low << ' ' << 1000;
    ASSERT_THROW(CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_high, 1000), RelearnException) << decay_amount_high << ' ' << 1000;
    ASSERT_THROW(CalciumCalculator cc2(TargetCalciumDecay::Relative, 1.0, 1000), RelearnException) << 0.0 << ' ' << 1000;
    ASSERT_THROW(CalciumCalculator cc3(TargetCalciumDecay::Relative, 0.5, 0), RelearnException) << 0.5 << ' ' << 0;
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorConstructorAbsoluteException) {
    const auto decay_amount = RandomAdapter::get_random_double(-1000.0, std::nextafter(0.0, -1.0), mt);

    ASSERT_THROW(CalciumCalculator cc1(TargetCalciumDecay::Absolute, 0.5, 0), RelearnException) << 0.5 << ' ' << 0;
    ASSERT_THROW(CalciumCalculator cc1(TargetCalciumDecay::Absolute, 0.0, 100), RelearnException) << 0.0 << ' ' << 100;
    ASSERT_THROW(CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount, 100), RelearnException) << decay_amount << ' ' << 100;
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorConstructurGetter) {
    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    ASSERT_EQ(cc1.get_decay_type(), TargetCalciumDecay::None);
    ASSERT_EQ(cc2.get_decay_type(), TargetCalciumDecay::Relative);
    ASSERT_EQ(cc3.get_decay_type(), TargetCalciumDecay::Absolute);

    ASSERT_EQ(cc1.get_decay_amount(), 0.0);
    ASSERT_EQ(cc2.get_decay_amount(), decay_amount_relative);
    ASSERT_EQ(cc3.get_decay_amount(), decay_amount_absolute);

    ASSERT_EQ(cc1.get_decay_step(), 0);
    ASSERT_EQ(cc2.get_decay_step(), decay_step);
    ASSERT_EQ(cc3.get_decay_step(), decay_step);
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorGetterSetter) {
    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    ASSERT_EQ(cc1.get_beta(), CalciumCalculator::default_beta);
    ASSERT_EQ(cc1.get_tau_C(), CalciumCalculator::default_tau_C);
    ASSERT_EQ(cc1.get_h(), CalciumCalculator::default_h);

    ASSERT_EQ(cc2.get_beta(), CalciumCalculator::default_beta);
    ASSERT_EQ(cc2.get_tau_C(), CalciumCalculator::default_tau_C);
    ASSERT_EQ(cc2.get_h(), CalciumCalculator::default_h);

    ASSERT_EQ(cc3.get_beta(), CalciumCalculator::default_beta);
    ASSERT_EQ(cc3.get_tau_C(), CalciumCalculator::default_tau_C);
    ASSERT_EQ(cc3.get_h(), CalciumCalculator::default_h);

    const auto beta1 = beta_distr(mt);
    const auto tau_C1 = tau_C_distr(mt);
    const auto h1 = h_distr(mt);

    ASSERT_NO_THROW(cc1.set_beta(beta1));
    ASSERT_NO_THROW(cc1.set_tau_C(tau_C1));
    ASSERT_NO_THROW(cc1.set_h(h1));

    ASSERT_NO_THROW(cc2.set_beta(beta1));
    ASSERT_NO_THROW(cc2.set_tau_C(tau_C1));
    ASSERT_NO_THROW(cc2.set_h(h1));

    ASSERT_NO_THROW(cc3.set_beta(beta1));
    ASSERT_NO_THROW(cc3.set_tau_C(tau_C1));
    ASSERT_NO_THROW(cc3.set_h(h1));

    ASSERT_EQ(cc1.get_beta(), beta1);
    ASSERT_EQ(cc1.get_tau_C(), tau_C1);
    ASSERT_EQ(cc1.get_h(), h1);

    ASSERT_EQ(cc2.get_beta(), beta1);
    ASSERT_EQ(cc2.get_tau_C(), tau_C1);
    ASSERT_EQ(cc2.get_h(), h1);

    ASSERT_EQ(cc3.get_beta(), beta1);
    ASSERT_EQ(cc3.get_tau_C(), tau_C1);
    ASSERT_EQ(cc3.get_h(), h1);

    const auto beta2 = beta_distr(mt);
    const auto tau_C2 = tau_C_distr(mt);
    const auto h2 = h_distr(mt);

    ASSERT_NO_THROW(cc1.set_beta(beta2));
    ASSERT_NO_THROW(cc1.set_tau_C(tau_C2));
    ASSERT_NO_THROW(cc1.set_h(h2));

    ASSERT_NO_THROW(cc2.set_beta(beta2));
    ASSERT_NO_THROW(cc2.set_tau_C(tau_C2));
    ASSERT_NO_THROW(cc2.set_h(h2));

    ASSERT_NO_THROW(cc3.set_beta(beta2));
    ASSERT_NO_THROW(cc3.set_tau_C(tau_C2));
    ASSERT_NO_THROW(cc3.set_h(h2));

    ASSERT_EQ(cc2.get_beta(), beta2);
    ASSERT_EQ(cc2.get_tau_C(), tau_C2);
    ASSERT_EQ(cc2.get_h(), h2);

    ASSERT_EQ(cc2.get_beta(), beta2);
    ASSERT_EQ(cc2.get_tau_C(), tau_C2);
    ASSERT_EQ(cc2.get_h(), h2);

    ASSERT_EQ(cc3.get_beta(), beta2);
    ASSERT_EQ(cc3.get_tau_C(), tau_C2);
    ASSERT_EQ(cc3.get_h(), h2);

    ASSERT_EQ(cc1.get_decay_type(), TargetCalciumDecay::None);
    ASSERT_EQ(cc2.get_decay_type(), TargetCalciumDecay::Relative);
    ASSERT_EQ(cc3.get_decay_type(), TargetCalciumDecay::Absolute);

    ASSERT_EQ(cc1.get_decay_amount(), 0.0);
    ASSERT_EQ(cc2.get_decay_amount(), decay_amount_relative);
    ASSERT_EQ(cc3.get_decay_amount(), decay_amount_absolute);

    ASSERT_EQ(cc1.get_decay_step(), 0);
    ASSERT_EQ(cc2.get_decay_step(), decay_step);
    ASSERT_EQ(cc3.get_decay_step(), decay_step);
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorGetterSetterException) {

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    ASSERT_EQ(cc1.get_beta(), CalciumCalculator::default_beta);
    ASSERT_EQ(cc1.get_tau_C(), CalciumCalculator::default_tau_C);
    ASSERT_EQ(cc1.get_h(), CalciumCalculator::default_h);

    ASSERT_EQ(cc2.get_beta(), CalciumCalculator::default_beta);
    ASSERT_EQ(cc2.get_tau_C(), CalciumCalculator::default_tau_C);
    ASSERT_EQ(cc2.get_h(), CalciumCalculator::default_h);

    ASSERT_EQ(cc3.get_beta(), CalciumCalculator::default_beta);
    ASSERT_EQ(cc3.get_tau_C(), CalciumCalculator::default_tau_C);
    ASSERT_EQ(cc3.get_h(), CalciumCalculator::default_h);

    const auto beta1 = beta_distr(mt);
    const auto tau_C1 = tau_C_distr(mt);
    const auto h1 = h_distr(mt);

    ASSERT_NO_THROW(cc1.set_beta(beta1));
    ASSERT_NO_THROW(cc1.set_tau_C(tau_C1));
    ASSERT_NO_THROW(cc1.set_h(h1));

    ASSERT_NO_THROW(cc2.set_beta(beta1));
    ASSERT_NO_THROW(cc2.set_tau_C(tau_C1));
    ASSERT_NO_THROW(cc2.set_h(h1));

    ASSERT_NO_THROW(cc3.set_beta(beta1));
    ASSERT_NO_THROW(cc3.set_tau_C(tau_C1));
    ASSERT_NO_THROW(cc3.set_h(h1));

    ASSERT_EQ(cc1.get_beta(), beta1);
    ASSERT_EQ(cc1.get_tau_C(), tau_C1);
    ASSERT_EQ(cc1.get_h(), h1);

    ASSERT_EQ(cc2.get_beta(), beta1);
    ASSERT_EQ(cc2.get_tau_C(), tau_C1);
    ASSERT_EQ(cc2.get_h(), h1);

    ASSERT_EQ(cc3.get_beta(), beta1);
    ASSERT_EQ(cc3.get_tau_C(), tau_C1);
    ASSERT_EQ(cc3.get_h(), h1);

    const auto beta2 = beta_distr(mt);
    const auto tau_C2 = tau_C_distr(mt);
    const auto h2 = h_distr(mt);

    ASSERT_NO_THROW(cc1.set_beta(beta2));
    ASSERT_NO_THROW(cc1.set_tau_C(tau_C2));
    ASSERT_NO_THROW(cc1.set_h(h2));

    ASSERT_NO_THROW(cc2.set_beta(beta2));
    ASSERT_NO_THROW(cc2.set_tau_C(tau_C2));
    ASSERT_NO_THROW(cc2.set_h(h2));

    ASSERT_NO_THROW(cc3.set_beta(beta2));
    ASSERT_NO_THROW(cc3.set_tau_C(tau_C2));
    ASSERT_NO_THROW(cc3.set_h(h2));

    ASSERT_EQ(cc2.get_beta(), beta2);
    ASSERT_EQ(cc2.get_tau_C(), tau_C2);
    ASSERT_EQ(cc2.get_h(), h2);

    ASSERT_EQ(cc2.get_beta(), beta2);
    ASSERT_EQ(cc2.get_tau_C(), tau_C2);
    ASSERT_EQ(cc2.get_h(), h2);

    ASSERT_EQ(cc3.get_beta(), beta2);
    ASSERT_EQ(cc3.get_tau_C(), tau_C2);
    ASSERT_EQ(cc3.get_h(), h2);

    ASSERT_EQ(cc1.get_decay_type(), TargetCalciumDecay::None);
    ASSERT_EQ(cc2.get_decay_type(), TargetCalciumDecay::Relative);
    ASSERT_EQ(cc3.get_decay_type(), TargetCalciumDecay::Absolute);

    ASSERT_EQ(cc1.get_decay_amount(), 0.0);
    ASSERT_EQ(cc2.get_decay_amount(), decay_amount_relative);
    ASSERT_EQ(cc3.get_decay_amount(), decay_amount_absolute);

    ASSERT_EQ(cc1.get_decay_step(), 0);
    ASSERT_EQ(cc2.get_decay_step(), decay_step);
    ASSERT_EQ(cc3.get_decay_step(), decay_step);
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorInitialTargetCalcium) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    cc1.set_beta(beta);
    cc1.set_tau_C(tau_C);
    cc1.set_h(h);
    cc1.set_initial_calcium_calculator(initiator);
    cc1.set_target_calcium_calculator(calculator);

    cc2.set_beta(beta);
    cc2.set_tau_C(tau_C);
    cc2.set_h(h);
    cc2.set_initial_calcium_calculator(initiator);
    cc2.set_target_calcium_calculator(calculator);

    cc3.set_beta(beta);
    cc3.set_tau_C(tau_C);
    cc3.set_h(h);
    cc3.set_initial_calcium_calculator(initiator);
    cc3.set_target_calcium_calculator(calculator);

    ASSERT_NO_THROW(cc1.init(number_neurons));
    ASSERT_NO_THROW(cc2.init(number_neurons));
    ASSERT_NO_THROW(cc3.init(number_neurons));

    const auto& calcium1 = cc1.get_calcium();
    const auto& targets1 = cc1.get_target_calcium();

    const auto& calcium2 = cc2.get_calcium();
    const auto& targets2 = cc2.get_target_calcium();

    const auto& calcium3 = cc3.get_calcium();
    const auto& targets3 = cc3.get_target_calcium();

    ASSERT_EQ(calcium1.size(), number_neurons);
    ASSERT_EQ(calcium2.size(), number_neurons);
    ASSERT_EQ(calcium3.size(), number_neurons);

    ASSERT_EQ(targets1.size(), number_neurons);
    ASSERT_EQ(targets2.size(), number_neurons);
    ASSERT_EQ(targets3.size(), number_neurons);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto initial = static_cast<double>(neuron_id) / 7.342;
        const auto target = static_cast<double>(neuron_id) * 5.98;

        ASSERT_EQ(calcium1[neuron_id], initial);
        ASSERT_EQ(targets1[neuron_id], target);

        ASSERT_EQ(calcium2[neuron_id], initial);
        ASSERT_EQ(targets2[neuron_id], target);

        ASSERT_EQ(calcium3[neuron_id], initial);
        ASSERT_EQ(targets3[neuron_id], target);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorCreate) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_created_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) * 5.98;
    };

    auto initiator_created = [number_neurons, number_created_neurons](MPIRank i, NeuronID::value_type v) {
        if (v < number_neurons) {
            return -1.0;
        }

        if (v >= number_neurons + number_created_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) / 1.52;
    };

    auto calculator_created = [number_neurons, number_created_neurons](MPIRank i, NeuronID::value_type v) {
        if (v < number_neurons) {
            return -1.0;
        }

        if (v >= number_neurons + number_created_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) * 86.2;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    cc1.set_beta(beta);
    cc1.set_tau_C(tau_C);
    cc1.set_h(h);
    cc1.set_initial_calcium_calculator(initiator);
    cc1.set_target_calcium_calculator(calculator);

    cc2.set_beta(beta);
    cc2.set_tau_C(tau_C);
    cc2.set_h(h);
    cc2.set_initial_calcium_calculator(initiator);
    cc2.set_target_calcium_calculator(calculator);

    cc3.set_beta(beta);
    cc3.set_tau_C(tau_C);
    cc3.set_h(h);
    cc3.set_initial_calcium_calculator(initiator);
    cc3.set_target_calcium_calculator(calculator);

    ASSERT_NO_THROW(cc1.init(number_neurons));
    ASSERT_NO_THROW(cc2.init(number_neurons));
    ASSERT_NO_THROW(cc3.init(number_neurons));

    cc1.set_initial_calcium_calculator(initiator_created);
    cc1.set_target_calcium_calculator(calculator_created);

    cc2.set_initial_calcium_calculator(initiator_created);
    cc2.set_target_calcium_calculator(calculator_created);

    cc3.set_initial_calcium_calculator(initiator_created);
    cc3.set_target_calcium_calculator(calculator_created);

    ASSERT_NO_THROW(cc1.create_neurons(number_created_neurons));
    ASSERT_NO_THROW(cc2.create_neurons(number_created_neurons));
    ASSERT_NO_THROW(cc3.create_neurons(number_created_neurons));

    const auto& calcium1 = cc1.get_calcium();
    const auto& targets1 = cc1.get_target_calcium();

    const auto& calcium2 = cc2.get_calcium();
    const auto& targets2 = cc2.get_target_calcium();

    const auto& calcium3 = cc3.get_calcium();
    const auto& targets3 = cc3.get_target_calcium();

    ASSERT_EQ(calcium1.size(), number_neurons + number_created_neurons);
    ASSERT_EQ(calcium2.size(), number_neurons + number_created_neurons);
    ASSERT_EQ(calcium3.size(), number_neurons + number_created_neurons);

    ASSERT_EQ(targets1.size(), number_neurons + number_created_neurons);
    ASSERT_EQ(targets2.size(), number_neurons + number_created_neurons);
    ASSERT_EQ(targets3.size(), number_neurons + number_created_neurons);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto initial = static_cast<double>(neuron_id) / 7.342;
        const auto target = static_cast<double>(neuron_id) * 5.98;

        ASSERT_EQ(calcium1[neuron_id], initial);
        ASSERT_EQ(targets1[neuron_id], target);

        ASSERT_EQ(calcium2[neuron_id], initial);
        ASSERT_EQ(targets2[neuron_id], target);

        ASSERT_EQ(calcium3[neuron_id], initial);
        ASSERT_EQ(targets3[neuron_id], target);
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons, number_neurons + number_created_neurons)) {
        const auto initial = static_cast<double>(neuron_id) / 1.52;
        const auto target = static_cast<double>(neuron_id) * 86.2;

        ASSERT_EQ(calcium1[neuron_id], initial);
        ASSERT_EQ(targets1[neuron_id], target);

        ASSERT_EQ(calcium2[neuron_id], initial);
        ASSERT_EQ(targets2[neuron_id], target);

        ASSERT_EQ(calcium3[neuron_id], initial);
        ASSERT_EQ(targets3[neuron_id], target);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorZeroNeurons) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    cc1.set_beta(beta);
    cc1.set_tau_C(tau_C);
    cc1.set_h(h);
    cc1.set_initial_calcium_calculator(initiator);
    cc1.set_target_calcium_calculator(calculator);

    cc2.set_beta(beta);
    cc2.set_tau_C(tau_C);
    cc2.set_h(h);
    cc2.set_initial_calcium_calculator(initiator);
    cc2.set_target_calcium_calculator(calculator);

    cc3.set_beta(beta);
    cc3.set_tau_C(tau_C);
    cc3.set_h(h);
    cc3.set_initial_calcium_calculator(initiator);
    cc3.set_target_calcium_calculator(calculator);

    ASSERT_THROW(cc1.init(0), RelearnException);
    ASSERT_THROW(cc2.init(0), RelearnException);
    ASSERT_THROW(cc3.init(0), RelearnException);

    ASSERT_THROW(cc1.create_neurons(0), RelearnException);
    ASSERT_THROW(cc2.create_neurons(0), RelearnException);
    ASSERT_THROW(cc3.create_neurons(0), RelearnException);
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorEmptyFunctions) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        if (v >= number_neurons) {
            return -1.0;
        }

        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    cc1.set_beta(beta);
    cc1.set_tau_C(tau_C);
    cc1.set_h(h);

    cc2.set_beta(beta);
    cc2.set_tau_C(tau_C);
    cc2.set_h(h);

    cc3.set_beta(beta);
    cc3.set_tau_C(tau_C);
    cc3.set_h(h);

    ASSERT_THROW(cc1.init(number_neurons), RelearnException);
    ASSERT_THROW(cc2.init(number_neurons), RelearnException);
    ASSERT_THROW(cc3.init(number_neurons), RelearnException);

    ASSERT_THROW(cc1.create_neurons(number_neurons), RelearnException);
    ASSERT_THROW(cc2.create_neurons(number_neurons), RelearnException);
    ASSERT_THROW(cc3.create_neurons(number_neurons), RelearnException);

    cc1.set_initial_calcium_calculator(initiator);
    cc2.set_initial_calcium_calculator(initiator);
    cc3.set_initial_calcium_calculator(initiator);

    ASSERT_THROW(cc1.init(number_neurons), RelearnException);
    ASSERT_THROW(cc2.init(number_neurons), RelearnException);
    ASSERT_THROW(cc3.init(number_neurons), RelearnException);

    ASSERT_THROW(cc1.create_neurons(number_neurons), RelearnException);
    ASSERT_THROW(cc2.create_neurons(number_neurons), RelearnException);
    ASSERT_THROW(cc3.create_neurons(number_neurons), RelearnException);

    cc1.set_initial_calcium_calculator({});
    cc2.set_initial_calcium_calculator({});
    cc3.set_initial_calcium_calculator({});

    cc1.set_target_calcium_calculator(calculator);
    cc2.set_target_calcium_calculator(calculator);
    cc3.set_target_calcium_calculator(calculator);

    ASSERT_THROW(cc1.init(number_neurons), RelearnException);
    ASSERT_THROW(cc2.init(number_neurons), RelearnException);
    ASSERT_THROW(cc3.init(number_neurons), RelearnException);

    ASSERT_THROW(cc1.create_neurons(number_neurons), RelearnException);
    ASSERT_THROW(cc2.create_neurons(number_neurons), RelearnException);
    ASSERT_THROW(cc3.create_neurons(number_neurons), RelearnException);
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateException) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc1(TargetCalciumDecay::None, 0.0, 0);
    CalciumCalculator cc2(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);
    CalciumCalculator cc3(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    cc1.set_beta(beta);
    cc1.set_tau_C(tau_C);
    cc1.set_h(h);
    cc1.set_initial_calcium_calculator(initiator);
    cc1.set_target_calcium_calculator(calculator);
    cc1.set_extra_infos(extra_info);

    cc2.set_beta(beta);
    cc2.set_tau_C(tau_C);
    cc2.set_h(h);
    cc2.set_initial_calcium_calculator(initiator);
    cc2.set_target_calcium_calculator(calculator);
    cc2.set_extra_infos(extra_info);

    cc3.set_beta(beta);
    cc3.set_tau_C(tau_C);
    cc3.set_h(h);
    cc3.set_initial_calcium_calculator(initiator);
    cc3.set_target_calcium_calculator(calculator);
    cc3.set_extra_infos(extra_info);

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_THROW(cc1.update_calcium(step, std::span<const FiredStatus>{ { FiredStatus::Fired } }), RelearnException);
    ASSERT_THROW(cc2.update_calcium(step, std::span<const FiredStatus>{ { FiredStatus::Fired } }), RelearnException);
    ASSERT_THROW(cc3.update_calcium(step, std::span<const FiredStatus>{ { FiredStatus::Fired } }), RelearnException);

    const auto fired_size = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto update_size = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<FiredStatus> fired_status(fired_size);

    extra_info->init(update_size == fired_size ? update_size + 1 : update_size);

    ASSERT_THROW(cc1.update_calcium(step, {}), RelearnException);
    ASSERT_THROW(cc2.update_calcium(step, {}), RelearnException);
    ASSERT_THROW(cc3.update_calcium(step, {}), RelearnException);

    ASSERT_NO_THROW(cc1.init(number_neurons));
    ASSERT_NO_THROW(cc2.init(number_neurons));
    ASSERT_NO_THROW(cc3.init(number_neurons));

    ASSERT_THROW(cc1.update_calcium(0, fired_status), RelearnException);
    ASSERT_THROW(cc1.update_calcium(step, fired_status), RelearnException);
    ASSERT_THROW(cc2.update_calcium(0, fired_status), RelearnException);
    ASSERT_THROW(cc2.update_calcium(step, fired_status), RelearnException);
    ASSERT_THROW(cc3.update_calcium(0, fired_status), RelearnException);
    ASSERT_THROW(cc3.update_calcium(step, fired_status), RelearnException);
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateNoneDisabled) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::None, 0.0, 0);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    const auto neuron_ids = NeuronID::range(number_neurons) | ranges::to_vector;
    extra_info->set_disabled_neurons(neuron_ids);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);
    std::vector<FiredStatus> fired_status2(number_neurons, FiredStatus::Fired);

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status));

    const auto now_calcium_2 = cc.get_calcium();
    const auto now_target_2 = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_2[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_2[neuron_id]);
    }

    const auto now_calcium_3 = cc.get_calcium();
    const auto now_target_3 = cc.get_target_calcium();

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status2));

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_3[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_3[neuron_id]);
    }

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status2));

    const auto now_calcium_4 = cc.get_calcium();
    const auto now_target_4 = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_4[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_4[neuron_id]);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateNoneStep0) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::None, 0.0, 0);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(number_neurons, mt);
    NeuronTypesAdapter::disable_neurons(number_neurons, extra_info, mt);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);

        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
            continue;
        }

        auto expected_calcium = previous_calcium[neuron_id];
        auto update_value = fired_status[neuron_id] == FiredStatus::Fired ? beta : 0.0;

        for (auto i = 0U; i < h; i++) {
            expected_calcium = expected_calcium + (1.0 / h) * (expected_calcium / -tau_C + update_value);
        }

        const auto now_value = now_calcium[neuron_id];

        ASSERT_NEAR(expected_calcium, now_value, eps);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateNone) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::None, 0.0, 0);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(number_neurons, mt);
    NeuronTypesAdapter::disable_neurons(number_neurons, extra_info, mt);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);

        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
            continue;
        }

        auto expected_calcium = previous_calcium[neuron_id];
        auto update_value = fired_status[neuron_id] == FiredStatus::Fired ? beta : 0.0;

        for (auto i = 0U; i < h; i++) {
            expected_calcium = expected_calcium + (1.0 / h) * (expected_calcium / -tau_C + update_value);
        }

        const auto now_value = now_calcium[neuron_id];

        ASSERT_NEAR(expected_calcium, now_value, eps);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateRelativeDisabled) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);
    std::vector<FiredStatus> fired_status2(number_neurons, FiredStatus::Fired);
    const auto neuron_ids = NeuronID::range(number_neurons) | ranges::to_vector;
    extra_info->set_disabled_neurons(neuron_ids);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status));

    const auto now_calcium_2 = cc.get_calcium();
    const auto now_target_2 = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_2[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_2[neuron_id]);
    }

    const auto now_calcium_3 = cc.get_calcium();
    const auto now_target_3 = cc.get_target_calcium();

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status2));

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_3[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_3[neuron_id]);
    }

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status2));

    const auto now_calcium_4 = cc.get_calcium();
    const auto now_target_4 = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_4[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_4[neuron_id]);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateRelativeStep0) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(number_neurons, mt);
    NeuronTypesAdapter::disable_neurons(number_neurons, extra_info, mt);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
            ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
            continue;
        }

        ASSERT_NEAR(previous_target[neuron_id] * decay_amount_relative, now_target[neuron_id], eps);

        auto expected_calcium = previous_calcium[neuron_id];
        auto update_value = fired_status[neuron_id] == FiredStatus::Fired ? beta : 0.0;

        for (auto i = 0U; i < h; i++) {
            expected_calcium = expected_calcium + (1.0 / h) * (expected_calcium / -tau_C + update_value);
        }

        const auto now_value = now_calcium[neuron_id];

        ASSERT_NEAR(expected_calcium, now_value, eps);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateRelative) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_relative = RandomAdapter::get_random_double(0.0, std::nextafter(1.0, 0.0), mt);
    const auto decay_step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::Relative, decay_amount_relative, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(number_neurons, mt);
    NeuronTypesAdapter::disable_neurons(number_neurons, extra_info, mt);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    if (step % decay_step == 0) {
        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            if (update_status[neuron_id] == UpdateStatus::Disabled) {
                ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
                continue;
            }
            ASSERT_NEAR(previous_target[neuron_id] * decay_amount_relative, now_target[neuron_id], eps);
        }
    } else {
        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
        }
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
            continue;
        }

        auto expected_calcium = previous_calcium[neuron_id];
        auto update_value = fired_status[neuron_id] == FiredStatus::Fired ? beta : 0.0;

        for (auto i = 0U; i < h; i++) {
            expected_calcium = expected_calcium + (1.0 / h) * (expected_calcium / -tau_C + update_value);
        }

        const auto now_value = now_calcium[neuron_id];

        ASSERT_NEAR(expected_calcium, now_value, eps);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateAbsoluteDisabled) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);
    std::vector<FiredStatus> fired_status2(number_neurons, FiredStatus::Fired);
    const auto neuron_ids = NeuronID::range(number_neurons) | ranges::to_vector;
    extra_info->set_disabled_neurons(neuron_ids);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status));

    const auto now_calcium_2 = cc.get_calcium();
    const auto now_target_2 = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_2[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_2[neuron_id]);
    }

    const auto now_calcium_3 = cc.get_calcium();
    const auto now_target_3 = cc.get_target_calcium();

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status2));

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_3[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_3[neuron_id]);
    }

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status2));

    const auto now_calcium_4 = cc.get_calcium();
    const auto now_target_4 = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(previous_calcium[neuron_id], now_calcium_4[neuron_id]);
        ASSERT_EQ(previous_target[neuron_id], now_target_4[neuron_id]);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateAbsoluteStep0) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(number_neurons, mt);
    const auto neuron_ids = NeuronID::range(number_neurons) | ranges::to_vector;
    extra_info->set_disabled_neurons(neuron_ids);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(0, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
            ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
            continue;
        }

        ASSERT_NEAR(previous_target[neuron_id] - decay_amount_absolute, now_target[neuron_id], eps);

        auto expected_calcium = previous_calcium[neuron_id];
        auto update_value = fired_status[neuron_id] == FiredStatus::Fired ? beta : 0.0;

        for (auto i = 0U; i < h; i++) {
            expected_calcium = expected_calcium + (1.0 / h) * (expected_calcium / -tau_C + update_value);
        }

        const auto now_value = now_calcium[neuron_id];

        ASSERT_NEAR(expected_calcium, now_value, eps);
    }
}

TEST_F(CalciumCalculatorTest, testCalciumCalculatorUpdateAbsolute) {
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    auto initiator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) / 7.342;
    };

    auto calculator = [number_neurons](MPIRank i, NeuronID::value_type v) {
        return static_cast<double>(v) * 5.98;
    };

    const auto decay_amount_absolute = RandomAdapter::get_random_double(std::nextafter(0.0, 1.0), 1000.0, mt);
    const auto decay_step
        = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    boost::random::uniform_real_distribution<double> beta_distr(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    boost::random::uniform_real_distribution<double> tau_C_distr(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    boost::random::uniform_int_distribution<unsigned int> h_distr(CalciumCalculator::min_h, CalciumCalculator::max_h);

    CalciumCalculator cc(TargetCalciumDecay::Absolute, decay_amount_absolute, decay_step);

    const auto beta = beta_distr(mt);
    const auto tau_C = tau_C_distr(mt);
    const auto h = h_distr(mt);

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 10000000, mt);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator(initiator);
    cc.set_target_calcium_calculator(calculator);
    cc.set_extra_infos(extra_info);

    ASSERT_NO_THROW(cc.init(number_neurons));

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(number_neurons, mt);
    const auto neuron_ids = NeuronID::range(number_neurons) | ranges::to_vector;
    extra_info->set_disabled_neurons(neuron_ids);
    const auto update_status = extra_info->get_disable_flags();

    const auto previous_calcium = vectorify_span(cc.get_calcium());
    const auto previous_target = vectorify_span(cc.get_target_calcium());

    ASSERT_NO_THROW(cc.update_calcium(step, fired_status));

    const auto now_calcium = cc.get_calcium();
    const auto now_target = cc.get_target_calcium();

    if (step % decay_step == 0) {
        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            if (update_status[neuron_id] == UpdateStatus::Disabled) {
                ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
                continue;
            }
            ASSERT_NEAR(previous_target[neuron_id] - decay_amount_absolute, now_target[neuron_id], eps);
        }
    } else {
        for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
            ASSERT_EQ(previous_target[neuron_id], now_target[neuron_id]);
        }
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(previous_calcium[neuron_id], now_calcium[neuron_id]);
            continue;
        }

        auto expected_calcium = previous_calcium[neuron_id];
        auto update_value = fired_status[neuron_id] == FiredStatus::Fired ? beta : 0.0;

        for (auto i = 0U; i < h; i++) {
            expected_calcium = expected_calcium + (1.0 / h) * (expected_calcium / -tau_C + update_value);
        }

        const auto now_value = now_calcium[neuron_id];

        ASSERT_NEAR(expected_calcium, now_value, eps);
    }
}
