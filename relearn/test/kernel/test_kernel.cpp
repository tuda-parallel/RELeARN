/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_kernel.h"

#include "adapter/random/RandomAdapter.h"

#include "adapter/kernel/KernelAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/kernel/KernelAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/Cells.h"
#include "algorithm/Kernel/Kernel.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

#include <gtest/gtest.h>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/indices.hpp>

TEST_F(KernelTest, testCalculateAttractivenessSameNode) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    OctreeNode<BarnesHutCell> node{};
    node.set_cell_neuron_id(neuron_id);
    node.set_rank(MPIRank::root_rank());

    const auto attractiveness = Kernel<BarnesHutCell>::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id }, position, &node, element_type, signal_type);
    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(KernelTest, testPickTargetEmpty2) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    auto* result = Kernel<BarnesHutCell>::pick_target({ MPIRank::root_rank(), neuron_id }, position, {}, element_type, signal_type);

    ASSERT_EQ(result, nullptr);
}