/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_barnes_hut.h"

#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/BarnesHutInternal/BarnesHutInverted.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "structure/Octree.h"

#include <memory>

TEST_F(BarnesHutInvertedTest, testBarnesHutInvertedGetterSetter) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    auto octree = std::make_shared<OctreeImplementation<additional_cell_attributes>>(min, max, 0);

    ASSERT_NO_THROW(BarnesHutInverted algorithm(octree););

    BarnesHutInverted algorithm(octree);

    ASSERT_EQ(algorithm.get_acceptance_criterion(), Constants::bh_default_theta);

    const auto random_acceptance_criterion = RandomAdapter::get_random_double<double>(0.0, Constants::bh_max_theta, mt);

    ASSERT_NO_THROW(algorithm.set_acceptance_criterion(random_acceptance_criterion));
    ASSERT_EQ(algorithm.get_acceptance_criterion(), random_acceptance_criterion);
}
