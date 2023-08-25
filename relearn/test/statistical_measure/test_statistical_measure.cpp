/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_statistical_measure.h"

#include "util/StatisticalMeasures.h"

TEST_F(StatisticalMeasuresTest, testDefaultValues) {
    StatisticalMeasures sm{};

    ASSERT_EQ(sm.avg, 0.0);
    ASSERT_EQ(sm.max, 0.0);
    ASSERT_EQ(sm.min, 0.0);
    ASSERT_EQ(sm.std, 0.0);
    ASSERT_EQ(sm.var, 0.0);
}
