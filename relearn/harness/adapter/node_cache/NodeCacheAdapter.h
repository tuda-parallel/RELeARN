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

#include "algorithm/Cells.h"
#include "structure/NodeCache.h"

class NodeCacheAdapter {
public:
    static void set_node_cache_testing_purposes() {
        NodeCache<BarnesHutCell>::is_already_downloaded = true;
        NodeCache<BarnesHutInvertedCell>::is_already_downloaded = true;
        NodeCache<NaiveCell>::is_already_downloaded = true;
    }

    static void reset_node_cache_testing_purposes() {
        NodeCache<BarnesHutCell>::is_already_downloaded = false;
        NodeCache<BarnesHutInvertedCell>::is_already_downloaded = false;
        NodeCache<NaiveCell>::is_already_downloaded = false;
    }
};
