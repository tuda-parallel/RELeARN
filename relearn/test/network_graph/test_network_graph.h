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

#include "Types.h"

#include <map>

class NetworkGraph;

class NetworkGraphTest : public RelearnTest {
protected:
    template <typename T, typename synapse_weight>
    void erase_empty(std::map<T, synapse_weight>& edges) {
        std::erase_if(edges, [](const auto& val) { return val.second == 0; });
    }

    template <typename T, typename synapse_weight>
    void erase_empties(std::map<T, std::map<T, synapse_weight>>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            erase_empty<T>(iterator->second);

            if (iterator->second.empty()) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    void assert_local_plastic_empty(const NetworkGraph& network_graph);

    void assert_distant_plastic_empty(const NetworkGraph& network_graph);

    void assert_plastic_empty(const NetworkGraph& network_graph);

    void assert_local_static_empty(const NetworkGraph& network_graph);

    void assert_distant_static_empty(const NetworkGraph& network_graph);

    void assert_static_empty(const NetworkGraph& network_graph);

    void assert_plastic_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type);

    void assert_static_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type);
};
