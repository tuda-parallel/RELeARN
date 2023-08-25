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

#include "gtest/gtest.h"

#include "util/MPIRank.h"
#include "mpi/CommunicationMap.h"

#include <cmath>
#include <random>

class MPIAdapter {
public:

    /**
     * @brief Simulates a MPI exchange_requests function call for tests without actual mpi
     * @tparam T Request type
     * @param requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template<class T>
    static std::vector<CommunicationMap<T>> exchange_requests(std::vector<CommunicationMap<T>>& requests) {
        const auto num_ranks = requests.size();
        std::vector<CommunicationMap<T>> answers;
        answers.resize(num_ranks, CommunicationMap<T>(num_ranks));

        for(auto sending_rank =0;sending_rank<num_ranks;sending_rank++) {
            for(auto receiving_rank=0; receiving_rank < num_ranks; receiving_rank++) {
                if(!requests[sending_rank].contains(MPIRank(receiving_rank))) {
                    continue;
                }
                RelearnException::check(sending_rank!=receiving_rank, "Sending != receiving");
                const auto& request_block = requests[sending_rank].get_requests(MPIRank(receiving_rank));
                auto& map = answers[receiving_rank];
                for(auto& request: request_block) {
                    map.append(MPIRank(sending_rank), request);
                }
            }
        }
        return answers;
    }
};
