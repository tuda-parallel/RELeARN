/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_rank_neuron_id.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "neurons/helper/RankNeuronId.h"

TEST_F(RankNeuronIdTest, testNeuronRankIdValid) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
        const auto id = NeuronIdAdapter::get_random_neuron_id(mt);

        const RankNeuronId rni{ rank, NeuronID{ id } };

        ASSERT_EQ(rni.get_neuron_id(), NeuronID{ id });
        ASSERT_TRUE(rni.get_rank() == rank);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdInvalidId) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);

        RankNeuronId rni(rank, NeuronID::uninitialized_id());

        ASSERT_NO_THROW(auto tmp = rni.get_rank());
        ASSERT_THROW(auto tmp = rni.get_neuron_id(), RelearnException);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdEquality) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank_1 = MPIRankAdapter::get_random_mpi_rank(mt);
        const auto id_1 = NeuronIdAdapter::get_random_neuron_id(mt);

        const auto rank_2 = MPIRankAdapter::get_random_mpi_rank(mt);
        const auto id_2 = NeuronIdAdapter::get_random_neuron_id(mt);

        const RankNeuronId rni_1(rank_1, NeuronID{ id_1 });
        const RankNeuronId rni_2(rank_2, NeuronID{ id_2 });

        if (rank_1 == rank_2 && id_1 == id_2) {
            ASSERT_EQ(rni_1, rni_2);
        } else {
            ASSERT_NE(rni_1, rni_2);
        }
    }
}
