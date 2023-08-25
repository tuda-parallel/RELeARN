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

#include "algorithm/Internal/ExchangingAlgorithm.h"

#include "Types.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "neurons/enums/SignalType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;
class SynapticElements;

/**
 * This class represents the implementation of the trivial O(n^2) algorithm.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class Naive : public ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse, NaiveCell> {
public:
    using AdditionalCellAttributes = NaiveCell;
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit Naive(const std::shared_ptr<OctreeImplementation<NaiveCell>>& octree)
        : ForwardAlgorithm(octree) { }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] CommunicationMap<SynapseCreationRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] PlasticDistantOutSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override;

private:
    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_position The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(const NeuronID& src_neuron_id, const position_type& axon_position, SignalType dendrite_type_needed);

    [[nodiscard]] static std::tuple<bool, bool> acceptance_criterion_test(
        const position_type& axon_position,
        const OctreeNode<NaiveCell>* node_with_dendrite,
        SignalType dendrite_type_needed);

    [[nodiscard]] static std::vector<OctreeNode<NaiveCell>*> get_nodes_for_interval(
        const position_type& axon_position,
        OctreeNode<NaiveCell>* root,
        SignalType dendrite_type_needed);
};
