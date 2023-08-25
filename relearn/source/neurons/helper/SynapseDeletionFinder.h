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

#include "Types.h"
#include "mpi/CommunicationMap.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <bitset>
#include <memory>
#include <vector>

class NetworkGraph;
class SynapticElements;

/**
 * This enums lists all types of synapse deletion finders
 */
enum class SynapseDeletionFinderType : char {
    Random
};

/**
 * @brief Pretty-prints the synapse deletion finder type to the chosen stream
 * @param out The stream to which to print the synapse deletion finder
 * @param element_type The synapse deletion finder to print
 * @return The argument out, now altered with the synapse deletion finder
 */
inline std::ostream& operator<<(std::ostream& out, const SynapseDeletionFinderType& synapse_deletion_type) {
    if (synapse_deletion_type == SynapseDeletionFinderType::Random) {
        return out << "Random";
    }
    return out;
}

template <>
struct fmt::formatter<SynapseDeletionFinderType> : ostream_formatter { };

/**
 * This class encapsulates the logic of finding and deleting synapses
 * based on the synaptic elements. It provides the communication via MPI
 * and other house keeping, as well as a virtual method to implement.
 */
class SynapseDeletionFinder {
public:
    virtual ~SynapseDeletionFinder() = default;

    /**
     * @brief Sets the network graph that stores the synapses
     * @param ng The new network graph, must not be empty
     * @exception Throws a RelearnException if ng is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> ng) {
        const auto full = ng.operator bool();
        RelearnException::check(full, "SynapseDeletionFinder::set_network_graph: The network graph is empty");

        network_graph = std::move(ng);
    }

    /**
     * @brief Sets the axons
     * @param se The axons, must not be empty
     * @exception Throws a RelearnException if se is empty
     */
    void set_axons(std::shared_ptr<SynapticElements> se) {
        const auto full = se.operator bool();
        RelearnException::check(full, "SynapseDeletionFinder::set_axons: The synaptic elements is empty");

        axons = std::move(se);
    }

    /**
     * @brief Sets the excitatory dendrites
     * @param se The excitatory dendrites, must not be empty
     * @exception Throws a RelearnException if se is empty
     */
    void set_dendrites_ex(std::shared_ptr<SynapticElements> se) {
        const auto full = se.operator bool();
        RelearnException::check(full, "SynapseDeletionFinder::set_dendrites_ex: The synaptic elements is empty");

        excitatory_dendrites = std::move(se);
    }

    /**
     * @brief Sets the inhibitory dendrites
     * @param se The inhibitory dendrites, must not be empty
     * @exception Throws a RelearnException if se is empty
     */
    void set_dendrites_in(std::shared_ptr<SynapticElements> se) {
        const auto full = se.operator bool();
        RelearnException::check(full, "SynapseDeletionFinder::set_dendrites_in: The synaptic elements is empty");

        inhibitory_dendrites = std::move(se);
    }

    /**
     * @brief Sets the extra information
     * @param new_extra_info The extra information, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto full = new_extra_info.operator bool();
        RelearnException::check(full, "SynapseDeletionFinder::set_extra_infos: new_extra_info is empty");

        extra_info = std::move(new_extra_info);
    }

    /**
     * @brief Commits the updates for the synaptic elements, deletes synapses in the network graph,
     *      exchanges the deletions between MPI ranks, and commits the deletions from other ranks as well
     * @return The number of deleted synapses that are initiated by (1) the local axons and (2) the local dendrites
     */
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> delete_synapses();

protected:
    [[nodiscard]] virtual CommunicationMap<SynapseDeletionRequest> find_synapses_to_delete(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete);

    [[nodiscard]] virtual std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) = 0;

    [[nodiscard]] std::uint64_t commit_deletions(const CommunicationMap<SynapseDeletionRequest>& deletions, MPIRank my_rank);

    [[nodiscard]] std::vector<RankNeuronId> register_synapses(NeuronID neuron_id, ElementType element_type, SignalType signal_type);

    std::shared_ptr<SynapticElements> axons{};
    std::shared_ptr<SynapticElements> excitatory_dendrites{};
    std::shared_ptr<SynapticElements> inhibitory_dendrites{};

    std::shared_ptr<NetworkGraph> network_graph{};
    std::shared_ptr<NeuronsExtraInfo> extra_info{};
};

/**
 * This class deletes synapses based on randomness, i.e., it picks the
 * synapses to delete uniformely at random.
 */
class RandomSynapseDeletionFinder : public SynapseDeletionFinder {
protected:
    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) override;
};
