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

#include "Config.h"
#include "Types.h"
#include "neurons/enums/SignalType.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <filesystem>
#include <functional>
#include <ostream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/transform.hpp>

/**
 * An object of type NetworkGraph stores the synaptic connections between neurons that are relevant for the current MPI rank.
 * The neurons are referred to by indices in the range [0, num_local_neurons).
 * The class does not perform any communication or synchronization with other MPI ranks when messing with edges.
 * NetworkGraph differentiates between local edges (from the current MPI rank to the current MPI rank) and
 * distant edges (another MPI rank is the owner of the target or source neuron).
 */
class NetworkGraph {
    template <typename synapse_weight>
    class NetworkGraphBase {
        friend class NetworkGraph;

    public:
        using number_neurons_type = RelearnTypes::number_neurons_type;

        using LocalEdges = std::vector<std::pair<NeuronID, synapse_weight>>;

        using NeuronLocalInNeighborhood = std::vector<LocalEdges>;
        using NeuronLocalOutNeighborhood = std::vector<LocalEdges>;

        using DistantEdges = std::vector<std::pair<RankNeuronId, synapse_weight>>;

        using NeuronDistantInNeighborhood = std::vector<DistantEdges>;
        using NeuronDistantOutNeighborhood = std::vector<DistantEdges>;

        NetworkGraphBase(const NetworkGraphBase& other) = delete;
        NetworkGraphBase(NetworkGraphBase&& other) = delete;

        NetworkGraphBase& operator=(const NetworkGraphBase& other) = delete;
        NetworkGraphBase& operator=(NetworkGraphBase&& other) = delete;

        /**
         * @brief Returns a constant reference to all distant in-edges to a neuron, i.e., a view on neurons that connect to the specified one via a synapse
         *      and belong to another MPI rank
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return A constant view of all distant in-edges.
         */
        [[nodiscard]] const DistantEdges& get_distant_in_edges(const NeuronID neuron_id) const {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < neuron_distant_in_neighborhood.size(),
                "NetworkGraph::NetworkGraphBase::get_distant_in_edges: Tried with a too large id of {}", neuron_id);

            return neuron_distant_in_neighborhood[local_neuron_id];
        }

        /**
         * @brief Returns a constant reference to all distant out-edges to a neuron, i.e., a view on all neurons that the specified one connects to via a synapse
         *      and belong to another MPI rank
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return A constant view of all distant out-edges.
         */
        [[nodiscard]] const DistantEdges& get_distant_out_edges(const NeuronID neuron_id) const {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < neuron_distant_out_neighborhood.size(),
                "NetworkGraph::NetworkGraphBase::get_distant_out_edges: Tried with a too large id of {}", neuron_id);

            return neuron_distant_out_neighborhood[local_neuron_id];
        }

        /**
         * @brief Returns a constant reference to all local in-edges to a neuron, i.e., a view on neurons that connect to the specified one via a synapse
         *      and belong to the current MPI rank
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return A constant view of all local in-edges.
         */
        [[nodiscard]] const LocalEdges& get_local_in_edges(const NeuronID neuron_id) const {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < neuron_local_in_neighborhood.size(),
                "NetworkGraph::NetworkGraphBase::get_local_in_edges: Tried with a too large id of {}", neuron_id);

            return neuron_local_in_neighborhood[local_neuron_id];
        }

        /**
         * @brief Returns a constant reference to all local out-edges to a neuron, i.e., a view on all neurons that the specified one connects to via a synapse
         *      and belong to the current MPI rank
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return A constant view of all local out-edges.
         */
        [[nodiscard]] const LocalEdges& get_local_out_edges(const NeuronID neuron_id) const {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < neuron_local_out_neighborhood.size(),
                "NetworkGraph::NetworkGraphBase::get_local_out_edges: Tried with a too large id of {}", neuron_id);

            return neuron_local_out_neighborhood[local_neuron_id];
        }

        /**
         * @brief Returns a constant reference to all local out-edges of all neurons on this mpi rank
         * @return Vector of edges. Edges from neuron with id i are at position i
         */
        [[nodiscard]] const NeuronLocalOutNeighborhood& get_all_local_out_edges() const {
            return neuron_local_out_neighborhood;
        }

        /**
         * @brief Returns a constant reference to all distant out-edges of all neurons on this mpi rank
         * @return Vector of edges. Edges from neuron with id i are at position i
         */
        [[nodiscard]] const NeuronDistantOutNeighborhood& get_all_distant_out_edges() const {
            return neuron_distant_out_neighborhood;
        }

        /**
         * @brief Returns a constant reference to all local in-edges of all neurons on this mpi rank
         * @return Vector of edges. Edges from neuron with id i are at position i
         */
        [[nodiscard]] const NeuronLocalInNeighborhood& get_all_local_in_edges() const {
            return neuron_local_in_neighborhood;
        }

        /**
         * @brief Returns a constant reference to all distant in-edges of all neurons on this mpi rank
         * @return Vector of edges. Edges from neuron with id i are at position i
         */
        [[nodiscard]] const NeuronDistantInNeighborhood& get_all_distant_in_edges() const {
            return neuron_distant_in_neighborhood;
        }

        /**
         * @brief Returns all identifiers of neurons which connect to the specified neuron, i.e.,
         *      <some_return_element> ---<signal_type>---> <neuron_id>
         * @param neuron_id The local neuron id
         * @param signal_type The type of synapse to search
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return A collection with all partnering neurons
         */
        [[nodiscard]] std::unordered_set<RankNeuronId> get_all_partners_incoming(const NeuronID neuron_id, const SignalType signal_type) const {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < number_local_neurons,
                "NetworkGraph::NetworkGraphBase::get_all_connecting_neurons: Tried with a too large id of {}", neuron_id);

            const auto& locals = neuron_local_in_neighborhood[local_neuron_id];
            const auto& distants = neuron_distant_in_neighborhood[local_neuron_id];

            const auto number_partners = locals.size() + distants.size();

            std::unordered_set<RankNeuronId> partners{};
            partners.reserve(number_partners * 2);

            for (const auto& [partner_id, weight] : locals) {
                if (signal_type == SignalType::Excitatory && weight < 0 || signal_type == SignalType::Inhibitory && weight > 0) {
                    continue;
                }

                partners.emplace(my_rank, partner_id);
            }

            for (const auto& [partner_id, weight] : distants) {
                if (signal_type == SignalType::Excitatory && weight < 0 || signal_type == SignalType::Inhibitory && weight > 0) {
                    continue;
                }

                partners.emplace(partner_id);
            }

            return partners;
        }

        /**
         * @brief Returns all identifiers of neurons to which the specified neuron connects, i.e.,
         *      <neuron_id> ------> <some_return_element>
         * @param neuron_id The local neuron id
         * @param signal_type The type of synapse to search
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return A collection with all partnering neurons
         */
        [[nodiscard]] std::unordered_set<RankNeuronId> get_all_partners_outgoing(const NeuronID neuron_id) const {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < number_local_neurons,
                "NetworkGraph::NetworkGraphBase::get_all_connecting_neurons: Tried with a too large id of {}", neuron_id);

            const auto& locals = neuron_local_out_neighborhood[local_neuron_id];
            const auto& distants = neuron_distant_out_neighborhood[local_neuron_id];

            const auto to_rank_neuron_id =
                [this](const auto partner_id) -> RankNeuronId {
                return { my_rank, partner_id };
            };

            return ranges::views::concat(
                       locals | ranges::views::keys | ranges::views::transform(to_rank_neuron_id),
                       distants | ranges::views::keys)
                | ranges::to<std::unordered_set>;
        }

        /**
         * @brief Returns the number of all in-edges to a neuron (counting multiplicities) from excitatory neurons
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return The number of incoming synapses that the specified neuron formed from excitatory neurons
         */
        [[nodiscard]] synapse_weight get_number_excitatory_in_edges(const NeuronID neuron_id) const {
            const auto& all_distant_edges = get_distant_in_edges(neuron_id);
            const auto& all_local_edges = get_local_in_edges(neuron_id);

            return ranges::accumulate(
                ranges::views::concat(all_distant_edges | ranges::views::values,
                    all_local_edges | ranges::views::values)
                    | ranges::views::filter(greater(0)),
                synapse_weight{ 0 });
        }

        /**
         * @brief Returns the number of all in-edges to a neuron (counting multiplicities) from inhibitory neurons
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return The number of incoming synapses that the specified neuron formed from inhibitory neurons
         */
        [[nodiscard]] synapse_weight get_number_inhibitory_in_edges(const NeuronID neuron_id) const {
            const auto& all_distant_edges = get_distant_in_edges(neuron_id);
            const auto& all_local_edges = get_local_in_edges(neuron_id);

            return ranges::accumulate(
                ranges::views::concat(all_distant_edges | ranges::views::values,
                    all_local_edges | ranges::views::values)
                    | ranges::views::filter(less(0)),
                synapse_weight{ 0 }, std::minus{});
        }

        /**
         * @brief Returns the number of all out-edges from a neuron (counting multiplicities)
         * @param neuron_id The id of the neuron
         * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
         * @return The number of outgoing synapses that the specified neuron formed
         */
        [[nodiscard]] synapse_weight get_number_out_edges(const NeuronID neuron_id) const {
            const auto& all_distant_edges = get_distant_out_edges(neuron_id);
            const auto& all_local_edges = get_local_out_edges(neuron_id);

            return ranges::accumulate(
                ranges::views::concat(all_distant_edges | ranges::views::values,
                    all_local_edges | ranges::views::values)
                    | ranges::views::transform(as_abs),
                synapse_weight{ 0 });
        }

        /**
         * @brief Returns directly if !Config::do_debug_checks
         *      Performs a debug check on the local portion of the network graph
         *      All stored ranks must be greater or equal to zero, no weight must be equal to zero,
         *      and all purely local edges must have a matching counterpart.
         * @exception Throws a RelearnException if any of the conditions is violated
         */
        void debug_check() const {
            if (!Config::do_debug_checks) {
                return;
            }

            struct NeuronIDPairHash {
            public:
                std::size_t operator()(const std::pair<NeuronID, NeuronID>& pair) const {
                    const std::hash<NeuronID> primitive_hash{};

                    const auto& [first_id, second_id] = pair;

                    const auto first_hash = primitive_hash(first_id);
                    const auto second_hash = primitive_hash(second_id);

                    // XOR might not be the best, but this is debug code
                    const auto combined_hash = first_hash ^ second_hash;
                    return combined_hash;
                }
            };

            for (const auto& distant_out_edges : NeuronID::range(number_local_neurons) | ranges::views::transform([this](const auto& neuron_id) { return get_distant_out_edges(neuron_id); })) {
                for (const auto& [target_id, edge_val] : distant_out_edges) {
                    const auto& [target_rank, target_neuron_id] = target_id;

                    RelearnException::check(edge_val != 0,
                        "NetworkGraph::debug_check: Distant synapse value is zero (out)");
                    RelearnException::check(target_rank.is_initialized(),
                        "NetworkGraph::debug_check: Distant synapse target rank is < 0");
                    RelearnException::check(target_rank != my_rank,
                        "NetworkGraph::debug_check: Distant synapse target rank is the local rank");
                }
            }

            for (const auto& distant_in_edges : NeuronID::range(number_local_neurons) | ranges::views::transform([this](const auto& neuron_id) { return get_distant_in_edges(neuron_id); })) {

                for (const auto& [source_id, edge_val] : distant_in_edges) {
                    const auto& [source_rank, source_neuron_id] = source_id;

                    RelearnException::check(edge_val != 0,
                        "NetworkGraph::debug_check: Distant synapse value is zero (out)");
                    RelearnException::check(source_rank.is_initialized(),
                        "NetworkGraph::debug_check: Distant synapse source rank is < 0");
                    RelearnException::check(source_rank != my_rank,
                        "NetworkGraph::debug_check: Distant synapse source rank is the local rank");
                }
            }

            // Golden map that stores all local edges
            std::unordered_map<std::pair<NeuronID, NeuronID>, synapse_weight, NeuronIDPairHash> edges{};
            edges.reserve(number_local_neurons);

            for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
                const auto& local_out_edges = get_local_out_edges(neuron_id);

                for (const auto& [target_neuron_id, edge_val] : local_out_edges) {
                    RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");
                    edges[std::make_pair(neuron_id, target_neuron_id)] = edge_val;
                }
            }

            for (const auto& id : NeuronID::range(number_local_neurons)) {
                const auto& local_in_edges = get_local_in_edges(id);

                for (const auto& [source_neuron_id, edge_val] : local_in_edges) {
                    RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");

                    const std::pair<NeuronID, NeuronID> id_pair(source_neuron_id, id);
                    const auto it = edges.find(id_pair);

                    const auto found = it != edges.cend();

                    RelearnException::check(found, "NetworkGraph::debug_check: Edge not found");

                    const auto golden_weight = it->second;
                    const auto weight_matches = golden_weight == edge_val;

                    RelearnException::check(weight_matches, "NetworkGraph::debug_check: Weight doesn't match");

                    edges.erase(id_pair);
                }
            }

            RelearnException::check(edges.empty(), "NetworkGraph::debug_check: Edges is not empty");
        }

        /**
         * @brief Returns the number of stored local neurons
         * @return The number of neurons
         */
        [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
            return number_local_neurons;
        }

    private:
        NetworkGraphBase() = default;

        /**
         * @brief Constructs an object and saves the given rank
         * @param mpi_rank The mpi rank that handles this portion of the graph, must be initialized
         * @exception Throws a RelearnException if mpi_rank is not initialized
         */
        NetworkGraphBase(const MPIRank mpi_rank)
            : my_rank(mpi_rank) {
            RelearnException::check(my_rank.is_initialized(), "NetworkGraph::NetworkGraphBase::NetworkGraphBase: The mpi rank must be initialized");
        }

        /**
         * @brief Initializes the base with the number of neurons
         * @param number_neurons The number of neurons, must be > 0
         * @exception Throws a RelearnException if already called or if number_neurons == 0
         */
        void init(const number_neurons_type number_neurons) {
            RelearnException::check(number_local_neurons == 0, "NetworkGraph::NetworkGraphBase::init: Was already initialized");
            RelearnException::check(number_neurons > 0, "NetworkGraph::NetworkGraphBase::init: Cannot initialize with 0 neurons");

            number_local_neurons = number_neurons;

            neuron_distant_in_neighborhood.resize(number_neurons);
            neuron_distant_out_neighborhood.resize(number_neurons);
            neuron_local_in_neighborhood.resize(number_neurons);
            neuron_local_out_neighborhood.resize(number_neurons);
        }

        /**
         * @brief Resizes the network graph by adding space for more neurons. Invalidates iterators
         * @param creation_count The number of additional neurons the network graph should handle, > 0
         * @exception Throws a RelearnException if creation_count == 0
         */
        void create_neurons(const number_neurons_type creation_count) {
            RelearnException::check(number_local_neurons > 0, "NetworkGraph::NetworkGraphBase::create_neurons: Was not initialized");
            RelearnException::check(creation_count > 0, "NetworkGraph::NetworkGraphBase::create_neurons: creation_count was 0");

            const auto old_size = number_local_neurons;
            const auto new_size = old_size + creation_count;

            neuron_distant_in_neighborhood.resize(new_size);
            neuron_distant_out_neighborhood.resize(new_size);

            neuron_local_in_neighborhood.resize(new_size);
            neuron_local_out_neighborhood.resize(new_size);

            number_local_neurons = new_size;
        }

        /**
         * @brief Adds a local synapse to the network graph
         * @param synapse The local synapse
         * @exception Throws a RelearnException if
         *      (a) The target is larger than the number neurons
         *      (b) The source is larger than the number neurons
         *      (c) The weight is equal to 0
         */
        void add_synapse(const Synapse<NeuronID, NeuronID, synapse_weight>& synapse) {
            const auto& [target, source, weight] = synapse;

            const auto local_target_id = target.get_neuron_id();
            const auto local_source_id = source.get_neuron_id();

            RelearnException::check(local_target_id < number_local_neurons,
                "NetworkGraph::NetworkGraphBase::add_synapse: Local synapse had a too large target: {} vs {}", target,
                number_local_neurons);
            RelearnException::check(local_source_id < number_local_neurons,
                "NetworkGraph::NetworkGraphBase::add_synapse: Local synapse had a too large source: {} vs {}", source,
                number_local_neurons);
            RelearnException::check(weight != 0, "NetworkGraph::NetworkGraphBase::add_synapse: Local synapse had weight 0");

            auto& in_edges = neuron_local_in_neighborhood[local_target_id];
            auto& out_edges = neuron_local_out_neighborhood[local_source_id];

            add_edge<decltype(in_edges), NeuronID>(in_edges, source, weight);
            add_edge<decltype(out_edges), NeuronID>(out_edges, target, weight);
        }

        /**
         * @brief Adds a distant in-synapse to the network graph (it might actually come from the same node, that's no problem)
         * @param synapse The distant in-synapse, must come from another rank
         * @exception Throws a RelearnException if
         *      (a) The target is larger than the number neurons
         *      (b) The weight is equal to 0
         *      (c) The distant rank is the same as the local one
         */
        void add_synapse(const Synapse<NeuronID, RankNeuronId, synapse_weight>& synapse) {
            const auto& [target, source_rni, weight] = synapse;
            const auto local_target_id = target.get_neuron_id();

            const auto& [source_rank, source_id] = source_rni;

            RelearnException::check(local_target_id < number_local_neurons,
                "NetworkGraph::NetworkGraphBase::add_synapse: Distant in-synapse had a too large target: {} vs {}",
                target, number_local_neurons);
            RelearnException::check(source_rank != my_rank,
                "NetworkGraph::NetworkGraphBase::add_synapse: Distant in-synapse was on my rank: {}", source_rank);
            RelearnException::check(weight != 0, "NetworkGraph::NetworkGraphBase::add_synapse: Local synapse had weight 0");

            auto& distant_in_edges = neuron_distant_in_neighborhood[local_target_id];
            add_edge<decltype(distant_in_edges), RankNeuronId>(distant_in_edges, source_rni, weight);
        }

        /**
         * @brief Adds a distant out-synapse to the network graph (it might actually come from the same node, that's no problem)
         * @param synapse The distant out-synapse, must come from another rank
         * @exception Throws a RelearnException if
         *      (a) The target rank is the same as the current rank
         *      (b) The weight is equal to 0
         *      (c) The distant rank is the same as the local one
         */
        void add_synapse(const Synapse<RankNeuronId, NeuronID, synapse_weight>& synapse) {
            const auto& [target_rni, source, weight] = synapse;
            const auto local_source_id = source.get_neuron_id();

            const auto& [target_rank, target_id] = target_rni;

            RelearnException::check(local_source_id < number_local_neurons,
                "NetworkGraph::NetworkGraphBase::add_synapse: Distant out-synapse had a too large target: {} vs {}",
                source, number_local_neurons);
            RelearnException::check(target_rank != my_rank,
                "NetworkGraph::NetworkGraphBase::add_synapse: Distant out-synapse was on my rank: {}", target_rank);
            RelearnException::check(weight != 0, "NetworkGraph::NetworkGraphBase::add_synapse: Local synapse had weight 0");

            auto& distant_out_edges = neuron_distant_out_neighborhood[local_source_id];
            add_edge<decltype(distant_out_edges), RankNeuronId>(distant_out_edges, target_rni, weight);
        }

        /**
         * @brief Adds all provided edges into the network graph at once.
         * @param local_edges All edges between two neurons on the current MPI rank
         * @param in_edges All edges that have a target on the current MPI rank and a source from another rank
         * @param out_edges All edges that have a source on the current MPI rank and a target from another rank
         */
        void add_edges(const std::vector<Synapse<NeuronID, NeuronID, synapse_weight>>& local_edges,
            const std::vector<Synapse<NeuronID, RankNeuronId, synapse_weight>>& in_edges,
            const std::vector<Synapse<RankNeuronId, NeuronID, synapse_weight>>& out_edges) {
            for (const auto& [target_id, source_id, weight] : local_edges) {
                const auto local_target_id = target_id.get_neuron_id();
                const auto local_source_id = source_id.get_neuron_id();

                RelearnException::check(local_target_id < neuron_local_in_neighborhood.size(),
                    "NetworkGraph::add_edges: local_in_neighborhood is too small: {} vs {}", target_id,
                    neuron_local_in_neighborhood.size());
                RelearnException::check(local_source_id < neuron_local_out_neighborhood.size(),
                    "NetworkGraph::add_edges: local_out_neighborhood is too small: {} vs {}", source_id,
                    neuron_distant_out_neighborhood.size());

                auto& in_edges = neuron_local_in_neighborhood[local_target_id];
                auto& out_edges = neuron_local_out_neighborhood[local_source_id];

                add_edge<decltype(in_edges), NeuronID>(in_edges, source_id, weight);
                add_edge<decltype(out_edges), NeuronID>(out_edges, target_id, weight);
            }

            for (const auto& [target_id, source_rni, weight] : in_edges) {
                const auto local_target_id = target_id.get_neuron_id();

                RelearnException::check(local_target_id < neuron_distant_in_neighborhood.size(),
                    "NetworkGraph::add_edges: distant_in_neighborhood is too small: {} vs {}",
                    target_id, neuron_distant_in_neighborhood.size());

                auto& distant_in_edges = neuron_distant_in_neighborhood[local_target_id];
                add_edge<decltype(distant_in_edges), RankNeuronId>(distant_in_edges, source_rni, weight);
            }

            for (const auto& [target_rni, source_id, weight] : out_edges) {
                const auto local_source_id = source_id.get_neuron_id();

                RelearnException::check(local_source_id < neuron_distant_out_neighborhood.size(),
                    "NetworkGraph::add_edges: distant_out_neighborhood is too small: {} vs {}",
                    source_id, neuron_distant_out_neighborhood.size());

                auto& distant_out_edges = neuron_distant_out_neighborhood[local_source_id];
                add_edge<decltype(distant_out_edges), RankNeuronId>(distant_out_edges, target_rni, weight);
            }
        }

        template <typename Edges, typename NeuronId>
        // NOLINTNEXTLINE
        static void add_edge(Edges& edges, const NeuronId& other_neuron_id, const synapse_weight weight) {
            size_t idx = 0;

            for (auto& [neuron_id, edge_weight] : edges) {
                if (neuron_id == other_neuron_id) {
                    const auto new_edge_weight = edge_weight + weight;
                    edge_weight = new_edge_weight;

                    if (new_edge_weight == 0) {
                        const auto idx_last = edges.size() - 1;
                        std::swap(edges[idx], edges[idx_last]);
                        edges.erase(edges.cend() - 1);
                    }

                    return;
                }

                idx++;
            }

            edges.emplace_back(other_neuron_id, weight);
        }

        NeuronDistantInNeighborhood neuron_distant_in_neighborhood{};
        NeuronDistantOutNeighborhood neuron_distant_out_neighborhood{};

        NeuronLocalInNeighborhood neuron_local_in_neighborhood{};
        NeuronLocalOutNeighborhood neuron_local_out_neighborhood{};

        number_neurons_type number_local_neurons{ 0 };
        MPIRank my_rank{ MPIRank::uninitialized_rank() };
    };

public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    using plastic_synapse_weight = RelearnTypes::plastic_synapse_weight;
    using static_synapse_weight = RelearnTypes::static_synapse_weight;

    /**
     * @brief Constructs an network graph for the MPI rank
     * @param mpi_rank The mpi rank that handles this portion of the graph, must be initialized
     * @exception Throws a RelearnException if mpi_rank is not initialized
     */
    NetworkGraph(const MPIRank mpi_rank)
        : plastic_network_graph(mpi_rank)
        , static_network_graph(mpi_rank) { }

    /**
     * @brief Initializes the network graph with the number of neurons
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if already called or if number_neurons == 0
     */
    void init(const number_neurons_type number_neurons) {
        plastic_network_graph.init(number_neurons);
        static_network_graph.init(number_neurons);
    }

    /**
     * @brief Resizes the network graph by adding space for more neurons. Invalidates iterators
     * @param creation_count The number of additional neurons the network graph should handle, > 0
     * @exception Throws a RelearnException if creation_count == 0
     */
    void create_neurons(const number_neurons_type creation_count) {
        plastic_network_graph.create_neurons(creation_count);
        static_network_graph.create_neurons(creation_count);
    }

    /**
     * @brief Returns the number of stored local neurons
     * @return The number of neurons
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        // plastic and static bases habe the same number
        return plastic_network_graph.get_number_neurons();
    }

    /**
     * @brief Returns constant references to all incoming edges to the specified neuron from other MPI ranks
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_distant_in_edges(const NeuronID neuron_id) const {
        return std::tie(plastic_network_graph.get_distant_in_edges(neuron_id), static_network_graph.get_distant_in_edges(neuron_id));
    }

    /**
     * @brief Returns constant references to all outgoing edges from the specified neuron to other MPI ranks
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_distant_out_edges(const NeuronID neuron_id) const {
        return std::tie(plastic_network_graph.get_distant_out_edges(neuron_id), static_network_graph.get_distant_out_edges(neuron_id));
    }

    /**
     * @brief Returns constant references to all incoming edges to the specified neuron from the current MPI rank
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_local_in_edges(const NeuronID neuron_id) const {
        return std::tie(plastic_network_graph.get_local_in_edges(neuron_id), static_network_graph.get_local_in_edges(neuron_id));
    }

    /**
     * @brief Returns constant references to all outgoing edges from the specified neuron to the current MPI rank
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_local_out_edges(const NeuronID neuron_id) const {
        return std::tie(plastic_network_graph.get_local_out_edges(neuron_id), static_network_graph.get_local_out_edges(neuron_id));
    }

    /**
     * @brief Returns constant references to all outgoing edges to the current MPI rank
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_all_local_out_edges() const {
        return std::tie(plastic_network_graph.get_all_local_out_edges(), static_network_graph.get_all_local_out_edges());
    }

    /**
     * @brief Returns constant references to all outgoing edges to other MPI ranks
     * @param neuron_id The local neuron id
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_all_distant_out_edges() const {
        return std::tie(plastic_network_graph.get_all_distant_out_edges(), static_network_graph.get_all_distant_out_edges());
    }

    /**
     * @brief Returns constant references to all outgoing edges from the current MPI rank
     * @param neuron_id The local neuron id
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_all_local_in_edges() const {
        return std::tie(plastic_network_graph.get_all_local_in_edges(), static_network_graph.get_all_local_in_edges());
    }

    /**
     * @brief Returns constant references to all outgoing edges from other MPI ranks
     * @param neuron_id The local neuron id
     * @return A tuple of (1) the plastic edges and (2) the static edges
     */
    [[nodiscard]] auto get_all_distant_in_edges() const {
        return std::tie(plastic_network_graph.get_all_distant_in_edges(), static_network_graph.get_all_distant_in_edges());
    }

    /**
     * @brief Returns the number of excitatory incoming edges to the specified neuron.
     *      Sums all weights (does not count the number of distinct partner neurons)
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the number of plastic edges and (2) the number of static edges
     */
    [[nodiscard]] auto get_number_excitatory_in_edges(const NeuronID neuron_id) const {
        return std::make_tuple(plastic_network_graph.get_number_excitatory_in_edges(neuron_id), static_network_graph.get_number_excitatory_in_edges(neuron_id));
    }

    /**
     * @brief Returns the number of inhibitory incoming edges to the specified neuron.
     *      Sums all weights (does not count the number of distinct partner neurons)
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the number of plastic edges and (2) the number of static edges
     */
    [[nodiscard]] auto get_number_inhibitory_in_edges(const NeuronID neuron_id) const {
        return std::make_tuple(plastic_network_graph.get_number_inhibitory_in_edges(neuron_id), static_network_graph.get_number_inhibitory_in_edges(neuron_id));
    }

    /**
     * @brief Returns the number of outgoing edges from the specified neuron.
     *      Sums all weights (does not count the number of distinct partner neurons)
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A tuple of (1) the number of plastic edges and (2) the number of static edges
     */
    [[nodiscard]] auto get_number_out_edges(const NeuronID neuron_id) const {
        return std::make_tuple(plastic_network_graph.get_number_out_edges(neuron_id), static_network_graph.get_number_out_edges(neuron_id));
    }

    /**
     * @brief Returns all identifiers of neurons which connect to the specified neuron via plastic edges, i.e.,
     *      <some_return_element> ---<signal_type>---> <neuron_id>
     * @param neuron_id The local neuron id
     * @param signal_type The type of synapse to search
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A collection with all partnering neurons
     */
    [[nodiscard]] std::unordered_set<RankNeuronId> get_all_plastic_partners_incoming(const NeuronID neuron_id, const SignalType signal_type) const {
        return plastic_network_graph.get_all_partners_incoming(neuron_id, signal_type);
    }

    /**
     * @brief Returns all identifiers of neurons to which the specified neuron connects via plastic edges, i.e.,
     *      <neuron_id> ------> <some_return_element>
     * @param neuron_id The local neuron id
     * @param signal_type The type of synapse to search
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A collection with all partnering neurons
     */
    [[nodiscard]] std::unordered_set<RankNeuronId> get_all_plastic_partners_outgoing(const NeuronID neuron_id) const {
        return plastic_network_graph.get_all_partners_outgoing(neuron_id);
    }

    /**
     * @brief Adds a local synapse to the network graph
     * @param synapse The local synapse
     * @exception Throws a RelearnException if
     *      (a) The target is larger than the number neurons
     *      (b) The source is larger than the number neurons
     *      (c) The weight is equal to 0
     */
    void add_synapse(const PlasticLocalSynapse& synapse) {
        plastic_network_graph.add_synapse(synapse);
    }

    /**
     * @brief Adds a distant in-synapse to the network graph (it might actually come from the same node, that's no problem)
     * @param synapse The distant in-synapse, must come from another rank
     * @exception Throws a RelearnException if
     *      (a) The target is larger than the number neurons
     *      (b) The weight is equal to 0
     *      (c) The distant rank is the same as the local one
     */
    void add_synapse(const PlasticDistantInSynapse& synapse) {
        plastic_network_graph.add_synapse(synapse);
    }

    /**
     * @brief Adds a distant out-synapse to the network graph (it might actually come from the same node, that's no problem)
     * @param synapse The distant out-synapse, must come from another rank
     * @exception Throws a RelearnException if
     *      (a) The target rank is the same as the current rank
     *      (b) The weight is equal to 0
     *      (c) The distant rank is the same as the local one
     */
    void add_synapse(const PlasticDistantOutSynapse& synapse) {
        plastic_network_graph.add_synapse(synapse);
    }

    /**
     * @brief Adds a local synapse to the network graph
     * @param synapse The local synapse
     * @exception Throws a RelearnException if
     *      (a) The target is larger than the number neurons
     *      (b) The source is larger than the number neurons
     *      (c) The weight is equal to 0
     */
    void add_synapse(const StaticLocalSynapse& synapse) {
        static_network_graph.add_synapse(synapse);
    }

    /**
     * @brief Adds a distant in-synapse to the network graph (it might actually come from the same node, that's no problem)
     * @param synapse The distant in-synapse, must come from another rank
     * @exception Throws a RelearnException if
     *      (a) The target is larger than the number neurons
     *      (b) The weight is equal to 0
     *      (c) The distant rank is the same as the local one
     */
    void add_synapse(const StaticDistantInSynapse& synapse) {
        static_network_graph.add_synapse(synapse);
    }

    /**
     * @brief Adds a distant out-synapse to the network graph (it might actually come from the same node, that's no problem)
     * @param synapse The distant out-synapse, must come from another rank
     * @exception Throws a RelearnException if
     *      (a) The target rank is the same as the current rank
     *      (b) The weight is equal to 0
     *      (c) The distant rank is the same as the local one
     */
    void add_synapse(const StaticDistantOutSynapse& synapse) {
        static_network_graph.add_synapse(synapse);
    }

    /**
     * @brief Adds all provided edges into the network graph at once.
     * @param local_edges All edges between two neurons on the current MPI rank
     * @param in_edges All edges that have a target on the current MPI rank and a source from another rank
     * @param out_edges All edges that have a source on the current MPI rank and a target from another rank
     */
    void add_edges(const PlasticLocalSynapses& local_edges, const PlasticDistantInSynapses& in_edges,
        const PlasticDistantOutSynapses& out_edges) {
        plastic_network_graph.add_edges(local_edges, in_edges, out_edges);
    }

    /**
     * @brief Adds all provided edges into the network graph at once.
     * @param local_edges All edges between two neurons on the current MPI rank
     * @param in_edges All edges that have a target on the current MPI rank and a source from another rank
     * @param out_edges All edges that have a source on the current MPI rank and a target from another rank
     */
    void add_edges(const StaticLocalSynapses& local_edges, const StaticDistantInSynapses& in_edges,
        const StaticDistantOutSynapses& out_edges) {
        static_network_graph.add_edges(local_edges, in_edges, out_edges);
    }

    /**
     * @brief Returns directly if !Config::do_debug_checks
     *      Performs a debug check on the local portion of the network graph
     *      All stored ranks must be greater or equal to zero, no weight must be equal to zero,
     *      and all purely local edges must have a matching counterpart.
     * @exception Throws a RelearnException if any of the conditions is violated
     */
    void debug_check() const {
        plastic_network_graph.debug_check();
        static_network_graph.debug_check();
    }

    NetworkGraphBase<plastic_synapse_weight> plastic_network_graph{};
    NetworkGraphBase<static_synapse_weight> static_network_graph{};
};
