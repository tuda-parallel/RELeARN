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
#include "mpi/MPIWrapper.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/SemiStableVector.h"

#include "fmt/ostream.h"

#include <array>
#include <map>
#include <type_traits>
#include <utility>

class NodeCacheAdapter;

enum class NodeCacheType : char {
    Combined,
    Separate,
};

/**
 * @brief Pretty-prints the node cache type to the chosen stream
 * @param out The stream to which to print the node cache type
 * @param cache_type The node cache type to print
 * @return The argument out, now altered with the node cache type
 */
inline std::ostream& operator<<(std::ostream& out, const NodeCacheType cache_type) {
    switch (cache_type) {
    case NodeCacheType::Combined:
        return out << "Combined";
    case NodeCacheType::Separate:
        return out << "Separate";
    }

    return out << "UNKNOWN";
}

template <>
struct fmt::formatter<NodeCacheType> : ostream_formatter { };

/**
 * DO NOT USE THIS CLASS DIRECTLY. USE NodeCache and set NodeCacheType::Combined.
 * This class caches octree nodes from other MPI ranks on the local MPI rank.
 * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
 */
template <typename AdditionalCellAttributes>
class NodeCacheCombined {
    using node_type = OctreeNode<AdditionalCellAttributes>;
    using children_type = std::array<node_type*, Constants::number_oct>;

public:
    using array_type = std::array<node_type, Constants::number_oct>;
    using NodesCacheKey = std::pair<MPIRank, node_type*>;
    using NodesCacheValue = children_type;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;

    /**
     * @brief Empties the cache that was built during the connection phase and frees all local copies
     */
    static void clear() {
        remote_nodes_cache.clear();
        memory.clear();
    }

    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded, must be virtual
     * @exception Throws a RelearnException if node is on the current MPI process or if the saved neuron_id is not virtual
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] static std::array<node_type*, Constants::number_oct> download_children(node_type* const node) {
        const auto target_rank = node->get_mpi_rank();
        RelearnException::check(node->get_cell_neuron_id().is_virtual(), "NodeCache::download_children: Tried to download from a non-virtual node");
        RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "NodeCache::download_children: Tried to download a local node");

        auto actual_download = [target_rank](node_type* const node) {
            children_type local_children{ nullptr };
            NodesCacheKey rank_address_pair{ target_rank, node };

            const auto& [iterator, inserted] = remote_nodes_cache.insert({ rank_address_pair, local_children });

            if (!inserted) {
                return iterator->second;
            }

            array_type& ref = memory.emplace_back();
            node_type* where_to_insert = ref.data();

            auto offset = node->get_cell_neuron_id().get_rma_offset();

            int first_valid_child_index = -1;
            for (auto child_index = 0; child_index < Constants::number_oct; child_index++) {
                if (node->get_child(child_index) != nullptr) {
                    first_valid_child_index = child_index;
                    break;
                }
            }
            RelearnException::check(first_valid_child_index >= 0, "NodeCache::download_children: No children found for node: {}", (void*)node);

            // Start access epoch to remote rank
            MPIWrapper::download_octree_node(where_to_insert, target_rank, offset, Constants::number_oct);

            // Retry download if something went wrong
            int retries = Constants::number_rma_download_retries;
            while (!ref[first_valid_child_index].get_mpi_rank().is_initialized() || ref[first_valid_child_index].get_mpi_rank() != target_rank) {
                retries--;
                if (retries > 0) {
                    LogFiles::print_message_rank(MPIWrapper::get_my_rank(), "Download of mpi node children from {} {} to {} {} is corrupted. Retries left: {}", target_rank, offset, MPIWrapper::get_my_rank(), (void*)where_to_insert, retries);
                    MPIWrapper::download_octree_node(where_to_insert, target_rank, offset, Constants::number_oct);
                } else {
                    RelearnException::fail("Download of mpi node children from {} {} to {} {} is corrupted. No retries left.", target_rank, offset, MPIWrapper::get_my_rank(), (void*)where_to_insert);
                }
            }

            for (auto child_index = 0; child_index < Constants::number_oct; child_index++) {
                if (node->get_child(child_index) == nullptr) {
                    local_children[child_index] = nullptr;
                    continue;
                }

                local_children[child_index] = &(ref[child_index]);
            }

            iterator->second = local_children;

            return local_children;
        };

        children_type local_children{ nullptr };

#pragma omp critical(node_cache_download)
        local_children = actual_download(node);

        return local_children;
    }

    /**
     * @brief Returns the currently used memory
     * @return The currently used memory
     */
    [[nodiscard]] static std::size_t get_memory_size() noexcept {
        return memory.size();
    }

    /**
     * @brief Returns the current number of cached values
     * @return The current number of cached values
     */
    [[nodiscard]] static std::size_t get_cache_size() noexcept {
        return remote_nodes_cache.size();
    }

private:
    static inline SemiStableVector<array_type> memory{}; // NOLINT
    static inline NodesCache remote_nodes_cache{};
};

/**
 * DO NOT USE THIS CLASS DIRECTLY. USE NodeCache and set NodeCacheType::Separate.
 * This class caches octree nodes from other MPI ranks on the local MPI rank.
 * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
 */
template <typename AdditionalCellAttributes>
class NodeCacheSeparate {
    using node_type = OctreeNode<AdditionalCellAttributes>;
    using children_type = std::array<node_type*, Constants::number_oct>;

public:
    using array_type = std::array<node_type, Constants::number_oct>;
    using NodesCacheKey = std::pair<MPIRank, node_type*>;
    using NodesCacheValue = node_type*;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;

    /**
     * @brief Empties the cache that was built during the connection phase and frees all local copies
     */
    static void clear() {
        remote_nodes_cache.clear();
        memory.clear();
    }

    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded, must be virtual
     * @exception Throws a RelearnException if node is on the current MPI process or if the saved neuron_id is not virtual
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] static std::array<node_type*, Constants::number_oct> download_children(node_type* const node) {
        const auto target_rank = node->get_mpi_rank();
        RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "NodeCache::download_children: Tried to download a local node");

        auto actual_download = [target_rank](node_type* node) {
            std::array<node_type*, Constants::number_oct> local_children{ nullptr };

            // Fetch remote children if they exist
            for (auto child_index = 0; child_index < Constants::number_oct; child_index++) {
                node_type* unusable_child_pointer_on_other_rank = node->get_child(child_index);
                if (nullptr == unusable_child_pointer_on_other_rank) {
                    // NOLINTNEXTLINE
                    local_children[child_index] = nullptr;
                    continue;
                }

                NodesCacheKey rank_addr_pair{ target_rank, unusable_child_pointer_on_other_rank };
                std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair{ rank_addr_pair, nullptr };

                // Get cache entry for "cache_key_val_pair"
                // It is created if it does not exist yet
                const auto& [iterator, inserted] = remote_nodes_cache.insert(cache_key_val_pair);

                // Cache entry just inserted as it was not in cache
                // So, we still need to init the entry by fetching
                // from the target rank
                if (inserted) {
                    node_type& ref = memory.emplace_back();
                    iterator->second = &ref;
                    node_type* local_child_addr = iterator->second;

                    MPIWrapper::download_octree_node<AdditionalCellAttributes>(local_child_addr, target_rank, unusable_child_pointer_on_other_rank, 1);
                }

                // Remember address of node
                // NOLINTNEXTLINE
                local_children[child_index] = iterator->second;
            }

            return local_children;
        };

        std::array<node_type*, Constants::number_oct> local_children{ nullptr };

#pragma omp critical(node_cache_download)
        local_children = actual_download(node);

        return local_children;
    }

    /**
     * @brief Returns the currently used memory
     * @return The currently used memory
     */
    [[nodiscard]] static std::size_t get_memory_size() noexcept {
        return memory.size();
    }

    /**
     * @brief Returns the current number of cached values
     * @return The current number of cached values
     */
    [[nodiscard]] static std::size_t get_cache_size() noexcept {
        return remote_nodes_cache.size();
    }

private:
    static inline SemiStableVector<node_type> memory{}; // NOLINT
    static inline NodesCache remote_nodes_cache{};
};

/**
 * This class acts as interface to different cache implementations.
 * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
 */
template <typename AdditionalCellAttributes>
class NodeCache {
    using node_type = OctreeNode<AdditionalCellAttributes>;

public:
    /**
     * @brief Sets the type of cache that shall be used
     * @param cache_type The cache type that from now on shall be used
     */
    static void set_cache_type(NodeCacheType cache_type) noexcept {
        currently_used_cache = cache_type;
    }

    /**
     * @brief Returns the currently used cache type
     * @return The currently used cache type
     */
    [[nodiscard]] NodeCacheType get_cache_type() noexcept {
        return currently_used_cache;
    }

    /**
     * @brief Empties the cache that was built during the connection phase and frees all local copies
     */
    static void clear() {
        switch (currently_used_cache) {
        case NodeCacheType::Combined:
            NodeCacheCombined<AdditionalCellAttributes>::clear();
            return;
        case NodeCacheType::Separate:
            NodeCacheSeparate<AdditionalCellAttributes>::clear();
            return;
        }

        RelearnException::fail("NodeCache::clear: {} is an unknown cache type!", currently_used_cache);
    }

    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded, must be virtual
     * @exception Throws a RelearnException if node is on the current MPI process or if the saved neuron_id is not virtual
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] static std::array<node_type*, Constants::number_oct> download_children(node_type* const node) {
        if (is_already_downloaded) {
            return node->get_children();
        }

        switch (currently_used_cache) {
        case NodeCacheType::Combined:
            return NodeCacheCombined<AdditionalCellAttributes>::download_children(node);
        case NodeCacheType::Separate:
            return NodeCacheSeparate<AdditionalCellAttributes>::download_children(node);
        }

        RelearnException::fail("NodeCache::download_children: {} is an unknown cache type!", currently_used_cache);

        return {};
    }

    /**
     * @brief Returns the children of the node. Downloads them from another MPI rank if necessary
     * @param node The node, must not be nullptr and not a leaf
     * @exception Throws a RelearnException if node is nullptr or node is a leaf
     * @return The children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] static std::array<node_type*, Constants::number_oct> get_children(node_type* const node) {
        RelearnException::check(node != nullptr, "NodeCache::get_children: node is nullptr");
        RelearnException::check(node->is_parent(), "NodeCache::get_children: node is a leaf");

        if (node->is_local()) {
            return node->get_children();
        }

        return download_children(node);
    }

    /**
     * @brief Returns the currently used memory
     * @return The currently used memory
     */
    [[nodiscard]] static std::size_t get_memory_size() noexcept {
        switch (currently_used_cache) {
        case NodeCacheType::Combined:
            return NodeCacheCombined<AdditionalCellAttributes>::get_memory_size();
        case NodeCacheType::Separate:
            return NodeCacheSeparate<AdditionalCellAttributes>::get_memory_size();
        }

        return 0;
    }

    /**
     * @brief Returns the current number of cached values
     * @return The current number of cached values
     */
    [[nodiscard]] static std::size_t get_cache_size() noexcept {
        switch (currently_used_cache) {
        case NodeCacheType::Combined:
            return NodeCacheCombined<AdditionalCellAttributes>::get_cache_size();
        case NodeCacheType::Separate:
            return NodeCacheSeparate<AdditionalCellAttributes>::get_cache_size();
        }

        return 0;
    }

private:
    friend class NodeCacheAdapter;

    static inline NodeCacheType currently_used_cache{ NodeCacheType::Combined };
    static inline bool is_already_downloaded{ false }; // For tests
};
