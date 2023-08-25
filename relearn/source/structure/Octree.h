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
#include "mpi/MPIWrapper.h"
#include "structure/OctreeNode.h"
#include "structure/OctreeNodeHelper.h"
#include "structure/SpaceFillingCurve.h"
#include "util/RelearnException.h"
#include "util/Stack.h"
#include "util/Timers.h"
#include "util/Vec3.h"

#include <climits>
#include <cstdint>
#include <functional>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

#include <range/v3/functional/indirect.hpp>
#include <range/v3/functional/not_fn.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/filter.hpp>

class Partition;

/**
 * This type represents the interface of the (spatial) Octree.
 */
class Octree {
public:
    using box_size_type = RelearnTypes::box_size_type;

    /**
     * @brief Constructs a new Octree with the the given size and constructs the "internal" part up to and including the level_of_branch_nodes
     * @param min The simulation box lower boundaries
     * @param max The simulation box upper boundaries
     * @param level_of_branch_nodes The level at which the branch nodes (that are exchanged via MPI) are
     * @exception Throws a RelearnException if min is not componentwise smaller than max
     */
    Octree(const box_size_type& min, const box_size_type& max, const std::uint16_t level_of_branch_nodes)
        : level_of_branch_nodes(level_of_branch_nodes) {

        const auto& [min_x, min_y, min_z] = min;
        const auto& [max_x, max_y, max_z] = max;

        RelearnException::check(min_x <= max_x, "Octree::Octree: The x component of the simulation box minimum was larger than that of the maximum");
        RelearnException::check(min_y <= max_y, "Octree::Octree: The y component of the simulation box minimum was larger than that of the maximum");
        RelearnException::check(min_z <= max_z, "Octree::Octree: The z component of the simulation box minimum was larger than that of the maximum");

        simulation_box_minimum = min;
        simulation_box_maximum = max;
    }

    virtual ~Octree() = default;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = delete;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = delete;

    /**
     * @brief Returns the minimum position in the Octree
     * @return The minimum position in the Octree
     */
    [[nodiscard]] const box_size_type& get_simulation_box_minimum() const noexcept {
        return simulation_box_minimum;
    }

    /**
     * @brief Returns the maximum position in the Octree
     * @return The maximum position in the Octree
     */
    [[nodiscard]] const box_size_type& get_simulation_box_maximum() const noexcept {
        return simulation_box_maximum;
    }

    /**
     * @brief Returns the level at which the branch nodes (that are exchanged via MPI) are
     * @return The level at which the branch nodes (that are exchanged via MPI) are
     */
    [[nodiscard]] std::uint16_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
    }

    /**
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] virtual size_t get_num_local_trees() const noexcept = 0;

    /**
     * @brief Inserts a neuron with the specified id and the specified position into the octree.
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron, < Constants::uninitialized (only use for actual neurons, virtual neurons are inserted automatically)
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the octree's boundaries
     *      (b) neuron_id >= Constants::uninitialized
     *      (c) Allocating a new object in the shared memory window fails
     *      (d) Something went wrong within the insertion
     * @return A pointer to the newly created and inserted node
     */
    virtual void insert(const box_size_type& position, const NeuronID& neuron_id) = 0;

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    virtual void initializes_leaf_nodes(RelearnTypes::number_neurons_type num_neurons) = 0;

    /**
     * Print a visualization of this tree to a file
     * @param file_path The file where the visualization will be stored
     */
    void print_to_file(const std::filesystem::path& file_path) const {
        std::ofstream out_stream{ file_path };
        RelearnException::check(out_stream.good() && !out_stream.bad(), "Octree::print_to_file: Unable to open stream for {}", file_path.string());
        std::stringstream ss;
        print(ss);
        out_stream << ss.rdbuf();
        out_stream.flush();
        out_stream.close();
    }

protected:
    virtual void print(std::stringstream& ss) const = 0;

private:
    box_size_type simulation_box_minimum{ 0 };
    box_size_type simulation_box_maximum{ 0 };

    std::uint16_t level_of_branch_nodes{ std::numeric_limits<std::uint16_t>::max() };
};

/**
 * This type represents the (spatial) Octree in which the neurons are organised.
 * It offers general information about the structure, the functionality to insert new neurons,
 * update from the bottom up, and synchronize parts with MPI.
 * It is templated by the additional cell attributes that the algorithm will need the cell to have.
 */
template <typename AdditionalCellAttributes>
class OctreeImplementation : public Octree {
public:
    /**
     * @brief Constructs a new OctreeImplementation with the the given size and constructs the "internal" part up to and including the level_of_branch_nodes
     * @param min The simulation box lower boundaries
     * @param max The simulation box upper boundaries
     * @param level_of_branch_nodes The level at which the branch nodes (that are exchanged via MPI) are
     * @exception Throws a RelearnException if min is not componentwise smaller than max
     */
    OctreeImplementation(const box_size_type& min, const box_size_type& max, const std::uint16_t level_of_branch_nodes)
        : Octree(min, max, level_of_branch_nodes) {

        const auto num_local_trees = 1ULL << (3U * level_of_branch_nodes);
        branch_nodes.resize(num_local_trees, nullptr);

        construct_global_tree_part();
    }

    /**
     * @brief Returns the root of the Octree
     * @return The root of the Octree. Ownership is not transferred
     */
    [[nodiscard]] const OctreeNode<AdditionalCellAttributes>* get_root() const noexcept {
        return &root;
    }

    /**
     * @brief Returns the root of the Octree
     * @return The root of the Octree. Ownership is not transferred
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_root() noexcept {
        return &root;
    }

    /**
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] size_t get_num_local_trees() const noexcept override {
        return branch_nodes.size();
    }

    /**
     * @brief Get all local branch nodes
     * @return All local branch nodes
     */
    [[nodiscard]] std::vector<const OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() const {
        return branch_nodes
            | ranges::views::filter(ranges::indirect(&OctreeNode<AdditionalCellAttributes>::is_local))
            | ranges::to_vector;
    }

    /**
     * @brief Get all local branch nodes
     * @return All local branch nodes
     */
    [[nodiscard]] std::vector<OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() {
        return branch_nodes
            | ranges::views::filter(ranges::indirect(&OctreeNode<AdditionalCellAttributes>::is_local))
            | ranges::to_vector;
    }

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    void initializes_leaf_nodes(const RelearnTypes::number_neurons_type num_neurons) override {
        std::vector<OctreeNode<AdditionalCellAttributes>*> leaf_nodes(num_neurons);

        Stack<OctreeNode<AdditionalCellAttributes>*> stack{ num_neurons };
        stack.emplace_back(&root);

        while (!stack.empty()) {
            OctreeNode<AdditionalCellAttributes>* node = stack.pop_back();

            if (node->is_leaf()) {
                const auto neuron_id = node->get_cell_neuron_id();
                RelearnException::check(neuron_id.get_neuron_id() < leaf_nodes.size(), "Octree::initializes_leaf_nodes: Neuron id was too large for leaf nodes: {}", neuron_id);

                leaf_nodes[neuron_id.get_neuron_id()] = node;
                continue;
            }

            for (auto* child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                if (const auto neuron_id = child->get_cell_neuron_id(); !child->is_parent() && (neuron_id.is_virtual() || !neuron_id.is_initialized())) {
                    continue;
                }

                stack.emplace_back(child);
            }
        }

        RelearnException::check(leaf_nodes.size() == num_neurons, "Octree::initializes_leaf_nodes: Less number of leaf nodes than number of local neurons {} != {}", leaf_nodes.size(), num_neurons);

        for (const auto neuron_id : NeuronID::range(num_neurons)) {
            const auto& node = leaf_nodes[neuron_id.get_neuron_id()];
            RelearnException::check(node != nullptr, "Octree::initializes_leaf_nodes: Leaf node {} is null", neuron_id);
            RelearnException::check(node->is_leaf(), "Octree::initializes_leaf_nodes: Leaf node {} is not a leaf node", neuron_id);
            RelearnException::check(node->is_local(), "Octree::initializes_leaf_nodes: Leaf node {} is not local", neuron_id);
            RelearnException::check(node->get_cell().get_neuron_id() == neuron_id, "Octree::initializes_leaf_nodes: Leaf node {} has wrong neuron id {}", neuron_id, node->get_cell().get_neuron_id());
        }

        all_leaf_nodes = std::move(leaf_nodes);
    }

    /**
     * @brief Synchronizes the octree with all MPI ranks
     */
    void synchronize_tree() {
        // Update my local trees bottom-up
        update_local_trees();

        // Exchange the local trees
        synchronize_local_trees();
    }

    /**
     * @brief Returns a constant reference to all leaf nodes
     *      The reference is never invalidated
     * @return All leaf nodes
     */
    [[nodiscard]] const std::vector<OctreeNode<AdditionalCellAttributes>*>& get_leaf_nodes() const noexcept {
        return all_leaf_nodes;
    }

    /**
     * @brief Returns the branch node with the (global) index, cast to a void*
     * @param index The global index of the requested branch node
     * @exception Throws a RelearnException if index is larger than or equal to the number of branch nodes
     * @return The requested branch node
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_branch_node_pointer(size_t index) {
        RelearnException::check(index < branch_nodes.size(), "OctreeImplementation::get_branch_node_pointer(): index ({}) is larger than or equal to the number of branch nodes ({}).", index, branch_nodes.size());
        return branch_nodes[index];
    }

    /**
     * @brief Inserts a neuron at the specified position with the neuron id and the position
     * @param position The position of the neuron
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if the root is nullptr, the position is not in the box,
     *      neuron_id is uninitialized, or OctreeNode::insert throws a RelearnException
     */
    void insert(const box_size_type& position, const NeuronID& neuron_id) override {
        const auto& [min_x, min_y, min_z] = get_simulation_box_minimum();
        const auto& [max_x, max_y, max_z] = get_simulation_box_maximum();
        const auto& [pos_x, pos_y, pos_z] = position;

        RelearnException::check(min_x <= pos_x && pos_x <= max_x, "Octree::insert: x was not in range: {} vs [{}, {}]", pos_x, min_x, max_x);
        RelearnException::check(min_y <= pos_y && pos_y <= max_y, "Octree::insert: y was not in range: {} vs [{}, {}]", pos_y, min_y, max_y);
        RelearnException::check(min_z <= pos_z && pos_z <= max_z, "Octree::insert: z was not in range: {} vs [{}, {}]", pos_z, min_z, max_z);

        RelearnException::check(neuron_id.is_initialized(), "Octree::insert: neuron_id {} was uninitialized", neuron_id);

        auto* res = root.insert(position, neuron_id);
        RelearnException::check(res != nullptr, "Octree::insert: res was nullptr");
    }

protected:
    /**
     * Print a visualization of this tree to a stringstream
     * @param ss stringstream
     */
    void print(std::stringstream& ss) const override {
        ss << root.to_string() << "\n";
        root.printSubtree(ss, "");
        ss << "\n";
    }

    /**
     * @brief Constructs the upper portion of the tree, i.e., all nodes at depths [0, level_of_branch_nodes].
     */
    void construct_global_tree_part() {
        const auto level_of_branch_nodes = get_level_of_branch_nodes();
        const auto num_cells_per_dimension = 1ULL << level_of_branch_nodes; // (2^level_of_branch_nodes)

        std::vector<box_size_type> branch_nodes_positions{};
        branch_nodes_positions.reserve(num_cells_per_dimension * num_cells_per_dimension * num_cells_per_dimension);

        const auto& xyz_min = get_simulation_box_minimum();
        const auto& xyz_max = get_simulation_box_maximum();

        const auto& [x_min, y_min, z_min] = xyz_min;
        const auto& [x_max, y_max, z_max] = xyz_max;

        const auto box_x = (x_max - x_min) / num_cells_per_dimension;
        const auto box_y = (y_max - y_min) / num_cells_per_dimension;
        const auto box_z = (z_max - z_min) / num_cells_per_dimension;

        const auto half_x = box_x / 2.0;
        const auto half_y = box_y / 2.0;
        const auto half_z = box_z / 2.0;

        for (auto z_it = 0; z_it < num_cells_per_dimension; z_it++) {
            for (auto y_it = 0; y_it < num_cells_per_dimension; y_it++) {
                for (auto x_it = 0; x_it < num_cells_per_dimension; x_it++) {
                    const auto x = x_it * box_x + half_x + x_min;
                    const auto y = y_it * box_y + half_y + y_min;
                    const auto z = z_it * box_z + half_z + z_min;
                    branch_nodes_positions.emplace_back(x, y, z);
                }
            }
        }

        root.set_cell_size(xyz_min, xyz_max);
        root.set_cell_neuron_id(NeuronID::virtual_id());
        root.set_cell_neuron_position(branch_nodes_positions[0]);
        root.set_rank(MPIWrapper::get_my_rank());
        root.set_level(0);

        for (auto pos_it = 1; pos_it < branch_nodes_positions.size(); pos_it++) {
            auto* ptr = root.insert(branch_nodes_positions[pos_it], NeuronID::virtual_id());
        }

        SpaceFillingCurve<Morton> space_curve{ static_cast<uint8_t>(level_of_branch_nodes) };

        Stack<std::pair<OctreeNode<AdditionalCellAttributes>*, Vec3s>> stack{ Constants::number_oct * level_of_branch_nodes };
        stack.emplace_back(&root, Vec3s{ 0, 0, 0 });

        while (!stack.empty()) {
            const auto [ptr, index3d] = stack.pop_back();

            if (!ptr->is_parent()) {
                const auto index1d = space_curve.map_3d_to_1d(index3d);
                branch_nodes[index1d] = ptr;
                continue;
            }

            for (size_t id = 0; id < Constants::number_oct; id++) {
                auto child_node = ptr->get_child(id);

                const auto larger_x = ((id & 1ULL) == 0) ? 0ULL : 1ULL;
                const auto larger_y = ((id & 2ULL) == 0) ? 0ULL : 1ULL;
                const auto larger_z = ((id & 4ULL) == 0) ? 0ULL : 1ULL;

                const Vec3s offset{ larger_x, larger_y, larger_z };
                const Vec3s pos = (index3d * 2) + offset;
                stack.emplace_back(child_node, pos);
            }
        }
    }

    /**
     * @brief Updates all local (!) branch nodes and their induced subtrees.
     * @exception Throws a RelearnException if the functor throws
     */
    void update_local_trees() {
        Timers::start(TimerRegion::UPDATE_LOCAL_TREES);

        const auto update_tree = [this](auto* local_tree) {
            update_tree_parallel(local_tree);
        };

        ranges::for_each(branch_nodes | ranges::views::filter(ranges::indirect(&OctreeNode<AdditionalCellAttributes>::is_local)), update_tree);

        Timers::stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);
    }

    /**
     * @brief Synchronizes all (locally) updated branch nodes with all other MPI ranks
     */
    void synchronize_local_trees() {
        Timers::start(TimerRegion::EXCHANGE_BRANCH_NODES);
        const auto number_branch_nodes = branch_nodes.size();

        // Copy local trees' root nodes to correct positions in receive buffer
        auto exchange_branch_nodes = branch_nodes | ranges::views::indirect | ranges::to_vector;

        // All-gather in-place branch nodes from every rank
        const auto number_local_branch_nodes = number_branch_nodes / MPIWrapper::get_num_ranks();
        RelearnException::check(number_local_branch_nodes < static_cast<size_t>(std::numeric_limits<int>::max()),
            "OctreeImplementation::synchronize_local_trees: Too many branch nodes: {}", number_local_branch_nodes);
        MPIWrapper::all_gather_inline(std::span{ exchange_branch_nodes.data(), number_local_branch_nodes });

        Timers::stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

        Timers::start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
        for (size_t i = 0; i < number_branch_nodes; i++) {
            auto& received_node = exchange_branch_nodes[i];
            if (received_node.is_parent()) {
                /*
                 *
                 * If the branch node is a leaf, it uses the leaf-case without problems.
                 * Otherwise, we need to store the index of the branch node so that we
                 * can later send it around.
                 */
                // received_node.set_cell_neuron_id(NeuronID::virtual_id(i));
            }

            *branch_nodes[i] = exchange_branch_nodes[i];
        }
        Timers::stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

        Timers::start(TimerRegion::UPDATE_GLOBAL_TREE);
        if (const auto level_of_branch_nodes = get_level_of_branch_nodes(); level_of_branch_nodes > 0) {
            // Only update whenever there are other branches to update
            // The nodes at level_of_branch_nodes are already updated (by other MPI ranks)
            update_tree_parallel(&root, level_of_branch_nodes - 1);
        }
        Timers::stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);
    }

    /**
     * @brief Updates the tree induced by local_tree_root until the desired level.
     *      Uses OctreeNode::get_level() to determine the depth. The nodes at that depth are still updated, but not their children.
     *      Potentially updates in parallel based on the depth of the updates.
     * @param local_tree_root The root of the tree from where to update
     * @param max_depth The depth where the updates shall stop
     * @exception Throws a RelearnException if local_tree_root is nullptr or if max_depth is smaller than the depth of local_tree_root
     */
    void update_tree_parallel(OctreeNode<AdditionalCellAttributes>* local_tree_root, const std::uint16_t max_depth = std::numeric_limits<std::uint16_t>::max()) {
        RelearnException::check(local_tree_root != nullptr, "OctreeImplementation::update_tree_parallel: local_tree_root was nullptr");
        RelearnException::check(local_tree_root->get_level() <= max_depth, "OctreeImplementation::update_tree_parallel: The root had a larger depth than max_depth.");

        if (const auto update_height = max_depth - local_tree_root->get_level(); update_height < 3) {
            // If the update concerns less than 3 levels, update serially
            OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(local_tree_root, max_depth);
            return;
        }

        // Gather all subtrees two levels down from the current node, update the induced trees in parallel, and then update the upper portion serially

        constexpr auto maximum_number_subtrees = 64;
        std::vector<OctreeNode<AdditionalCellAttributes>*> subtrees{};
        subtrees.reserve(maximum_number_subtrees);

        constexpr auto maximum_number_nodes = 64 + 8 + 1;
        Stack<OctreeNode<AdditionalCellAttributes>*> tree_upper_part{ maximum_number_nodes };
        tree_upper_part.emplace_back(local_tree_root);

        for (const auto& root_child : local_tree_root->get_children()) {
            if (root_child == nullptr) {
                continue;
            }

            tree_upper_part.emplace_back(root_child);

            for (const auto& root_child_child : root_child->get_children()) {
                if (root_child_child == nullptr) {
                    continue;
                }

                tree_upper_part.emplace_back(root_child_child);
                subtrees.emplace_back(root_child_child);
            }
        }

#pragma omp parallel for shared(subtrees, max_depth) default(none)
        for (auto i = 0; i < subtrees.size(); i++) {
            auto* local_tree_root = subtrees[i];
            OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(local_tree_root, max_depth);
        }

        while (!tree_upper_part.empty()) {
            auto* node = tree_upper_part.top();
            tree_upper_part.pop();

            if (node->is_parent()) {
                OctreeNodeUpdater<AdditionalCellAttributes>::update_node(node);
            }
        }
    }

private:
    // Root of the tree
    OctreeNode<AdditionalCellAttributes> root{};

    std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> all_leaf_nodes{};
};
