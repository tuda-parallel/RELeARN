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
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "util/RelearnException.h"

#include <optional>
#include <ostream>
#include <tuple>

/**
 * This class summarizes all 'octree-relevant' data from a neuron.
 * It contains a size in the octree (min and max), a neuron id (value Constants::uninitialized for virtual neurons, aka. inner nodes in the octree).
 * Depending on the template type, it also stores dendrite and axon positions, as well as calculated HermiteCoefficients.
 * AdditionalCellAttributes should be BarnesHutCell
 */
template <typename AdditionalCellAttributes>
class Cell {
public:
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;
    using box_size_type = RelearnTypes::box_size_type;

    constexpr static bool has_excitatory_dendrite = AdditionalCellAttributes::has_excitatory_dendrite;
    constexpr static bool has_inhibitory_dendrite = AdditionalCellAttributes::has_inhibitory_dendrite;
    constexpr static bool has_excitatory_axon = AdditionalCellAttributes::has_excitatory_axon;
    constexpr static bool has_inhibitory_axon = AdditionalCellAttributes::has_inhibitory_axon;

    /**
     * @brief Sets the neuron id
     * @param neuron_id The neuron id
     * @exception Throws a RelearnException if the neuron_id is not initialized
     */
    constexpr void set_neuron_id(const NeuronID neuron_id) {
        RelearnException::check(neuron_id.is_initialized(), "Cell::set:neuron_id: The neuron id was not initialized: {}", neuron_id);
        this->neuron_id = neuron_id;
    }

    /**
     * @brief Returns the neuron id
     * @return The neuron id
     */
    [[nodiscard]] constexpr NeuronID get_neuron_id() const noexcept {
        return neuron_id;
    }

    /**
     * @brief Sets the size of this cell
     * @param min The minimum index, y, and z of the cell
     * @param max The maximum index, y, and z of the cell
     * @exception Throws a RelearnException if one component of min is larger than the respective component of max
     */
    constexpr void set_size(const box_size_type& min, const box_size_type& max) {
        const auto& [min_x, min_y, min_z] = min;
        const auto& [max_x, max_y, max_z] = max;

        RelearnException::check(min_x <= max_x, "Cell::set_size: x was not ok");
        RelearnException::check(min_y <= max_y, "Cell::set_size: y was not ok");
        RelearnException::check(min_z <= max_z, "Cell::set_size: z was not ok");

        minimum_position = min;
        maximum_position = max;
    }

    /**
     * @brief Returns the size of the cell as tuple of (1) min and (2) max
     * @return The size of the cell as tuple of (1) min and (2) max
     */
    [[nodiscard]] constexpr std::tuple<box_size_type, box_size_type> get_size() const noexcept {
        return std::make_tuple(minimum_position, maximum_position);
    }

    /**
     * @brief Returns maximum edge length of the cell, i.e., ||max - min||_1
     * @return The maximum edge length of the cell
     */
    [[nodiscard]] constexpr double get_maximal_dimension_difference() const noexcept {
        const auto diff_vector = maximum_position - minimum_position;
        const auto diff = diff_vector.get_maximum();

        return diff;
    }

    /**
     * @brief Calculates the octant for the position.
     * @param position The position inside the current cell whose octant position should be found
     * @exception Throws a RelearnException if the position is not within the current cell
     * @return A value from 0 to 7 that indicates which octant the position is
     *
     * The binary numbering is computed as follows:
     *
     * 		   110 ----- 111
     *		   /|        /|
     *		  / |       / |
     *		 /  |      /  |
     *	   010 ----- 011  |    y
     *		|  100 ---|- 101   ^   z
     *		|  /      |  /     |
     *		| /       | /      | /
     *		|/        |/       |/
     *	   000 ----- 001       +-----> x
     */
    [[nodiscard]] unsigned char get_octant_for_position(const box_size_type& position) const {
        const auto& [x, y, z] = position;

        /**
         * Sanity check: Make sure that the position is within this cell
         * This check returns false if negative coordinates are used.
         * Thus make sure to use positions >= 0.
         */
        const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
        RelearnException::check(is_in_box, "Cell::get_octant_for_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);

        const auto& [min_x, min_y, min_z] = minimum_position;
        const auto& [max_x, max_y, max_z] = maximum_position;

        unsigned char idx = 0;
        // NOLINTNEXTLINE
        idx = idx | ((x < (min_x + max_x) / 2.0) ? 0 : 1); // idx | (pos_x < midpoint_dim_x) ? 0 : 1

        // NOLINTNEXTLINE
        idx = idx | ((y < (min_y + max_y) / 2.0) ? 0 : 2); // idx | (pos_y < midpoint_dim_y) ? 0 : 2

        // NOLINTNEXTLINE
        idx = idx | ((z < (min_z + max_z) / 2.0) ? 0 : 4); // idx | (pos_z < midpoint_dim_z) ? 0 : 4

        RelearnException::check(idx < Constants::number_oct, "Cell::get_octant_for_position: Calculated octant is too large: {}", idx);

        return idx;
    }

    /**
     * @brief Returns the size of the cell in the in the given octant
     * @param octant The octant, between 0 and 7
     * @exception Throws a RelearnException if octant > Constants::number_oct
     * @return A tuple with (min, max) for the cell in the given octant
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_size_for_octant(const unsigned char octant) const {
        RelearnException::check(octant <= Constants::number_oct, "Cell::get_size_for_octant: Octant was too large: {}", octant);

        const bool x_over_halfway_point = (octant & 1U) != 0;
        const bool y_over_halfway_point = (octant & 2U) != 0;
        const bool z_over_halfway_point = (octant & 4U) != 0;

        auto octant_xyz_min = this->minimum_position;
        auto octant_xyz_max = this->maximum_position;
        // NOLINTNEXTLINE
        const auto& octant_xyz_middle = octant_xyz_min.get_midpoint(octant_xyz_max);
        const auto& [middle_x, middle_y, middle_z] = octant_xyz_middle;

        if (x_over_halfway_point) {
            octant_xyz_min.set_x(middle_x);
        } else {
            octant_xyz_max.set_x(middle_x);
        }

        if (y_over_halfway_point) {
            octant_xyz_min.set_y(middle_y);
        } else {
            octant_xyz_max.set_y(middle_y);
        }

        if (z_over_halfway_point) {
            octant_xyz_min.set_z(middle_z);
        } else {
            octant_xyz_max.set_z(middle_z);
        }

        return std::make_tuple(octant_xyz_min, octant_xyz_max);
    }

    /**
     * @brief Prints the cell to the output stream
     * @param output_stream The output stream
     * @param cell The cell to print
     * @return The output stream after printing the cell
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Cell<AdditionalCellAttributes>& cell) {
        const auto number_excitatory_dendrites = cell.get_number_excitatory_dendrites();
        const auto number_inhibitory_dendrites = cell.get_number_inhibitory_dendrites();

        const auto& position_excitatory_dendrites_opt = cell.get_excitatory_dendrites_position();
        const auto& position_inhibitory_dendrites_opt = cell.get_inhibitory_dendrites_position();

        const auto& position_excitatory_dendrites = position_excitatory_dendrites_opt.value();
        const auto& position_inhibitory_dendrites = position_inhibitory_dendrites_opt.value();

        const auto& [minimum_position, maximum_position] = cell.get_size();

        // NOLINTNEXTLINE
        output_stream << "  == Cell (" << reinterpret_cast<size_t>(&cell) << " ==\n";
        output_stream << "\tMin: " << minimum_position << "\n\tMax: " << maximum_position << '\n';
        output_stream << "\tNeuronID: " << cell.get_neuron_id() << '\n';
        output_stream << cell.additional_cell_attributes;

        return output_stream;
    }

private:
    /**
     * ID of the neuron in the cell.
     * This is only valid for cells that contain a normal neuron.
     * For those with a super neuron, it has no meaning.
     * This info is used to identify (return) the target neuron for a given axon
     */
    NeuronID neuron_id{ NeuronID::uninitialized_id() };

    // Two points describe size of cell
    box_size_type minimum_position{ Constants::uninitialized };
    box_size_type maximum_position{ Constants::uninitialized };

    AdditionalCellAttributes additional_cell_attributes{};

public:
    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param number_dendrites The number of free excitatory dendrites
     */
    constexpr void set_number_excitatory_dendrites(const counter_type number_dendrites) noexcept {
        additional_cell_attributes.set_number_excitatory_dendrites(number_dendrites);
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] constexpr counter_type get_number_excitatory_dendrites() const noexcept {
        return additional_cell_attributes.get_number_excitatory_dendrites();
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param number_dendrites The number of free inhibitory dendrites
     */
    constexpr void set_number_inhibitory_dendrites(const counter_type number_dendrites) noexcept {
        additional_cell_attributes.set_number_inhibitory_dendrites(number_dendrites);
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] constexpr counter_type get_number_inhibitory_dendrites() const noexcept {
        return additional_cell_attributes.get_number_inhibitory_dendrites();
    }

    /**
     * @brief Returns the number of free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @return The number of free dendrites for the associated type
     */
    [[nodiscard]] counter_type get_number_dendrites_for(const SignalType dendrite_type) const {
        return additional_cell_attributes.get_number_dendrites_for(dendrite_type);
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory dendrite
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_excitatory_dendrites_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_excitatory_dendrites_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_excitatory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory dendrite
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] constexpr std::optional<position_type> get_excitatory_dendrites_position() const noexcept {
        return additional_cell_attributes.get_excitatory_dendrites_position();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory dendrite
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_inhibitory_dendrites_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_inhibitory_dendrites_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory dendrite
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] constexpr std::optional<position_type> get_inhibitory_dendrites_position() const noexcept {
        return additional_cell_attributes.get_inhibitory_dendrites_position();
    }

    /**
     * @brief Returns the position of the dendrite with the given signal type
     * @param dendrite_type The type of dendrite whose position should be returned
     * @return The position of the associated dendrite, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_dendrites_position_for(const SignalType dendrite_type) const {
        return additional_cell_attributes.get_dendrites_position_for(dendrite_type);
    }

    /**
     * @brief Sets the dendrite position for both inhibitory and excitatory
     * @param opt_position The dendrite position, can be empty
     */
    void set_dendrites_position(const std::optional<position_type>& opt_position) {
        set_excitatory_dendrites_position(opt_position);
        set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the dendrite position, for which either both positions must be empty or equal
     * @exception Throws a RelearnException if one position is valid and the other one invalid or if both are valid with different values
     * @return The position of the dendrite, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_dendrites_position() const {
        const auto& excitatory_dendrites_position_opt = get_excitatory_dendrites_position();
        const auto& inhibitory_dendrites_position_opt = get_inhibitory_dendrites_position();

        const bool ex_valid = excitatory_dendrites_position_opt.has_value();
        const bool in_valid = inhibitory_dendrites_position_opt.has_value();
        if (!ex_valid && !in_valid) {
            return {};
        }

        if (ex_valid && in_valid) {
            const auto& pos_ex = excitatory_dendrites_position_opt.value();
            const auto& pos_in = inhibitory_dendrites_position_opt.value();

            const auto diff = pos_ex - pos_in;
            const bool exc_position_equals_inh_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(exc_position_equals_inh_position, "Cell::get_dendrites_positions: positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("Cell::get_dendrites_positions: one pos was valid and one was not");

        return {};
    }

    /**
     * @brief Sets the number of free excitatory axons in this cell
     * @param number_axons The number of free excitatory axons
     */
    constexpr void set_number_excitatory_axons(const counter_type number_axons) noexcept {
        additional_cell_attributes.set_number_excitatory_axons(number_axons);
    }

    /**
     * @brief Returns the number of free excitatory axons in this cell
     * @return The number of free excitatory axons
     */
    [[nodiscard]] constexpr counter_type get_number_excitatory_axons() const noexcept {
        return additional_cell_attributes.get_number_excitatory_axons();
    }

    /**
     * @brief Sets the number of free inhibitory axons in this cell
     * @param number_dendrites The number of free inhibitory axons
     */
    constexpr void set_number_inhibitory_axons(const counter_type number_axons) noexcept {
        additional_cell_attributes.set_number_inhibitory_axons(number_axons);
    }

    /**
     * @brief Returns the number of free inhibitory axons in this cell
     * @return The number of free inhibitory axons
     */
    [[nodiscard]] constexpr counter_type get_number_inhibitory_axons() const noexcept {
        return additional_cell_attributes.get_number_inhibitory_axons();
    }

    /**
     * @brief Returns the number of free axons for the associated type in this cell
     * @param axon_type The requested axons type
     * @return The number of free axons for the associated type
     */
    [[nodiscard]] counter_type get_number_axons_for(const SignalType axon_type) const {
        return additional_cell_attributes.get_number_axons_for(axon_type);
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory axons
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_excitatory_axons_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_excitatory_axons_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_excitatory_axons_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory axons
     * @return The position of the excitatory axons
     */
    [[nodiscard]] constexpr std::optional<position_type> get_excitatory_axons_position() const noexcept {
        return additional_cell_attributes.get_excitatory_axons_position();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory axons
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_inhibitory_axons_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_inhibitory_axons_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory axons
     * @return The position of the inhibitory axons
     */
    [[nodiscard]] constexpr std::optional<position_type> get_inhibitory_axons_position() const noexcept {
        return additional_cell_attributes.get_inhibitory_axons_position();
    }

    /**
     * @brief Returns the position of the axons with the given signal type
     * @param dendrite_type The type of axons whose position should be returned
     * @return The position of the associated axons, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_axons_position_for(const SignalType axon_type) const {
        return additional_cell_attributes.get_axons_position_for(axon_type);
    }

    /**
     * @brief Sets the axon position for both inhibitory and excitatory
     * @param opt_position The axon position, can be empty
     */
    void set_axons_position(const std::optional<position_type>& opt_position) {
        set_excitatory_axons_position(opt_position);
        set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the axons position, for which either both positions must be empty or equal
     * @exception Throws a RelearnException if one position is valid and the other one invalid or if both are valid with different values
     * @return The position of the axons, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_axons_position() const {
        const auto& excitatory_axons_position_opt = get_excitatory_axons_position();
        const auto& inhibitory_axons_position_opt = get_inhibitory_axons_position();

        const bool ex_valid = excitatory_axons_position_opt.has_value();
        const bool in_valid = inhibitory_axons_position_opt.has_value();

        if (!ex_valid && !in_valid) {
            return {};
        }

        if (ex_valid && in_valid) {
            const auto& pos_ex = excitatory_axons_position_opt.value();
            const auto& pos_in = inhibitory_axons_position_opt.value();

            const auto diff = pos_ex - pos_in;
            const bool exc_position_equals_inh_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(exc_position_equals_inh_position, "Cell::get_axons_position: positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("Cell::get_axons_position: one pos was valid and one was not");

        return {};
    }

    /**
     * @brief Sets the position of the neuron for every necessary part of the cell
     * @param opt_position The position, can be empty
     * @exception Throws a RelearnException if the position is outside of the size
     */
    void set_neuron_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_neuron_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_neuron_position(opt_position);
    }

    /**
     * @brief Gets the position of the neuron for every necessary part of the cell
     * @exception Throws a RelearnException if one position if the positions do not agree with one another
     * @return opt_position The position, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_neuron_position() const {
        return additional_cell_attributes.get_neuron_position();
    }

    /**
     * @brief Returns the number of free elements for the associated type in this cell
     * @param axon_type The requested element type
     * @param signal_type The requested signal type
     * @exception Might throw a RelearnException if this operation is not supported
     * @return The number of free elements for the associated signal type
     */
    [[nodiscard]] counter_type get_number_elements_for(const ElementType element_type, const SignalType signal_type) const {
        return additional_cell_attributes.get_number_elements_for(element_type, signal_type);
    }

    /**
     * @brief Returns the position of the specified element with the given signal type
     * @param axon_type The requested element type
     * @param signal_type The requested signal type
     * @exception Might throw a RelearnException if this operation is not supported
     * @return The position of the associated element, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_position_for(const ElementType element_type, const SignalType signal_type) const {
        return additional_cell_attributes.get_position_for(element_type, signal_type);
    }

    /**
     * @brief Sets the number of free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @param num_dendrites The number of free dendrites
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_number_dendrites_for(const SignalType dendrite_type, const counter_type num_dendrites) {
        additional_cell_attributes.set_number_dendrites_for(dendrite_type, num_dendrites);
    }

    /**
     * @brief Sets the number of free axons for the associated type in this cell
     * @param axon_type The requested axon type
     * @param num_axons The number of free axons
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_number_axons_for(const SignalType axon_type, const counter_type num_axons) {
        additional_cell_attributes.set_number_axons_for(axon_type, num_axons);
    }

    /**
     * @brief Sets the number of free elements for the associated type in this cell
     * @param element_type The requested elements' type
     * @param signal_type The requested elements' signal type
     * @param num_elements The number of free elements
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_number_elements_for(const ElementType element_type, const SignalType signal_type, const counter_type num_elements) {
        additional_cell_attributes.set_number_elements_for(element_type, signal_type, num_elements);
    }

    /**
     * @brief Sets the position of the free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @param opt_position The position, can be empty
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_dendrites_position_for(const SignalType dendrite_type, const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_dendrites_position_for: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_dendrites_position_for(dendrite_type, opt_position);
    }

    /**
     * @brief Sets the position of the free axons for the associated type in this cell
     * @param axon_type The requested axon type
     * @param opt_position The position, can be empty
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_axons_position_for(const SignalType axon_type, const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_axons_position_for: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_axons_position_for(axon_type, opt_position);
    }

    /**
     * @brief Sets the position of the free elements for the associated type in this cell
     * @param element_type The requested elements' type
     * @param signal_type The requested elements' signal type
     * @param virtual_position The position of the free elements
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_position_for(const ElementType element_type, const SignalType signal_type, const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_position_for: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_position_for(element_type, signal_type, opt_position);
    }
};

#undef RELEARN_NUA
