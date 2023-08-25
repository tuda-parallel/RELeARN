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
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "neurons/models/ModelParameter.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"

#include <cmath>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <range/v3/view/transform.hpp>

class NeuronMonitor;
class NeuronsExtraInfo;

/**
 * @brief A gaussian curve that is compressed by growth-factor nu and intersects the x-axis at
 *      eta (left intersection) and epsilon (right intersection).
 *      It is positive on (eta, epsilon) and negative at (-inf, eta), (epsilon, inf)
 *      Its maximum is at m := (eta + epsilon) / 2, and it is symmetric wrt. m.
 *      It attracts (-inf, eta) to -inf and (eta, inf) to epsilon.
 *      See Butz and van Ooyen, 2013 PloS Comp Biol, Equation 4.
 * @param current The current value (of calcium in the neuron)
 * @param eta The left intersection with the x-axis
 * @param epsilon The right intersection with the x-axis
 * @param growth_rate A linear scaling factor, i.e., the maximum of the function at (eta + epsilon) / 2
 */
inline double gaussian_growth_curve(const double current, const double eta, const double epsilon, const double growth_rate) noexcept {
    if (eta == epsilon) {
        // This is a corner case when using decaying target calcium
        if (current == eta) {
            return 0.0;
        }
        return -growth_rate;
    }

    constexpr auto factor = 1.6651092223153955127063292897904020952611777045288814583336582344;
    // 1.6651092223153955127063292897904020952611777045288814583336582344... = (2 * sqrt(-log(0.5)))

    const auto xi = (eta + epsilon) / 2;
    const auto zeta = (eta - epsilon) / factor;

    const auto difference = current - xi;
    const auto quotient = difference / zeta;
    const auto product = quotient * quotient;

    const auto dz = growth_rate * (2 * std::exp(-product) - 1);
    return dz;
}

/**
 * This type is a SoA for synaptic elements (can be used for axons and dendrites, both for excitatory and inhibitory).
 * It stores the number of grown and connected elements, and a delta that is accumulated during the electrical updates and committed during the synaptic updates.
 */
class SynapticElements {
    friend class NeuronMonitor;

public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Creates a new object with the given parameters. Does not initialize the vectors
     * @param type The type (axon or dendrite) of the elements stored in this object
     * @param min_C_level_to_grow The minimum calcium value for the elements to grow
     * @param nu The growth rate for the elements in ms^-1
     * @param vacant_retract_ratio The retract ratio for unconnected elements
     * @param initial_vacant_elements_lb The minimum number of free vacant elements during initialization
     * @param initial_vacant_elements_ub The maximum number of free vacant elements during initialization
     */
    SynapticElements(const ElementType type, const double min_C_level_to_grow,
        const double nu = SynapticElements::default_nu,
        const double vacant_retract_ratio = SynapticElements::default_vacant_retract_ratio,
        const double initial_vacant_elements_lb = SynapticElements::default_vacant_elements_initially_lower_bound,
        const double initial_vacant_elements_ub = SynapticElements::default_vacant_elements_initially_upper_bound)
        : type(type)
        , min_C_level_to_grow(min_C_level_to_grow)
        , nu(nu)
        , vacant_retract_ratio(vacant_retract_ratio)
        , initial_vacant_elements_lower_bound(initial_vacant_elements_lb)
        , initial_vacant_elements_upper_bound(initial_vacant_elements_ub) {
    }

    SynapticElements(const SynapticElements& other) = delete;
    SynapticElements(SynapticElements&& other) = default;

    SynapticElements& operator=(const SynapticElements& other) = delete;
    SynapticElements& operator=(SynapticElements&& other) = default;

    ~SynapticElements() = default;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its elements
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info.operator bool();
        RelearnException::check(is_filled, "SynapticElements::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Initializes the object to contain number_neurons elements.
     *      Creates initially free elements, uniformly and independently drawn from [initial_vacant_elements_lb, initial_vacant_elements_ub].
     *      Sets the number of connected elements to 0 for all neurons
     * @param number_neurons The number of that should be stored
     * @exception Throws a RelearnException if initial_vacant_elements_ub < initial_vacant_elements_lb
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Creates additional creation_count elements.
     *      For those, creates initially free elements, uniformly and independently drawn from [initial_vacant_elements_lb, initial_vacant_elements_ub].
     *      For those, sets the number of connected elements to 0 for all neurons.
     *      All previous elements are not changed.
     * @param number_neurons The number of that should be created
     * @exception Throws a RelearnException if initial_vacant_elements_ub < initial_vacant_elements_lb
     */
    void create_neurons(const number_neurons_type creation_count);

    /**
     * @brief Updates the counts for the specified neuron by the specified delta.
     *      This bypasses the commit-step. Use this only if necessary
     * @param neuron_id The local neuron id
     * @param delta The delta by which the number of elements changes (can be positive and negative)
     * @exception Throws a RelearnException if (a) neuron_id is too large, (b) the counts for the neuron are negative afterwards
     */
    void update_grown_elements(const NeuronID neuron_id, const double delta) {
        if (nu <= 0.0) {
            return;
        }

        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < grown_elements.size(), "SynapticElements::update_grown_elements: neuron_id is too large: {}", neuron_id);
        grown_elements[local_neuron_id] += delta;
        RelearnException::check(grown_elements[local_neuron_id] >= 0.0, "SynapticElements::update_grown_elements: Grown elements for {} are now negative", neuron_id);
    }

    /**
     * @brief Updates the connected elements for the specified neuron by the specified delta.
     * @param neuron_id The local neuron id
     * @param delta The delta by which the number of elements changes (can be positive and negative)
     * @exception Throws a RelearnException if (a) neuron_id is too large, (b) the counts for the neuron are negative afterwards
     */
    void update_connected_elements(const NeuronID neuron_id, const int delta) {
        if (nu <= 0.0) {
            return;
        }

        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < connected_elements.size(), "SynapticElements::update_connected_elements: neuron_id is too large: {}", neuron_id);
        if (delta < 0) {
            const unsigned int abs_delta = -delta;
            RelearnException::check(connected_elements[local_neuron_id] >= abs_delta, "SynapticElements::update_connected_elements: {} {} {}: {} {}", this->type, this->signal_types[local_neuron_id], neuron_id, delta, connected_elements[local_neuron_id]);
        }

        if (delta > 0) {
            const auto number_free_elements = get_free_elements(neuron_id);
            RelearnException::check(number_free_elements >= static_cast<unsigned int>(delta), "SynapticElements::update_connected_elements: There are not enough free elements {}: {} vs {}", neuron_id, delta, number_free_elements);
        }

        connected_elements[local_neuron_id] += delta;
    }

    /**
     * @brief Decreases the number of connected elements for each neuron by the specified amount.
     *      Disables all specified neurons, i.e., sets all values to 0
     * @param changes The deletions for each neuron, must be >= 0
     * @param disabled_neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if
     *      (a) changes.size() is not equal to the number of stored neurons
     *      (b) changes[i] is larger than the connected counts for i
     *      (c) disabled_neuron_ids[i] is larger than the number of stored neurons
     */
    void update_after_deletion(const std::span<const unsigned int> changes, const std::span<const NeuronID> disabled_neuron_ids) {
        if (nu <= 0.0) {
            return;
        }

        RelearnException::check(changes.size() == size, "SynapticElements::update_after_deletion: The number of changes does not match the number of elements");

        for (auto neuron_id = 0; neuron_id < size; neuron_id++) {
            const auto change = changes[neuron_id];
            RelearnException::check(connected_elements[neuron_id] >= change,
                "SynapticElements::update_after_deletion: Cannot delete more connections than present for neuron {}: {} vs {}", neuron_id, change, connected_elements[neuron_id]);

            connected_elements[neuron_id] -= change;
        }

        for (const auto local_neuron_id : disabled_neuron_ids | ranges::views::transform(&NeuronID::get_neuron_id)) {
            RelearnException::check(local_neuron_id < size, "SynapticElements::update_after_deletion: Cannot disable a neuron with a too large id");
            connected_elements[local_neuron_id] = 0;
            grown_elements[local_neuron_id] = 0.0;
            deltas_since_last_update[local_neuron_id] = 0.0;
        }
    }

    /**
     * @brief Commits the accumulated differences for all neurons that are not disabled and returns the number of total deletions and neuron-wise deletions
     * @exception Throws a RelearnException if the number of neurons in the extra infos does not match the number of stored neurons
     * @return Returns a tuple with (1) the number of deletions and (2) the number of deletions for each neuron
     */
    [[nodiscard]] std::pair<unsigned int, std::vector<unsigned int>> commit_updates();

    /**
     * @brief Updates the accumulated delta for each enabled neuron based on its current calcium value
     * @param calcium The current calcium value for each neuron
     * @param target_calcium The target calcium value for each neuron
     * @exception Throws a RelearnException if calcium.size() or the number of neurons in the extra infos does not match the number of stored neurons
     */
    void update_number_elements_delta(std::span<const double> calcium, std::span<const double> target_calcium);

    /**
     * @brief Clones this instance and creates a new SynapticElements with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::shared_ptr<SynapticElements> clone() const {
        return std::make_shared<SynapticElements>(type, min_C_level_to_grow, nu, vacant_retract_ratio, initial_vacant_elements_lower_bound, initial_vacant_elements_upper_bound);
    }

    /**
     * @brief Returns a vector with an std::shared_ptr for each significant instance (axons, dendrites (excitatory, inhibitory))
     * @return A vector with all significant instances
     */
    [[nodiscard]] static std::vector<std::shared_ptr<SynapticElements>> get_elements() {
        std::vector<std::shared_ptr<SynapticElements>> res{};
        res.emplace_back(std::make_shared<SynapticElements>(ElementType::Axon, SynapticElements::default_eta_Axons));
        res.emplace_back(std::make_shared<SynapticElements>(ElementType::Dendrite, SynapticElements::default_eta_Dendrites_exc));
        res.emplace_back(std::make_shared<SynapticElements>(ElementType::Dendrite, SynapticElements::default_eta_Dendrites_inh));
        return res;
    }

    /**
     * @brief Returns a vector with all adjustable ModelParameter
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() {
        return {
            Parameter<double>{ "Minimum calcium to grow", min_C_level_to_grow, SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow },
            Parameter<double>{ "nu", nu, SynapticElements::min_nu, SynapticElements::max_nu },
            Parameter<double>{ "Vacant synapse retract ratio", vacant_retract_ratio, SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio },
            Parameter<double>{ "Initial vacant elements lower bound", initial_vacant_elements_lower_bound, SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially },
            Parameter<double>{ "Initial vacant elements upper bound", initial_vacant_elements_upper_bound, SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially },
        };
    }

    /**
     * @brief Sets the signal type for the specified neuron
     * @param neuron_id The neuron
     * @param type The new signal type
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void set_signal_type(const NeuronID neuron_id, const SignalType type) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < signal_types.size(), "SynapticElements::set_signal_type: neuron_id is too large: {}", neuron_id);
        signal_types[local_neuron_id] = type;
    }

    /**
     * @brief Sets the signal type for all neurons at once
     * @param types The new signal types
     * @exception Throws a RelearnException if types.size() is not the same as the number of stored neurons
     */
    void set_signal_types(std::vector<SignalType>&& types) {
        RelearnException::check(types.size() == size, "SynapticElements::set_signal_type: Mismatching size of type vectors");
        signal_types = std::move(types);
    }

    /**
     * @brief Returns the number of grown elements, indexed by the local neuron id
     * @return The number of grown elements
     */
    [[nodiscard]] std::span<const double> get_grown_elements() const noexcept {
        return grown_elements;
    }

    /**
     * @brief Returns the number of connected elements, indexed by the local neuron id (how many elements from the neuron are connected via synapses)
     * @return The connected elements
     */
    [[nodiscard]] std::span<const unsigned int> get_connected_elements() const noexcept {
        return connected_elements;
    }

    /**
     * @brief Returns the accumulated changes to the grown elements, indexed by the local neuron id (the built-up difference from the electrical updates)
     * @return The accumulated changes
     */
    [[nodiscard]] std::span<const double> get_deltas() const noexcept {
        return deltas_since_last_update;
    }

    /**
     * @brief Returns the signal types of the elements, indexed by the local neuron id
     * @return The signal types
     */
    [[nodiscard]] std::span<const SignalType> get_signal_types() const noexcept {
        return signal_types;
    }

    /**
     * @brief Returns the number of grown elements for the specified neuron id
     * @param neuron_id The neuron
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The number of grown elements
     */
    [[nodiscard]] double get_grown_elements(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < grown_elements.size(), "SynapticElements::get_grown_elements: neuron_id is too large: {}", neuron_id);
        return grown_elements[local_neuron_id];
    }

    /**
     * @brief Returns the number of connected elements for the specified neuron id
     * @param neuron_id The neuron
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The number of connected elements
     */
    [[nodiscard]] unsigned int get_connected_elements(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < connected_elements.size(), "SynapticElements::get_connected_elements: neuron_id is too large: {}", neuron_id);
        return connected_elements[local_neuron_id];
    }

    /**
     * @brief Returns the number of free elements for the specified neuron id
     * @param neuron_id The neuron
     * @exception Throws a RelearnException if neuron_id is too large or if the number of connected elements exceeds the number of grown elements
     * @return The number of free elements
     */
    [[nodiscard]] unsigned int get_free_elements(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < connected_elements.size(), "SynapticElements::get_free_elements: neuron_id is too large: {}", neuron_id);
        RelearnException::check(connected_elements[local_neuron_id] <= grown_elements[local_neuron_id], "SynapticElements::get_free_elements: More elements were connected then free: {}, {} vs {}", neuron_id, connected_elements[local_neuron_id], grown_elements[local_neuron_id]);

        return static_cast<unsigned int>(grown_elements[local_neuron_id] - connected_elements[local_neuron_id]);
    }

    /**
     * @brief Returns the number of free elements for the specified neuron id and a signal type
     * @param neuron_id The neuron
     * @param signal_type The signal type
     * @exception Throws a RelearnException if neuron_id is too large or if the number of connected elements exceeds the number of grown elements
     * @return The number of free elements for the signal type
     */
    [[nodiscard]] unsigned int get_free_elements(const NeuronID neuron_id, const SignalType signal_type) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < signal_types.size(), "SynapticElements::get_free_elements: neuron_id is too large for the signal types: {}", neuron_id);

        if (signal_type != signal_types[local_neuron_id]) {
            return 0;
        }

        RelearnException::check(local_neuron_id < connected_elements.size(), "SynapticElements::get_free_elements: neuron_id is too large: {}", neuron_id);
        RelearnException::check(connected_elements[local_neuron_id] <= grown_elements[local_neuron_id], "SynapticElements::get_free_elements: More elements were connected then free: {}, {} vs {}", neuron_id, connected_elements[local_neuron_id], grown_elements[local_neuron_id]);

        return static_cast<unsigned int>(grown_elements[local_neuron_id] - connected_elements[local_neuron_id]);
    }

    /**
     * @brief Returns the accumulated delta for the specified neuron id
     * @param neuron_id The neuron
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The accumulated delta
     */
    [[nodiscard]] double get_delta(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < deltas_since_last_update.size(), "SynapticElements::get_delta: neuron_id is too large: {}", neuron_id);
        return deltas_since_last_update[local_neuron_id];
    }

    /**
     * @brief Returns the signal type for the specified neuron id
     * @param neuron_id The neuron
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The signal type
     */
    [[nodiscard]] SignalType get_signal_type(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < signal_types.size(), "SynapticElements::get_signal_type: neuron_id is too large: {}", neuron_id);
        return signal_types[local_neuron_id];
    }

    /**
     * @brief Returns the element type
     * @return The element type
     */
    [[nodiscard]] ElementType get_element_type() const noexcept {
        return type;
    }

    /**
     * @brief Returns the size
     * @return The size
     */
    [[nodiscard]] number_neurons_type get_size() const noexcept {
        return size;
    }

    /**
     * @brief Calculates and returns the histogram of the synaptic elements, i.e.,
     *      a mapping (x, y) -> z which indicates that there are z neurons
     *      that have x connected elements and y total elements (rounded down).
     * @exception Throws a RelearnException if the internal data structures are corrupted
     * @return The connection histogram
     */
    [[nodiscard]] std::map<std::pair<unsigned int, unsigned int>, uint64_t> get_histogram() const {
        RelearnException::check(size == grown_elements.size(), "SynapticElements::get_histogram: size did not match the number of grown elements");
        RelearnException::check(size == deltas_since_last_update.size(), "SynapticElements::get_histogram: size did not match the number of deltas");
        RelearnException::check(size == connected_elements.size(), "SynapticElements::get_histogram: size did not match the number of connected elements");
        RelearnException::check(size == signal_types.size(), "SynapticElements::get_histogram: size did not match the number of signal types");

        std::map<std::pair<unsigned int, unsigned int>, uint64_t> result{};

        for (number_neurons_type i = 0; i < size; i++) {
            const auto number_connected_elements = connected_elements[i];
            const auto number_grown_elements = static_cast<unsigned int>(grown_elements[i]);

            result[{ number_connected_elements, number_grown_elements }] += 1;
        }

        return result;
    }

    /**
     * @brief Returns the total amount of additions to these synaptic elements over the whole simulation
     * @return The additions
     */
    [[nodiscard]] double get_total_additions() const noexcept {
        return total_additions;
    }

    /**
     * @brief Returns the total amount of deletions to these synaptic elements over the whole simulation
     * @return The deletions
     */
    [[nodiscard]] double get_total_deletions() const noexcept {
        return total_deletions;
    }

private:
    /**
     * @brief Updates the number of synaptic elements for the specified neuron.
     *      Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
     * @param neuron_id The neuron
     * @exception Throws a RelearnException if neuron_id is too large or there was a calculation error
     * @return The number of synapses that must be deleted as a consequence
     *
     * Synaptic elements are deleted based on "deltas_since_last_update" in the following way:
     * 1. Delete vacant elements
     * 2. Delete bound elements
     */
    [[nodiscard]] unsigned int update_number_elements(NeuronID neuron_id);

public:
    static constexpr double default_eta_Axons{ 0.4 }; // In Sebastian's work: 0.0
    static constexpr double default_eta_Dendrites_exc{ 0.1 }; // In Sebastian's work: 0.0
    static constexpr double default_eta_Dendrites_inh{ 0.0 }; // In Sebastian's work: 0.0
    static constexpr double default_nu{ 1e-5 }; // In Sebastian's work: 1e-5
    static constexpr double default_vacant_retract_ratio{ 0.0 };
    static constexpr double default_vacant_elements_initially_lower_bound{ 0.0 };
    static constexpr double default_vacant_elements_initially_upper_bound{ 0.0 };

    static constexpr double min_min_C_level_to_grow{ 0.0 };
    static constexpr double min_C_target{ 0.0 };
    static constexpr double min_nu{ 0.0 };
    static constexpr double min_vacant_retract_ratio{ 0.0 };
    static constexpr double min_vacant_elements_initially{ 0.0 };

    static constexpr double max_min_C_level_to_grow{ 10.0 };
    static constexpr double max_C_target{ 100.0 };
    static constexpr double max_nu{ 1.0 };
    static constexpr double max_vacant_retract_ratio{ 1.0 };
    static constexpr double max_vacant_elements_initially{ 1000.0 };

private:
    double total_additions{ 0.0 };
    double total_deletions{ 0.0 };

    ElementType type{}; // Denotes the type of all synaptic elements, which is Axon or Dendrite
    number_neurons_type size{ 0 };

    std::shared_ptr<NeuronsExtraInfo> extra_infos{};
    std::vector<double> grown_elements{};
    std::vector<double> deltas_since_last_update{}; // Keeps track of changes in number of elements until those changes are applied in next connectivity update
    std::vector<unsigned int> connected_elements{};
    std::vector<SignalType> signal_types{}; // Signal type of synaptic elements, i.e., Excitatory or Inhibitory.
                                            // Note: Given that current exc. and inh. dendrites are in different objects, this would only be needed for axons.
                                            //       A more memory-efficient solution would be to use a different class for axons which has the signal_types array.

    // Parameters
    double min_C_level_to_grow{ 0.0 }; // Minimum level of calcium needed for elements to grow
    double nu{ default_nu }; // Growth rate for synaptic elements in ms^-1. Needs to be much smaller than 1 to separate activity and structural dynamics.
    double vacant_retract_ratio{ default_vacant_retract_ratio }; // Percentage of how many vacant synaptic elements should be deleted during each connectivity update

    double initial_vacant_elements_lower_bound{ default_vacant_elements_initially_lower_bound }; // Minimum number of vacant elements that are available at the start of the simulation
    double initial_vacant_elements_upper_bound{ default_vacant_elements_initially_upper_bound }; // Maximum number of vacant elements that are available at the start of the simulation
};
