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

#include "fmt/ostream.h"

#include <ostream>

/**
 * An instance of this enum classifies the synaptic elements of a neuron.
 * In this simulation, there exists exactly two different ones: axonal elements and dendritic elements.
 * The distinction excitatory / inhibitory is made by the type SignalType.
 */
enum class ElementType : char {
    Axon,
    Dendrite
};

/**
 * @brief Returns the other element type, i.e., Axon for Dendrite and vice versa
 * @param The current element type
 * @return The other element type
 */
[[nodiscard]] inline constexpr ElementType get_other_element_type(const ElementType type) noexcept {
    if (type == ElementType::Axon) {
        return ElementType::Dendrite;
    }

    return ElementType::Axon;
}

/**
 * @brief Pretty-prints the element type to the chosen stream
 * @param out The stream to which to print the element type
 * @param element_type The element type to print
 * @return The argument out, now altered with the element type
 */
inline std::ostream& operator<<(std::ostream& out, const ElementType element_type) {
    if (element_type == ElementType::Axon) {
        return out << "Axon";
    }

    if (element_type == ElementType::Dendrite) {
        return out << "Dendrite";
    }

    return out;
}

template <>
struct fmt::formatter<ElementType> : ostream_formatter { };
