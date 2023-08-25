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
 * An instance of this enum classifies a synaptic elements as either excitatory or inhibitory.
 * An axon of a specific type pairs only with dendrites of that type.
 * A spiked transfered via an excitatory axon to an excitatory dendrite increases the electrical activity,
 * while the same for an inhibitory decreases the electrical activity.
 */
enum class SignalType : char {
    Excitatory,
    Inhibitory
};

/**
 * @brief Returns the other signal type, i.e., Excitatory for Inhibitory and vice versa
 * @param The current signal type
 * @return The other signal type
 */
[[nodiscard]] inline constexpr SignalType get_other_signal_type(const SignalType type) noexcept {
    if (type == SignalType::Excitatory) {
        return SignalType::Inhibitory;
    }

    return SignalType::Excitatory;
}

/**
 * @brief Pretty-prints the signal type to the chosen stream
 * @param out The stream to which to print the signal type
 * @param signal_type The signal type to print
 * @return The argument out, now altered with the signal type
 */
inline std::ostream& operator<<(std::ostream& out, const SignalType signal_type) {
    if (signal_type == SignalType::Excitatory) {
        return out << "Excitatory";
    }

    if (signal_type == SignalType::Inhibitory) {
        return out << "Inhibitory";
    }

    return out;
}

template <>
struct fmt::formatter<SignalType> : ostream_formatter { };
