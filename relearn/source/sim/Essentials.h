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

#include <map>
#include <ostream>
#include <string>
#include <type_traits>

/**
 * Provides the functionality to gather descriptions of the simulation
 * and print them out sorted in one flush.
 */
class Essentials {
public:
    using description_type = std::string;
    using value_type = std::string;

    /**
     * @brief Adds a description and a value to the store.
     *      If the descriptions is already present, the previous value is overwritten.
     * @param description The (new) description
     * @param value The value
     * @tparam T The type of the value, must support std::to_string(T)
     */
    template <typename T>
    void insert(description_type description, T value) {
        using T_type = std::decay_t<T>;

        if constexpr (std::is_same_v<T_type, value_type>) {
            dictionary[std::move(description)] = std::move(value);
        } else if constexpr (std::is_same_v<T_type, const char*>) {
            dictionary[std::move(description)] = std::string(value);
        } else if constexpr (std::is_same_v<T_type, char*>) {
            dictionary[std::move(description)] = std::string(value);
        } else {
            dictionary[std::move(description)] = std::to_string(value);
        }
    }

    /**
     * @brief Prints all stored entries in the form
     *      <key>: <value>\n
     *      sorted by <key>
     * @param out The outstream to print
     */
    void print(std::ostream& out) {
        for (const auto& [key, value] : dictionary) {
            out << key << ": " << value << '\n';
        }
    }

private:
    std::map<description_type, value_type> dictionary{};
};
