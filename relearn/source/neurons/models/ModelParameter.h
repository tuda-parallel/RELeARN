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

#include "util/RelearnException.h"

#include <functional>
#include <string>
#include <variant>

/**
 * An object of type Parameter<T> represents a parameter of the model, the synaptic elements, etc.
 * It has a min and a max value, a name and supports setting a value to the specified parameter via a reference. Ensure that an object of type Parameter does not outlive referenced parameter.
 * Checks that min <= x <= max whenever a value is set through the object.
 */
template <typename T>
class Parameter {
public:
    /**
     * Type definition
     */
    using value_type = T;

    /**
     * @brief Constructs a Parameter that holds a min, a max, a name and the current value. The current value can lie outside of [min, max].
     * @param name The name of the parameter (to display it in the GUI)
     * @param value A reference to the current value. Is not checked against min or max
     * @param min The minimal acceptable value when set from the outside
     * @param max The maximal acceptable value when set from the outside
     * @exception Throws a RelearnException if max < min
     */
    Parameter(std::string&& name, T& value, const T& min, const T& max) noexcept(false)
        : name_{ std::move(name) }
        , value_{ value }
        , min_{ min }
        , max_{ max } {
        RelearnException::check(min <= max, "Parameter::Parameter: min was larger than max: {} vs {}", min, max);
    }

    Parameter(const Parameter& other) = default;
    Parameter(Parameter& other) = default;

    Parameter& operator=(const Parameter& other) = default;
    Parameter& operator=(Parameter&& other) = default;

    bool operator==(const Parameter<T>& other) const noexcept = default;

    /**
     * @brief Returns the name for the parameter
     * @return The name
     */
    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

    /**
     * @brief Sets the associated parameter to the value
     * @param val The value to which the parameter should be set
     * @exception Throws a RelearnException if min() <= val <= max() is violated
     */
    void set_value(const value_type& val) {
        RelearnException::check(min_ <= val, "Parameter::set_value: val was smaller than min_");
        RelearnException::check(val <= max_, "Parameter::set_value: val was larger than max_");

        value_.get() = val;
    }

    /**
     * @brief Returns the current value of the parameter
     * @return The value
     */
    [[nodiscard]] const value_type& value() const noexcept {
        return value_.get();
    }

    /**
     * @brief Returns the minimal value for the parameter
     * @return The minimal value
     */
    [[nodiscard]] const value_type& min() const noexcept {
        return min_;
    }

    /**
     * @brief Returns the maximal value for the parameter
     * @return The maximal value
     */
    [[nodiscard]] const value_type& max() const noexcept {
        return max_;
    }

private:
    std::string name_{}; // name of the parameter
    std::reference_wrapper<T> value_{}; // value of the parameter
    T min_{}; // minimum value of the parameter
    T max_{}; // maximum value of the parameter
};

/**
 * Variant of every Parameter of type T that is currently used in the simulation
 */
using ModelParameter = std::variant<Parameter<unsigned int>, Parameter<double>, Parameter<size_t>>;
