/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */
#include "util/Utility.h"

#include "neurons/NeuronsExtraInfo.h"

template <typename T>
std::tuple<T, T, T, size_t> Util::min_max_acc(const std::span<const T>& values, const std::shared_ptr<const NeuronsExtraInfo> extra_infos) {
    static_assert(std::is_arithmetic<T>::value);

    RelearnException::check(!values.empty(), "Util::min_max_acc: values are empty");
    RelearnException::check(values.size() == extra_infos->get_size(), "Util::min_max_acc: values and extra_infos had different sizes");

    size_t first_index = 0;

    while (first_index < values.size() && !(extra_infos->does_update_electrical_actvity(NeuronID{ first_index }) && extra_infos->does_update_plasticity(NeuronID{ first_index }))) {
        first_index++;
    }

    RelearnException::check(first_index != values.size(), "Util::min_max_acc: all were disabled");

    T min = values[first_index];
    T max = values[first_index];
    T acc = values[first_index];

    size_t num_values = 1;

    for (auto i = first_index + 1; i < values.size(); i++) {
        if (!(extra_infos->does_update_electrical_actvity(NeuronID{ i }) && extra_infos->does_update_plasticity(NeuronID{ i }))) {
            continue;
        }

        const T& current_value = values[i];

        if (current_value < min) {
            min = current_value;
        } else if (current_value > max) {
            max = current_value;
        }

        acc += current_value;
        num_values++;
    }

    return std::make_tuple(min, max, acc, num_values);
}

template std::tuple<double, double, double, size_t> Util::min_max_acc(const std::span<const double>& values, const std::shared_ptr<const NeuronsExtraInfo> extra_infos);
template std::tuple<float, float, float, size_t> Util::min_max_acc(const std::span<const float>& values, const std::shared_ptr<const NeuronsExtraInfo> extra_infos);
template std::tuple<size_t, size_t, size_t, size_t> Util::min_max_acc(const std::span<const size_t>& values, const std::shared_ptr<const NeuronsExtraInfo> extra_infos);
template std::tuple<int, int, int, size_t> Util::min_max_acc(const std::span<const int>& values, const std::shared_ptr<const NeuronsExtraInfo> extra_infos);
