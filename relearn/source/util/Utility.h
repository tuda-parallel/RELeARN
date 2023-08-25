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

#include "neurons/enums/UpdateStatus.h"
#include "util/RelearnException.h"
#include "util/StringUtil.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

class NeuronsExtraInfo;

namespace Util {

/**
 * @brief Calculates the minimum, maximum, and sum over all values in the span, for which the disable flags are not disabled
 * @tparam T Must be a arithmetic (floating point or integral)
 * @param values The values that should be reduced
 * @param disable_flags The flags that indicate which values to skip
 * @exception Throws a RelearnException if (a) values.empty(), (b) values.size() != disable_flags.size(), (c) all values are disabled
 * @return Returns a tuple with (1) minimum and (2) maximum value from values, (3) the sum of all enabled values and (4) the number of enabled values
 */
template <typename T>
std::tuple<T, T, T, size_t> min_max_acc(const std::span<const T>& values, const std::shared_ptr<const NeuronsExtraInfo> extra_infos);

/**
 * @brief Counts the number of digits necessary to print the value
 * @tparam T Must be integral
 * @param val The value to print
 * @return The number of digits of val
 */
template <typename T>
constexpr unsigned int num_digits(T val) noexcept {
    static_assert(std::is_integral<T>::value);

    constexpr const auto number_system_base = 10;
    unsigned int num_digits = 1;

    while (val >= T(number_system_base)) {
        ++num_digits;
        // NOLINTNEXTLINE
        val /= number_system_base;
    }

    return num_digits;
}

/**
 * @brief Calculates the faculty.
 * @param value
 * @tparam T Type of which a faculty should be calculated. Must fulfill std::is_unsigned_v<T>
 * @return Returns the faculty of the parameter value.
 */
template <typename T>
static constexpr T factorial(T value) noexcept {
    static_assert(std::is_unsigned_v<T>, "bad T");
    if (value < 2) {
        return 1;
    }

    T result = 1;
    while (value > 1) {
        result *= value;
        value--;
    }

    return result;
}

/**
 * @brief Looks for a given file in a directory. Path: directory / prefix rank suffix. Tries different formats for the rank.
 * @param directory The directory where it looks for the file
 * @param rank The mpi rank
 * @param prefix Filename part before the mpi rank
 * @param suffix Filename after the mpi rank
 * @param max_digits Max width of the string which represents the rank
 * @return The file path for the found file
 * @throws RelearnException When no file was found
 */
static std::filesystem::path find_file_for_rank(const std::filesystem::path& directory, const int rank,
    const std::string& prefix, const std::string& suffix, const unsigned int max_digits) {

    RelearnException::check(!directory.empty(), "Utility::find_file_for_rank: Path is empty");
    RelearnException::check(std::filesystem::exists(directory), "Utility::find_file_for_rank: Path is not existent");

    if (std::filesystem::is_regular_file(directory)) {
        RelearnException::check(rank == 0, "Utility::find_file_for_rank: Single positions file is only allowed for a single mpi rank");
        return directory;
    }

    const auto num_files_in_directory = std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{});
    if (num_files_in_directory == 1) {
        RelearnException::check(rank == 0, "Utility::find_file_for_rank: Single file is only allowed for a single mpi rank");
        return directory / std::filesystem::directory_iterator(directory)->path().filename();
    }

    for (auto nr_digits = 1U; nr_digits < max_digits; nr_digits++) {
        const auto my_position_filename = prefix + StringUtil::format_int_with_leading_zeros(rank, nr_digits) + suffix;
        std::filesystem::path path_to_file = directory / my_position_filename;
        if (std::filesystem::exists(path_to_file)) {
            return path_to_file;
        }
    }

    RelearnException::fail("Util::find_file_for_rank: No file found for {}{}{}", prefix, rank, suffix);
}

} // namespace Util
