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

#include "util/ranges/Functional.hpp"
#include "util/shuffle/shuffle.h"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/transform.hpp>

class RandomAdapter {
public:
    template <typename T>
    static T get_random_double(T min, T max, std::mt19937& mt) {
        boost::random::uniform_real_distribution<double> urd(min, max);
        return urd(mt);
    }

    template <typename T>
    static T get_random_integer(T min, T max, std::mt19937& mt) {
        boost::random::uniform_int_distribution<T> uid(min, max);
        return uid(mt);
    }

    template <typename T>
    static T get_random_percentage(std::mt19937& mt) {
        return get_random_double<T>(0.0, std::nextafter(1.0, 1.1), mt);
    }

    static std::string get_random_string(size_t length, std::mt19937& mt) {
        auto randchar = [&mt]() -> char {
            const char charset[] = "0123456789"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "abcdefghijklmnopqrstuvwxyz";
            const size_t max_index = (sizeof(charset) - 2);
            return charset[get_random_integer(size_t{ 0 }, max_index, mt)];
        };
        std::string str(length, 0);
        std::generate_n(str.begin(), length, randchar);
        return str;
    }

    static bool get_random_bool(std::mt19937& mt) {
        const auto val = get_random_integer(0, 1, mt);
        return val == 0;
    }

    static std::vector<size_t> get_random_derangement(size_t size, std::mt19937& mt) {
        auto derangement = ranges::views::indices(size) | ranges::to_vector;

        auto check = [](const std::vector<size_t>& vec) {
            const auto index_equals_value = [](const auto& IndexValuePair) {
                const auto& [Index, Value] = IndexValuePair;
                return Index == Value;
            };

            return ranges::any_of(vec | ranges::views::enumerate, index_equals_value);
        };

        if (size <= 1) {
            return derangement;
        }

        do {
            shuffle(derangement, mt);
        } while (!check(derangement));

        return derangement;
    }

    template <typename Iterator>
    static void shuffle(Iterator begin, Iterator end, std::mt19937& mt) {
        ::shuffle(begin, end, mt);
    }

    template <typename Range>
    static void shuffle(Range& range, std::mt19937& mt) {
        ::shuffle(range, mt);
    }

    template <typename T>
    static std::vector<T> sample(const std::vector<T> vector, size_t sample_size, std::mt19937& mt) {
        std::unordered_set<size_t> indices{};
        while (indices.size() != sample_size) {
            size_t index;
            do {
                index = RandomAdapter::get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
            } while (indices.contains(index));
            indices.insert(index);
        }

        return indices
            | ranges::views::transform(lookup(vector))
            | ranges::to_vector
            | actions::shuffle(mt);
    }

    template <typename T>
    static std::vector<T> sample(const std::vector<T> vector, std::mt19937& mt) {
        const size_t sample_size = RandomAdapter::get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
        return sample(vector, sample_size, mt);
    }
};
