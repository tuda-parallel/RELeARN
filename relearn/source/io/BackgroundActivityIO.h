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
#include "io/parser/MonitorParser.h"
#include "util/MPIRank.h"
#include "util/ranges/views/IO.hpp"

#include <filesystem>
#include <functional>
#include <utility>
#include <range/v3/action/sort.hpp>
#include <range/v3/view/getlines.hpp>

class BackgroundActivityIO {
public:
    [[nodiscard]] static std::vector<std::tuple<RelearnTypes::step_type, std::string, std::vector<NeuronID>>> load_background_activity(const std::filesystem::path& file_path, const MPIRank& my_rank, const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
        std::ifstream file{ file_path };

        const bool file_is_good = file.good();
        const bool file_is_not_good = file.fail() || file.eof();

        const auto& parse_line = [my_rank, local_area_translator](const std::string& line) {
            std::stringstream sstream(line);

            RelearnTypes::step_type step{};
            std::string delim{};

            bool success = (sstream >> step) && (sstream >> delim);

            std::vector<std::string> descriptions{};

            for (std::string current_value{}; sstream >> current_value;) {
                descriptions.emplace_back(current_value);
            }
            const auto& parsed_ids = MonitorParser::parse_my_ids(descriptions, my_rank, local_area_translator);

            return std::tuple{ step, delim, parsed_ids };
        };

        RelearnException::check(file_is_good && !file_is_not_good,
            "BackgroundActivityIO::load_background_activity: Opening the file was not successful");

        auto background_activities = ranges::getlines(file)
            | views::filter_not_comment_not_empty_line
            | ranges::views::transform(parse_line)
            | ranges::to_vector
            | ranges::actions::sort([](const auto& t1, const auto& t2) {
                  return std::get<0>(t1) < std::get<0>(t2);
              })
            | ranges::to_vector;

        return background_activities;
    }
};