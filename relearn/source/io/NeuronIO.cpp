/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronIO.h"

#include "neurons/LocalAreaTranslator.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "spdlog/spdlog.h"
#include "util/ranges/views/IO.hpp"

#include <climits>
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <iostream>
#include <utility>

#include <range/v3/iterator/operations.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/view/getlines.hpp>

std::vector<std::string> NeuronIO::read_comments(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_comments: Opening the file was not successful");

    std::vector<std::string> comments{};

    for (std::string line{}; std::getline(file, line);) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            comments.emplace_back(std::move(line));
            continue;
        }

        break;
    }

    return comments;
}

AdditionalPositionInformation NeuronIO::parse_additional_position_information(const std::vector<std::string>& comments) {
    auto search_multiple = [&comments](const std::string& specifier) -> std::vector<std::string> {
        std::vector<std::string> results{};

        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position == std::string::npos) {
                continue;
            }
            results.push_back(comment.substr(specifier.size()));
        }

        return results;
    };
    auto search = [&comments](const std::string& specifier) -> double {
        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position != 0) {
                continue;
            }
            return std::stod(comment.substr(specifier.size()));
        }

        RelearnException::fail("MultipleSubdomainsFromFile::read_neurons_from_file: Did not find comment containing {}", specifier);
    };

    auto search_contains = [&comments](const std::string& specifier) -> std::string {
        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position == std::string::npos) {
                continue;
            }
            return comment;
        }

        RelearnException::fail("MultipleSubdomainsFromFile::read_neurons_from_file: Did not find comment containing {}", specifier);
    };

    auto parse_coordinates = [](std::string str) -> RelearnTypes::position_type {
        std::replace(str.begin(), str.end(), '(', ' ');
        std::replace(str.begin(), str.end(), ')', ' ');
        auto coords = StringUtil::split_string(str, ',');
        RelearnException::check(coords.size() == 3, "NeuronIO::parse_additional_position_information: Subdomains have invalid coordinates: {}", str);
        auto x = std::stod(coords[0]);
        auto y = std::stod(coords[1]);
        auto z = std::stod(coords[2]);
        return { x, y, z };
    };

    const auto min_x = search("# Minimum x:");
    const auto min_y = search("# Minimum y:");
    const auto min_z = search("# Minimum z:");
    const auto max_x = search("# Maximum x:");
    const auto max_y = search("# Maximum y:");
    const auto max_z = search("# Maximum z:");
    const RelearnTypes::bounding_box_type sim_box{ { min_x, min_y, min_z }, { max_x, max_y, max_z } };

    auto subdomain_strings = search_multiple("# Local subdomain ");
    std::vector<RelearnTypes::bounding_box_type> subdomains{};
    auto subdomain_id_expected = 0;
    for (auto i = 0; i < subdomain_strings.size(); i++) {
        const auto& subdomain_string = subdomain_strings[i];
        auto tokens = StringUtil::split_string(subdomain_string, ' ');
        if (!StringUtil::is_number(tokens[0])) {
            continue;
        }
        auto subdomain_id = std::stoi(tokens[0]);
        RelearnException::check(subdomain_id_expected == subdomain_id, "NeuronIO::parse_additional_position_information: Expected subdomain id {} not {}", i, subdomain_id);
        subdomain_id_expected++;

        auto i1 = subdomain_string.find('(');
        auto i2 = subdomain_string.find(')');
        auto i3 = subdomain_string.find('(', i2 + 1);
        auto i4 = subdomain_string.find(')', i3 + 1);

        auto min_subdomain = parse_coordinates(subdomain_string.substr(i1 + 1, i2 - i1));
        auto max_subdomain = parse_coordinates(subdomain_string.substr(i3 + 1, i4 - i3));

        RelearnException::check(min_subdomain.get_x() < max_subdomain.get_x(), "NeuronIO::parse_additional_position_information: Min subdomain larger than max subdomain");
        RelearnException::check(min_subdomain.get_y() < max_subdomain.get_y(), "NeuronIO::parse_additional_position_information: Min subdomain larger than max subdomain");
        RelearnException::check(min_subdomain.get_z() < max_subdomain.get_z(), "NeuronIO::parse_additional_position_information: Min subdomain larger than max subdomain");

        subdomains.emplace_back(min_subdomain, max_subdomain);
    }

    const auto neurons_str = search_contains(" of ");
    auto i1 = neurons_str.find(" of ");
    auto local_neurons = std::stoi(neurons_str.substr(2, i1 - 2));
    auto total_neurons = std::stoi(neurons_str.substr(i1 + 4));

    return { sim_box, subdomains, static_cast<number_neurons_type>(total_neurons), static_cast<number_neurons_type>(local_neurons) };
}

bool NeuronIO::is_valid_area_name(const RelearnTypes::area_name& area_name) {
    auto is_valid_char = [](const char c) {
        return std::isalpha(c) || std::isdigit(c) || c == '_';
    };

    return std::find_if(area_name.begin(), area_name.end(),
               [is_valid_char](char c) { return !is_valid_char(c); })
        == area_name.end();
}

std::tuple<std::vector<LoadedNeuron>, std::vector<RelearnTypes::area_name>, LoadedNeuronsInfo, AdditionalPositionInformation> NeuronIO::read_neurons(const std::filesystem::path& file_path) {
    RelearnException::check(std::filesystem::is_regular_file(file_path), "NeuronIO::read_neurons: Path is not a file");
    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neurons: Opening the file was not successful");

    position_type minimum(std::numeric_limits<position_type::value_type>::max());
    position_type maximum(std::numeric_limits<position_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    std::vector<LoadedNeuron> nodes{};

    number_neurons_type expected_id = 0;

    std::vector<RelearnTypes::area_name> area_names{};

    std::vector<std::string> comments{};

    for (const auto& line : ranges::getlines(file)) {

        // Skip line with comments
        if (line.empty()) {
            continue;
        }
        if ('#' == line[0]) {
            comments.emplace_back(std::move(line));
        }

        NeuronID::value_type id{};
        position_type::value_type pos_x{};
        position_type::value_type pos_y{};
        position_type::value_type pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(pos_x >= 0, "NeuronIO::read_neurons: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "NeuronIO::read_neurons: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "NeuronIO::read_neurons: z position of neuron {} was negative: {}", id, pos_z);
        RelearnException::check(id >= 1, "NeuronIO::read_neurons: neuron id is too small {}", id);

        id--;
        RelearnException::check(id == expected_id, "NeuronIO::read_neurons: Loaded neuron with id {} but expected: {}", id, expected_id);

        expected_id++;

        position_type position{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        const auto area_id_it = ranges::find(area_names, area_name);
        RelearnTypes::area_id area_id{ 0 };
        if (area_id_it == area_names.end()) {
            // Area name not known
            // Check if it is valid
            RelearnException::check(NeuronIO::is_valid_area_name(area_name), "NeuronIO::read_neurons: Area name {} is invalid", area_name);
            area_names.emplace_back(std::move(area_name));
            area_id = area_names.size() - 1;
        } else {
            area_id = static_cast<RelearnTypes::area_id>(ranges::distance(area_names.begin(), area_id_it));
        }

        if (signal_type == "in") {
            found_in_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Inhibitory, area_id);
        } else {
            found_ex_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Excitatory, area_id);
        }
    }

    const auto additional_position_information = parse_additional_position_information(comments);

    return { std::move(nodes), std::move(area_names), LoadedNeuronsInfo{ minimum, maximum, found_ex_neurons, found_in_neurons }, additional_position_information };
}

std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<RelearnTypes::area_id>, std::vector<RelearnTypes::area_name>, std::vector<SignalType>, LoadedNeuronsInfo>
NeuronIO::read_neurons_componentwise(const std::filesystem::path& file_path) {

    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neurons_componentwise: Opening the file was not successful");

    position_type minimum(std::numeric_limits<position_type::value_type>::max());
    position_type maximum(std::numeric_limits<position_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    std::vector<NeuronID> ids{};
    std::vector<position_type> positions{};
    std::vector<RelearnTypes::area_id> area_ids{};
    std::vector<RelearnTypes::area_name> area_names{};
    std::vector<SignalType> signal_types{};

    NeuronID::value_type expected_id = 0;

    for (const auto& line : ranges::getlines(file) | views::filter_not_comment_not_empty_line) {
        NeuronID::value_type id{};
        position_type::value_type pos_x{};
        position_type::value_type pos_y{};
        position_type::value_type pos_z{};
        RelearnTypes::area_name area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(pos_x >= 0, "NeuronIO::read_neurons_componentwise: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "NeuronIO::read_neurons_componentwise: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "NeuronIO::read_neurons_componentwise: z position of neuron {} was negative: {}", id, pos_z);

        id--;

        RelearnException::check(id == expected_id, "NeuronIO::read_neurons_componentwise: Loaded neuron with id {} but expected: {}", id, expected_id);

        expected_id++;

        position_type position{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        ids.emplace_back(false, id);
        positions.emplace_back(position);

        const auto area_id_it = ranges::find(area_names, area_name);
        RelearnTypes::area_id area_id{ 0 };
        if (area_id_it == area_names.end()) {
            // Area name not known
            area_names.emplace_back(std::move(area_name));
            area_id = area_names.size() - 1;
        } else {
            area_id = static_cast<RelearnTypes::area_id>(ranges::distance(area_names.begin(), area_id_it));
        }
        area_ids.emplace_back(area_id);

        if (signal_type == "in") {
            found_in_neurons++;
            signal_types.emplace_back(SignalType::Inhibitory);
        } else {
            found_ex_neurons++;
            signal_types.emplace_back(SignalType::Excitatory);
        }
    }

    return { std::move(ids), std::move(positions), std::move(area_ids), std::move(area_names), std::move(signal_types), LoadedNeuronsInfo{ minimum, maximum, found_ex_neurons, found_in_neurons } };
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
    write_neurons(neurons, file_path, local_area_translator, nullptr);
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::shared_ptr<Partition> partition) {
    std::stringstream ss{};
    write_neurons(neurons, ss, local_area_translator, partition);
    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");

    of << ss.str();
    of.close();
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, std::stringstream& ss, const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::shared_ptr<Partition>& partition) {
    RelearnTypes::bounding_box_type sim_box{};
    std::vector<RelearnTypes::bounding_box_type> subdomain_boxes{};
    RelearnTypes::number_neurons_type total_number_neurons;

    size_t number_local_subdomains;
    size_t first_local_subdomain_index;
    size_t last_local_subdomain_index;

    if (partition != nullptr) {
        total_number_neurons = partition->get_total_number_neurons();

        sim_box = partition->get_simulation_box_size();

        number_local_subdomains = partition->get_number_local_subdomains();
        first_local_subdomain_index = partition->get_local_subdomain_id_start();
        last_local_subdomain_index = partition->get_local_subdomain_id_end();

        for (auto i = first_local_subdomain_index; i <= last_local_subdomain_index; i++) {
            subdomain_boxes.push_back(partition->get_subdomain_boundaries(i));
        }

    } else {
        auto min_x = std::numeric_limits<double>::max();
        auto min_y = std::numeric_limits<double>::max();
        auto min_z = std::numeric_limits<double>::max();
        auto max_x = std::numeric_limits<double>::min();
        auto max_y = std::numeric_limits<double>::min();
        auto max_z = std::numeric_limits<double>::min();

        for (const auto& neuron : neurons) {
            const auto& [x, y, z] = neuron.pos;
            if (x < min_x) {
                min_x = x;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (y > max_y) {
                max_y = y;
            }
            if (z < min_z) {
                min_z = z;
            }
            if (z > max_z) {
                max_z = z;
            }
        }
        sim_box = RelearnTypes::bounding_box_type{ { min_x, min_y, min_z }, { max_x, max_y, max_z } };
        subdomain_boxes.push_back(sim_box);

        number_local_subdomains = 1;
        first_local_subdomain_index = 0;
        last_local_subdomain_index = 0;

        total_number_neurons = neurons.size();
    }

    ss << std::setprecision(std::numeric_limits<double>::digits10);

    // Write total number of neurons to log file
    ss << "# " << neurons.size() << " of " << total_number_neurons << '\n';
    ss << "# Minimum x: " << sim_box.get_minimum().get_x() << '\n';
    ss << "# Minimum y: " << sim_box.get_minimum().get_y() << '\n';
    ss << "# Minimum z: " << sim_box.get_minimum().get_z() << '\n';
    ss << "# Maximum x: " << sim_box.get_maximum().get_x() << '\n';
    ss << "# Maximum y: " << sim_box.get_maximum().get_y() << '\n';
    ss << "# Maximum z: " << sim_box.get_maximum().get_z() << '\n';
    ss << "# <local id> <pos x> <pos y> <pos z> <area> <type>\n";

    ss << "# Local subdomain index start: " << first_local_subdomain_index << "\n";
    ss << "# Local subdomain index end: " << last_local_subdomain_index << "\n";
    ss << "# Number of local subdomains: " << number_local_subdomains << "\n";

    for (auto i = 0; i < subdomain_boxes.size(); i++) {
        const auto local_subdomain_index = first_local_subdomain_index + i;
        const auto& subdomain_bb = subdomain_boxes[i];
        ss << "# Local subdomain " << local_subdomain_index << " boundaries (" << subdomain_bb.get_minimum().get_x() << ", " << subdomain_bb.get_minimum().get_y() << ", " << subdomain_bb.get_minimum().get_z() << ") - (";
        ss << subdomain_bb.get_maximum().get_x() << ", " << subdomain_bb.get_maximum().get_y() << ", " << subdomain_bb.get_maximum().get_z() << ")\n";
    }

    for (const auto& neuron : neurons) {
        const auto& [x, y, z] = neuron.pos;
        const auto& signal_type_name = (neuron.signal_type == SignalType::Excitatory) ? "ex" : "in";
        const auto& area_name = local_area_translator->get_area_name_for_neuron_id(neuron.id.get_neuron_id());

        ss << fmt::format("{1:<} {2:<.{0}} {3:<.{0}} {4:<.{0}} {5:<} {6:<}",
            Constants::print_precision, (neuron.id.get_neuron_id() + 1), x, y, z, area_name, signal_type_name)
           << '\n';
    }
}

void NeuronIO::write_area_names(std::stringstream& ss, const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
    ss << "# <area id>\t<ara_name>\t<num_neurons_in_area>\n";

    const auto num_areas = local_area_translator->get_number_of_areas();

    for (size_t area_id = 0; area_id < num_areas; area_id++) {
        const auto& area_name = local_area_translator->get_area_name_for_area_id(area_id);

        ss << area_id << '\t' << area_name << '\t' << local_area_translator->get_number_neurons_in_area(area_id) << '\n';
    }
}

void NeuronIO::write_neurons_componentwise(const std::span<const NeuronID> ids, const std::span<const position_type> positions,
    const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::span<const SignalType> signal_types, std::stringstream& ss, size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box,
    std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries) {

    const auto size_ids = ids.size();
    const auto size_positions = positions.size();
    const auto size_signal_types = signal_types.size();

    if (simulation_box.get_minimum().get_x() == simulation_box.get_maximum().get_x()) {
        auto min_x = std::numeric_limits<double>::max();
        auto min_y = std::numeric_limits<double>::max();
        auto min_z = std::numeric_limits<double>::max();
        auto max_x = std::numeric_limits<double>::min();
        auto max_y = std::numeric_limits<double>::min();
        auto max_z = std::numeric_limits<double>::min();

        for (const auto& [x, y, z] : positions) {
            if (x < min_x) {
                min_x = x;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (y > max_y) {
                max_y = y;
            }
            if (z < min_z) {
                min_z = z;
            }
            if (z > max_z) {
                max_z = z;
            }
        }
        simulation_box = RelearnTypes::bounding_box_type{ { min_x, min_y, min_z }, { max_x, max_y, max_z } };
    }

    if (local_subdomain_boundaries.empty()) {
        local_subdomain_boundaries.push_back(simulation_box);
    }

    const auto all_same_size = size_ids == size_positions && size_ids == size_signal_types;

    RelearnException::check(all_same_size, "NeuronIO::write_neurons_componentwise: The vectors had different sizes.");

    // Write total number of neurons to log file
    if (total_number_neurons > 0) {
        ss << "# " << ids.size() << " of " << total_number_neurons << '\n';
    }

    ss << std::setprecision(std::numeric_limits<double>::digits10);
    const auto& [simulation_box_min, simulation_box_max] = simulation_box;
    const auto& [min_x, min_y, min_z] = simulation_box_min;
    const auto& [max_x, max_y, max_z] = simulation_box_max;

    ss << "# Minimum x: " << min_x << '\n';
    ss << "# Minimum y: " << min_y << '\n';
    ss << "# Minimum z: " << min_z << '\n';
    ss << "# Maximum x: " << max_x << '\n';
    ss << "# Maximum y: " << max_y << '\n';
    ss << "# Maximum z: " << max_z << '\n';
    ss << "# <local id> <pos x> <pos y> <pos z> <area> <type>\n";

    const auto number_local_subdomains = local_subdomain_boundaries.size();
    ss << "# Number of local subdomains: " << number_local_subdomains << "\n";

    for (auto local_subdomain_index = 0; local_subdomain_index < number_local_subdomains; local_subdomain_index++) {
        const auto& [subdomain_bounding_box_min, subdomain_bounding_box_max] = local_subdomain_boundaries[local_subdomain_index];
        ss << "# Local subdomain " << local_subdomain_index << " boundaries (" << subdomain_bounding_box_min.get_x() << ", " << subdomain_bounding_box_min.get_y() << ", " << subdomain_bounding_box_min.get_z() << ") - (";
        ss << subdomain_bounding_box_max.get_x() << ", " << subdomain_bounding_box_max.get_y() << ", " << subdomain_bounding_box_max.get_z() << ")\n";
    }

    for (const auto& neuron_id : ids) {
        RelearnException::check(neuron_id.get_neuron_id() < ids.size(), "NeuronIO::write_neurons_componentwise: Neuron id {} is too large", neuron_id);
        const auto& [x, y, z] = positions[neuron_id.get_neuron_id()];
        const auto& signal_type_name = (signal_types[neuron_id.get_neuron_id()] == SignalType::Excitatory) ? "ex" : "in";
        const auto& area_name = local_area_translator->get_area_name_for_neuron_id(neuron_id.get_neuron_id());
        ss << (neuron_id.get_neuron_id() + 1) << " " << x << " " << y << " " << z << " " << area_name << " " << signal_type_name << '\n';
    }
}

void NeuronIO::write_neurons_componentwise(const std::span<const NeuronID> ids, const std::span<const position_type> positions,
    const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::span<const SignalType> signal_types, std::filesystem::path& file_path) {
    std::stringstream ss;
    write_neurons_componentwise(ids, positions, local_area_translator, signal_types, ss, 0, {}, {});
    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");

    of << ss.str();
    of.close();
}

void NeuronIO::write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
    const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::span<const SignalType> signal_types, const std::filesystem::path& path,
    size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box, std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries) {
    std::stringstream ss;
    write_neurons_componentwise(ids, positions, local_area_translator, signal_types, ss, total_number_neurons, simulation_box, std::move(local_subdomain_boundaries));
    std::ofstream of(path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");

    of << ss.str();
    of.close();
}

std::optional<std::vector<NeuronID>> NeuronIO::read_neuron_ids(const std::filesystem::path& file_path) {
    std::ifstream local_file(file_path);

    const bool file_is_good = local_file.good();
    const bool file_is_not_good = local_file.fail() || local_file.eof();

    if (!file_is_good || file_is_not_good) {
        return {};
    }

    std::vector<NeuronID> ids{};

    for (const auto& line : ranges::getlines(local_file) | views::filter_not_comment_not_empty_line) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type id{};
        position_type::value_type pos_x{};
        position_type::value_type pos_y{};
        position_type::value_type pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            return {};
        }

        id--;

        if (!ids.empty()) {
            const auto last_id = ids[ids.size() - 1].get_neuron_id();

            if (last_id + 1 != id) {
                return {};
            }
        }

        ids.emplace_back(false, id);
    }

    return ids;
}

NeuronIO::InSynapses NeuronIO::read_in_synapses(const std::filesystem::path& file_path,
    number_neurons_type number_local_neurons, MPIRank my_rank, size_t number_mpi_ranks) {
    StaticLocalSynapses local_in_synapses_static{};
    StaticDistantInSynapses distant_in_synapses_static{};
    PlasticLocalSynapses local_in_synapses_plastic{};
    PlasticDistantInSynapses distant_in_synapses_plastic{};

    std::ifstream file_synapses(file_path, std::ios::binary | std::ios::in);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::read_in_synapses: The ofstream failed to open {}", file_path.string());

    for (const auto& line : ranges::getlines(file_synapses) | views::filter_not_comment_not_empty_line) {
        int read_target_rank = 0;
        NeuronID::value_type read_target_id = 0;
        int read_source_rank = 0;
        NeuronID::value_type read_source_id = 0;
        RelearnTypes::static_synapse_weight weight = 0;
        bool plastic;

        std::stringstream sstream(line);
        const bool success = (sstream >> read_target_rank) && (sstream >> read_target_id) && (sstream >> read_source_rank) && (sstream >> read_source_id) && (sstream >> weight) && (sstream >> plastic);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(read_target_rank == my_rank.get_rank(), "NeuronIO::read_in_synapses: target_rank is not equal to my_rank: {} vs {}", read_target_rank, my_rank);
        RelearnException::check(read_target_id > 0 && read_target_id <= number_local_neurons, "NeuronIO::read_in_synapses: target_id was not from [1, {}]: {}", number_local_neurons, read_target_id);

        RelearnException::check(read_source_rank < number_mpi_ranks, "NeuronIO::read_in_synapses: source rank is not smaller than the number of mpi ranks: {} vs {}", read_source_rank, number_mpi_ranks);

        RelearnException::check(weight != 0, "NeuronIO::read_in_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;

        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        if (read_source_rank != my_rank.get_rank()) {
            if (plastic) {
                distant_in_synapses_plastic.emplace_back(target_id, RankNeuronId{ MPIRank(read_source_rank), source_id }, RelearnTypes::plastic_synapse_weight(weight));
            } else {
                distant_in_synapses_static.emplace_back(target_id, RankNeuronId{ MPIRank(read_source_rank), source_id }, weight);
            }
        } else {
            // if (target_id == source_id) {
            //     spdlog::info("Skipping line: {}", line);
            //     continue;
            // }

            if (plastic) {
                local_in_synapses_plastic.emplace_back(target_id, source_id, RelearnTypes::plastic_synapse_weight(weight));
            } else {
                local_in_synapses_static.emplace_back(target_id, source_id, weight);
            }
        }
    }

    return { { local_in_synapses_static, distant_in_synapses_static }, { local_in_synapses_plastic, distant_in_synapses_plastic } };
}

NeuronIO::OutSynapses NeuronIO::read_out_synapses(const std::filesystem::path& file_path,
    number_neurons_type number_local_neurons, MPIRank my_rank, size_t number_mpi_ranks) {
    StaticLocalSynapses local_out_synapses_static{};
    StaticDistantOutSynapses distant_out_synapses_static{};
    PlasticLocalSynapses local_out_synapses_plastic{};
    PlasticDistantOutSynapses distant_out_synapses_plastic{};

    std::ifstream file_synapses(file_path, std::ios::binary | std::ios::in);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::read_out_synapses: The ofstream failed to open");

    for (const auto& line : ranges::getlines(file_synapses) | views::filter_not_comment_not_empty_line) {
        int read_target_rank = 0;
        NeuronID::value_type read_target_id = 0;
        int read_source_rank = 0;
        NeuronID::value_type read_source_id = 0;
        RelearnTypes::static_synapse_weight weight = 0;

        std::stringstream sstream(line);
        bool plastic;
        const bool success = (sstream >> read_target_rank) && (sstream >> read_target_id) && (sstream >> read_source_rank) && (sstream >> read_source_id) && (sstream >> weight) && (sstream >> plastic);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(read_source_rank == my_rank.get_rank(), "NeuronIO::read_out_synapses: source_rank is not equal to my_rank: {} vs {}", read_target_rank, my_rank);
        RelearnException::check(read_source_id > 0 && read_source_id <= number_local_neurons, "NeuronIO::read_out_synapses: source_id was not from [1, {}]: {}", number_local_neurons, read_source_id);

        RelearnException::check(read_target_rank < number_mpi_ranks, "NeuronIO::read_out_synapses: target rank is not smaller than the number of mpi ranks: {} vs {}", read_source_rank, number_mpi_ranks);

        RelearnException::check(weight != 0, "NeuronIO::read_out_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;

        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        if (read_target_rank != my_rank.get_rank()) {
            if (plastic) {
                distant_out_synapses_plastic.emplace_back(RankNeuronId{ MPIRank(read_target_rank), target_id }, source_id, RelearnTypes::plastic_synapse_weight(weight));
            } else {
                distant_out_synapses_static.emplace_back(RankNeuronId{ MPIRank(read_target_rank), target_id }, source_id, weight);
            }
        } else {
            // if (target_id == source_id) {
            //     spdlog::info("Skipping line: {}", line);
            //     continue;
            // }

            if (plastic) {
                local_out_synapses_plastic.emplace_back(target_id, source_id, RelearnTypes::plastic_synapse_weight(weight));
            } else {
                local_out_synapses_static.emplace_back(target_id, source_id, weight);
            }
        }
    }

    return { { local_out_synapses_static, distant_out_synapses_static }, { local_out_synapses_plastic, distant_out_synapses_plastic } };
}

void NeuronIO::write_out_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_out_edges_static,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_out_edges_static,
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_out_edges_plastic,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_out_edges_plastic,
    const MPIRank my_rank, const size_t mpi_ranks, const RelearnTypes::number_neurons_type number_local_neurons, const RelearnTypes::number_neurons_type number_total_neurons,
    std::stringstream& ss, const size_t step) {
    const auto is_good = ss.good();
    const auto is_bad = ss.bad();

    const auto my_rank_int = my_rank.get_rank();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_distant_out_synapses: The ofstream failed to open");

    ss << "# Total number neurons: " << number_total_neurons << '\n';
    ss << "# Local number neurons: " << number_local_neurons << '\n';
    ss << "# Number MPI ranks: " << mpi_ranks << '\n';
    ss << "# Current simulation step: " << step << '\n';
    ss << "# <target rank> <target neuron id>\t<source rank> <source neuron id>\t<weight>\t<plastic>\n";

    for (const auto& source_local_id : NeuronID::range_id(number_local_neurons)) {
        for (const auto& [target_id, weight] : local_out_edges_static[source_local_id]) {
            const auto& target_local_id = target_id.get_neuron_id();

            ss << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [target_neuron, weight] : distant_out_edges_static[source_local_id]) {
            const auto& [target_rank, target_id] = target_neuron;
            const auto& target_local_id = target_id.get_neuron_id();

            RelearnException::check(target_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            ss << target_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [target_id, weight] : local_out_edges_plastic[source_local_id]) {
            const auto& target_local_id = target_id.get_neuron_id();

            ss << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }

        for (const auto& [target_neuron, weight] : distant_out_edges_plastic[source_local_id]) {
            const auto& [target_rank, target_id] = target_neuron;
            const auto& target_local_id = target_id.get_neuron_id();

            RelearnException::check(target_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            ss << target_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }
    }
}

void NeuronIO::write_in_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_in_edges_static,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_in_edges_static,
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_in_edges_plastic,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_in_edges_plastic,
    const MPIRank my_rank, const size_t mpi_ranks, const RelearnTypes::number_neurons_type number_local_neurons, const RelearnTypes::number_neurons_type number_total_neurons,
    std::stringstream& ss, const size_t step) {
    const auto is_good = ss.good();
    const auto is_bad = ss.bad();

    const auto my_rank_int = my_rank.get_rank();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_distant_out_synapses: The ofstream failed to open");

    ss << "# Total number neurons: " << number_total_neurons << '\n';
    ss << "# Local number neurons: " << number_local_neurons << '\n';
    ss << "# Number MPI ranks: " << mpi_ranks << '\n';
    ss << "# Current simulation step: " << step << '\n';
    ss << "# <target rank> <target neuron id>\t<source rank> <source neuron id>\t<weight>\t<plastic>\n";

    for (const auto& target_local_id : NeuronID::range_id(number_local_neurons)) {
        for (const auto& [source_id, weight] : local_in_edges_static[target_local_id]) {
            const auto& source_local_id = source_id.get_neuron_id();

            ss << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [source_neuron, weight] : distant_in_edges_static[target_local_id]) {
            const auto& [source_rank, source_id] = source_neuron;
            const auto& source_local_id = source_id.get_neuron_id();

            RelearnException::check(source_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            ss << my_rank_int << ' ' << (target_local_id + 1) << '\t' << source_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [source_id, weight] : local_in_edges_plastic[target_local_id]) {
            const auto& source_local_id = source_id.get_neuron_id();

            ss << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }

        for (const auto& [source_neuron, weight] : distant_in_edges_plastic[target_local_id]) {
            const auto& [source_rank, source_id] = source_neuron;
            const auto& source_local_id = source_id.get_neuron_id();

            RelearnException::check(source_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            ss << my_rank_int << ' ' << (target_local_id + 1) << '\t' << source_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }
    }
}

void NeuronIO::write_out_synapses(const StaticLocalSynapses& local_out_synapses_static, const StaticDistantOutSynapses& distant_out_synapses_static, const PlasticLocalSynapses& local_out_synapses_plastic,
    const PlasticDistantOutSynapses& distant_out_synapses_plastic, MPIRank my_rank, RelearnTypes::number_neurons_type number_neurons, const std::filesystem::path& file_path) {
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>> local_neighborhood_static{};
    local_neighborhood_static.resize(number_neurons);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>> distant_neighborhood_static{};
    distant_neighborhood_static.resize(number_neurons);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_neighborhood_plastic{};
    local_neighborhood_plastic.resize(number_neurons);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_neighborhood_plastic{};
    distant_neighborhood_plastic.resize(number_neurons);

    for (const auto& [target_id, source_id, weight] : local_out_synapses_static) {
        local_neighborhood_static[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : local_out_synapses_plastic) {
        local_neighborhood_plastic[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_out_synapses_static) {
        distant_neighborhood_static[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id, weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_out_synapses_plastic) {
        distant_neighborhood_plastic[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id, weight));
    }

    std::stringstream ss{};
    write_out_synapses(local_neighborhood_static, distant_neighborhood_static, local_neighborhood_plastic, distant_neighborhood_plastic, my_rank, 1, number_neurons, number_neurons, ss, 0);
    std::ofstream of(file_path, std::ios::binary | std::ios::out);
    const auto is_good = of.good();
    const auto is_bad = of.bad();
    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");
    of << ss.str();
    of.close();
}

void NeuronIO::write_in_synapses(const StaticLocalSynapses& local_in_synapses_static, const StaticDistantInSynapses& distant_in_synapses_static, const PlasticLocalSynapses& local_in_synapses_plastic,
    const PlasticDistantInSynapses& distant_in_synapses_plastic, MPIRank my_rank, RelearnTypes::number_neurons_type number_neurons, const std::filesystem::path& file_path) {
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>> local_neighborhood_static{};
    local_neighborhood_static.resize(number_neurons);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>> distant_neighborhood_static{};
    distant_neighborhood_static.resize(number_neurons);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_neighborhood_plastic{};
    local_neighborhood_plastic.resize(number_neurons);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_neighborhood_plastic{};
    distant_neighborhood_plastic.resize(number_neurons);

    for (const auto& [target_id, source_id, weight] : local_in_synapses_static) {
        local_neighborhood_static[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : local_in_synapses_plastic) {
        local_neighborhood_plastic[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_in_synapses_static) {
        distant_neighborhood_static[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id, weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_in_synapses_plastic) {
        distant_neighborhood_plastic[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id, weight));
    }

    std::stringstream ss{};
    write_in_synapses(local_neighborhood_static, distant_neighborhood_static, local_neighborhood_plastic, distant_neighborhood_plastic, my_rank, 1, number_neurons, number_neurons, ss, 0);
    std::ofstream of(file_path, std::ios::binary | std::ios::out);
    const auto is_good = of.good();
    const auto is_bad = of.bad();
    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");
    of << ss.str();
    of.close();
}
