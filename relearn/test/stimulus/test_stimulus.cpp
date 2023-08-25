/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_stimulus.h"

#include "adapter/neuron_assignment/NeuronAssignmentAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/stimulus/StimulusAdapter.h"

#include "io/parser/StimulusParser.h"
#include "io/InteractiveNeuronIO.h"
#include "neurons/LocalAreaTranslator.h"

#include <range/v3/range/conversion.hpp>

#include <unordered_set>

static void write_stimuli_to_file(std::filesystem::path path, std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>> stimuli) {
    std::ofstream of(path, std::ios::binary | std::ios::out);

    for (const auto& [begin, end, frequency, intensity, names] : stimuli) {
        of << begin << "-" << end << ":" << frequency << " " << intensity;
        for (const auto name : names) {
            of << " " << name;
        }
        of << "\n";
    }
    of.close();
}

TEST_F(StimulusTest, testStimulusWithNeuronIds) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    const auto num_neurons = local_area_translator->get_number_neurons_in_total();
    std::filesystem::path path = "stimulus.tmp";
    const auto num_rank = RandomAdapter::get_random_integer(2, 10, mt);
    const auto num_stimuli = RandomAdapter::get_random_integer(1, 10, mt) * num_rank;
    const auto num_steps = RandomAdapter::get_random_integer(1000U, 100000U, mt);
    const auto my_rank = MPIRank(RandomAdapter::get_random_integer(0, num_rank - 1, mt));

    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>> stimuli;
    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<NeuronID>>> my_stimuli;
    std::vector<Interval> intervals = StimulusAdapter::get_random_non_overlapping_intervals(num_stimuli, num_steps, mt);

    for (const auto i : ranges::views::indices(num_stimuli)) {
        Interval interval = intervals[i];

        const auto intensity = RandomAdapter::get_random_double(0.001, 100.0, mt);
        const auto& ids = NeuronIdAdapter::get_random_neuron_ids(num_neurons, RandomAdapter::get_random_integer(RelearnTypes::number_neurons_type(1), num_neurons, mt), mt);
        std::unordered_set<std::string> rank_ids{};
        std::unordered_set<NeuronID> my_ids{};
        for (const auto& neuron_id : ids) {
            const auto rank = MPIRank(RandomAdapter::get_random_integer(0, num_rank - 1, mt));
            rank_ids.insert(std::to_string(rank.get_rank()) + ":" + std::to_string(neuron_id.get_neuron_id() + 1));
            if (rank == my_rank) {
                my_ids.insert(neuron_id);
            }
        }
        stimuli.emplace_back(std::make_tuple(interval.begin, interval.end, 1U, intensity, rank_ids));
        my_stimuli.emplace_back(std::make_tuple(interval.begin, interval.end, 1U, intensity, my_ids));
    }

    write_stimuli_to_file(path, stimuli);

    RelearnTypes::stimuli_function_type stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, my_rank, local_area_translator);
    for (auto step = 0U; step < num_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);
        auto read_stimulated_neurons = size_t(0);
        auto stimulated_neurons = size_t(0);
        for (const auto& [neuron_ids, intensity] : read_stimuli) {
            read_stimulated_neurons += neuron_ids.size();
            bool found_my_stimuli = false;
            for (const auto& [begin, end, frequency, my_intensity, my_ids] : my_stimuli) {
                if (begin <= step && end >= step) {
                    if (my_ids == neuron_ids) {
                        // Same stimulus
                        stimulated_neurons += my_ids.size();
                        found_my_stimuli = true;
                        ASSERT_NEAR(my_intensity, intensity, eps);
                        break;
                    }
                }
            }
            ASSERT_TRUE(found_my_stimuli);
        }

        ASSERT_EQ(stimulated_neurons, read_stimulated_neurons);
    }
}

TEST_F(StimulusTest, testStimulusWithAreas) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    const auto num_neurons = local_area_translator->get_number_neurons_in_total();
    const auto& area_names = local_area_translator->get_all_area_names();
    std::filesystem::path path = "stimulus.tmp";
    const auto num_rank = RandomAdapter::get_random_integer(2, 10, mt);
    const auto num_stimuli = RandomAdapter::get_random_integer(1, 10, mt) * num_rank;
    const auto num_steps = RandomAdapter::get_random_integer(1000U, 100000U, mt);
    const auto my_rank = MPIRank(RandomAdapter::get_random_integer(0, num_rank - 1, mt));

    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>> stimuli;
    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<NeuronID>>> my_stimuli;
    std::vector<Interval> intervals = StimulusAdapter::get_random_non_overlapping_intervals(num_stimuli, num_steps, mt);

    for (const auto i : ranges::views::indices(num_stimuli)) {
        Interval interval = intervals[i];

        const auto intensity = RandomAdapter::get_random_double(0.001, 100.0, mt);
        const auto chosen_area_names = RandomAdapter::sample(area_names, mt);
        std::unordered_set<NeuronID> my_ids{};
        for (const auto& area_name : chosen_area_names) {
            for (const auto& neuron_id : local_area_translator->get_neuron_ids_in_area(local_area_translator->get_area_id_for_area_name(area_name))) {
                my_ids.insert(neuron_id);
            }
        }
        stimuli.emplace_back(std::make_tuple(interval.begin, interval.end, 1U, intensity, chosen_area_names | ranges::to<std::unordered_set>));
        my_stimuli.emplace_back(std::make_tuple(interval.begin, interval.end, 1U, intensity, my_ids));
    }

    write_stimuli_to_file(path, stimuli);

    RelearnTypes::stimuli_function_type stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, my_rank, local_area_translator);
    for (auto step = 0U; step < num_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);
        auto read_stimulated_neurons = size_t(0);
        auto stimulated_neurons = size_t(0);
        for (const auto& [neuron_ids, intensity] : read_stimuli) {
            read_stimulated_neurons += neuron_ids.size();
            bool found_my_stimuli = false;
            for (const auto& [begin, end, frequency, my_intensity, my_ids] : my_stimuli) {
                if (begin <= step && end >= step) {
                    if (my_ids == neuron_ids) {
                        // Same stimulus
                        stimulated_neurons += my_ids.size();
                        found_my_stimuli = true;
                        ASSERT_NEAR(my_intensity, intensity, eps);
                        break;
                    }
                }
            }
            ASSERT_TRUE(found_my_stimuli);
        }

        ASSERT_EQ(stimulated_neurons, read_stimulated_neurons);
    }
}

TEST_F(StimulusTest, testFrequency) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    const auto num_neurons = local_area_translator->get_number_neurons_in_total();
    std::filesystem::path path = "stimulus.tmp";
    const auto num_steps = RandomAdapter::get_random_integer(20000U, 100000U, mt);

    std::vector<Interval> intervals{};

    const auto begin = RandomAdapter::get_random_integer(0U, num_steps - 10000, mt);
    const auto end = RandomAdapter::get_random_integer(begin + 50, num_steps, mt);
    const auto frequency = RandomAdapter::get_random_integer(2, 10, mt);

    const auto intensity = RandomAdapter::get_random_double(0.001, 100.0, mt);
    const auto& ids = NeuronIdAdapter::get_random_neuron_ids(num_neurons, RandomAdapter::get_random_integer(RelearnTypes::number_neurons_type(1), num_neurons, mt), mt);
    std::unordered_set<std::string> rank_ids{};
    for (const auto& neuron_id : ids) {
        rank_ids.insert("0:" + std::to_string(neuron_id.get_neuron_id() + 1));
    }

    write_stimuli_to_file(path, { std::make_tuple(begin, end, frequency, intensity, rank_ids) });

    RelearnTypes::stimuli_function_type stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, MPIRank(0), local_area_translator);
    for (auto step = 0U; step < num_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);
        if (step < begin || step > end || (step - begin) % frequency != 0) {
            ASSERT_EQ(0, read_stimuli.size());
        } else {
            ASSERT_EQ(1, read_stimuli.size());
            const auto& [read_neuron_ids, read_intensity] = read_stimuli[0];
            ASSERT_EQ(ids, read_neuron_ids);
            ASSERT_FALSE(read_neuron_ids.empty());
            ASSERT_NEAR(intensity, read_intensity, eps);
        }
    }
}

TEST_F(StimulusTest, testEmptyNeurons) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    const auto num_neurons = local_area_translator->get_number_neurons_in_total();
    std::filesystem::path path = "stimulus.tmp";
    const auto num_stimuli = RandomAdapter::get_random_integer(2, 10, mt);
    const auto num_steps = RandomAdapter::get_random_integer(1000U, 100000U, mt);

    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>> stimuli;
    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<NeuronID>>> my_stimuli;
    std::vector<Interval> intervals = StimulusAdapter::get_random_non_overlapping_intervals(num_stimuli, num_steps, mt);

    for (const auto i : ranges::views::indices(num_stimuli)) {
        Interval interval = intervals[i];
        const auto intensity = RandomAdapter::get_random_double(0.001, 100.0, mt);

        const auto& ids = (i % 2 == 0) ? NeuronIdAdapter::get_random_neuron_ids(num_neurons, RandomAdapter::get_random_integer(RelearnTypes::number_neurons_type(1), num_neurons, mt), mt) : std::unordered_set<NeuronID>{};
        std::unordered_set<std::string> rank_ids{};
        for (const auto& neuron_id : ids) {
            rank_ids.insert("0:" + std::to_string(neuron_id.get_neuron_id() + 1));
        }
        stimuli.emplace_back(std::make_tuple(interval.begin, interval.end, 1U, intensity, rank_ids));
        my_stimuli.emplace_back(std::make_tuple(interval.begin, interval.end, 1U, intensity, ids));
    }

    write_stimuli_to_file(path, stimuli);

    RelearnTypes::stimuli_function_type stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, MPIRank(0), local_area_translator);
    for (auto step = 0U; step < num_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);
        auto read_stimulated_neurons = size_t(0);
        auto stimulated_neurons = size_t(0);
        for (const auto& [neuron_ids, intensity] : read_stimuli) {
            read_stimulated_neurons += neuron_ids.size();
            bool found_my_stimuli = false;
            for (const auto& [begin, end, frequency, my_intensity, my_ids] : my_stimuli) {
                if (begin <= step && end >= step) {
                    if (my_ids == neuron_ids) {
                        // Same stimulus
                        stimulated_neurons += my_ids.size();
                        found_my_stimuli = true;
                        ASSERT_NEAR(my_intensity, intensity, eps);
                        break;
                    }
                }
            }
            ASSERT_TRUE(found_my_stimuli);
        }

        ASSERT_EQ(stimulated_neurons, read_stimulated_neurons);
    }
}

TEST_F(StimulusTest, testNoFile) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    std::filesystem::path path = "stimulus.tmp";
    ASSERT_THROW(std::ignore = InteractiveNeuronIO::load_stimulus_interrupts(path, MPIRank(0), local_area_translator), RelearnException);
}

TEST_F(StimulusTest, testEmptyFile) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    std::filesystem::path path = "stimulus.tmp";

    write_stimuli_to_file(path, {});

    RelearnTypes::stimuli_function_type stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, MPIRank(0), local_area_translator);
    for (auto step = 0U; step < 1000; step++) {
        const auto& read_stimuli = stimulus_function(step);
        ASSERT_EQ(0, read_stimuli.size());
    }
}

TEST_F(StimulusTest, testInvalidNeuronId) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    const auto num_neurons = local_area_translator->get_number_neurons_in_total();
    std::filesystem::path path = "stimulus.tmp";
    const auto num_steps = RandomAdapter::get_random_integer(1000U, 100000U, mt);
    const auto intensity = RandomAdapter::get_random_double(0.001, 100.0, mt);
    std::unordered_set<std::string> rank_ids{ "0:" + std::to_string(num_neurons + 1) };
    Interval interval = StimulusAdapter::get_random_interval(num_steps, 1U, mt);

    write_stimuli_to_file(path, { std::make_tuple(interval.begin, interval.end, 1U, intensity, rank_ids) });

    ASSERT_THROW(std::ignore = InteractiveNeuronIO::load_stimulus_interrupts(path, MPIRank(0), local_area_translator), RelearnException);
}

TEST_F(StimulusTest, testInvalidAreaName) {
    const auto local_area_translator = NeuronAssignmentAdapter::get_randomized_area_translator(mt);
    std::filesystem::path path = "stimulus.tmp";

    const auto invalid_area_name = NeuronAssignmentAdapter::get_invalid_area_name(local_area_translator->get_all_area_names(), mt);
    const auto num_steps = RandomAdapter::get_random_integer(1000U, 100000U, mt);
    const auto intensity = RandomAdapter::get_random_double(0.001, 100.0, mt);
    Interval interval = StimulusAdapter::get_random_interval(num_steps, 1U, mt);

    write_stimuli_to_file(path, { std::make_tuple(interval.begin, interval.end, 1U, intensity, std::unordered_set{ invalid_area_name }) });

    RelearnTypes::stimuli_function_type stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, MPIRank(0), local_area_translator);
    for (auto step = 0U; step < num_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);
        ASSERT_EQ(0, read_stimuli.size());
    }
}