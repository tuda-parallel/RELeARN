/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cell.h"

#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "util/Vec3.h"

#include <numeric>
#include <random>

template <typename AdditionalCellAttributes>
void CellTest::test_cell_size() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min_1, max_1] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min_1, max_1);

    const auto& [res_min_1, res_max_1] = cell.get_size();

    ASSERT_EQ(min_1, res_min_1);
    ASSERT_EQ(max_1, res_max_1);

    const auto& [min_2, max_2] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min_2, max_2);

    const auto& [res_min_2, res_max_2] = cell.get_size();

    ASSERT_EQ(min_2, res_min_2);
    ASSERT_EQ(max_2, res_max_2);

    ASSERT_EQ(cell.get_maximal_dimension_difference(), (max_2 - min_2).get_maximum());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_dendrites_position() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_ex_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_dendrites_position(pos_ex_1);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());

    cell.set_excitatory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());

    const auto& pos_ex_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_dendrites_position(pos_ex_2);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());

    cell.set_excitatory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());

    const auto& pos_in_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_dendrites_position(pos_in_1);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_1, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_1, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());

    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());

    cell.set_inhibitory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());

    const auto& pos_in_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_dendrites_position(pos_in_2);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_2, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_2, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());

    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());

    cell.set_inhibitory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());

    const auto& pos_ex_3 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_dendrites_position(pos_ex_3);

    const auto& pos_in_3 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_dendrites_position(pos_in_3);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_3, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_3, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_3, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_3, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());

    const auto& pos_ex_4 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_4);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_4, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_3, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());

    const auto& pos_in_4 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_4);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_4, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_4, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());

    const auto& pos_ex_5 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_5);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_5, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_4, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());

    const auto& pos_in_5 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_5);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_5, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_5, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_5, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_5, cell.get_position_for(ElementType::Dendrite, SignalType::Inhibitory).value());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_dendrites_position_exception() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_ex_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_dendrites_position(pos_ex_1);

    const auto& pos_ex_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_ex_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_ex_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_z_max), RelearnException);

    const auto& pos_ex_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_ex_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_ex_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    const auto& pos_ex_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_dendrites_position(pos_ex_2);

    const auto& pos_ex_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_ex_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_ex_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_2_invalid_z_max), RelearnException);

    const auto& pos_ex_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_ex_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_ex_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Excitatory, pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Excitatory, pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_dendrites_position_for(SignalType::Excitatory).value());

    const auto& pos_in_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_dendrites_position(pos_in_1);

    const auto& pos_in_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_in_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_in_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_1_invalid_z_max), RelearnException);

    const auto& pos_in_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_in_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_in_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_1_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_1, cell.get_dendrites_position_for(SignalType::Inhibitory).value());

    const auto& pos_in_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_dendrites_position(pos_in_2);

    const auto& pos_in_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_in_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_in_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_2_invalid_z_max), RelearnException);

    const auto& pos_in_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_in_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_in_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_dendrites_position_for(SignalType::Inhibitory, pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Dendrite, SignalType::Inhibitory, pos_in_2_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_2, cell.get_dendrites_position_for(SignalType::Inhibitory).value());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_set_number_dendrites() {
    Cell<AdditionalCellAttributes> cell{};

    const auto num_dends_ex_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_dends_in_1 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_excitatory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_ex_1));
    cell.set_number_inhibitory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_in_1));

    ASSERT_EQ(num_dends_ex_1, cell.get_number_excitatory_dendrites());
    ASSERT_EQ(num_dends_ex_1, cell.get_number_dendrites_for(SignalType::Excitatory));
    ASSERT_EQ(num_dends_ex_1, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(num_dends_in_1, cell.get_number_inhibitory_dendrites());
    ASSERT_EQ(num_dends_in_1, cell.get_number_dendrites_for(SignalType::Inhibitory));
    ASSERT_EQ(num_dends_in_1, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Inhibitory));

    const auto num_dends_ex_2 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_dends_in_2 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_excitatory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_ex_2));
    cell.set_number_inhibitory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_in_2));

    ASSERT_EQ(num_dends_ex_2, cell.get_number_excitatory_dendrites());
    ASSERT_EQ(num_dends_ex_2, cell.get_number_dendrites_for(SignalType::Excitatory));
    ASSERT_EQ(num_dends_ex_2, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(num_dends_in_2, cell.get_number_inhibitory_dendrites());
    ASSERT_EQ(num_dends_in_2, cell.get_number_dendrites_for(SignalType::Inhibitory));
    ASSERT_EQ(num_dends_in_2, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Inhibitory));

    const auto num_dends_ex_3 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_dends_in_3 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_dendrites_for(SignalType::Excitatory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_ex_3));
    cell.set_number_dendrites_for(SignalType::Inhibitory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_in_3));

    ASSERT_EQ(num_dends_ex_3, cell.get_number_excitatory_dendrites());
    ASSERT_EQ(num_dends_ex_3, cell.get_number_dendrites_for(SignalType::Excitatory));
    ASSERT_EQ(num_dends_ex_3, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(num_dends_in_3, cell.get_number_inhibitory_dendrites());
    ASSERT_EQ(num_dends_in_3, cell.get_number_dendrites_for(SignalType::Inhibitory));
    ASSERT_EQ(num_dends_in_3, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Inhibitory));

    const auto num_dends_ex_4 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_dends_in_4 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_elements_for(ElementType::Dendrite, SignalType::Excitatory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_ex_4));
    cell.set_number_elements_for(ElementType::Dendrite, SignalType::Inhibitory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_in_4));

    ASSERT_EQ(num_dends_ex_4, cell.get_number_excitatory_dendrites());
    ASSERT_EQ(num_dends_ex_4, cell.get_number_dendrites_for(SignalType::Excitatory));
    ASSERT_EQ(num_dends_ex_4, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(num_dends_in_4, cell.get_number_inhibitory_dendrites());
    ASSERT_EQ(num_dends_in_4, cell.get_number_dendrites_for(SignalType::Inhibitory));
    ASSERT_EQ(num_dends_in_4, cell.get_number_elements_for(ElementType::Dendrite, SignalType::Inhibitory));
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_dendrites_position_combined() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    const auto& pos_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    const auto& pos_3 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    const auto& pos_4 = SimulationAdapter::get_random_position_in_box(min, max, mt);

    cell.set_dendrites_position({});

    ASSERT_FALSE(cell.get_dendrites_position().has_value());

    cell.set_excitatory_dendrites_position(pos_1);
    cell.set_inhibitory_dendrites_position(pos_1);

    ASSERT_TRUE(cell.get_dendrites_position().has_value());
    ASSERT_EQ(cell.get_dendrites_position().value(), pos_1);

    cell.set_excitatory_dendrites_position({});
    cell.set_inhibitory_dendrites_position({});

    ASSERT_FALSE(cell.get_dendrites_position().has_value());

    cell.set_excitatory_dendrites_position(pos_2);

    ASSERT_THROW(auto tmp = cell.get_dendrites_position(), RelearnException);

    cell.set_inhibitory_dendrites_position(pos_3);

    if (pos_2 == pos_3) {
        ASSERT_TRUE(cell.get_dendrites_position().has_value());
        ASSERT_EQ(cell.get_dendrites_position().value(), pos_2);
    } else {
        ASSERT_THROW(auto tmp = cell.get_dendrites_position(), RelearnException);
    }

    cell.set_dendrites_position({});

    ASSERT_FALSE(cell.get_dendrites_position().has_value());

    cell.set_excitatory_dendrites_position(pos_4);
    cell.set_inhibitory_dendrites_position(pos_4);

    ASSERT_TRUE(cell.get_dendrites_position().has_value());
    ASSERT_EQ(cell.get_dendrites_position().value(), pos_4);
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_axons_position() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_ex_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_axons_position(pos_ex_1);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_1, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());

    cell.set_excitatory_axons_position({});
    ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());

    const auto& pos_ex_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_axons_position(pos_ex_2);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_2, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());

    cell.set_excitatory_axons_position({});
    ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());

    const auto& pos_in_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_axons_position(pos_in_1);

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_1, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_1, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_1, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());

    ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());

    cell.set_inhibitory_axons_position({});
    ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());

    const auto& pos_in_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_axons_position(pos_in_2);

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_2, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_2, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_2, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());

    ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());

    cell.set_inhibitory_axons_position({});
    ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());

    ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_FALSE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_FALSE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());

    const auto& pos_ex_3 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_axons_position(pos_ex_3);

    const auto& pos_in_3 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_axons_position(pos_in_3);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_3, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_3, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_3, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_3, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());

    const auto& pos_ex_4 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_axons_position_for(SignalType::Excitatory, pos_ex_4);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_4, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_3, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_3, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());

    const auto& pos_in_4 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_axons_position_for(SignalType::Inhibitory, pos_in_4);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_4, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_4, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_4, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());

    const auto& pos_ex_5 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_5);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_5, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_4, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_4, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());

    const auto& pos_in_5 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_5);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_5, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_axons_position_for(SignalType::Excitatory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_5, cell.get_position_for(ElementType::Axon, SignalType::Excitatory).value());

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_5, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_5, cell.get_axons_position_for(SignalType::Inhibitory).value());

    ASSERT_NO_THROW(auto val = cell.get_position_for(ElementType::Axon, SignalType::Inhibitory));
    ASSERT_TRUE(cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_5, cell.get_position_for(ElementType::Axon, SignalType::Inhibitory).value());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_axons_position_exception() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_ex_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_axons_position(pos_ex_1);

    const auto& pos_ex_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_ex_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_ex_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_1_invalid_z_max), RelearnException);

    const auto& pos_ex_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_ex_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_ex_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_1, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_axons_position_for(SignalType::Excitatory).value());

    const auto& pos_ex_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_excitatory_axons_position(pos_ex_2);

    const auto& pos_ex_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_ex_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_ex_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_2_invalid_z_max), RelearnException);

    const auto& pos_ex_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_ex_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_ex_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_axons_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Excitatory, pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Excitatory, pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
    ASSERT_EQ(pos_ex_2, cell.get_excitatory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Excitatory).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_axons_position_for(SignalType::Excitatory).value());

    const auto& pos_in_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_axons_position(pos_in_1);

    const auto& pos_in_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_in_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_in_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_1_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_1_invalid_z_max), RelearnException);

    const auto& pos_in_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_in_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_in_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_1_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_1_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_1, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_1, cell.get_axons_position_for(SignalType::Inhibitory).value());

    const auto& pos_in_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    cell.set_inhibitory_axons_position(pos_in_2);

    const auto& pos_in_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_in_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_in_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_2_invalid_z_max), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_2_invalid_z_max), RelearnException);

    const auto& pos_in_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_in_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_in_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_axons_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_neuron_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_axons_position_for(SignalType::Inhibitory, pos_in_2_invalid_z_min), RelearnException);

    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_position_for(ElementType::Axon, SignalType::Inhibitory, pos_in_2_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
    ASSERT_EQ(pos_in_2, cell.get_inhibitory_axons_position().value());

    ASSERT_TRUE(cell.get_axons_position_for(SignalType::Inhibitory).has_value());
    ASSERT_EQ(pos_in_2, cell.get_axons_position_for(SignalType::Inhibitory).value());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_set_number_axons() {
    Cell<AdditionalCellAttributes> cell{};

    const auto num_axs_ex_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_axs_in_1 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_excitatory_axons(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_ex_1));
    cell.set_number_inhibitory_axons(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_in_1));

    ASSERT_EQ(num_axs_ex_1, cell.get_number_excitatory_axons());
    ASSERT_EQ(num_axs_ex_1, cell.get_number_axons_for(SignalType::Excitatory));
    ASSERT_EQ(num_axs_ex_1, cell.get_number_elements_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(num_axs_in_1, cell.get_number_inhibitory_axons());
    ASSERT_EQ(num_axs_in_1, cell.get_number_axons_for(SignalType::Inhibitory));
    ASSERT_EQ(num_axs_in_1, cell.get_number_elements_for(ElementType::Axon, SignalType::Inhibitory));

    const auto num_axs_ex_2 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_axs_in_2 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_excitatory_axons(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_ex_2));
    cell.set_number_inhibitory_axons(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_in_2));

    ASSERT_EQ(num_axs_ex_2, cell.get_number_excitatory_axons());
    ASSERT_EQ(num_axs_ex_2, cell.get_number_axons_for(SignalType::Excitatory));
    ASSERT_EQ(num_axs_ex_2, cell.get_number_elements_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(num_axs_in_2, cell.get_number_inhibitory_axons());
    ASSERT_EQ(num_axs_in_2, cell.get_number_axons_for(SignalType::Inhibitory));
    ASSERT_EQ(num_axs_in_2, cell.get_number_elements_for(ElementType::Axon, SignalType::Inhibitory));

    const auto num_axs_ex_3 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_axs_in_3 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_axons_for(SignalType::Excitatory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_ex_3));
    cell.set_number_axons_for(SignalType::Inhibitory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_in_3));

    ASSERT_EQ(num_axs_ex_3, cell.get_number_excitatory_axons());
    ASSERT_EQ(num_axs_ex_3, cell.get_number_axons_for(SignalType::Excitatory));
    ASSERT_EQ(num_axs_ex_3, cell.get_number_elements_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(num_axs_in_3, cell.get_number_inhibitory_axons());
    ASSERT_EQ(num_axs_in_3, cell.get_number_axons_for(SignalType::Inhibitory));
    ASSERT_EQ(num_axs_in_3, cell.get_number_elements_for(ElementType::Axon, SignalType::Inhibitory));

    const auto num_axs_ex_4 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto num_axs_in_4 = NeuronIdAdapter::get_random_number_neurons(mt);

    cell.set_number_elements_for(ElementType::Axon, SignalType::Excitatory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_ex_4));
    cell.set_number_elements_for(ElementType::Axon, SignalType::Inhibitory, static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_axs_in_4));

    ASSERT_EQ(num_axs_ex_4, cell.get_number_excitatory_axons());
    ASSERT_EQ(num_axs_ex_4, cell.get_number_axons_for(SignalType::Excitatory));
    ASSERT_EQ(num_axs_ex_4, cell.get_number_elements_for(ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(num_axs_in_4, cell.get_number_inhibitory_axons());
    ASSERT_EQ(num_axs_in_4, cell.get_number_axons_for(SignalType::Inhibitory));
    ASSERT_EQ(num_axs_in_4, cell.get_number_elements_for(ElementType::Axon, SignalType::Inhibitory));
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_axons_position_combined() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_1 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    const auto& pos_2 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    const auto& pos_3 = SimulationAdapter::get_random_position_in_box(min, max, mt);
    const auto& pos_4 = SimulationAdapter::get_random_position_in_box(min, max, mt);

    cell.set_axons_position({});

    ASSERT_FALSE(cell.get_axons_position().has_value());

    cell.set_excitatory_axons_position(pos_1);
    cell.set_inhibitory_axons_position(pos_1);

    ASSERT_TRUE(cell.get_axons_position().has_value());
    ASSERT_EQ(cell.get_axons_position().value(), pos_1);

    cell.set_excitatory_axons_position({});
    cell.set_inhibitory_axons_position({});

    ASSERT_FALSE(cell.get_axons_position().has_value());

    cell.set_excitatory_axons_position(pos_2);

    ASSERT_THROW(auto tmp = cell.get_axons_position(), RelearnException);

    cell.set_inhibitory_axons_position(pos_3);

    if (pos_2 == pos_3) {
        ASSERT_TRUE(cell.get_axons_position().has_value());
        ASSERT_EQ(cell.get_axons_position().value(), pos_2);
    } else {
        ASSERT_THROW(auto tmp = cell.get_axons_position(), RelearnException);
    }

    cell.set_axons_position({});

    ASSERT_FALSE(cell.get_axons_position().has_value());

    cell.set_excitatory_axons_position(pos_4);
    cell.set_inhibitory_axons_position(pos_4);

    ASSERT_TRUE(cell.get_axons_position().has_value());
    ASSERT_EQ(cell.get_axons_position().value(), pos_4);
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_set_neuron_id() {
    Cell<AdditionalCellAttributes> cell{};

    const auto neuron_id_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto id1 = NeuronID{ neuron_id_1 };
    cell.set_neuron_id(id1);
    ASSERT_EQ(id1, cell.get_neuron_id());

    const auto neuron_id_2 = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto id2 = NeuronID{ neuron_id_2 };
    cell.set_neuron_id(id2);
    ASSERT_EQ(id2, cell.get_neuron_id());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_octants() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto midpoint = (min + max) / 2;

    for (auto id = 0; id < 1000; id++) {
        const auto& position = SimulationAdapter::get_random_position_in_box(min, max, mt);

        const auto larger_x = position.get_x() >= midpoint.get_x() ? 1 : 0;
        const auto larger_y = position.get_y() >= midpoint.get_y() ? 2 : 0;
        const auto larger_z = position.get_z() >= midpoint.get_z() ? 4 : 0;

        const auto expected_octant_idx = larger_x + larger_y + larger_z;

        const auto received_idx = cell.get_octant_for_position(position);

        ASSERT_EQ(expected_octant_idx, received_idx);
    }
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_octants_exception() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto& pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    const auto& pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_x_max), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_y_max), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_z_max), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_x_min), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_y_min), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_z_min), RelearnException);
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_octants_size() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    cell.set_size(min, max);

    const auto midpoint = (min + max) / 2;

    for (auto id = 0; id < 8; id++) {
        const auto larger_x = ((id & 1) == 0) ? 0 : 1;
        const auto larger_y = ((id & 2) == 0) ? 0 : 1;
        const auto larger_z = ((id & 4) == 0) ? 0 : 1;

        auto subcell_min = min;
        auto subcell_max = midpoint;

        if (larger_x == 1) {
            subcell_min += Vec3d{ midpoint.get_x() - min.get_x(), 0, 0 };
            subcell_max += Vec3d{ midpoint.get_x() - min.get_x(), 0, 0 };
        }

        if (larger_y == 1) {
            subcell_min += Vec3d{ 0, midpoint.get_y() - min.get_y(), 0 };
            subcell_max += Vec3d{ 0, midpoint.get_y() - min.get_y(), 0 };
        }

        if (larger_z == 1) {
            subcell_min += Vec3d{ 0, 0, midpoint.get_z() - min.get_z() };
            subcell_max += Vec3d{ 0, 0, midpoint.get_z() - min.get_z() };
        }

        const auto& [subcell_received_min, subcell_received_max] = cell.get_size_for_octant(id);

        const auto diff_subcell_min = subcell_min - subcell_received_min;
        const auto diff_subcell_max = subcell_max - subcell_received_max;

        ASSERT_NEAR(diff_subcell_min.calculate_p_norm(2), 0.0, eps);
        ASSERT_NEAR(diff_subcell_max.calculate_p_norm(2), 0.0, eps);
    }
}

template <typename VirtualPlasticityElement>
void CellTest::test_vpe_number_elements() {
    VirtualPlasticityElement vpe{};

    const auto& number_initially_free_elements = vpe.get_number_free_elements();
    ASSERT_EQ(number_initially_free_elements, 0) << number_initially_free_elements;

    const auto nfe_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    vpe.set_number_free_elements(static_cast<typename VirtualPlasticityElement::counter_type>(nfe_1));

    const auto& number_free_elements_1 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_1, nfe_1) << number_free_elements_1 << ' ' << nfe_1;

    const auto nfe_2 = NeuronIdAdapter::get_random_number_neurons(mt);
    vpe.set_number_free_elements(static_cast<typename VirtualPlasticityElement::counter_type>(nfe_2));

    const auto& number_free_elements_2 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_2, nfe_2) << number_free_elements_2 << ' ' << nfe_2;
}

template <typename VirtualPlasticityElement>
void CellTest::test_vpe_position() {
    VirtualPlasticityElement vpe{};

    const auto& initial_position = vpe.get_position();
    ASSERT_FALSE(initial_position.has_value());

    const auto& [pos_1, pos_3] = SimulationAdapter::get_random_simulation_box_size(mt);

    vpe.set_position(pos_1);
    const auto& position_1 = vpe.get_position();
    ASSERT_TRUE(position_1.has_value());
    ASSERT_EQ(position_1.value(), pos_1);

    vpe.set_position({});
    const auto& position_2 = vpe.get_position();
    ASSERT_FALSE(position_2.has_value());

    vpe.set_position(pos_3);
    const auto& position_3 = vpe.get_position();
    ASSERT_TRUE(position_3.has_value());
    ASSERT_EQ(position_3.value(), pos_3);
}

template <typename VirtualPlasticityElement>
void CellTest::test_vpe_mixed() {
    VirtualPlasticityElement vpe{};

    const auto& initial_position = vpe.get_position();
    ASSERT_FALSE(initial_position.has_value());

    const auto& [pos_1, pos_3] = SimulationAdapter::get_random_simulation_box_size(mt);

    vpe.set_position(pos_1);
    const auto& position_1 = vpe.get_position();
    ASSERT_TRUE(position_1.has_value());
    ASSERT_EQ(position_1.value(), pos_1);

    const auto& number_initially_free_elements = vpe.get_number_free_elements();
    ASSERT_EQ(number_initially_free_elements, 0) << number_initially_free_elements;

    const auto nfe_1 = NeuronIdAdapter::get_random_number_neurons(mt);
    vpe.set_number_free_elements(nfe_1);

    const auto& number_free_elements_1 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_1, nfe_1) << number_free_elements_1 << ' ' << nfe_1;

    vpe.set_position({});
    const auto& position_2 = vpe.get_position();
    ASSERT_FALSE(position_2.has_value());

    vpe.set_position(pos_3);
    const auto& position_3 = vpe.get_position();
    ASSERT_TRUE(position_3.has_value());
    ASSERT_EQ(position_3.value(), pos_3);

    const auto nfe_2 = NeuronIdAdapter::get_random_number_neurons(mt);
    vpe.set_number_free_elements(nfe_2);

    const auto& number_free_elements_2 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_2, nfe_2) << number_free_elements_2 << ' ' << nfe_2;
}

TEST_F(CellTest, testBarnesHutCellSize) {
    test_cell_size<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellDendritesPosition) {
    test_cell_dendrites_position<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellDendritesPositionException) {
    test_cell_dendrites_position_exception<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellDendritesPositionCombined) {
    test_cell_dendrites_position_combined<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellSetNumberDendrites) {
    test_cell_set_number_dendrites<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellSetNeuronId) {
    test_cell_set_neuron_id<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellOctants) {
    test_cell_octants<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellOctantsException) {
    test_cell_octants_exception<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellOctantsSize) {
    test_cell_octants_size<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellSize) {
    test_cell_size<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellAxonsPosition) {
    test_cell_axons_position<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellAxonsPositionException) {
    test_cell_axons_position_exception<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellAxonsPositionCombined) {
    test_cell_axons_position_combined<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellSetNumberAxons) {
    test_cell_set_number_axons<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellSetNeuronId) {
    test_cell_set_neuron_id<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellOctants) {
    test_cell_octants<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellOctantsException) {
    test_cell_octants_exception<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testBarnesHutInvertedCellOctantsSize) {
    test_cell_octants_size<BarnesHutInvertedCell>();
}

TEST_F(CellTest, testVPEManualNumberFreeElements) {
    test_vpe_number_elements<VirtualPlasticityElementManual>();
}

TEST_F(CellTest, testVPEManualPosition) {
    test_vpe_position<VirtualPlasticityElementManual>();
}

TEST_F(CellTest, testVPEOptionalNumberFreeElements) {
    test_vpe_number_elements<VirtualPlasticityElementOptional>();
}

TEST_F(CellTest, testVPEOptionalPosition) {
    test_vpe_position<VirtualPlasticityElementOptional>();
}
