#include "test_neurons.h"

#include "adapter/mpi/MpiAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/input/BackgroundActivityCalculators.h"
#include "neurons/input/Stimulus.h"
#include "neurons/input/SynapticInputCalculator.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "neurons/Neurons.h"
#include "adapter/random/RandomAdapter.h"
#include "structure/Partition.h"
#include "util/Utility.h"
#include "util/ranges/Functional.hpp"

#include <span>
#include <vector>

#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/generate_n.hpp>

TEST_F(NeuronsTest, testNeuronsConstructor) {
    auto partition = std::make_shared<Partition>(1, MPIRank::root_rank());

    auto model = std::make_unique<models::PoissonModel>();
    auto calcium = std::make_unique<CalciumCalculator>();
    auto network_graph = std::make_shared<NetworkGraph>(MPIRank::root_rank());
    auto dends_ex = std::make_shared<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_shared<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_shared<SynapticElements>(ElementType::Axon, 0.2);

    auto sdf = std::make_unique<RandomSynapseDeletionFinder>();
    sdf->set_axons(axs);
    sdf->set_dendrites_ex(dends_ex);
    sdf->set_dendrites_in(dends_in);

    Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(network_graph), std::move(axs), std::move(dends_ex),
        std::move(dends_in), std::move(sdf) };
}

TEST_F(NeuronsTest, testSignalTypeCheck) {
    const auto num_test_synapses = RandomAdapter::get_random_integer(50, 500, mt);
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 10;
    const auto num_synapses = RandomAdapter::get_random_integer(10, 100, mt);
    const auto num_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto network_graph = std::make_shared<NetworkGraph>(MPIRank::root_rank());
    network_graph->init(num_neurons);

    const auto signal_types = ranges::views::generate_n([this]() { return NeuronTypesAdapter::get_random_signal_type(mt); }, num_neurons)
        | ranges::to_vector;

    for (int synapse_nr = 0; synapse_nr < num_synapses; synapse_nr++) {
        auto weight = RandomAdapter::get_random_double(0.1, 20.0, mt);
        const auto src = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
        const auto tgt = NeuronIdAdapter::get_random_neuron_id(num_neurons, src, mt);
        const auto tgt_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);
        if (signal_types[src.get_neuron_id()] == SignalType::Inhibitory) {
            weight = -weight;
        }
        if (tgt_rank == MPIRank::root_rank()) {
            network_graph->add_synapse(StaticLocalSynapse{ tgt, src, weight });
        } else {
            network_graph->add_synapse(StaticDistantOutSynapse{ RankNeuronId(tgt_rank, tgt), src, weight });
        }
    }

    ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, MPIRank::root_rank()));

    for (int test_synapse = 0; test_synapse < num_test_synapses; test_synapse++) {
        const auto src = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
        const auto tgt_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);
        const auto tgt = NeuronIdAdapter::get_random_neuron_id(num_neurons, src, mt);
        const RankNeuronId tgt_rni(tgt_rank, tgt);
        auto weight = std::abs(NetworkGraphAdapter::get_random_plastic_synapse_weight(mt));
        if (signal_types[src.get_neuron_id()] == SignalType::Excitatory) {
            weight = -weight;
        }

        if (tgt_rank == MPIRank::root_rank()) {
            network_graph->add_synapse(PlasticLocalSynapse{ tgt, src, weight });
        } else {
            network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId(tgt_rank, tgt), src, weight });
        }
        ASSERT_THROW(Neurons::check_signal_types(network_graph, signal_types, MPIRank::root_rank()), RelearnException);
        if (tgt_rank == MPIRank::root_rank()) {
            network_graph->add_synapse(PlasticLocalSynapse{ tgt, src, -weight });
        } else {
            network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId(tgt_rank, tgt), src, -weight });
        }
        ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, MPIRank::root_rank()));
    }
}

TEST_F(NeuronsTest, testStaticConnectionsChecker) {
    auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 30;
    auto num_static_neurons = RandomAdapter::get_random_integer(15, static_cast<int>(num_neurons) - 10, mt);

    std::vector<NeuronID> static_neurons{};
    for (const auto i : NeuronID::range_id(num_static_neurons)) {
        NeuronID static_neuron;
        do {
            static_neuron = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
        } while (ranges::contains(static_neurons, static_neuron));
        static_neurons.emplace_back(static_neuron);
    }

    auto partition = std::make_shared<Partition>(1, MPIRank::root_rank());
    auto model = std::make_unique<models::PoissonModel>(models::PoissonModel::default_h,
        std::make_unique<LinearSynapticInputCalculator>(SynapticInputCalculator::default_conductance, std::make_unique<FiredStatusCommunicationMap>(1)),
        std::make_unique<NullBackgroundActivityCalculator>(),
        std::make_unique<Stimulus>(),
        models::PoissonModel::default_x_0,
        models::PoissonModel::default_tau_x,
        models::PoissonModel::default_refractory_period);

    auto calcium = std::make_unique<CalciumCalculator>();
    calcium->set_initial_calcium_calculator(
        [](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    calcium->set_target_calcium_calculator(
        [](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    auto dends_ex = std::make_shared<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_shared<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_shared<SynapticElements>(ElementType::Axon, 0.2);

    auto sdf = std::make_unique<RandomSynapseDeletionFinder>();
    sdf->set_axons(axs);
    sdf->set_dendrites_ex(dends_ex);
    sdf->set_dendrites_in(dends_in);

    auto network_graph = std::make_shared<NetworkGraph>(MPIRank::root_rank());

    Neurons neurons{ partition, std::move(model), std::move(calcium), network_graph, std::move(axs), std::move(dends_ex),
        std::move(dends_in), std::move(sdf) };
    neurons.init(num_neurons);

    auto num_synapses_static = RandomAdapter::get_random_integer(30, 100, mt);
    auto num_synapses_plastic = RandomAdapter::get_random_integer(30, 100, mt);

    for (const auto i : ranges::views::indices(num_synapses_static)) {
        auto src = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
        auto tgt = NeuronIdAdapter::get_random_neuron_id(num_neurons, NeuronID(src), mt);
        const auto weight = std::abs(NetworkGraphAdapter::get_random_static_synapse_weight(mt));
        network_graph->add_synapse(StaticLocalSynapse{ tgt, src, weight });
    }

    for (const auto i : ranges::views::indices(num_synapses_plastic)) {
        NeuronID src, tgt;
        do {
            src = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
            tgt = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
        } while (ranges::contains(static_neurons, src) || ranges::contains(static_neurons, tgt) || src == tgt);
        const auto weight = std::abs(NetworkGraphAdapter::get_random_plastic_synapse_weight(mt));
        network_graph->add_synapse(PlasticLocalSynapse{ NeuronID{ tgt }, NeuronID{ src }, weight });
    }

    neurons.set_static_neurons(static_neurons);

    const auto num_tries = RandomAdapter::get_random_integer(10, 100, mt);
    for (const auto i : ranges::views::indices(num_tries)) {
        const bool src_is_static = RandomAdapter::get_random_bool(mt);
        const bool tgt_is_static = !src_is_static || RandomAdapter::get_random_bool(mt);

        NeuronID src, tgt;
        if (src_is_static) {
            src = static_neurons[RandomAdapter::get_random_integer(size_t(0), static_neurons.size() - 1, mt)];
        } else {
            do {
                src = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
            } while (ranges::contains(static_neurons, src));
        }
        if (tgt_is_static) {
            tgt = static_neurons[RandomAdapter::get_random_integer(size_t(0), static_neurons.size() - 1, mt)];
        } else {
            do {
                tgt = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
            } while (ranges::contains(static_neurons, tgt));
        }
        if (tgt == src)
            continue;

        const auto weight = NetworkGraphAdapter::get_random_plastic_synapse_weight(mt);

        network_graph->add_synapse(PlasticLocalSynapse{ tgt, src, weight });

        ASSERT_THROW(neurons.set_static_neurons(static_neurons), RelearnException);
        network_graph->add_synapse(PlasticLocalSynapse{ tgt, src, -weight });
        neurons.set_static_neurons(static_neurons);
    }
}

TEST_F(NeuronsTest, testDisableNeuronsWithoutMPI) {
    auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 30;

    auto partition = std::make_shared<Partition>(1, MPIRank::root_rank());
    auto [neurons, network_graph] = create_neurons_object(partition, MPIRank::root_rank());

    neurons->init(num_neurons);
    NetworkGraphAdapter::create_dense_plastic_network(network_graph, neurons->get_axons().get_signal_types(),
        num_neurons, 8, 1, MPIRank(0), mt);
    neurons->init_synaptic_elements({}, {}, {});

    const auto disable_id = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
    const auto disabled_neurons = std::vector<NeuronID>{ disable_id };
    const auto enabled_neurons = std::vector<NeuronID>{ disable_id };

    const auto [out_edges_ref, _1] = network_graph->get_local_out_edges(disable_id);
    auto out_edges = out_edges_ref;
    ASSERT_GT(out_edges.size(), 0);

    const auto [in_edges_ref, _2] = network_graph->get_local_in_edges(disable_id);
    auto in_edges = in_edges_ref;
    ASSERT_GT(in_edges.size(), 0);

    const std::set<NeuronID> out_ids = out_edges | ranges::views::keys | ranges::to<std::set>;
    const std::set<NeuronID> in_ids = in_edges | ranges::views::keys | ranges::to<std::set>;

    ASSERT_THROW(neurons->enable_neurons(enabled_neurons), RelearnException);

    const auto& [num_deletions,number_deleted_distant_out_axons , number_deleted_distant_in
            , number_deleted_in_edges_from_outside , number_deleted_out_edges_to_outside
            , number_deleted_out_edges_within, synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons, 1);
    ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(), 0);
    ASSERT_EQ(num_deletions, out_edges.size() + in_edges.size());

    const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests, MPIRank(0));
    ASSERT_EQ(num_distant_deletions, 0);

    ASSERT_THROW(neurons->disable_neurons(1, disabled_neurons, 1), RelearnException);

    ASSERT_EQ(out_edges_ref.size(), 0);
    ASSERT_EQ(in_edges_ref.size(), 0);

    for (const auto& id : out_ids) {
        auto [edges, _1] = network_graph->get_local_in_edges(id);

        const bool contains = ranges::all_of(edges, not_equal_to(disable_id), element<0>);
        ASSERT_TRUE(contains);

        ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[id.get_neuron_id()], UpdateStatus::Enabled);
    }

    for (const auto& id : in_ids) {
        auto [edges, _1] = network_graph->get_local_out_edges(id);

        const bool contains = ranges::all_of(edges, not_equal_to(disable_id), element<0>);
        ASSERT_TRUE(contains);

        ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[id.get_neuron_id()], UpdateStatus::Enabled);
        ASSERT_EQ(neurons->get_axons().get_connected_elements(id), 7);
    }

    ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()], UpdateStatus::Disabled);
    ASSERT_EQ(neurons->get_axons().get_connected_elements(disable_id), 0);
    ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(disable_id), 0);
    ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(disable_id), 0);

    const std::vector<std::vector<SignalType>> signal_types{ neurons->get_axons().get_signal_types() | ranges::to_vector };
    NetworkGraphAdapter::check_validity_of_network_graphs({ network_graph }, signal_types, num_neurons);
}

TEST_F(NeuronsTest, testDisableMultipleNeuronsWithoutMPI) {
    auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 30;

    auto partition = std::make_shared<Partition>(1, MPIRank::root_rank());
    auto [neurons, network_graph] = create_neurons_object(partition, MPIRank::root_rank());

    neurons->init(num_neurons);

    const auto mpi_rank = MPIRank{ 0 };
    const auto disabled_neurons = NeuronIdAdapter::get_random_neuron_ids(num_neurons, 15, mt);
    std::vector<RankNeuronId> disabled_rank_neurons = disabled_neurons | ranges::views::transform([&mpi_rank](const auto& neuron_id) { return RankNeuronId(mpi_rank, neuron_id); }) | ranges::to_vector;
    for (const auto& neuron1 : disabled_neurons) {
        for (const auto& neuron2 : disabled_neurons) {
            if (neuron1 == neuron2) {
                continue;
            }
            const auto weight = neurons->get_axons().get_signal_type(neuron1) == SignalType::Excitatory ? 1 : -1;
            network_graph->add_synapse(PlasticLocalSynapse(neuron2, neuron1, weight));
        }
    }
    NetworkGraphAdapter::create_dense_plastic_network(network_graph, neurons->get_axons().get_signal_types(),
                                                      num_neurons, 8, 1, MPIRank(0), mt);

    const std::vector<NeuronID> disabled_neurons_vector = disabled_neurons | ranges::to_vector;

    std::vector<double> to_delete_axons{};
    std::vector<double> to_delete_den_ex{};
    std::vector<double> to_delete_den_inh{};
    to_delete_axons.resize(num_neurons, 0.0);
    to_delete_den_ex.resize(num_neurons, 0.0);
    to_delete_den_inh.resize(num_neurons, 0.0);
    auto expected_num_deletions = 0;
    auto expected_number_deleted_in_edges_from_outside =0;
    auto expected_number_deleted_out_edges_to_outside =0;
    auto expected_number_deleted_out_edges_within = 0;
    for (const auto& disabled_id : disabled_neurons) {
        const auto [plastic_local_out_edges, _1] = network_graph->get_local_out_edges(disabled_id);
        for (const auto& [target, weight] : plastic_local_out_edges) {
            expected_num_deletions+=std::abs(weight);
            if (weight > 0) {
                to_delete_den_ex[target.get_neuron_id()]+=std::abs(weight);

            } else {
                to_delete_den_inh[target.get_neuron_id()]+=std::abs(weight);
            }

            if (!disabled_neurons.contains(target)) {
                expected_number_deleted_out_edges_to_outside+=std::abs(weight);
            }
            else {
                expected_number_deleted_out_edges_within+=std::abs(weight);
            }
        }

        const auto [in_edges, _4] = network_graph->get_local_in_edges(disabled_id);
        for (const auto& [source, weight] : in_edges) {
            to_delete_axons[source.get_neuron_id()] += std::abs(weight);
            if (!disabled_neurons.contains(source)) {
                expected_num_deletions+=std::abs(weight);
                expected_number_deleted_in_edges_from_outside += std::abs(weight);
            }
        }
    }

    neurons->init_synaptic_elements({}, {}, {});

    const auto axons = neurons->get_axons().get_connected_elements();
    const auto den_ex = neurons->get_dendrites_exc().get_connected_elements();
    const auto den_inh = neurons->get_dendrites_inh().get_connected_elements();
    std::vector<unsigned int> axons_old = axons | ranges::to_vector;
    std::vector<unsigned int> den_ex_old = den_ex | ranges::to_vector;
    std::vector<unsigned int> den_inh_old = den_inh | ranges::to_vector;

    const auto& [num_deletions,number_deleted_distant_out_axons , number_deleted_distant_in
            , number_deleted_in_edges_from_outside , number_deleted_out_edges_to_outside
            , number_deleted_out_edges_within, synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons_vector, 1);
    ASSERT_EQ(0,number_deleted_distant_in);
    ASSERT_EQ(0,number_deleted_distant_out_axons);
    ASSERT_EQ(expected_number_deleted_in_edges_from_outside, number_deleted_in_edges_from_outside);
    ASSERT_EQ(expected_number_deleted_out_edges_within, number_deleted_out_edges_within);
    ASSERT_EQ(expected_number_deleted_out_edges_to_outside, number_deleted_out_edges_to_outside);
    ASSERT_EQ(num_deletions,number_deleted_distant_out_axons + number_deleted_distant_in
                            + number_deleted_in_edges_from_outside + number_deleted_out_edges_to_outside
                            + number_deleted_out_edges_within );

    ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(), 0);
    ASSERT_EQ(num_deletions, expected_num_deletions);
    const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests, MPIRank(0));
    ASSERT_EQ(num_distant_deletions, 0);
    for (const auto& disable_id : disabled_neurons) {
        ASSERT_EQ(std::get<0>(network_graph->get_local_in_edges(disable_id)).size(), 0);
        ASSERT_EQ(std::get<0>(network_graph->get_local_out_edges(disable_id)).size(), 0);
        ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()], UpdateStatus::Disabled);
        ASSERT_EQ(neurons->get_axons().get_connected_elements(disable_id), 0);
        ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(disable_id), 0);
        ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(disable_id), 0);
    }

    for (const auto& neuron_id : NeuronID::range(num_neurons)) {
        if (disabled_neurons.contains(neuron_id)) {
            continue;
        }

        ASSERT_EQ(neurons->get_axons().get_connected_elements(neuron_id),
            axons_old[neuron_id.get_neuron_id()] - to_delete_axons[neuron_id.get_neuron_id()]);
        ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(neuron_id),
            den_ex_old[neuron_id.get_neuron_id()] - to_delete_den_ex[neuron_id.get_neuron_id()]);
        ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(neuron_id),
            den_inh_old[neuron_id.get_neuron_id()] - to_delete_den_inh[neuron_id.get_neuron_id()]);
    }

    const std::vector<std::vector<SignalType>> signal_types{ neurons->get_axons().get_signal_types() | ranges::to_vector };
    NetworkGraphAdapter::check_validity_of_network_graphs({network_graph }, signal_types, num_neurons);
}

TEST_F(NeuronsTest, testDisableNeuronsWithRanks) {
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 30;
    const auto num_ranks = MPIRankAdapter::get_random_number_ranks(mt) + 1;

    std::vector<std::shared_ptr<Neurons>> rank_to_neurons;
    std::vector<std::shared_ptr<NetworkGraph>> network_graphs;
    std::vector<std::unordered_set<NeuronID>> rank_to_disabled_neurons;
    std::vector<size_t> expected_distant_out_deletions_received{};
    std::vector<size_t> expected_distant_in_deletions_received{};
    expected_distant_out_deletions_received.resize(num_ranks, 0);
    expected_distant_in_deletions_received.resize(num_ranks, 0);
    std::vector<size_t> expected_distant_out_deletions_initiated{};
    std::vector<size_t> expected_distant_in_deletions_initiated{};
    expected_distant_out_deletions_initiated.resize(num_ranks, 0);
    expected_distant_in_deletions_initiated.resize(num_ranks, 0);
    for (int rank = 0; rank < num_ranks; rank++) {
        const MPIRank mpi_rank{ rank };

        auto partition = std::make_shared<Partition>(1, MPIRank(0));
        auto [neurons, network_graph_plastic] = create_neurons_object(partition, mpi_rank);

        neurons->init(num_neurons);

        const auto disabled_neurons = NeuronIdAdapter::get_random_neuron_ids(num_neurons, 15, mt);
        for (const auto& neuron1 : disabled_neurons) {
            for (const auto& neuron2 : disabled_neurons) {
                if (neuron1 == neuron2) {
                    continue;
                }
                const auto target_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);
                const auto weight = neurons->get_axons().get_signal_type(neuron1) == SignalType::Excitatory ? 1 : -1;

                if (target_rank == mpi_rank) {
                    network_graph_plastic->add_synapse(PlasticLocalSynapse(neuron2, neuron1, weight));
                } else {
                    network_graph_plastic->add_synapse(
                        PlasticDistantOutSynapse(RankNeuronId(target_rank, neuron2), neuron1, weight));
                }
            }
        }
        NetworkGraphAdapter::create_dense_plastic_network(network_graph_plastic, neurons->get_axons().get_signal_types(),
            num_neurons, 8, num_ranks, mpi_rank, mt);

        Neurons::check_signal_types(network_graph_plastic, neurons->get_axons().get_signal_types(), mpi_rank);

        rank_to_neurons.push_back(std::move(neurons));
        rank_to_disabled_neurons.push_back(disabled_neurons);
        network_graphs.push_back(network_graph_plastic);
    }
    NetworkGraphAdapter::harmonize_network_graphs_from_different_ranks(network_graphs, num_neurons);

    std::vector<std::vector<SignalType>> signal_types = rank_to_neurons | ranges::views::transform([](const auto& neurons_of_rank) { return neurons_of_rank->get_axons().get_signal_types() | ranges::to_vector; }) | ranges::to_vector;

    std::vector<std::vector<unsigned int>> expected_axons{};
    std::vector<std::vector<unsigned int>> expected_den_ex{};
    std::vector<std::vector<unsigned int>> expected_den_inh{};
    expected_axons.resize(num_ranks);
    expected_den_ex.resize(num_ranks);
    expected_den_inh.resize(num_ranks);
    for (auto rank = 0; rank < num_ranks; rank++) {
        auto& neurons = rank_to_neurons[rank];
        neurons->init_synaptic_elements({}, {}, {});

        const auto axons = neurons->get_axons().get_connected_elements();
        const auto den_ex = neurons->get_dendrites_exc().get_connected_elements();
        const auto den_inh = neurons->get_dendrites_inh().get_connected_elements();
        expected_axons[rank] = axons | ranges::to_vector;
        expected_den_ex[rank] = den_ex | ranges::to_vector;
        expected_den_inh[rank] = den_inh | ranges::to_vector;
    }

    std::vector<CommunicationMap<SynapseDeletionRequest>> outgoing_requests;

    for (int rank = 0; rank < num_ranks; rank++) {
        const auto& network_graph = network_graphs[rank];
        const auto& disabled_neurons = rank_to_disabled_neurons[rank];
        std::vector<NeuronID> disabled_neurons_vector = disabled_neurons | ranges::to_vector;
        auto& neurons = rank_to_neurons[rank];

        auto expected_num_local_deletions = 0;
        auto expected_num_distant_in_deletions = 0;
        auto expected_num_distant_out_deletions = 0;

        for (const auto neuron_id : rank_to_disabled_neurons[rank]) {
            const auto& [distant_out_edges, _3] = network_graph->get_distant_out_edges(neuron_id);
            for (const auto& [target, weight] : distant_out_edges) {
                expected_distant_out_deletions_initiated[rank]++;
                expected_distant_in_deletions_received[target.get_rank().get_rank()]++;
                expected_num_distant_out_deletions += std::abs(weight);
                if (weight > 0) {
                    expected_den_ex[target.get_rank().get_rank()][target.get_neuron_id().get_neuron_id()]--;
                } else {
                    expected_den_inh[target.get_rank().get_rank()][target.get_neuron_id().get_neuron_id()]--;
                }
            }

            const auto& [distant_in_edges, _2] = network_graph->get_distant_in_edges(neuron_id);

            for (const auto& [source,weight] : distant_in_edges) {
                expected_distant_in_deletions_initiated[rank] ++;
                expected_distant_out_deletions_received[source.get_rank().get_rank()] ++;
                expected_num_distant_in_deletions += std::abs(weight);
                expected_axons[source.get_rank().get_rank()][source.get_neuron_id().get_neuron_id()]-= std::abs(weight);
            }

            const auto [out_edges, _1] = network_graph->get_local_out_edges(neuron_id);
            for (const auto& [target, weight] : out_edges) {
                if (weight > 0) {
                    expected_den_ex[rank][target.get_neuron_id()]-= std::abs(weight);
                } else {
                    expected_den_inh[rank][target.get_neuron_id()]-=std::abs(weight);
                }
                expected_num_local_deletions+= std::abs(weight);
            }

            const auto [in_edges, _4] = network_graph->get_local_in_edges(neuron_id);
            for (const auto& [source,weight] : in_edges ) {
                expected_axons[rank][source.get_neuron_id()]-=std::abs(weight);
                if (!disabled_neurons.contains(source)) {
                    expected_num_local_deletions+=std::abs(weight);
                }
            }
        }

        const auto& [num_deletions_sum, number_deleted_distant_out_axons , number_deleted_distant_in
                            , number_deleted_in_edges_from_outside , number_deleted_out_edges_to_outside
                            , number_deleted_out_edges_within,
                synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons_vector,
            num_ranks);

        ASSERT_EQ(number_deleted_distant_out_axons, expected_num_distant_out_deletions);
        ASSERT_EQ(number_deleted_distant_in, expected_num_distant_in_deletions);
        ASSERT_EQ(number_deleted_in_edges_from_outside + number_deleted_out_edges_to_outside+ number_deleted_out_edges_within,expected_num_local_deletions);
        ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(),
            expected_distant_out_deletions_initiated[rank] + expected_distant_in_deletions_initiated[rank]);
        ASSERT_EQ(num_deletions_sum, expected_num_local_deletions + expected_num_distant_out_deletions + expected_num_distant_in_deletions);

        ASSERT_FALSE(synapse_deletion_Requests.contains(MPIRank(rank)));

        outgoing_requests.push_back(synapse_deletion_Requests);

        for (const auto& disable_id : disabled_neurons) {
            ASSERT_EQ(std::get<0>(network_graph->get_local_in_edges(disable_id)).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_local_out_edges(disable_id)).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_in_edges(disable_id)).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_out_edges(disable_id)).size(), 0);
            ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()],
                UpdateStatus::Disabled);
            ASSERT_EQ(neurons->get_axons().get_connected_elements(disable_id), 0);
            ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(disable_id), 0);
            ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(disable_id), 0);
        }
    }

    const auto& ingoing_requests = MPIAdapter::exchange_requests(outgoing_requests);

    for (int rank = 0; rank < num_ranks; rank++) {
        const MPIRank mpi_rank{ rank };

        const auto& synapse_deletion_Requests = ingoing_requests[rank];

        const auto& network_graph = network_graphs[rank];
        const auto& disabled_neurons = rank_to_disabled_neurons[rank];
        auto& neurons = rank_to_neurons[rank];
        const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests,
            mpi_rank);
        ASSERT_EQ(num_distant_deletions,
            expected_distant_in_deletions_received[rank] + expected_distant_out_deletions_received[rank]);
        for (const auto& disable_id : disabled_neurons) {
            ASSERT_EQ(std::get<0>(network_graph->get_local_in_edges(disable_id)).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_local_out_edges(disable_id)).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_in_edges(disable_id)).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_out_edges(disable_id)).size(), 0);
            ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()],
                UpdateStatus::Disabled);
            ASSERT_EQ(neurons->get_axons().get_connected_elements(disable_id), 0);
            ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(disable_id), 0);
            ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(disable_id), 0);
        }

        for (const auto& neuron_id : NeuronID::range(num_neurons)) {
            if (disabled_neurons.contains(neuron_id)) {
                continue;
            }

            ASSERT_EQ(neurons->get_axons().get_connected_elements(neuron_id),
                expected_axons[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(neuron_id),
                expected_den_ex[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(neuron_id),
                expected_den_inh[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(neurons->get_disable_flags()[neuron_id.get_neuron_id()], UpdateStatus::Enabled);
        }
    }

    NetworkGraphAdapter::check_validity_of_network_graphs(network_graphs, signal_types, num_neurons);
}

TEST_F(NeuronsTest, testDisableNeuronsWithRanksAndOnlyOneDisabledNeuron) {
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 30;
    const auto num_ranks = MPIRankAdapter::get_random_number_ranks(mt) + 1;

    std::vector<std::shared_ptr<Neurons>> rank_to_neurons;
    std::vector<std::shared_ptr<NetworkGraph>> network_graphs;

    for (const auto mpi_rank : MPIRank::range(num_ranks)) {
        auto partition = std::make_shared<Partition>(1, MPIRank(0));
        auto [neurons, network_graph_plastic] = create_neurons_object(partition, mpi_rank);

        neurons->init(num_neurons);

        NetworkGraphAdapter::create_dense_plastic_network(network_graph_plastic, neurons->get_axons().get_signal_types(),
            num_neurons, 8, num_ranks, mpi_rank, mt);

        Neurons::check_signal_types(network_graph_plastic, neurons->get_axons().get_signal_types(), mpi_rank);

        rank_to_neurons.push_back(std::move(neurons));
        network_graphs.push_back(network_graph_plastic);
    }

    const auto& disabled_neuron_id = NeuronIdAdapter::get_random_neuron_id(num_neurons, mt);
    const auto& disabled_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);

    NetworkGraphAdapter::harmonize_network_graphs_from_different_ranks(network_graphs, num_neurons);

    std::vector<std::vector<SignalType>> signal_types = rank_to_neurons | ranges::views::transform([](const auto& neurons_of_rank) { return neurons_of_rank->get_axons().get_signal_types() | ranges::to_vector; }) | ranges::to_vector;

    std::vector<std::vector<unsigned int>> expected_axons;
    std::vector<std::vector<unsigned int>> expected_den_ex;
    std::vector<std::vector<unsigned int>> expected_den_inh;
    expected_axons.resize(num_ranks);
    expected_den_ex.resize(num_ranks);
    expected_den_inh.resize(num_ranks);
    for (auto rank = 0; rank < num_ranks; rank++) {
        auto& neurons = rank_to_neurons[rank];
        neurons->init_synaptic_elements({}, {}, {});

        const auto axons = neurons->get_axons().get_connected_elements();
        const auto den_ex = neurons->get_dendrites_exc().get_connected_elements();
        const auto den_inh = neurons->get_dendrites_inh().get_connected_elements();
        expected_axons[rank] = axons | ranges::to_vector;
        expected_den_ex[rank] = den_ex | ranges::to_vector;
        expected_den_inh[rank] = den_inh | ranges::to_vector;
    }

    std::vector<CommunicationMap<SynapseDeletionRequest>> outgoing_requests;

    size_t num_distant_deletions = 0;
    size_t num_local_deletions = 0;

    const auto& [distant_out_edges, _3] = network_graphs[disabled_rank.get_rank()]->get_distant_out_edges(disabled_neuron_id);
    for (const auto& [target, weight] : distant_out_edges) {
        if (weight > 0) {
            expected_den_ex[target.get_rank().get_rank()][target.get_neuron_id().get_neuron_id()]--;
        } else {
            expected_den_inh[target.get_rank().get_rank()][target.get_neuron_id().get_neuron_id()]--;
        }
        num_distant_deletions++;
    }

    const auto& [distant_in_edges, _2] = network_graphs[disabled_rank.get_rank()]->get_distant_in_edges(disabled_neuron_id);
    for (const auto& source : distant_in_edges | ranges::views::keys) {
        expected_axons[source.get_rank().get_rank()][source.get_neuron_id().get_neuron_id()]--;
        num_distant_deletions++;
    }

    const auto& [local_out_edges, _7] = network_graphs[disabled_rank.get_rank()]->get_local_out_edges(disabled_neuron_id);
    for (const auto& [target, weight] : local_out_edges) {
        if (weight > 0) {
            expected_den_ex[disabled_rank.get_rank()][target.get_neuron_id()]--;
        } else {
            expected_den_inh[disabled_rank.get_rank()][target.get_neuron_id()]--;
        }
        num_local_deletions++;
    }

    const auto& [local_in_edges, _8] = network_graphs[disabled_rank.get_rank()]->get_local_in_edges(disabled_neuron_id);
    for (const auto& source : local_in_edges | ranges::views::keys) {
        expected_axons[disabled_rank.get_rank()][source.get_neuron_id()]--;
        num_local_deletions++;
    }

    expected_axons[disabled_rank.get_rank()][disabled_neuron_id.get_neuron_id()] = 0;
    expected_den_ex[disabled_rank.get_rank()][disabled_neuron_id.get_neuron_id()] = 0;
    expected_den_inh[disabled_rank.get_rank()][disabled_neuron_id.get_neuron_id()] = 0;

    for (int rank = 0; rank < num_ranks; rank++) {
        const auto& network_graph = network_graphs[rank];
        std::vector<NeuronID> disabled_neurons_vector;
        if (disabled_rank.get_rank() == rank) {
            disabled_neurons_vector.push_back(disabled_neuron_id);
        }
        auto& neurons = rank_to_neurons[rank];

        const auto& [num_deletions,number_deleted_distant_out_axons , number_deleted_distant_in
                , number_deleted_in_edges_from_outside , number_deleted_out_edges_to_outside
                , number_deleted_out_edges_within, synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons_vector,
            num_ranks);
        ASSERT_FALSE(synapse_deletion_Requests.contains(MPIRank(rank)));

        if (rank == disabled_rank.get_rank()) {
            ASSERT_EQ(num_deletions, num_local_deletions + num_distant_deletions);
            ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(), num_distant_deletions);
        } else {
            ASSERT_EQ(num_deletions, 0);
            ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(), 0);
        }

        outgoing_requests.push_back(synapse_deletion_Requests);
    }

    const auto& ingoing_requests = MPIAdapter::exchange_requests(outgoing_requests);

    for (int rank = 0; rank < num_ranks; rank++) {
        const MPIRank mpi_rank{ rank };

        const auto& synapse_deletion_Requests = ingoing_requests[rank];

        const auto& network_graph = network_graphs[rank];
        auto& neurons = rank_to_neurons[rank];
        const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests,
            mpi_rank);

        for (const auto& neuron_id : NeuronID::range(num_neurons)) {
            ASSERT_EQ(neurons->get_axons().get_connected_elements(neuron_id),
                expected_axons[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(neurons->get_dendrites_exc().get_connected_elements(neuron_id),
                expected_den_ex[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(neurons->get_dendrites_inh().get_connected_elements(neuron_id),
                expected_den_inh[rank][neuron_id.get_neuron_id()]);
        }
    }

    NetworkGraphAdapter::check_validity_of_network_graphs(network_graphs, signal_types, num_neurons);
}
