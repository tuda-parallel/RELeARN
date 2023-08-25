// #include "gtest/gtest.h"
//
// #include "RelearnTest.hpp"
//
// #include "neurons/Neurons.h"
// #include "neurons/enums/UpdateStatus.h"
// #include "neurons/models/NeuronModels.h"
//
// #include "neurons/NetworkGraph.h"
//
// #include "structure/Partition.h"
//
// #include <algorithm>
// #include <map>
// #include <numeric>
// #include <random>
// #include <stack>
// #include <tuple>
// #include <vector>
//
// void assert_getter_equality(const std::unique_ptr<NeuronModel>& model) {
//    const auto number_neurons = model->get_number_neurons();
//
//    const auto& all_fired = model->get_fired();
//    const auto& all_i_syn = model->get_synaptic_input();
//    const auto& all_x = model->get_x();
//
//    ASSERT_EQ(all_fired.size(), number_neurons);
//    ASSERT_EQ(all_i_syn.size(), number_neurons);
//    ASSERT_EQ(all_x.size(), number_neurons);
//
//    const auto& valid_ids = NeuronID::range(number_neurons);
//    for (const auto& neuron_id : valid_ids) {
//        ASSERT_NO_THROW(const auto fired = model->get_fired(neuron_id););
//        ASSERT_NO_THROW(const auto i_syn = model->get_synaptic_input(neuron_id););
//        ASSERT_NO_THROW(const auto x = model->get_x(neuron_id););
//        ASSERT_NO_THROW(const auto secondary = model->get_secondary_variable(neuron_id););
//    }
//
//    for (const auto& neuron_id : valid_ids) {
//        ASSERT_EQ(model->get_fired(neuron_id), all_fired[neuron_id.get_neuron_id()] == FiredStatus::Fired);
//        ASSERT_EQ(model->get_synaptic_input(neuron_id), all_i_syn[neuron_id.get_neuron_id()]);
//        ASSERT_EQ(model->get_x(neuron_id), all_x[neuron_id.get_neuron_id()]);
//    }
//}
//
// void assert_getter_throws(const std::unique_ptr<NeuronModel>& model) {
//    const auto number_neurons = model->get_number_neurons();
//
//    const auto& invalid_local_ids = NeuronID::range(number_neurons, number_neurons + number_neurons);
//    for (const auto& neuron_id : invalid_local_ids) {
//        ASSERT_THROW(const auto fired = model->get_fired(neuron_id), RelearnException);
//        ASSERT_THROW(const auto i_syn = model->get_synaptic_input(neuron_id), RelearnException);
//        ASSERT_THROW(const auto x = model->get_x(neuron_id), RelearnException);
//        ASSERT_THROW(const auto secondary = model->get_secondary_variable(neuron_id), RelearnException);
//    }
//}
//
// void test_initialization(std::unique_ptr<NeuronModel> model, size_t number_neurons) {
//    const auto number_neurons_pre_init = model->get_number_neurons();
//    ASSERT_EQ(number_neurons_pre_init, 0);
//
//    model->init(number_neurons);
//
//    assert_getter_equality(model);
//
//    const auto& valid_ids = NeuronID::range(number_neurons);
//    for (const auto& neuron_id : valid_ids) {
//        ASSERT_EQ(model->get_fired(neuron_id), false);
//        ASSERT_EQ(model->get_synaptic_input(neuron_id), 0.0);
//    }
//
//    assert_getter_throws(model);
//}
//
// void test_creation(std::unique_ptr<NeuronModel> model, size_t number_neurons_init, size_t number_neurons_create) {
//    model->init(number_neurons_init);
//
//    // Copy on purpose
//    const auto initial_fired = model->get_fired();
//    const auto initial_i_syn = model->get_synaptic_input();
//    const auto initial_x = model->get_x();
//    std::vector<double> initial_secondary(number_neurons_init);
//
//    const auto& valid_init_ids = NeuronID::range(number_neurons_init);
//
//    for (const auto& neuron_id : valid_init_ids) {
//        const auto secondary = model->get_secondary_variable(neuron_id);
//        initial_secondary[neuron_id.get_neuron_id()] = secondary;
//    }
//
//    model->create_neurons(number_neurons_create);
//
//    const auto golden_number_neurons = number_neurons_init + number_neurons_create;
//    const auto total_number_neurons = model->get_number_neurons();
//
//    ASSERT_EQ(golden_number_neurons, total_number_neurons);
//
//    const auto& all_fired = model->get_fired();
//    const auto& all_i_syn = model->get_synaptic_input();
//    const auto& all_x = model->get_x();
//
//    const auto& valid_ids = NeuronID::range(golden_number_neurons);
//
//    for (const auto& neuron_id : valid_ids) {
//        ASSERT_NO_THROW(const auto fired = model->get_fired(neuron_id););
//        ASSERT_NO_THROW(const auto i_syn = model->get_synaptic_input(neuron_id););
//        ASSERT_NO_THROW(const auto x = model->get_x(neuron_id););
//        ASSERT_NO_THROW(const auto secondary = model->get_secondary_variable(neuron_id););
//    }
//
//    for (const auto& neuron_id : valid_ids) {
//        ASSERT_EQ(model->get_fired(neuron_id), all_fired[neuron_id.get_neuron_id()] == FiredStatus::Fired);
//        ASSERT_EQ(model->get_synaptic_input(neuron_id), all_i_syn[neuron_id.get_neuron_id()]);
//        ASSERT_EQ(model->get_fired(neuron_id), false);
//        ASSERT_EQ(model->get_synaptic_input(neuron_id), 0.0);
//        ASSERT_EQ(model->get_x(neuron_id), all_x[neuron_id.get_neuron_id()]);
//    }
//
//    for (const auto& neuron_id : valid_init_ids) {
//        const auto id = neuron_id.get_neuron_id();
//
//        ASSERT_EQ(initial_fired[id], all_fired[id]);
//        ASSERT_EQ(initial_i_syn[id], all_i_syn[id]);
//        ASSERT_EQ(initial_x[id], all_x[id]);
//        ASSERT_EQ(initial_secondary[id], model->get_secondary_variable(neuron_id));
//    }
//
//    assert_getter_throws(model);
//}
//
// void NeuronModelsTest::test_update(std::unique_ptr<NeuronModel> model, std::shared_ptr<NetworkGraph> ng, size_t number_neurons) {
//    model->init(number_neurons);
//
//    const auto conductance = model->get_k();
//    const auto number_disabled_neurons = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt).get_neuron_id() * 0;
//
//    const auto status_flags = get_update_status(number_neurons, number_disabled_neurons);
//
//    for (const auto& input : model->get_synaptic_input()) {
//        ASSERT_EQ(input, 0.0);
//    }
//
//    const auto& fired_status = model->get_fired();
//
//    for (const auto& fired : fired_status) {
//        ASSERT_EQ(fired, FiredStatus::Inactive);
//    }
//
//    model->update_electrical_activity(*ng, status_flags);
//
//    std::vector<double> expected_input(number_neurons, 0.0);
//
//    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
//        if (status_flags[neuron_id] == UpdateStatus::Disabled) {
//            expected_input[neuron_id] = 0.0;
//        }
//
//        const auto& in_edges = ng->get_local_in_edges(NeuronID(neuron_id));
//
//        for (const auto& [source_id, weight] : in_edges) {
//            const auto id = source_id.get_neuron_id();
//
//            if (fired_status[id] == FiredStatus::Inactive) {
//                continue;
//            }
//
//            const auto transferred = conductance * weight;
//            expected_input[neuron_id] += transferred;
//        }
//    }
//
//    model->update_electrical_activity(*ng, status_flags);
//
//    const auto& input = model->get_synaptic_input();
//
//    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
//        const auto golden_input = expected_input[neuron_id];
//        const auto calculated_input = input[neuron_id];
//
//        ASSERT_EQ(golden_input, calculated_input);
//    }
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorPoisson) {
//    using namespace models;
//
//    const auto expected_k = NeuronModel::default_k;
//    const auto expected_h = NeuronModel::default_h;
//    const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
//    const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
//    const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;
//
//    auto model = std::make_unique<PoissonModel>();
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    const auto expected_x0 = PoissonModel::default_x_0;
//    const auto expected_refrac = PoissonModel::default_refractory_period;
//    const auto expected_tau_x = PoissonModel::default_tau_x;
//
//    ASSERT_EQ(expected_x0, model->get_x_0());
//    ASSERT_EQ(expected_refrac, model->get_refractory_time());
//    ASSERT_EQ(expected_tau_x, model->get_tau_x());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorIzhikevich) {
//    using namespace models;
//
//    const auto expected_k = NeuronModel::default_k;
//    const auto expected_h = NeuronModel::default_h;
//    const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
//    const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
//    const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;
//
//    auto model = std::make_unique<IzhikevichModel>();
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    const auto expected_a = IzhikevichModel::default_a;
//    const auto expected_b = IzhikevichModel::default_b;
//    const auto expected_c = IzhikevichModel::default_c;
//    const auto expected_d = IzhikevichModel::default_d;
//    const auto expected_V_spike = IzhikevichModel::default_V_spike;
//    const auto expected_k1 = IzhikevichModel::default_k1;
//    const auto expected_k2 = IzhikevichModel::default_k2;
//    const auto expected_k3 = IzhikevichModel::default_k3;
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_c, model->get_c());
//    ASSERT_EQ(expected_d, model->get_d());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//    ASSERT_EQ(expected_k1, model->get_k1());
//    ASSERT_EQ(expected_k2, model->get_k2());
//    ASSERT_EQ(expected_k3, model->get_k3());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorFitzHughNagumo) {
//    using namespace models;
//
//    const auto expected_k = NeuronModel::default_k;
//    const auto expected_h = NeuronModel::default_h;
//    const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
//    const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
//    const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;
//
//    auto model = std::make_unique<FitzHughNagumoModel>();
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    const auto expected_a = FitzHughNagumoModel::default_a;
//    const auto expected_b = FitzHughNagumoModel::default_b;
//    const auto expected_phi = FitzHughNagumoModel::default_phi;
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_phi, model->get_phi());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorAEIF) {
//    using namespace models;
//
//    const auto expected_k = NeuronModel::default_k;
//    const auto expected_h = NeuronModel::default_h;
//    const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
//    const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
//    const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;
//
//    auto model = std::make_unique<AEIFModel>();
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    const auto expected_C = AEIFModel::default_C;
//    const auto expected_g_L = AEIFModel::default_g_L;
//    const auto expected_E_L = AEIFModel::default_E_L;
//    const auto expected_V_T = AEIFModel::default_V_T;
//    const auto expected_d_T = AEIFModel::default_d_T;
//    const auto expected_tau_w = AEIFModel::default_tau_w;
//    const auto expected_a = AEIFModel::default_a;
//    const auto expected_b = AEIFModel::default_b;
//    const auto expected_V_spike = AEIFModel::default_V_spike;
//
//    ASSERT_EQ(expected_C, model->get_C());
//    ASSERT_EQ(expected_g_L, model->get_g_L());
//    ASSERT_EQ(expected_E_L, model->get_E_L());
//    ASSERT_EQ(expected_V_T, model->get_V_T());
//    ASSERT_EQ(expected_d_T, model->get_d_T());
//    ASSERT_EQ(expected_tau_w, model->get_tau_w());
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorPoisson) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
//    uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refractory_time, PoissonModel::max_refractory_time);
//    uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_x0 = urd_desired_x0(mt);
//    const auto expected_refrac = urd_desired_refrac(mt);
//    const auto expected_tau_x = urd_desired_tau_x(mt);
//
//    auto model = std::make_unique<PoissonModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_x0, expected_tau_x, expected_refrac);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_x0, model->get_x_0());
//    ASSERT_EQ(expected_refrac, model->get_refractory_time());
//    ASSERT_EQ(expected_tau_x, model->get_tau_x());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorIzhikevich) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
//    uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
//    uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
//    uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
//    uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
//    uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
//    uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_c = urd_desired_c(mt);
//    const auto expected_d = urd_desired_d(mt);
//    const auto expected_V_spike = urd_desired_V_spike(mt);
//    const auto expected_k1 = urd_desired_k1(mt);
//    const auto expected_k2 = urd_desired_k2(mt);
//    const auto expected_k3 = urd_desired_k3(mt);
//
//    auto model = std::make_unique<IzhikevichModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_c, model->get_c());
//    ASSERT_EQ(expected_d, model->get_d());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//    ASSERT_EQ(expected_k1, model->get_k1());
//    ASSERT_EQ(expected_k2, model->get_k2());
//    ASSERT_EQ(expected_k3, model->get_k3());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorFitzHughNagumo) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
//    uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_phi = urd_desired_phi(mt);
//
//    auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_a, expected_b, expected_phi);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_phi, model->get_phi());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorAEIF) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
//    uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
//    uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
//    uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
//    uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
//    uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
//    uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
//    uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_C = urd_desired_C(mt);
//    const auto expected_g_L = urd_desired_g_L(mt);
//    const auto expected_E_L = urd_desired_E_L(mt);
//    const auto expected_V_T = urd_desired_V_T(mt);
//    const auto expected_d_T = urd_desired_d_T(mt);
//    const auto expected_tau_w = urd_desired_tau_w(mt);
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_V_spike = urd_desired_V_spike(mt);
//
//    auto model = std::make_unique<AEIFModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_C, model->get_C());
//    ASSERT_EQ(expected_g_L, model->get_g_L());
//    ASSERT_EQ(expected_E_L, model->get_E_L());
//    ASSERT_EQ(expected_V_T, model->get_V_T());
//    ASSERT_EQ(expected_d_T, model->get_d_T());
//    ASSERT_EQ(expected_tau_w, model->get_tau_w());
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsClonePoisson) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
//    uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refractory_time, PoissonModel::max_refractory_time);
//    uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_x0 = urd_desired_x0(mt);
//    const auto expected_refrac = urd_desired_refrac(mt);
//    const auto expected_tau_x = urd_desired_tau_x(mt);
//
//    auto model = std::make_unique<PoissonModel>(expected_k,expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_x0, expected_tau_x, expected_refrac);
//
//    auto cloned_model = model->clone();
//    std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
//    auto cast_cloned_model = std::dynamic_pointer_cast<PoissonModel>(shared_version);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_x0, model->get_x_0());
//    ASSERT_EQ(expected_refrac, model->get_refractory_time());
//    ASSERT_EQ(expected_tau_x, model->get_tau_x());
//
//    ASSERT_EQ(expected_k, cast_cloned_model->get_k());
//    ASSERT_EQ(expected_h, cast_cloned_model->get_h());
//    ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_x0, cast_cloned_model->get_x_0());
//    ASSERT_EQ(expected_refrac, cast_cloned_model->get_refractory_time());
//    ASSERT_EQ(expected_tau_x, cast_cloned_model->get_tau_x());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCloneIzhikevich) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
//    uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
//    uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
//    uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
//    uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
//    uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
//    uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_c = urd_desired_c(mt);
//    const auto expected_d = urd_desired_d(mt);
//    const auto expected_V_spike = urd_desired_V_spike(mt);
//    const auto expected_k1 = urd_desired_k1(mt);
//    const auto expected_k2 = urd_desired_k2(mt);
//    const auto expected_k3 = urd_desired_k3(mt);
//
//    auto model = std::make_unique<IzhikevichModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);
//
//    auto cloned_model = model->clone();
//    std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
//    auto cast_cloned_model = std::dynamic_pointer_cast<IzhikevichModel>(shared_version);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_c, model->get_c());
//    ASSERT_EQ(expected_d, model->get_d());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//    ASSERT_EQ(expected_k1, model->get_k1());
//    ASSERT_EQ(expected_k2, model->get_k2());
//    ASSERT_EQ(expected_k3, model->get_k3());
//
//    ASSERT_EQ(expected_k, cast_cloned_model->get_k());
//    ASSERT_EQ(expected_h, cast_cloned_model->get_h());
//    ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, cast_cloned_model->get_a());
//    ASSERT_EQ(expected_b, cast_cloned_model->get_b());
//    ASSERT_EQ(expected_c, cast_cloned_model->get_c());
//    ASSERT_EQ(expected_d, cast_cloned_model->get_d());
//    ASSERT_EQ(expected_V_spike, cast_cloned_model->get_V_spike());
//    ASSERT_EQ(expected_k1, cast_cloned_model->get_k1());
//    ASSERT_EQ(expected_k2, cast_cloned_model->get_k2());
//    ASSERT_EQ(expected_k3, cast_cloned_model->get_k3());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCloneFitzHughNagumo) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
//    uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_phi = urd_desired_phi(mt);
//
//    auto model = std::make_unique<FitzHughNagumoModel>(expected_k,expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_a, expected_b, expected_phi);
//
//    auto cloned_model = model->clone();
//    std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
//    auto cast_cloned_model = std::dynamic_pointer_cast<FitzHughNagumoModel>(shared_version);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_phi, model->get_phi());
//
//    ASSERT_EQ(expected_k, cast_cloned_model->get_k());
//    ASSERT_EQ(expected_h, cast_cloned_model->get_h());
//    ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, cast_cloned_model->get_a());
//    ASSERT_EQ(expected_b, cast_cloned_model->get_b());
//    ASSERT_EQ(expected_phi, cast_cloned_model->get_phi());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCloneAEIF) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
//    uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
//    uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
//    uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
//    uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
//    uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
//    uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
//    uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_C = urd_desired_C(mt);
//    const auto expected_g_L = urd_desired_g_L(mt);
//    const auto expected_E_L = urd_desired_E_L(mt);
//    const auto expected_V_T = urd_desired_V_T(mt);
//    const auto expected_d_T = urd_desired_d_T(mt);
//    const auto expected_tau_w = urd_desired_tau_w(mt);
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_V_spike = urd_desired_V_spike(mt);
//
//    auto model = std::make_unique<AEIFModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);
//
//    auto cloned_model = model->clone();
//    std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
//    auto cast_cloned_model = std::dynamic_pointer_cast<AEIFModel>(shared_version);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_C, model->get_C());
//    ASSERT_EQ(expected_g_L, model->get_g_L());
//    ASSERT_EQ(expected_E_L, model->get_E_L());
//    ASSERT_EQ(expected_V_T, model->get_V_T());
//    ASSERT_EQ(expected_d_T, model->get_d_T());
//    ASSERT_EQ(expected_tau_w, model->get_tau_w());
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//
//    ASSERT_EQ(expected_k, cast_cloned_model->get_k());
//    ASSERT_EQ(expected_h, cast_cloned_model->get_h());
//    ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_C, cast_cloned_model->get_C());
//    ASSERT_EQ(expected_g_L, cast_cloned_model->get_g_L());
//    ASSERT_EQ(expected_E_L, cast_cloned_model->get_E_L());
//    ASSERT_EQ(expected_V_T, cast_cloned_model->get_V_T());
//    ASSERT_EQ(expected_d_T, cast_cloned_model->get_d_T());
//    ASSERT_EQ(expected_tau_w, cast_cloned_model->get_tau_w());
//    ASSERT_EQ(expected_a, cast_cloned_model->get_a());
//    ASSERT_EQ(expected_b, cast_cloned_model->get_b());
//    ASSERT_EQ(expected_V_spike, cast_cloned_model->get_V_spike());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreatePoisson) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
//    uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refractory_time, PoissonModel::max_refractory_time);
//    uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_x0 = urd_desired_x0(mt);
//    const auto expected_refrac = urd_desired_refrac(mt);
//    const auto expected_tau_x = urd_desired_tau_x(mt);
//
//    auto model = NeuronModel::create<PoissonModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_x0, expected_tau_x, expected_refrac);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_x0, model->get_x_0());
//    ASSERT_EQ(expected_refrac, model->get_refractory_time());
//    ASSERT_EQ(expected_tau_x, model->get_tau_x());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateIzhikevich) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
//    uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
//    uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
//    uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
//    uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
//    uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
//    uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_c = urd_desired_c(mt);
//    const auto expected_d = urd_desired_d(mt);
//    const auto expected_V_spike = urd_desired_V_spike(mt);
//    const auto expected_k1 = urd_desired_k1(mt);
//    const auto expected_k2 = urd_desired_k2(mt);
//    const auto expected_k3 = urd_desired_k3(mt);
//
//    auto model = NeuronModel::create<IzhikevichModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_c, model->get_c());
//    ASSERT_EQ(expected_d, model->get_d());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//    ASSERT_EQ(expected_k1, model->get_k1());
//    ASSERT_EQ(expected_k2, model->get_k2());
//    ASSERT_EQ(expected_k3, model->get_k3());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateFitzHughNagumo) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
//    uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_phi = urd_desired_phi(mt);
//
//    auto model = NeuronModel::create<FitzHughNagumoModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_a, expected_b, expected_phi);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_phi, model->get_phi());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateAEIF) {
//    using namespace models;
//    using urd = uniform_real_distribution<double>;
//    using uid = uniform_int_distribution<unsigned int>;
//
//    uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
//    uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
//    uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
//    uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
//    uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);
//
//    uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
//    uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
//    uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
//    uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
//    uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
//    uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
//    uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
//    uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
//    uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);
//
//    const auto expected_k = urd_desired_k(mt);
//    const auto expected_h = uid_desired_h(mt);
//    const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
//    const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
//    const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);
//
//    const auto expected_C = urd_desired_C(mt);
//    const auto expected_g_L = urd_desired_g_L(mt);
//    const auto expected_E_L = urd_desired_E_L(mt);
//    const auto expected_V_T = urd_desired_V_T(mt);
//    const auto expected_d_T = urd_desired_d_T(mt);
//    const auto expected_tau_w = urd_desired_tau_w(mt);
//    const auto expected_a = urd_desired_a(mt);
//    const auto expected_b = urd_desired_b(mt);
//    const auto expected_V_spike = urd_desired_V_spike(mt);
//
//    auto model = NeuronModel::create<AEIFModel>(expected_k, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
//        expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);
//
//    ASSERT_EQ(expected_k, model->get_k());
//    ASSERT_EQ(expected_h, model->get_h());
//    ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
//    ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
//    ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());
//
//    ASSERT_EQ(expected_C, model->get_C());
//    ASSERT_EQ(expected_g_L, model->get_g_L());
//    ASSERT_EQ(expected_E_L, model->get_E_L());
//    ASSERT_EQ(expected_V_T, model->get_V_T());
//    ASSERT_EQ(expected_d_T, model->get_d_T());
//    ASSERT_EQ(expected_tau_w, model->get_tau_w());
//    ASSERT_EQ(expected_a, model->get_a());
//    ASSERT_EQ(expected_b, model->get_b());
//    ASSERT_EQ(expected_V_spike, model->get_V_spike());
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsInitPoisson) {
//    using namespace models;
//    auto model = std::make_unique<PoissonModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    test_initialization(std::move(model), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsInitIzhikevich) {
//    using namespace models;
//    auto model = std::make_unique<IzhikevichModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    test_initialization(std::move(model), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsInitFitzHughNagumo) {
//    using namespace models;
//    auto model = std::make_unique<FitzHughNagumoModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    test_initialization(std::move(model), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsInitAEIF) {
//    using namespace models;
//    auto model = std::make_unique<AEIFModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    test_initialization(std::move(model), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsPoisson) {
//    using namespace models;
//    auto model = std::make_unique<PoissonModel>();
//
//    const auto number_neurons_init = get_random_number_neurons();
//    const auto number_neurons_create = get_random_number_neurons();
//
//    test_creation(std::move(model), number_neurons_init, number_neurons_create);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsIzhikevich) {
//    using namespace models;
//    auto model = std::make_unique<IzhikevichModel>();
//
//    const auto number_neurons_init = get_random_number_neurons();
//    const auto number_neurons_create = get_random_number_neurons();
//
//    test_creation(std::move(model), number_neurons_init, number_neurons_create);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsFitzHughNagumo) {
//    using namespace models;
//    auto model = std::make_unique<FitzHughNagumoModel>();
//
//    const auto number_neurons_init = get_random_number_neurons();
//    const auto number_neurons_create = get_random_number_neurons();
//
//    test_creation(std::move(model), number_neurons_init, number_neurons_create);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsAEIF) {
//    using namespace models;
//    auto model = std::make_unique<AEIFModel>();
//
//    const auto number_neurons_init = get_random_number_neurons();
//    const auto number_neurons_create = get_random_number_neurons();
//
//    test_creation(std::move(model), number_neurons_init, number_neurons_create);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputEmptyPoisson) {
//    using namespace models;
//    auto model = std::make_unique<PoissonModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_empty_network_graph(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputEmptyIzhikevich) {
//    using namespace models;
//    auto model = std::make_unique<IzhikevichModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_empty_network_graph(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputEmptyFitzHughNagumo) {
//    using namespace models;
//    auto model = std::make_unique<FitzHughNagumoModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_empty_network_graph(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputEmptyAEIF) {
//    using namespace models;
//    auto model = std::make_unique<AEIFModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_empty_network_graph(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputSomePoisson) {
//    using namespace models;
//    auto model = std::make_unique<PoissonModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//    const auto number_synapses = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
//
//    auto ng = create_network_graph(number_neurons, 0, number_synapses.get_neuron_id());
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputSomeIzhikevich) {
//    using namespace models;
//    auto model = std::make_unique<IzhikevichModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//    const auto number_synapses = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
//
//    auto ng = create_network_graph(number_neurons, 0, number_synapses.get_neuron_id());
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputSomeFitzHughNagumo) {
//    using namespace models;
//    auto model = std::make_unique<FitzHughNagumoModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//    const auto number_synapses = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
//
//    auto ng = create_network_graph(number_neurons, 0, number_synapses.get_neuron_id());
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputSomeAEIF) {
//    using namespace models;
//    auto model = std::make_unique<AEIFModel>();
//
//    const auto number_neurons = get_random_number_neurons();
//    const auto number_synapses = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
//
//    auto ng = create_network_graph(number_neurons, 0, number_synapses.get_neuron_id());
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputFullPoisson) {
//    using namespace models;
//    auto model = std::make_unique<PoissonModel>(
//        NeuronModel::default_k,
//        NeuronModel::default_h,
//        0.5, 0.0, 0.0,
//        PoissonModel::default_x_0, PoissonModel::default_tau_x, PoissonModel::default_refractory_period);
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_network_graph_all_to_all(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputFullIzhikevich) {
//    using namespace models;
//    auto model = std::make_unique<IzhikevichModel>(
//        NeuronModel::default_k,
//        NeuronModel::default_h,
//        60.0, 0.0, 0.0,
//        IzhikevichModel::default_a, IzhikevichModel::default_b,
//        IzhikevichModel::default_c, IzhikevichModel::default_d,
//        IzhikevichModel::default_V_spike, IzhikevichModel::default_k1,
//        IzhikevichModel::default_k2, IzhikevichModel::default_k3);
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_network_graph_all_to_all(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputFullFitzHughNagumo) {
//    using namespace models;
//    auto model = std::make_unique<FitzHughNagumoModel>(
//        NeuronModel::default_k,
//        NeuronModel::default_h,
//        60.0, 0.0, 0.0,
//        FitzHughNagumoModel::default_a, FitzHughNagumoModel::default_b, FitzHughNagumoModel::default_phi);
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_network_graph_all_to_all(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
//
// TEST_F(NeuronModelsTest, testNeuronModelsSynapticInputFullAEIF) {
//    using namespace models;
//    auto model = std::make_unique<AEIFModel>(
//        NeuronModel::default_k,
//        NeuronModel::default_h,
//        60.0, 0.0, 0.0,
//        AEIFModel::default_C, AEIFModel::default_g_L,
//        AEIFModel::default_E_L, AEIFModel::default_V_T,
//        AEIFModel::default_d_T, AEIFModel::default_tau_w,
//        AEIFModel::default_a, AEIFModel::default_b, AEIFModel::default_V_spike);
//
//    const auto number_neurons = get_random_number_neurons();
//
//    auto ng = create_network_graph_all_to_all(number_neurons, 0);
//
//    test_update(std::move(model), std::move(ng), number_neurons);
//}
