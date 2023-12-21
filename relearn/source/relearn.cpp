/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "Types.h"
#include "algorithm/Algorithms.h"
#include "algorithm/Kernel/Kernel.h"
#include "io/parser/MonitorParser.h"
#include "io/CalciumIO.h"
#include "io/InteractiveNeuronIO.h"
#include "io/LogFiles.h"
#include "mpi/CommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/enums/ElementType.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/input/BackgroundActivityCalculator.h"
#include "neurons/input/BackgroundActivityCalculators.h"
#include "neurons/input/FiredStatusCommunicationMap.h"
#include "neurons/input/FiredStatusCommunicator.h"
#include "neurons/input/SynapticInputCalculator.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "sim/Essentials.h"
#include "sim/Simulation.h"
#include "sim/file/MultipleSubdomainsFromFile.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "sim/random/SubdomainFromNeuronPerRank.h"
#include "structure/BaseCell.h"
#include "structure/NodeCache.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/StringUtil.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "spdlog/spdlog.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
void omp_set_num_threads(int num) { }
#endif

#include <array>
#include <bitset>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>

struct empty_t {
    using position_type = VirtualPlasticityElement::position_type;
    using counter_type = VirtualPlasticityElement::counter_type;

    constexpr static bool has_excitatory_dendrite = false;
    constexpr static bool has_inhibitory_dendrite = false;
    constexpr static bool has_excitatory_axon = false;
    constexpr static bool has_inhibitory_axon = false;
};

void print_sizes() {
    constexpr auto number_bits_in_byte = CHAR_BIT;

    constexpr auto sizeof_vec3_double = sizeof(Vec3d);
    constexpr auto sizeof_vec3_size_t = sizeof(Vec3s);

    constexpr auto sizeof_virtual_plasticity_element = sizeof(VirtualPlasticityElement);

    constexpr auto sizeof_empty_t = sizeof(empty_t);
    constexpr auto sizeof_bh_cell_attributes = sizeof(BarnesHutCell);
    constexpr auto sizeof_bh_naive_attributes = sizeof(NaiveCell);

    constexpr auto sizeof_empty_cell = sizeof(Cell<empty_t>);
    constexpr auto sizeof_bh_cell = sizeof(Cell<BarnesHutCell>);
    constexpr auto sizeof_naive_cell = sizeof(Cell<NaiveCell>);

    constexpr auto sizeof_octree_node = sizeof(OctreeNode<empty_t>);
    constexpr auto sizeof_bh_octree_node = sizeof(OctreeNode<BarnesHutCell>);
    constexpr auto sizeof_naive_octree_node = sizeof(OctreeNode<NaiveCell>);

    constexpr auto sizeof_neuron_id = sizeof(NeuronID);
    constexpr auto sizeof_rank_neuron_id = sizeof(RankNeuronId);

    constexpr auto sizeof_mpi_rank = sizeof(MPIRank);
    constexpr auto sizeof_int = sizeof(int);

    constexpr auto sizeof_plastic_local_synapse = sizeof(PlasticLocalSynapse);
    constexpr auto sizeof_plastic_distant_in_synapse = sizeof(PlasticDistantInSynapse);
    constexpr auto sizeof_plastic_distant_out_synapse = sizeof(PlasticDistantOutSynapse);

    constexpr auto sizeof_static_local_synapse = sizeof(StaticLocalSynapse);
    constexpr auto sizeof_static_distant_in_synapse = sizeof(StaticDistantInSynapse);
    constexpr auto sizeof_static_distant_out_synapse = sizeof(StaticDistantOutSynapse);

    constexpr auto sizeof_empty_base_cell = sizeof(BaseCell<false, false, false, false>);
    constexpr auto sizeof_full_base_cell = sizeof(BaseCell<true, true, true, true>);
    constexpr auto sizeof_dendrites_base_cell = sizeof(BaseCell<true, true, false, false>);
    constexpr auto sizeof_axons_base_cell = sizeof(BaseCell<false, false, true, true>);

    std::stringstream ss{};

    ss << '\n';

    ss << "Number of bits in a byte: " << number_bits_in_byte << '\n';

    ss << "Size of Vec3d: " << sizeof_vec3_double << '\n';
    ss << "Size of Vec3s: " << sizeof_vec3_size_t << '\n';

    ss << "Size of VirtualPlasticityElement: " << sizeof_virtual_plasticity_element << '\n';

    ss << "Size of empty_t: " << sizeof_empty_t << '\n';
    ss << "Size of BarnesHutCell: " << sizeof_bh_cell_attributes << '\n';
    ss << "Size of NaiveCell: " << sizeof_bh_naive_attributes << '\n';

    ss << "Size of Cell<empty_t>: " << sizeof_empty_cell << '\n';
    ss << "Size of Cell<BarnesHutCell>: " << sizeof_bh_cell << '\n';
    ss << "Size of Cell<NaiveCell>: " << sizeof_naive_cell << '\n';

    ss << "Size of OctreeNode<empty_t>: " << sizeof_octree_node << '\n';
    ss << "Size of OctreeNode<BarnesHutCell>: " << sizeof_bh_octree_node << '\n';
    ss << "Size of OctreeNode<NaiveCell>: " << sizeof_naive_octree_node << '\n';

    ss << "Size of NeuronID: " << sizeof_neuron_id << '\n';
    ss << "Size of RankNeuronID: " << sizeof_rank_neuron_id << '\n';

    ss << "Size of MPIRank: " << sizeof_mpi_rank << '\n';
    ss << "Size of int: " << sizeof_int << '\n';

    ss << "Size of PlasticLocalSynapse: " << sizeof_plastic_local_synapse << '\n';
    ss << "Size of PlasticDistantInSynapse: " << sizeof_plastic_distant_in_synapse << '\n';
    ss << "Size of PlasticDistantOutSynapse: " << sizeof_plastic_distant_out_synapse << '\n';

    ss << "Size of StaticLocalSynapse: " << sizeof_static_local_synapse << '\n';
    ss << "Size of StaticDistantInSynapse: " << sizeof_static_distant_in_synapse << '\n';
    ss << "Size of StaticDistantOutSynapse: " << sizeof_static_distant_out_synapse << '\n';

    ss << "Size of BaseCell<false, false, false, false>: " << sizeof_empty_base_cell << '\n';
    ss << "Size of BaseCell<true, true, true, true>: " << sizeof_full_base_cell << '\n';
    ss << "Size of BaseCell<true, true, false, false>: " << sizeof_dendrites_base_cell << '\n';
    ss << "Size of BaseCell<false, false, true, true>: " << sizeof_axons_base_cell << '\n';

    LogFiles::print_message_rank(MPIRank::root_rank(), ss.str());
}

void print_arguments(int argc, char** argv) {
    std::stringstream ss{};

    for (auto i = 0; i < argc; i++) {
        ss << argv[i] << ' ';
    }

    LogFiles::print_message_rank(MPIRank::root_rank(), ss.str());
}

int main(int argc, char** argv) {
    /**
     * Init MPI and store some MPI infos
     */
    MPIWrapper::init(argc, argv);
    MPIWrapper::stop_measureing_communication();

    print_arguments(argc, argv);
    print_sizes();

    if constexpr (Config::do_debug_checks) {
        LogFiles::print_message_rank(MPIRank::root_rank(), "I'm performing Debug Checks");
    } else {
        LogFiles::print_message_rank(MPIRank::root_rank(), "I'm skipping Debug Checks");
    }

    const auto my_rank = MPIWrapper::get_my_rank();
    const auto num_ranks = MPIWrapper::get_num_ranks();

    // Command line arguments
    CLI::App app{ "" };

    AlgorithmEnum chosen_algorithm = AlgorithmEnum::BarnesHut;
    std::map<std::string, AlgorithmEnum> cli_parse_algorithm{
        { "naive", AlgorithmEnum::Naive },
        { "barnes-hut", AlgorithmEnum::BarnesHut },
        { "barnes-hut-inverted", AlgorithmEnum::BarnesHutInverted },
    };

    NeuronModelEnum chosen_neuron_model = NeuronModelEnum::Poisson;
    std::map<std::string, NeuronModelEnum> cli_parse_neuron_model{
        { "poisson", NeuronModelEnum::Poisson },
        { "izhikevich", NeuronModelEnum::Izhikevich },
        { "aeif", NeuronModelEnum::AEIF },
        { "fitzhughnagumo", NeuronModelEnum::FitzHughNagumo }
    };

    SynapseDeletionFinderType chosen_synapse_deleter = SynapseDeletionFinderType::Random;
    std::map<std::string, SynapseDeletionFinderType> cli_parse_synapse_deleter{
        { "random", SynapseDeletionFinderType::Random },
    };

    KernelType chosen_kernel_type = KernelType::Gaussian;
    std::map<std::string, KernelType> cli_parse_kernel_type{
        { "gamma", KernelType::Gamma },
        { "gaussian", KernelType::Gaussian },
        { "linear", KernelType::Linear },
        { "weibull", KernelType::Weibull }
    };

    TargetCalciumDecay target_calcium_decay_type = TargetCalciumDecay::None;
    std::map<std::string, TargetCalciumDecay> cli_parse_decay_type{
        { "none", TargetCalciumDecay::None },
        { "relative", TargetCalciumDecay::Relative },
        { "absolute", TargetCalciumDecay::Absolute }
    };

    NodeCacheType chosen_cache_type = NodeCacheType::Combined;
    std::map<std::string, NodeCacheType> cli_parse_cache_type{
        { "combined", NodeCacheType::Combined },
        { "separate", NodeCacheType::Separate },
    };

    SynapticInputCalculatorType chosen_synapse_input_calculator_type = SynapticInputCalculatorType::Linear;
    std::map<std::string, SynapticInputCalculatorType> cli_parse_synapse_input_calculator_type{
        { "linear", SynapticInputCalculatorType::Linear },
    };

    FiredStatusCommunicatorType chosen_fired_status_communicator_type = FiredStatusCommunicatorType::Map;
    std::map<std::string, FiredStatusCommunicatorType> cli_parse_fired_status_communicator_type{
        { "map", FiredStatusCommunicatorType::Map },
    };

    BackgroundActivityCalculatorType chosen_background_activity_calculator_type = BackgroundActivityCalculatorType::Null;
    std::map<std::string, BackgroundActivityCalculatorType> cli_parse_background_activity_calculator_type{
        { "null", BackgroundActivityCalculatorType::Null },
        { "constant", BackgroundActivityCalculatorType::Constant },
        { "normal", BackgroundActivityCalculatorType::Normal },
        { "fast-normal", BackgroundActivityCalculatorType::FastNormal },
        { "flexible", BackgroundActivityCalculatorType::Flexible }
    };

    RelearnTypes::step_type simulation_steps{};
    app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms.")->required();

    RelearnTypes::step_type first_plasticity_step{ 0 };
    app.add_option("--first-plasticity-step", first_plasticity_step, "The first step in which the plasticity is updated.");

    RelearnTypes::step_type last_plasticity_step{ std::numeric_limits<RelearnTypes::step_type>::max() };
    auto* opt_last_plasticity_update_step = app.add_option("--last-plasticity-step", last_plasticity_step, "The last step in which the plasticity is updated.");

    RelearnTypes::step_type plasticity_update_step{ Config::plasticity_update_step };
    auto* opt_plasticity_update_step = app.add_option("--plasticity-update-step", plasticity_update_step, "The interval of steps between a plasticity update.");

    RelearnTypes::step_type calcium_log_step{ Config::calcium_log_step };
    app.add_option("--calcium-log-step", calcium_log_step, "Sets the interval for logging all calcium values.");

    RelearnTypes::step_type synaptic_input_log_step{ Config::synaptic_input_log_step };
    app.add_option("--synaptic-input-log-step", synaptic_input_log_step, "Sets the interval for logging all synaptic inputs.");

    RelearnTypes::step_type network_log_step = Config::network_log_step;
    auto* const opt_network_log_step = app.add_option("--network-log-step", network_log_step, "Steps between saving the network graph");

    auto* flag_area_monitor = app.add_flag("--enable-area-monitor", "Enables the area monitor");
    auto* flag_area_monitor_connectivity = app.add_flag("--enable-area-monitor-connectivity", "Enables the monitoring of the connectivity by the area monitor");
    flag_area_monitor_connectivity->needs(flag_area_monitor);

    RelearnTypes::step_type monitor_steps{ Config::neuron_monitor_log_step };
    auto* opt_monitor_steps = app.add_option("--monitor-steps", monitor_steps, "Every time the neuron state is captured");

    const auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

    unsigned int random_seed{ 0 };
    app.add_option("-r,--random-seed", random_seed, "Random seed. Default: 0.");

    int openmp_threads{ 1 };
    app.add_option("--openmp", openmp_threads, "Number of OpenMP Threads.");

    std::filesystem::path log_path{};
    auto* const opt_log_path = app.add_option("-l,--log-path", log_path, "Path for log files.");

    std::string log_prefix{};
    const auto* opt_log_prefix = app.add_option("-p,--log-prefix", log_prefix, "Prefix for log files.");

    const auto* flag_enable_printing_events = app.add_flag("--print-events", "Enables printing the events to a file.");
    const auto* flag_disable_printing_positions = app.add_flag("--no-print-positions", "Disables printing the positions to a file.");
    const auto* flag_disable_printing_network = app.add_flag("--no-print-network", "Disables printing the network to a file.");
    const auto* flag_disable_printing_plasticity = app.add_flag("--no-print-plasticity", "Disables printing the plasticity changes to a file.");
    const auto* flag_disable_printing_calcium = app.add_flag("--no-print-calcium", "Disables printing the calcium changes to a file.");
    const auto* flag_disable_printing_fire_rate = app.add_flag("--no-print-fire-rate", "Disables printing the fire rate changes to a file.");
    const auto* flag_disable_printing_overview = app.add_flag("--no-print-overview", "Disables printing the overviews to a file.");
    const auto* flag_disable_printing_area_mapping = app.add_flag("--no-print-mapping", "Disables printing the area mapping to a file.");

    RelearnTypes::number_neurons_type number_neurons{};
    auto* const opt_num_neurons = app.add_option("-n,--num-neurons", number_neurons, "Number of neurons. This option only works with one MPI rank!");

    RelearnTypes::number_neurons_type number_neurons_per_rank{};
    auto* const opt_num_neurons_per_rank = app.add_option("--num-neurons-per-rank", number_neurons_per_rank, "Number neurons per MPI rank.");

    double fraction_excitatory_neurons{ 1.0 };
    app.add_option("--fraction-excitatory-neurons", fraction_excitatory_neurons, "The fraction of excitatory neurons, must be from [0.0, 1.0]. Requires --num-neurons or --num-neurons-per-rank to take effect.");

    double um_per_neuron{ 1.0 };
    app.add_option("--um-per-neuron", um_per_neuron, "The micrometer per neuron in one dimension, must be from (0.0, \\inf). Requires --num-neurons or --num-neurons-per-rank to take effect.");

    std::filesystem::path file_positions{};
    auto* const opt_file_positions = app.add_option("-f,--file", file_positions, "File or directory with neuron positions.");

    std::filesystem::path file_network{};
    auto* const opt_file_network = app.add_option("-g,--graph", file_network, "Folder that contains the files with the networks. The network files must be names rank_0_in_network.txt and rank_0_out_network.txt. This option only works with one MPI rank!");

    std::filesystem::path file_enable_interrupts{};
    auto* const opt_file_enable_interrupts = app.add_option("--enable-interrupts", file_enable_interrupts, "File with the enable interrupts.");

    std::filesystem::path file_disable_interrupts{};
    auto* const opt_file_disable_interrupts = app.add_option("--disable-interrupts", file_disable_interrupts, "File with the disable interrupts.");

    std::filesystem::path file_creation_interrupts{};
    auto* const opt_file_creation_interrupts = app.add_option("--creation-interrupts", file_creation_interrupts, "File with the creation interrupts.");

    auto* const opt_algorithm = app.add_option("-a,--algorithm", chosen_algorithm, "The algorithm that is used for finding the targets");
    opt_algorithm->required()->transform(CLI::CheckedTransformer(cli_parse_algorithm, CLI::ignore_case));

    auto* const opt_node_cache_type = app.add_option("--node-cache-type", chosen_cache_type, "The type of cache for the nodes of other ranks.");
    opt_node_cache_type->transform(CLI::CheckedTransformer(cli_parse_cache_type, CLI::ignore_case));

    double accept_criterion{ Constants::bh_default_theta };
    const auto* const opt_accept_criterion = app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion for Barnes-Hut. Default: 0.3. Requires Barnes-Hut or inverted Barnes-Hut.");

    auto* const opt_kernel_type = app.add_option("--kernel-type", chosen_kernel_type, "The probability kernel type.");
    opt_kernel_type->transform(CLI::CheckedTransformer(cli_parse_kernel_type, CLI::ignore_case));

    double gamma_k{ GammaDistributionKernel::default_k };
    app.add_option("--gamma-k", gamma_k, "Shape parameter for the gamma probability kernel.");

    double gamma_theta{ GammaDistributionKernel::default_theta };
    app.add_option("--gamma-theta", gamma_theta, "Scale parameter for the gamma probability kernel.");

    double gaussian_sigma{ GaussianDistributionKernel::default_sigma };
    app.add_option("--gaussian-sigma", gaussian_sigma, "Scaling parameter for the gaussian probability kernel. Default: 750");

    double gaussian_mu{ GaussianDistributionKernel::default_mu };
    app.add_option("--gaussian-mu", gaussian_mu, "Translation parameter for the gaussian probability kernel. Default: 0");

    double linear_cutoff{ LinearDistributionKernel::default_cutoff };
    app.add_option("--linear-cutoff", linear_cutoff, "Cut-off parameter for the linear probability kernel. Default: +inf");

    double weibull_k{ WeibullDistributionKernel::default_k };
    app.add_option("--weibull-k", weibull_k, "Shape parameter for the weibull probability kernel.");

    double weibull_b{ WeibullDistributionKernel::default_b };
    app.add_option("--weibull-b", weibull_b, "Scale parameter for the weibull probability kernel.");

    auto* const opt_neuron_model = app.add_option("--neuron-model", chosen_neuron_model, "The neuron model.");
    opt_neuron_model->transform(CLI::CheckedTransformer(cli_parse_neuron_model, CLI::ignore_case));

    auto* const opt_synapse_deleter = app.add_option("--synapse-deleter", chosen_synapse_deleter, "The algorithm for deleting synapses.");
    opt_synapse_deleter->transform(CLI::CheckedTransformer(cli_parse_synapse_deleter, CLI::ignore_case));

    std::string static_neurons_str{};
    auto* opt_static_neurons = app.add_option("--static-neurons", static_neurons_str, "String with neuron ids for static neurons. Format is <mpi_rank>:<neuron_id>;<mpi_rank>:<neuron_id>;... where <mpi_rank> can be -1 to indicate \"on every rank\". Alternatively use area names instead of neuron ids");

    std::filesystem::path file_external_stimulation{};
    auto* opt_file_external_stimulation = app.add_option("--external-stimulation", file_external_stimulation, "File with the external stimulation.");

    auto* const opt_background_activity = app.add_option("--background-activity", chosen_background_activity_calculator_type, "The type of background activity");
    opt_background_activity->transform(CLI::CheckedTransformer(cli_parse_background_activity_calculator_type, CLI::ignore_case));

    double base_background_activity{ BackgroundActivityCalculator::default_base_background_activity };
    auto* const opt_base_background_activity = app.add_option("--base-background-activity", base_background_activity,
        "The base background activity by which all neurons are excited");

    std::filesystem::path flexible_file_path{};
    auto* const opt_background_activity_file_path = app.add_option("--background-activity-file-path", flexible_file_path,
        "The file path for the flexible background activity");

    double background_activity_mean{ BackgroundActivityCalculator::default_background_activity_mean };
    auto* const opt_mean_background_activity = app.add_option("--background-activity-mean", background_activity_mean,
        "The mean background activity by which all neurons are excited. The background activity is calculated N(mean, stddev)");

    double background_activity_stddev{ BackgroundActivityCalculator::default_background_activity_stddev };
    auto* const opt_stddev_background_activity = app.add_option("--background-activity-stddev", background_activity_stddev,
        "The standard deviation of the background activity by which all neurons are excited. The background activity is calculated as N(mean, stddev)");

    double synapse_conductance{ SynapticInputCalculator::default_conductance };
    app.add_option("--synapse-conductance", synapse_conductance, "The activity that is transferred to its neighbors when a neuron spikes. Default is 0.03");

    auto* const opt_synapse_input_calculator_type = app.add_option("--synapse-input-calculator-type", chosen_synapse_input_calculator_type, "The type calculator that transforms the synapse input.");
    opt_synapse_input_calculator_type->transform(CLI::CheckedTransformer(cli_parse_synapse_input_calculator_type, CLI::ignore_case));

    auto* const opt_fired_status_communicator_type = app.add_option("--fired-status-communicator-type", chosen_fired_status_communicator_type, "The type of communicator between MPI ranks that exchange the fired status.");
    opt_fired_status_communicator_type->transform(CLI::CheckedTransformer(cli_parse_fired_status_communicator_type, CLI::ignore_case));

    double calcium_decay{ CalciumCalculator::default_tau_C };
    app.add_option("--calcium-decay", calcium_decay, "The decay constant for the intercellular calcium. Must be greater than 0.0");

    RelearnTypes::step_type first_decay_step{ 0 };
    app.add_option("--target-calcium-first-decay-step", first_decay_step, "The first decay step of the calcium.");

    RelearnTypes::step_type last_decay_step{ std::numeric_limits<RelearnTypes::step_type>::max() };
    app.add_option("--target-calcium-last-decay-step", last_decay_step, "The last decay step of the calcium.");

    RelearnTypes::step_type target_calcium_decay_step{ 0 };
    app.add_option("--target-calcium-decay-step", target_calcium_decay_step, "The decay step for the target calcium values.");

    double target_calcium_decay_amount{ 0.0 };
    app.add_option("--target-calcium-amount", target_calcium_decay_amount, "The decay amount for the target calcium values.");

    auto* const opt_decay_type = app.add_option("--decay-type", target_calcium_decay_type, "The decay type for the target calcium values.");
    opt_decay_type->transform(CLI::CheckedTransformer(cli_parse_decay_type, CLI::ignore_case));

    double target_calcium{ CalciumCalculator::default_C_target };
    auto* const opt_target_calcium = app.add_option("--target-ca", target_calcium, "The target Ca2+ ions in each neuron. Default is 0.7.");

    double initial_calcium{ 0.0 };
    auto* const opt_initial_calcium = app.add_option("--initial-ca", initial_calcium, "The initial Ca2+ ions in each neuron. Default is 0.0.");

    std::string file_calcium{};
    auto* const opt_file_calcium = app.add_option("--file-calcium", file_calcium, "File with calcium values.");

    double beta{ CalciumCalculator::default_beta };
    app.add_option("--beta", beta, "The amount of calcium ions gathered when a neuron fires. Default is 0.001.");

    unsigned int h{ NeuronModel::default_h };
    app.add_option("--integration-step-size", h, "The step size for the numerical integration of the electrical activity. Default is 10.");

    double retract_ratio{ SynapticElements::default_vacant_retract_ratio };
    app.add_option("--retract-ratio", retract_ratio, "The ratio by which vacant synapses retract.");

    double synaptic_elements_init_lb{ 0.0 };
    app.add_option("--synaptic-elements-lower-bound", synaptic_elements_init_lb, "The minimum number of vacant synaptic elements per neuron. Must be smaller of equal to synaptic-elements-upper-bound.");

    double synaptic_elements_init_ub{ 0.0 };
    app.add_option("--synaptic-elements-upper-bound", synaptic_elements_init_ub, "The maximum number of vacant synaptic elements per neuron. Must be larger or equal to synaptic-elements-lower-bound.");

    double nu_axon{ SynapticElements::default_nu };
    app.add_option("--growth-rate-axon", nu_axon, "The growth rate for the axons. Default is 1e-5");

    double nu_dend_inh{ SynapticElements::default_nu };
    app.add_option("--growth-rate-dendrite-inh", nu_dend_inh, "The growth rate for the inhibitory dendrites. Default is 1e-5");

    double nu_dend_ex{ SynapticElements::default_nu };
    app.add_option("--growth-rate-dendrite-exc", nu_dend_ex, "The growth rate for the excitatory dendrites. Default is 1e-5");

    double min_calcium_axons{ SynapticElements::default_eta_Axons };
    app.add_option("--min-calcium-axons", min_calcium_axons, "The minimum intercellular calcium for axons to grow. Default is 0.4");

    double min_calcium_excitatory_dendrites{ SynapticElements::default_eta_Dendrites_exc };
    app.add_option("--min-calcium-excitatory-dendrites", min_calcium_excitatory_dendrites, "The minimum intercellular calcium for excitatory dendrites to grow. Default is 0.1");

    double min_calcium_inhibitory_dendrites{ SynapticElements::default_eta_Dendrites_inh };
    app.add_option("--min-calcium-inhibitory-dendrites", min_calcium_inhibitory_dendrites, "The minimum intercellular calcium for inhibitory dendrites to grow. Default is 0.0");

    std::string neuron_monitors_file{};
    auto* const monitor_option = app.add_option("--neuron-monitors", neuron_monitors_file,
        "The description which neurons to monitor. Format is <mpi_rank>:<neuron_id>;<mpi_rank>:<neuron_id>;...<area_name>;... where <mpi_rank> can be -1 to indicate \"on every rank\"");

    auto* const flag_monitor_all = app.add_flag("--neuron-monitors-all", "Monitors all neurons.");

    auto* const opt_flush_monintor = app.add_option("--monitor-flush-step", Config::flush_area_monitor_step, "The steps when to flush the neuron monitors. Must be > 0");

    double percentage_initial_fired_neurons{ 0.0 };
    app.add_option("--percentage-initial-fired-neurons", percentage_initial_fired_neurons, "The percentage of neurons that fired in the (imaginary) 0th step. Must be from [0.0, 1.0]. Default ist 0.0");

    monitor_option->excludes(flag_monitor_all);
    flag_monitor_all->excludes(monitor_option);

    opt_num_neurons->excludes(opt_file_positions);
    opt_num_neurons->excludes(opt_file_network);
    opt_num_neurons->excludes(opt_num_neurons_per_rank);

    opt_num_neurons_per_rank->excludes(opt_num_neurons);
    opt_num_neurons_per_rank->excludes(opt_file_positions);
    opt_num_neurons_per_rank->excludes(opt_file_network);

    opt_file_positions->excludes(opt_num_neurons);
    opt_file_network->excludes(opt_num_neurons);
    opt_file_positions->excludes(opt_num_neurons_per_rank);
    opt_file_network->excludes(opt_num_neurons_per_rank);

    opt_file_network->needs(opt_file_positions);

    opt_file_positions->check(CLI::ExistingPath);
    opt_file_network->check(CLI::ExistingDirectory);

    opt_file_calcium->excludes(opt_initial_calcium);
    opt_file_calcium->excludes(opt_target_calcium);
    opt_initial_calcium->excludes(opt_file_calcium);
    opt_target_calcium->excludes(opt_file_calcium);

    opt_file_calcium->check(CLI::ExistingFile);

    opt_file_enable_interrupts->check(CLI::ExistingFile);
    opt_file_disable_interrupts->check(CLI::ExistingFile);
    opt_file_creation_interrupts->check(CLI::ExistingFile);

    opt_file_external_stimulation->check(CLI::ExistingFile);

    opt_log_path->check(CLI::ExistingDirectory);

    opt_file_external_stimulation->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    if (static_cast<bool>(*opt_accept_criterion)) {
        RelearnException::check(is_barnes_hut(chosen_algorithm), "Acceptance criterion can only be set if Barnes-Hut is used");
        RelearnException::check(accept_criterion <= Constants::bh_max_theta, "Acceptance criterion must be smaller or equal to {}", Constants::bh_max_theta);
        RelearnException::check(accept_criterion > 0.0, "Acceptance criterion must be larger than 0.0");
    }

    if (static_cast<bool>(*opt_num_neurons)) {
        RelearnException::check(num_ranks == 1, "The option --num-neurons can only be used for one MPI rank. There are {} ranks.", num_ranks);
    }

    RelearnException::check(fraction_excitatory_neurons >= 0.0 && fraction_excitatory_neurons <= 1.0, "The fraction of excitatory neurons must be from [0.0, 1.0]");
    RelearnException::check(um_per_neuron > 0.0, "The micrometer per neuron must be greater than 0.0.");

    RelearnException::check(synaptic_elements_init_lb >= 0.0, "The minimum number of vacant synaptic elements must not be negative");
    RelearnException::check(synaptic_elements_init_ub >= synaptic_elements_init_lb, "The minimum number of vacant synaptic elements must not be larger than the maximum number");
    RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions) || static_cast<bool>(*opt_num_neurons_per_rank),
        "Missing command line option, need a total number of neurons (-n,--num-neurons), a number of neurons per rank (--num-neurons-per-rank), or file_positions (-f,--file).");
    RelearnException::check(openmp_threads > 0, "Number of OpenMP Threads must be greater than 0 (or not set).");
    RelearnException::check(calcium_decay > 0.0, "The calcium decay constant must be greater than 0.");

    RelearnException::check(percentage_initial_fired_neurons >= 0.0 && percentage_initial_fired_neurons <= 1.0, "The percentage of neurons that fired in the 0th step must be from [0.0, 1.0]: {}", percentage_initial_fired_neurons);

    if (static_cast<bool>(*opt_target_calcium)) {
        RelearnException::check(target_calcium >= SynapticElements::min_C_target, "Target calcium is smaller than {}", SynapticElements::min_C_target);
        RelearnException::check(target_calcium <= SynapticElements::max_C_target, "Target calcium is larger than {}", SynapticElements::max_C_target);
    }

    if (target_calcium_decay_type == TargetCalciumDecay::Relative) {
        RelearnException::check(target_calcium_decay_step > 0, "The target calcium decay step is 0 but must be larger than 0.");
        RelearnException::check(target_calcium_decay_amount < 1.0, "The target calcium decay amount must be smaller than 1.0 for relative decay.");
        RelearnException::check(target_calcium_decay_amount >= 0.0, "The target calcium decay amount must be larger than or equal to 0.0 for relative decay.");
    }

    if (target_calcium_decay_type == TargetCalciumDecay::Absolute) {
        RelearnException::check(target_calcium_decay_step > 0, "The target calcium decay step is 0 but must be larger than 0.");
        RelearnException::check(target_calcium_decay_amount > 0.0, "The target calcium decay amount must be larger than 0.0 for absolute decay.");
    }

    RelearnException::check(nu_axon >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
    RelearnException::check(nu_axon <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);
    RelearnException::check(nu_dend_inh >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
    RelearnException::check(nu_dend_inh <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);
    RelearnException::check(nu_dend_ex >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
    RelearnException::check(nu_dend_ex <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);

    RelearnException::check(Config::flush_area_monitor_step > 0, "The step for flushing the neuron monitors must be > 0.");

    omp_set_num_threads(openmp_threads);

    std::size_t current_seed = 0;
    boost::hash_combine(current_seed, my_rank.get_rank());
    boost::hash_combine(current_seed, random_seed);

    RandomHolder::seed_all(current_seed);

    auto init_log_files = [&]() -> void {
        if (static_cast<bool>(*opt_log_path)) {
            LogFiles::set_output_path(log_path);
        }
        if (static_cast<bool>(*opt_log_prefix)) {
            LogFiles::set_general_prefix(log_prefix);
        }

        LogFiles::set_log_status(LogFiles::EventType::Events, !static_cast<bool>(*flag_enable_printing_events));

        if (static_cast<bool>(*flag_disable_printing_positions)) {
            LogFiles::set_log_status(LogFiles::EventType::Positions, true);
        }

        if (static_cast<bool>(*flag_disable_printing_network)) {
            LogFiles::set_log_status(LogFiles::EventType::InNetwork, true);
            LogFiles::set_log_status(LogFiles::EventType::OutNetwork, true);
            LogFiles::set_log_status(LogFiles::EventType::NetworkInExcitatoryHistogramLocal, true);
            LogFiles::set_log_status(LogFiles::EventType::NetworkInInhibitoryHistogramLocal, true);
            LogFiles::set_log_status(LogFiles::EventType::NetworkOutHistogramLocal, true);
        }

        if (static_cast<bool>(*flag_disable_printing_plasticity)) {
            LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdate, true);
            LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdateCSV, true);
            LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdateLocal, true);
        }

        if (static_cast<bool>(*flag_disable_printing_calcium)) {
            LogFiles::set_log_status(LogFiles::EventType::CalciumValues, true);
            LogFiles::set_log_status(LogFiles::EventType::ExtremeCalciumValues, true);
        }

        if (static_cast<bool>(*flag_disable_printing_fire_rate)) {
            LogFiles::set_log_status(LogFiles::EventType::FireRates, true);
        }

        if (static_cast<bool>(*flag_disable_printing_overview)) {
            LogFiles::set_log_status(LogFiles::EventType::SynapticInput, true);
            LogFiles::set_log_status(LogFiles::EventType::NeuronsOverview, true);
            LogFiles::set_log_status(LogFiles::EventType::NeuronsOverviewCSV, true);
        }

        if (static_cast<bool>(*flag_disable_printing_area_mapping)) {
            LogFiles::set_log_status(LogFiles::EventType::AreaMapping, true);
        }

        LogFiles::init();
    };
    init_log_files();

    auto essentials = std::make_unique<Essentials>();

    // Rank 0 prints start time of simulation
    MPIWrapper::barrier();
    if (MPIRank::root_rank() == my_rank) {
        essentials->insert("Start", Timers::wall_clock_time());
        essentials->insert("Number-of-Ranks", num_ranks);
        essentials->insert("Number-of-Steps", simulation_steps);
        essentials->insert("Initial-Elements-Lower-Bound", synaptic_elements_init_lb);
        essentials->insert("Initial-Elements-Upper-Bound", synaptic_elements_init_ub);
        essentials->insert("Calcium-Target", target_calcium);
        essentials->insert("Beta", beta);
        essentials->insert("Calcium-Decay", calcium_decay);
        essentials->insert("Nu-Axons", nu_axon);
        essentials->insert("Nu-Dendrites inh", nu_dend_inh);
        essentials->insert("Nu-Dendrites ex", nu_dend_ex);
        essentials->insert("Retract-Ratio", retract_ratio);
        essentials->insert("Synapse-Conductance", synapse_conductance);
        essentials->insert("Background-Base", base_background_activity);
        essentials->insert("Background-Mean", background_activity_mean);
        essentials->insert("Background-Stddev", background_activity_stddev);

        essentials->insert("Log-path", log_path.string());
        essentials->insert("Algorithm", stringify(chosen_algorithm));
        essentials->insert("Neuron-model", stringify(chosen_neuron_model));
        essentials->insert("First-plasticity-step", first_plasticity_step);
        essentials->insert("Last-plasticity-step", last_plasticity_step);

        essentials->insert("Calcium-Minimum-Axons", min_calcium_axons);
        essentials->insert("Calcium-Minimum-Excitatory-Dendrites", min_calcium_excitatory_dendrites);
        essentials->insert("Calcium-Minimum-Inhibitory-Dendrites", min_calcium_inhibitory_dendrites);

        if (chosen_synapse_input_calculator_type == SynapticInputCalculatorType::Linear) {
            essentials->insert("Synapse-Input", "Linear");
        }

        if (chosen_kernel_type == KernelType::Gamma) {
            essentials->insert("Kernel-Type", "Gamma");
            essentials->insert("Kernel-Shape-Parameter", gamma_k);
            essentials->insert("Kernel-Scale-Parameter", gamma_theta);
        } else if (chosen_kernel_type == KernelType::Gaussian) {
            essentials->insert("Kernel-Type", "Gaussian");
            essentials->insert("Kernel-Translation-Parameter", gaussian_mu);
            essentials->insert("Kernel-Scale-Parameter", gaussian_sigma);
        } else if (chosen_kernel_type == KernelType::Linear) {
            essentials->insert("Kernel-Type", "Linear");
            essentials->insert("Kernel-Cut-off-Parameter", linear_cutoff);
        } else if (chosen_kernel_type == KernelType::Weibull) {
            essentials->insert("Kernel-Type", "Weibull");
            essentials->insert("Kernel-Shape-Parameter", weibull_k);
            essentials->insert("Kernel-Scale-Parameter", weibull_b);
        }

        if (static_cast<bool>(*opt_num_neurons)) {
            essentials->insert("number neurons", number_neurons);
            essentials->insert("Fraction-excitatory-neurons", fraction_excitatory_neurons);
            essentials->insert("um-per-neuron", um_per_neuron);
        } else if (static_cast<bool>(*opt_num_neurons_per_rank)) {
            essentials->insert("number neurons per rank", number_neurons_per_rank);
            essentials->insert("Fraction-excitatory-neurons", fraction_excitatory_neurons);
            essentials->insert("um-per-neuron", um_per_neuron);
        } else {
            essentials->insert("positions directory", file_positions.string());

            essentials->insert("network directory",
                file_network.string());
        }
        essentials->insert("external stimulation file", file_external_stimulation.string());
        essentials->insert("static neurons",
            static_neurons_str);
    }

    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "#step: creations deletions net");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "#step;creations;deletions;net");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "#step: creations deletions net");

    Timers::start(TimerRegion::INITIALIZATION);

    auto prepare_algorithm = [&]() -> void {
        // Set the correct kernel and initialize the MPIWrapper to return the correct type
        if (chosen_algorithm == AlgorithmEnum::BarnesHut) {
            Kernel<BarnesHutCell>::set_kernel_type(chosen_kernel_type);
            NodeCache<BarnesHutCell>::set_cache_type(chosen_cache_type);
            MPIWrapper::init_buffer_octree<BarnesHutCell>();
        } else if (chosen_algorithm == AlgorithmEnum::BarnesHutInverted) {
            Kernel<BarnesHutInvertedCell>::set_kernel_type(chosen_kernel_type);
            NodeCache<BarnesHutInvertedCell>::set_cache_type(chosen_cache_type);
            MPIWrapper::init_buffer_octree<BarnesHutInvertedCell>();
        } else {
            RelearnException::check(chosen_algorithm == AlgorithmEnum::Naive, "An algorithm was chosen that is not supported");
            Kernel<NaiveCell>::set_kernel_type(chosen_kernel_type);
            NodeCache<NaiveCell>::set_cache_type(chosen_cache_type);
            MPIWrapper::init_buffer_octree<NaiveCell>();
        }

        // Set the parameters for all kernel types, even though only one is used later one
        GammaDistributionKernel::set_k(gamma_k);
        GammaDistributionKernel::set_theta(gamma_theta);

        GaussianDistributionKernel::set_sigma(gaussian_sigma);
        GaussianDistributionKernel::set_mu(gaussian_mu);

        LinearDistributionKernel::set_cutoff(linear_cutoff);

        WeibullDistributionKernel::set_b(weibull_b);
        WeibullDistributionKernel::set_k(weibull_k);
    };
    prepare_algorithm();

    auto partition = std::make_shared<Partition>(num_ranks, my_rank);

    auto construct_subdomain = [&]() -> std::unique_ptr<NeuronToSubdomainAssignment> {
        if (static_cast<bool>(*opt_num_neurons)) {
            return std::make_unique<SubdomainFromNeuronDensity>(number_neurons, fraction_excitatory_neurons, um_per_neuron, partition);
        }

        if (static_cast<bool>(*opt_num_neurons_per_rank)) {
            return std::make_unique<SubdomainFromNeuronPerRank>(number_neurons_per_rank, fraction_excitatory_neurons, um_per_neuron, partition);
        }

        std::optional<std::filesystem::path> path_to_network{};
        if (static_cast<bool>(*opt_file_network)) {
            path_to_network = file_network;
        }

        return std::make_unique<MultipleSubdomainsFromFile>(file_positions, std::move(path_to_network), partition);
    };
    auto subdomain = construct_subdomain();

    auto construct_background_activity_calculator = [&]() -> std::unique_ptr<BackgroundActivityCalculator> {
        if (chosen_background_activity_calculator_type == BackgroundActivityCalculatorType::Null) {
            RelearnException::check(!static_cast<bool>(*opt_base_background_activity), "Setting the base background activity is not valid when choosing the null-background calculator (or not setting it at all).");
            RelearnException::check(!static_cast<bool>(*opt_mean_background_activity), "Setting the mean background activity is not valid when choosing the null-background calculator (or not setting it at all).");
            RelearnException::check(!static_cast<bool>(*opt_stddev_background_activity), "Setting the stddev background activity is not valid when choosing the null-background calculator (or not setting it at all).");
            return std::make_unique<NullBackgroundActivityCalculator>();
        }

        if (chosen_background_activity_calculator_type == BackgroundActivityCalculatorType::Constant) {
            RelearnException::check(!static_cast<bool>(*opt_mean_background_activity), "Setting the mean background activity is not valid when choosing the constant-background calculator.");
            RelearnException::check(!static_cast<bool>(*opt_stddev_background_activity), "Setting the stddev background activity is not valid when choosing the constant-background calculator.");
            return std::make_unique<ConstantBackgroundActivityCalculator>(base_background_activity);
        }

        if (chosen_background_activity_calculator_type == BackgroundActivityCalculatorType::Normal) {
            RelearnException::check(background_activity_stddev > 0.0, "When choosing the normal-background calculator, the standard deviation must be set to > 0.0.");
            return std::make_unique<NormalBackgroundActivityCalculator>(background_activity_mean, background_activity_stddev);
        }

        if (chosen_background_activity_calculator_type == BackgroundActivityCalculatorType::Flexible) {
            return std::make_unique<FlexibleBackgroundActivityCalculator>(flexible_file_path, my_rank, subdomain->get_local_area_translator());
        }

        RelearnException::check(chosen_background_activity_calculator_type == BackgroundActivityCalculatorType::FastNormal, "Chose a background activity calculator that is not implemented");
        RelearnException::check(background_activity_stddev > 0.0, "When choosing the fast-normal-background calculator, the standard deviation must be set to > 0.0.");

        // TODO Choose multiplier
        return std::make_unique<FastNormalBackgroundActivityCalculator>(background_activity_mean, background_activity_stddev, 1000);
    };
    auto background_activity_calculator = construct_background_activity_calculator();

    auto construct_stimulus = [&]() -> std::unique_ptr<Stimulus> {
        if (static_cast<bool>(*opt_file_external_stimulation)) {
            return std::make_unique<Stimulus>(file_external_stimulation, my_rank, subdomain->get_local_area_translator());
        }

        return std::make_unique<Stimulus>();
    };
    auto stimulus_calculator = construct_stimulus();

    auto construct_fired_status_communicator = [&]() -> std::unique_ptr<FiredStatusCommunicator> {
        if (chosen_fired_status_communicator_type == FiredStatusCommunicatorType::Map) {
            return std::make_unique<FiredStatusCommunicationMap>(MPIWrapper::get_num_ranks());
        }
        RelearnException::fail("Unknown fired status communicator");
    };

    auto construct_input = [&]() -> std::unique_ptr<SynapticInputCalculator> {
        auto fired_status_communicator = construct_fired_status_communicator();

        if (chosen_synapse_input_calculator_type == SynapticInputCalculatorType::Linear) {
            return std::make_unique<LinearSynapticInputCalculator>(synapse_conductance, std::move(fired_status_communicator));
        }
        RelearnException::fail("Unknown synaptic input calculator");
    };
    auto input_calculator = construct_input();

    auto construct_neuron_model = [&]() -> std::unique_ptr<NeuronModel> {
        if (chosen_neuron_model == NeuronModelEnum::Poisson) {
            return std::make_unique<models::PoissonModel>(h, std::move(input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                models::PoissonModel::default_x_0, models::PoissonModel::default_tau_x, models::PoissonModel::default_refractory_period);
        }

        if (chosen_neuron_model == NeuronModelEnum::Izhikevich) {
            return std::make_unique<models::IzhikevichModel>(h, std::move(input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                models::IzhikevichModel::default_a, models::IzhikevichModel::default_b, models::IzhikevichModel::default_c,
                models::IzhikevichModel::default_d, models::IzhikevichModel::default_V_spike, models::IzhikevichModel::default_k1,
                models::IzhikevichModel::default_k2, models::IzhikevichModel::default_k3);
        }

        if (chosen_neuron_model == NeuronModelEnum::FitzHughNagumo) {
            return std::make_unique<models::FitzHughNagumoModel>(h, std::move(input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
                models::FitzHughNagumoModel::default_a, models::FitzHughNagumoModel::default_b, models::FitzHughNagumoModel::default_phi);
        }

        RelearnException::check(chosen_neuron_model == NeuronModelEnum::AEIF, "Chose a neuron model that is not implemented");
        return std::make_unique<models::AEIFModel>(h, std::move(input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator),
            models::AEIFModel::default_C, models::AEIFModel::default_g_L, models::AEIFModel::default_E_L, models::AEIFModel::default_V_T,
            models::AEIFModel::default_d_T, models::AEIFModel::default_tau_w, models::AEIFModel::default_a, models::AEIFModel::default_b,
            models::AEIFModel::default_V_spike);
    };
    auto neuron_model = construct_neuron_model();

    auto construct_calcium_calculator = [&]() -> std::unique_ptr<CalciumCalculator> {
        auto calcium_calculator = std::make_unique<CalciumCalculator>(target_calcium_decay_type, target_calcium_decay_amount, target_calcium_decay_step, first_decay_step, last_decay_step);
        calcium_calculator->set_beta(beta);
        calcium_calculator->set_tau_C(calcium_decay);
        calcium_calculator->set_h(h);

        if (*opt_file_calcium) {
            auto [initial_calcium_calculator, target_calcium_calculator] = CalciumIO::load_initial_and_target_function(file_calcium);

            calcium_calculator->set_initial_calcium_calculator(std::move(initial_calcium_calculator));
            calcium_calculator->set_target_calcium_calculator(std::move(target_calcium_calculator));
        } else {
            auto initial_calcium_calculator = [initial = initial_calcium](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return initial; };
            calcium_calculator->set_initial_calcium_calculator(std::move(initial_calcium_calculator));

            auto target_calcium_calculator = [target = target_calcium](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return target; };
            calcium_calculator->set_target_calcium_calculator(std::move(target_calcium_calculator));
        }

        return calcium_calculator;
    };
    auto calcium_calculator = construct_calcium_calculator();

    auto axons_model = std::make_shared<SynapticElements>(ElementType::Axon, min_calcium_axons,
        nu_axon, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto excitatory_dendrites_model = std::make_shared<SynapticElements>(ElementType::Dendrite, min_calcium_excitatory_dendrites,
        nu_dend_ex, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto inhibitory_dendrites_model = std::make_shared<SynapticElements>(ElementType::Dendrite, min_calcium_inhibitory_dendrites,
        nu_dend_inh, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto construct_synapse_deletion_finder = [&]() -> std::unique_ptr<SynapseDeletionFinder> {
        if (chosen_synapse_deleter == SynapseDeletionFinderType::Random) {
            return std::make_unique<RandomSynapseDeletionFinder>();
        }
        RelearnException::fail("Unknown synapse deletion finder");
    };
    auto synapse_deletion_finder = construct_synapse_deletion_finder();

    Simulation sim(std::move(essentials), partition);
    sim.set_neuron_model(std::move(neuron_model));
    sim.set_calcium_calculator(std::move(calcium_calculator));
    sim.set_synapse_deletion_finder(std::move(synapse_deletion_finder));
    sim.set_axons(std::move(axons_model));
    sim.set_dendrites_ex(std::move(excitatory_dendrites_model));
    sim.set_dendrites_in(std::move(inhibitory_dendrites_model));

    sim.set_percentage_initial_fired_neurons(percentage_initial_fired_neurons);

    if (*opt_static_neurons) {
        auto static_neurons = MonitorParser::parse_my_ids(static_neurons_str, my_rank, subdomain->get_local_area_translator());
        sim.set_static_neurons(static_neurons);
    }

    if (is_barnes_hut(chosen_algorithm)) {
        sim.set_acceptance_criterion_for_barnes_hut(accept_criterion);
    }

    sim.set_algorithm(chosen_algorithm);
    sim.set_subdomain_assignment(std::move(subdomain));

    if (*opt_file_enable_interrupts) {
        auto enable_interrupts = InteractiveNeuronIO::load_enable_interrupts(file_enable_interrupts, my_rank);
        sim.set_enable_interrupts(std::move(enable_interrupts));
    }

    if (*opt_file_disable_interrupts) {
        auto disable_interrupts = InteractiveNeuronIO::load_disable_interrupts(file_disable_interrupts, my_rank);
        sim.set_disable_interrupts(std::move(disable_interrupts));
    }

    if (*opt_file_creation_interrupts) {
        auto creation_interrupts = InteractiveNeuronIO::load_creation_interrupts(file_creation_interrupts);
        sim.set_creation_interrupts(std::move(creation_interrupts));
    }

    RelearnException::check(plasticity_update_step > 0, "update-plasticity-step must be greater than 0");

    sim.set_update_plasticity_interval(Interval{ first_plasticity_step, last_plasticity_step, plasticity_update_step });
    sim.set_update_synaptic_elements_interval(Interval{ first_plasticity_step, last_plasticity_step, 1 });
    sim.set_log_calcium_interval(Interval{ 0, std::numeric_limits<RelearnTypes::step_type>::max(), calcium_log_step });
    sim.set_log_synaptic_input_interval(Interval{ 0, std::numeric_limits<RelearnTypes::step_type>::max(), synaptic_input_log_step });
    sim.set_log_network_interval(Interval{ 0, std::numeric_limits<RelearnTypes::step_type>::max(), network_log_step });
    sim.set_update_neuron_monitor_interval(Interval{ 0, std::numeric_limits<RelearnTypes::step_type>::max(), monitor_steps });
    sim.enable_area_monitor(static_cast<bool>(*flag_area_monitor), static_cast<bool>(*flag_area_monitor_connectivity));

    NeuronMonitor::log_frequency = monitor_steps;

    const auto steps_per_simulation = simulation_steps / monitor_steps;
    sim.increase_monitoring_capacity(steps_per_simulation);

    /**********************************************************************************/

    // The barrier ensures that every rank finished its local stores.
    // Otherwise, a "fast" rank might try to read from the RMA window of another
    // rank which has not finished (or even begun) its local stores
    MPIWrapper::barrier(); // TODO(future) Really needed?

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(MPIWindow::Window::Octree, my_rank, MPI_Locktype::Exclusive);

    sim.initialize();

    if (static_cast<bool>(*flag_monitor_all)) {
        const auto number_local_neurons = partition->get_number_local_neurons();
        for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
            sim.register_neuron_monitor(neuron_id);
        }
    } else if (!neuron_monitors_file.empty()) {
        const auto& my_neuron_ids_to_monitor = InteractiveNeuronIO::load_neuron_monitors(neuron_monitors_file, sim.get_neurons()->get_local_area_translator(), my_rank);
        for (const auto& neuron_id : my_neuron_ids_to_monitor) {
            sim.register_neuron_monitor(neuron_id);
        }
    }

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(MPIWindow::Window::Octree, my_rank);

    Timers::stop_and_add(TimerRegion::INITIALIZATION);

    MPIWrapper::barrier();

    auto simulate = [&sim, &simulation_steps]() {
        sim.simulate(simulation_steps);

        MPIWrapper::barrier();

        sim.finalize();
    };

    simulate();

    if (static_cast<bool>(*flag_interactive)) {
        while (true) {
            spdlog::info("Interactive run. Run another {} simulation steps? [y/n]\n", simulation_steps);
            char yn{ 'n' };
            std::cin >> std::ws >> yn;

            if (yn == 'n' || yn == 'N') {
                break;
            }

            if (yn == 'y' || yn == 'Y') {
                sim.increase_monitoring_capacity(steps_per_simulation);
                simulate();
            } else {
                RelearnException::fail("Input for question to run another {} simulation steps was not valid.", simulation_steps);
            }
        }
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, false, "Number of bytes send: {}, Number  of bytes received: {}, Number  of bytes accessed remotely: {}",
        MPIWrapper::get_number_bytes_sent(), MPIWrapper::get_number_bytes_received(), MPIWrapper::get_number_bytes_remote_accessed());

    MPIWrapper::finalize();

    return 0;
}
