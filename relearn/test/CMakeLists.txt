enable_testing()

# add custom check target, builds and tests all tests

# processor count detection and option
include(ProcessorCount)
ProcessorCount(proc_count)
set(CTEST_JOB_COUNT
        "${proc_count}"
        CACHE STRING "number of jobs that run the tests")

set(check_args -j${CTEST_JOB_COUNT} --output-on-failure ${CTEST_EXTRA_ARGS})

add_custom_target(
        check
        COMMAND
        ${CMAKE_COMMAND} -E env
        $<$<BOOL:${ENABLE_SANITIZER_ADDRESS}>:ASAN_OPTIONS=fast_unwind_on_malloc=0,symbolize=1>
        $<$<BOOL:${ENABLE_SANITIZER_LEAK}>:LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/cmake/sanitizers/lsan.supp>
        ${CMAKE_CTEST_COMMAND} ${check_args}
        USES_TERMINAL)

if (${COVERAGE_REPORT_AVAILABLE})
    add_custom_target(
            check-coverage
            DEPENDS check
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
            coverage-report)
endif ()

# # googletest# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG release-1.12.1)

# google test
set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_target_property(gtest_main_includes gtest_main
        INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gtest_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${gtest_main_includes}")

file(
        WRITE ${CMAKE_BINARY_DIR}/test_get_relearn_path.cpp
        "#include <filesystem>\nstd::filesystem::path get_relearn_path() { return \"${CMAKE_SOURCE_DIR}\"; }"
)

add_executable(relearn_tests ${SRC_TST})
target_sources(
        relearn_tests
        PRIVATE ${CMAKE_BINARY_DIR}/test_get_relearn_path.cpp
        RelearnTest.cpp

        background_activity/test_background_activity.cpp

        barnes_hut/test_barnes_hut.cpp
        barnes_hut/test_barnes_hut_base.cpp
        barnes_hut/test_barnes_hut_base_inverted.cpp
        barnes_hut/test_barnes_hut_inverted.cpp

        calcium_calculator/test_calcium_calculator.cpp

        cell/test_cell.cpp

        connector/test_connector.cpp

        event/test_event.cpp


        helper/test_distant_neuron_request.cpp
        helper/test_rank_neuron_id.cpp
        helper/test_synapse_creation_request.cpp
        helper/test_synapse_deletion_request.cpp

        interval/test_interval.cpp

        kernel/test_kernel.cpp
        kernel/test_kernel_gamma.cpp
        kernel/test_kernel_gaussian.cpp
        kernel/test_kernel_linear.cpp
        kernel/test_kernel_weibull.cpp

        local_area_translator/test_local_area_translator.cpp

        memory_holder/test_memory_holder.cpp

        misc/test_misc.cpp

        mpi/test_mpi_rank.cpp

        network_graph/test_network_graph.cpp

        neuron_assignment/test_neuron_assignment.cpp

        neuron_extra_info/test_neuron_extra_info.cpp

        neuron_io/test_neuron_io.cpp
        neuron_io/test_background_activity_io.cpp

        neuron_models/test_neuron_models.cpp

        neurons/test_neurons.cpp

        neurons/test_neurons.cpp

        neuron_id/test_neuron_id.cpp

        octree/test_octree.cpp

        octree_node/test_octree_node.cpp

        parser/test_interval_parser.cpp
        parser/test_monitor_parser.cpp
        parser/test_neuronid_parser.cpp
        parser/test_step_parser.cpp

        partition/test_partition.cpp

        probability_picker/test_probability_picker.cpp

        random/test_random.cpp

        relearn_exception/test_relearn_exception.cpp

        semi_stable_vector/test_semi_stable_vector.cpp

        space_filling_curve/test_space_filling_curve.cpp

        stack/test_stack.cpp

        statistical_measure/test_statistical_measure.cpp

        stimulus/test_stimulus.cpp

        string_util/test_string_util.cpp

        synaptic_elements/test_synaptic_elements.cpp

        synaptic_input/test_synaptic_input.cpp

        timers/test_timers.cpp

        vector/test_vector.cpp

        # for visual studio
        ${relearn_tests_additional_files})

target_include_directories(relearn_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(relearn_tests PRIVATE project_options project_libraries)
target_link_libraries(relearn_tests PRIVATE relearn_lib relearn_harness)
target_link_libraries(relearn_tests PRIVATE gtest)

add_test(NAME SerialTests COMMAND relearn_tests)
add_dependencies(check relearn_tests)

get_target_property(relearn_tests_sources relearn_tests SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_tests_sources})
