add_library(relearn_harness STATIC)
target_sources(
        relearn_harness
        PRIVATE
        harness.cpp
        # for visual studio
        ${relearn_harness_additional_files})

target_include_directories(relearn_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(relearn_harness PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(relearn_harness PUBLIC project_options project_libraries)
target_link_libraries(relearn_harness PRIVATE relearn_lib)

get_target_property(relearn_harness_sources relearn_harness SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_harness_sources})
