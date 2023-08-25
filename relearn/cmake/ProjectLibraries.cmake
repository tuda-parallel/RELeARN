# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

add_library(project_libraries INTERFACE)

include(FetchContent)

find_package(Threads REQUIRED)
target_link_libraries(project_libraries INTERFACE Threads::Threads)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(project_options INTERFACE OpenMP::OpenMP_CXX)
endif()

if(UNIX)
  target_link_libraries(project_options INTERFACE stdc++fs)
endif()

option(ENABLE_MPI "Enable mpi" ON)
if(ENABLE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    target_compile_definitions(project_options
                               INTERFACE -DMPI_FOUND=$<BOOL:${MPI_CXX_FOUND}>)

    # fix CI build issue
    get_target_property(mpi_cxx_compile_options MPI::MPI_CXX
                        INTERFACE_COMPILE_OPTIONS)

    if("${mpi_cxx_compile_options}" MATCHES "-flto=auto")
      message(
        WARNING
          "MPI_CXX was compiled with -flto=auto and -ffat-lto-objects, removing lto flags to prevent CI build failure"
      )
      string(
        REPLACE "-flto=auto"
                ""
                mpi_cxx_compile_options
                ${mpi_cxx_compile_options})
      string(
        REPLACE "-ffat-lto-objects"
                ""
                mpi_cxx_compile_options
                ${mpi_cxx_compile_options})
    endif()

    set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS
                                                  "${mpi_cxx_compile_options}")

    target_link_libraries(project_libraries INTERFACE MPI::MPI_CXX)
  endif()
endif()

if(WIN32)
 #  FetchContent_Declare(
 #    boostrandom
 #    GIT_REPOSITORY https://github.com/boostorg/random.git
 #    GIT_TAG master)

 #  FetchContent_GetProperties(boostrandom)
 #  if(NOT boostrandom_POPULATED)
 #    FetchContent_Populate(boostrandom)
 #    add_subdirectory(${boostrandom_SOURCE_DIR} ${boostrandom_BINARY_DIR})
 #  endif()

  target_include_directories(project_options INTERFACE SYSTEM external)
  # target_link_libraries(project_libraries INTERFACE boostorg::random)
else()
  set(BOOST_ENABLE_CMAKE ON)
  find_package(Boost REQUIRED COMPONENTS RANDOM)
  # target_link_libraries(project_options INTERFACE Boost::random)

  target_link_libraries(project_options INTERFACE Boost::random)
endif()

# declaration

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 9.1.0)

# spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.11.0)

# range-v3
FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3
  GIT_TAG 0.12.0)

# make available

# fmt
FetchContent_MakeAvailable(fmt)
get_target_property(fmt_includes fmt INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(fmt PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                     "${fmt_includes}")
target_link_libraries(project_libraries INTERFACE fmt)

# spdlog
FetchContent_MakeAvailable(spdlog)
get_target_property(spdlog_includes spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                        "${spdlog_includes}")
target_link_libraries(project_libraries INTERFACE spdlog)

# range-v3
FetchContent_MakeAvailable(range-v3)
get_target_property(range-v3_includes range-v3 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(range-v3 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                          "${range-v3_includes}")
target_link_libraries(project_libraries INTERFACE range-v3)

target_link_libraries(project_options INTERFACE Boost::random)

# set compile commands back to on
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
