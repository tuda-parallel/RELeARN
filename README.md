# RELeARN

RELeARN (REwiring of LARge-scale Neural networks) is a project developed at Technische Universität Darmstadt under the supervision of [Prof. Dr. Felix Wolf](https://www.informatik.tu-darmstadt.de/parallel/parallel_programming/index.en.jsp). It allows fast simulations of structural plasticity as described by Markus Butz-Ostendorf and Arjen van Ooyen in *A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization after Focal Retinal Lesions*.

## Installation
Using the project requires CMake and a C++ compiler capable of C++20.
Optionally, you can use MPI and OpenMP to parallelize the computation.

## Organisation

The simulation is found in the subdirectory ./relearn/
In ./graph/ is a tool that analyzes the generated networks with respect to different graph metrics
./paper/ includes multiple publications which are relevant to the implemented algorithm and model

## Dependencies

- Parsing the command line arguments is done with [CLI11](https://github.com/CLIUtils/CLI11)
- Logging is done with [spdlog](https://github.com/gabime/spdlog)
- Tests are written with [GoogleTest](https://github.com/google/googletest)
- [Boost](https://github.com/boostorg/boost)

## Tested module combination

* cmake/3.26.1
* gcc/11.2.0
* mpich/3.4.2
* boost/1.77.0
* openucx/1.11.2