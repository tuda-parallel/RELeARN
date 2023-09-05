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

## Citation
```
@article{rinke2018,
author = {Rinke, Sebastian and Butz-Ostendorf, Markus and Hermanns, Marc-Andr{\'{e}} and Naveau, Mika{\"{e}}l and Wolf, Felix},
title = {A Scalable Algorithm for Simulating the Structural Plasticity of the Brain},
journal = {Journal of Parallel and Distributed Computing},
volume = {120},
year = {2018},
pages = {251--266},
doi = {10.1016/j.jpdc.2017.11.019}
}
```

## Publications
1) Rinke, S., Butz-Ostendorf, M., Hermanns, M.A., Naveau, M., & Wolf, F. (2018). _A Scalable Algorithm for Simulating the Structural Plasticity of the Brain_. Journal of Parallel and Distributed Computing, 120, 251–266. [PDF](https://apps.fz-juelich.de/jsc-pubsystem/aigaion/attachments/rinke_ea.pdf-5c72f91c90128cfe0433a70f61fa4693.pdf)
2) Czappa, F., Geiß, A., & Wolf, F. (2023). _Simulating Structural Plasticity of the Brain more Scalable than Expected_. Journal of Parallel and Distributed Computing, 171, 24–27. [PDF](https://arxiv.org/pdf/2210.05267.pdf)
