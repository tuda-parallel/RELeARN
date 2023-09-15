# RELeARN
The connectivity of the brain is constantly changing. Even in the mature brain, new connections between neurons are formed, and existing ones are deleted, a phenomenon called structural plasticity.
Understanding the dynamics of these neuronal networks is crucial to understanding learning, memory, and diseases such as Alzheimer’s.
The Model of Structural Plasticity enables simulation with structural plasticity as described by  Markus Butz-Ostendorf and Arjen van Ooyen in [*A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization after Focal Retinal Lesions*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003259).
However, with a naive approach to modeling structural plasticity, we need to calculate the probability of forming a synapse between each neuron pair in each plasticity step.
This is unfeasible for a larger number of neurons.
The RELeARN (REwiring of LARge-scale Neural networks) code addresses this challenge via an approximation, reducing the required computations from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$. Using it, we could conduct simulations with up to one billion neurons. RELeARN is implemented in C++ and parallelized with MPI and OpenMP, making it suitable for almost every HPC system.

## Installation
Using the project requires CMake and a C++ compiler capable of C++20.
Optionally, you can use MPI and OpenMP to parallelize the computation.

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
Please cite RELeARN in your publications if it helps your research:
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


