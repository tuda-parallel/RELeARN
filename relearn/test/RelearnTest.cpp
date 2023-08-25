/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "algorithm/Cells.h"
#include "io/LogFiles.h"

#include "mpi/MPIWrapper.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"

#include <chrono>
#include <iostream>

int RelearnTest::iterations = 10;
double RelearnTest::eps = 0.001;

bool RelearnTest::use_predetermined_seed = false;
unsigned int RelearnTest::predetermined_seed = 2818124801;

std::vector<OctreeNode<BarnesHutCell>> holder_bh_cells{};
std::vector<OctreeNode<BarnesHutInvertedCell>> holder_bhi_cells{};

RelearnTest::RelearnTest() {
}

RelearnTest::~RelearnTest() {
}

void RelearnTest::SetUp() {
    if (use_predetermined_seed) {
        std::cerr << "Using predetermined seed: " << predetermined_seed << '\n';
        mt.seed(predetermined_seed);
    } else {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto time_since_epoch = now.time_since_epoch();
        const auto time_since_epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();

        const auto seed = static_cast<unsigned int>(time_since_epoch_ns);

        std::cerr << "Test seed: " << seed << '\n';
        mt.seed(seed);
    }
}

void RelearnTest::TearDown() {
    // Remove tmp files
    for (auto const& entry : std::filesystem::recursive_directory_iterator("./")) {
        if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".tmp") {
            std::filesystem::remove(entry);
            std::cerr << "REMOVED " << entry.path() << std::endl;
        }
    }

    std::cerr << "Test finished\n";
}

RelearnMemoryTest::RelearnMemoryTest() {
    MemoryHolder<BarnesHutCell>::init(holder_bh_cells);
    MemoryHolder<BarnesHutInvertedCell>::init(holder_bhi_cells);
}

RelearnMemoryTest::~RelearnMemoryTest() {
}

int main(int argc, char** argv) {
    MPIWrapper::init(1, argv);
    ::testing::InitGoogleTest(&argc, argv);

    holder_bh_cells.resize(1024 * 1024);
    holder_bhi_cells.resize(1024 * 1024);

    RelearnException::hide_messages = true;
    LogFiles::disable = true;

    MemoryHolder<BarnesHutCell>::init(holder_bh_cells);
    MemoryHolder<BarnesHutInvertedCell>::init(holder_bhi_cells);

    const auto tests_return_code = RUN_ALL_TESTS();

    RelearnException::hide_messages = false;
    LogFiles::disable = false;

    MPIWrapper::finalize();

    return tests_return_code;
}
