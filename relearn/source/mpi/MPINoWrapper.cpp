#include "MPINoWrapper.h"

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#if !RELEARN_MPI_FOUND

#include "algorithm/Cells.h"
#include "io/LogFiles.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"
#include "util/Utility.h"

#include <bitset>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

void MPINoWrapper::init(int argc, char** argv) {
    LogFiles::print_message_rank(MPIRank::root_rank(), "I'm using the MPINoWrapper");
}

void MPINoWrapper::barrier() {
}

[[nodiscard]] double MPINoWrapper::reduce(double value, ReduceFunction /*function*/, MPIRank /*root_rank*/) {
    return value;
}

[[nodiscard]] double MPINoWrapper::all_reduce_double(double value, ReduceFunction /*function*/) {
    return value;
}

[[nodiscard]] uint64_t MPINoWrapper::all_reduce_uint64(uint64_t value, ReduceFunction /*function*/) {
    return value;
}

std::vector<size_t> MPINoWrapper::all_to_all(const std::vector<size_t>& src) {
    return src;
}

void MPINoWrapper::reduce(const void* src, void* dst, int size, ReduceFunction /*function*/, int /*root_rank*/) {
    std::memcpy(dst, src, size);
}

void MPINoWrapper::all_gather(const void* own_data, void* buffer, int size) {
    std::memcpy(buffer, own_data, size);
}

int MPINoWrapper::get_num_ranks() {
    return 1;
}

MPIRank MPINoWrapper::get_my_rank() {
    return MPIRank::root_rank();
}

std::string MPINoWrapper::get_my_rank_str() {
    return my_rank_str;
}

void MPINoWrapper::lock_window(MPIWindow::Window window, MPIRank rank, MPI_Locktype /*lock_type*/) {
    RelearnException::check(rank.is_initialized(), "rank was: %d", rank);
}

void MPINoWrapper::unlock_window(MPIWindow::Window window, MPIRank rank) {
    RelearnException::check(rank.is_initialized(), "rank was: %d", rank);
}

void MPINoWrapper::lock_window_all(MPIWindow::Window window) {
}

void MPINoWrapper::unlock_window_all(MPIWindow::Window window) {
}

void MPINoWrapper::sync_window(MPIWindow::Window window) {
}

void MPINoWrapper::finalize() {
}

#endif
