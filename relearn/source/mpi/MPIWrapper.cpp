#include "MPIWrapper.h"
/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MPIWrapper.h"

#if RELEARN_MPI_FOUND

#include "util/Utility.h"

#include <mpi.h>

#include <bitset>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * IMPORTANT: MPI expects int array with receive counts and displacements for vector operations
 *
 * The int receive count limits the number of neurons per rank to ~2e9 (2^31 - 1).
 * The int displacement limits the *total* number of neurons to ~2e9 (2^31 - 1) which is a problem.
 * A way to solve this problem is to use communication operations without displacement.
 *
 * Solving the problem is future work. Until it is solved the total number of neurons is limited to 2^31-1.
 */

static std::unique_ptr<MPI_Op> minsummax{ nullptr };

static std::map<MPIWrapper::AsyncToken, MPI_Request> translation_map{};
static size_t current_token{ 0 };

std::unique_ptr<MPI_Op> translate_reduce_function(const MPIWrapper::ReduceFunction rf) {
    switch (rf) {
    case MPIWrapper::ReduceFunction::Min:
        return std::make_unique<MPI_Op>(MPI_MIN);

    case MPIWrapper::ReduceFunction::Max:
        return std::make_unique<MPI_Op>(MPI_MAX);

    case MPIWrapper::ReduceFunction::Sum:
        return std::make_unique<MPI_Op>(MPI_SUM);

    case MPIWrapper::ReduceFunction::MinSumMax:
        return std::make_unique<MPI_Op>(*minsummax);

    default:
        RelearnException::fail("In reduce, got wrong function");
        return nullptr;
    }
}

void MPIWrapper::init(int argc, char** argv) {
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_level_provided);

    init_globals();

    // Number of ranks must be 2^n so that
    // the connectivity update works correctly
    const std::bitset<sizeof(int) * 8> bitset_num_ranks(num_ranks);
    if (1 != bitset_num_ranks.count() && (MPIRank::root_rank() == my_rank)) {
        RelearnException::fail("MPIWrapper::init: Number of ranks must be of the form 2^n");
    }

    register_custom_function();

    const unsigned int num_digits = Util::num_digits(num_ranks - 1);
    my_rank_str = fmt::format("{1:0>{0}}", num_digits, my_rank.get_rank());

    LogFiles::print_message_rank(MPIRank::root_rank(), "I'm using the MPIWrapper");
}

void MPIWrapper::init_globals() {
    int num_ranks_int{};
    // NOLINTNEXTLINE
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks_int);

    int my_rank_int{};
    // NOLINTNEXTLINE
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_int);

    num_ranks = static_cast<size_t>(num_ranks_int);
    my_rank = MPIRank(my_rank_int);
}

void MPIWrapper::barrier() {
    const int error_code = MPI_Barrier(MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::barrier: Error code received: {}", error_code);
}

double MPIWrapper::reduce(double value, const ReduceFunction function, const MPIRank root_rank) {
    RelearnException::check(root_rank.is_initialized(), "MPIWrapper::reduce: root_rank was negative");
    const auto mpi_reduce_function = translate_reduce_function(function);

    double result = 0.0;
    const int error_code = MPI_Reduce(&value, &result, 1, MPI_DOUBLE, *mpi_reduce_function, root_rank.get_rank(), MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::reduce: Error code received: {}", error_code);

    add_to_sent(sizeof(double));
    if (my_rank == root_rank) {
        add_to_received(sizeof(double));
    }

    return result;
}

double MPIWrapper::all_reduce_double(const double value, const ReduceFunction function) {
    const auto mpi_reduce_function = translate_reduce_function(function);

    double result = 0.0;
    const int error_code = MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, *mpi_reduce_function, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::all_reduce_double: Error code received: {}", error_code);

    add_to_sent(sizeof(double));
    add_to_received(sizeof(double));

    return result;
}

uint64_t MPIWrapper::all_reduce_uint64(const uint64_t value, const ReduceFunction function) {
    const auto mpi_reduce_function = translate_reduce_function(function);

    uint64_t result = 0;
    const int error_code = MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, *mpi_reduce_function, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::all_reduce_uint64: Error code received: {}", error_code);

    add_to_sent(sizeof(std::uint64_t));
    add_to_received(sizeof(std::uint64_t));

    return result;
}

void MPIWrapper::reduce_double(const double* src, double* dst, const size_t size, const ReduceFunction function, const int root_rank) {
    const auto mpi_reduce_function = translate_reduce_function(function);

    RelearnException::check(size < static_cast<size_t>(std::numeric_limits<int>::max()), "MPIWrapper::reduce_double: Too much to reduce");

    const int error_code = MPI_Reduce(src, dst, static_cast<int>(size), MPI_DOUBLE, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::reduce_double: Error code received: {}", error_code);

    add_to_sent(sizeof(double) * size);
    if (my_rank.get_rank() == root_rank) {
        add_to_received(sizeof(double) * size);
    }
}

void MPIWrapper::reduce_int64(const int64_t* src, int64_t* dst, const size_t size, const ReduceFunction function, const int root_rank) {
    const auto mpi_reduce_function = translate_reduce_function(function);

    RelearnException::check(size < static_cast<size_t>(std::numeric_limits<int>::max()), "MPIWrapper::reduce_int64: Too much to reduce");

    const int error_code = MPI_Reduce(src, dst, static_cast<int>(size), MPI_INT64_T, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::reduce_int64: Error code received: {}", error_code);

    add_to_sent(sizeof(std::uint64_t) * size);
    if (my_rank.get_rank() == root_rank) {
        add_to_received(sizeof(std::uint64_t) * size);
    }
}

std::vector<size_t> MPIWrapper::all_to_all(const std::vector<size_t>& src) {
    const size_t count_src = src.size();
    std::vector<size_t> dst(count_src, 0);

    const int error_code = MPI_Alltoall(src.data(), sizeof(size_t), MPI_CHAR, dst.data(), sizeof(size_t), MPI_CHAR, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::all_to_all: Error code received: {}", error_code);

    add_to_sent(sizeof(std::uint64_t) * count_src);
    add_to_received(sizeof(std::uint64_t) * count_src);

    return dst;
}

MPIWrapper::AsyncToken MPIWrapper::async_s(const void* buffer, const int count, const int rank) {
    RelearnException::check(rank >= 0, "MPIWrapper::async_s: Error in async s, rank is <= 0");

    const auto token = current_token++;
    MPI_Request& translated_token = translation_map[token];

    const int error_code = MPI_Isend(buffer, count, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &translated_token);
    RelearnException::check(error_code == 0, "MPIWrapper::async_s: Error code received: {}", error_code);

    add_to_sent(count);

    return token;
}

MPIWrapper::AsyncToken MPIWrapper::async_recv(void* buffer, const int count, const int rank) {
    RelearnException::check(rank >= 0, "MPIWrapper::async_recv: Error in async recv, rank is <= 0");

    const auto token = current_token++;
    MPI_Request& translated_token = translation_map[token];

    const int error_code = MPI_Irecv(buffer, count, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &translated_token);
    RelearnException::check(error_code == 0, "MPIWrapper::async_recv: Error code received: {}", error_code);

    add_to_received(count);

    return token;
}

int MPIWrapper::translate_lock_type(const MPI_Locktype lock_type) {
    switch (lock_type) {
    case MPI_Locktype::Exclusive:
        return MPI_LOCK_EXCLUSIVE;
    case MPI_Locktype::Shared:
        return MPI_LOCK_SHARED;
    }

    return 0;
}

void MPIWrapper::reduce(const void* src, void* dst, const int size, const ReduceFunction function, const int root_rank) {
    const auto mpi_reduce_function = translate_reduce_function(function);

    const auto* s_ptr = static_cast<const int64_t*>(src);

    const int error_code = MPI_Reduce(src, dst, size, MPI_CHAR, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::reduce: Error code received: {}", error_code);

    add_to_sent(size);
    if (my_rank.get_rank() == root_rank) {
        add_to_received(size);
    }
}

void MPIWrapper::all_gather(const void* own_data, void* buffer, const int size) {
    const int error_code = MPI_Allgather(own_data, size, MPI_CHAR, buffer, size, MPI_CHAR, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::all_gather: Error code received: {}", error_code);

    add_to_sent(size);
    add_to_received(size);
}

void MPIWrapper::all_gather_inl(void* ptr, const int count) {
    RelearnException::check(count > 0, "MPIWrapper::all_gather_inl: Error in all gather , count is not greater than 0");

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    const int error_code = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ptr, count, MPI_CHAR, MPI_COMM_WORLD);
    RelearnException::check(error_code == 0, "MPIWrapper::all_gather_inl: Error code received: {}", error_code);

    add_to_sent(count);
    add_to_received(count);
}

void MPIWrapper::get(MPIWindow::Window window_type, void* origin, const size_t size, const int target_rank, const uint64_t displacement, const int number_elements) {
    const MPI_Aint displacement_mpi(displacement);
    const auto& window = MPIWindow::mpi_windows[window_type]; // NOLINT(readability-qualified-auto, llvm-qualified-auto)
    const auto download_size = size * number_elements;

    RelearnException::check(download_size < std::numeric_limits<int>::max(), "MPIWrapper::get: Too much to download via RMA");

    const auto download_size_int = static_cast<int>(download_size);

    MPI_Request request{};
    const auto error_code = MPI_Rget(origin, download_size_int, MPI_CHAR, target_rank, displacement_mpi, download_size_int, MPI_CHAR, window->window, &request);
    RelearnException::check(error_code == 0, "MPIWrapper::get: Error code received: {}", error_code);

    MPI_Status mpi_status{};
    const auto error_code_wait = MPI_Wait(&request, &mpi_status);
    RelearnException::check(error_code_wait == MPI_SUCCESS, "MPIWrapper::get: Wait error code received: {}", error_code_wait);
    RelearnException::check(mpi_status.MPI_ERROR == MPI_SUCCESS, "MPIWrapper::get: Wait error code received: {}", mpi_status.MPI_ERROR);

    add_to_remotely_accessed(download_size);
}

int MPIWrapper::get_num_ranks() {
    RelearnException::check(num_ranks >= 1, "MPIWrapper::get_num_ranks: MPIWrapper is not initialized");
    return num_ranks;
}

MPIRank MPIWrapper::get_my_rank() {
    RelearnException::check(my_rank.is_initialized(), "MPIWrapper::get_my_rank: MPIWrapper is not initialized");
    return my_rank;
}

std::string MPIWrapper::get_my_rank_str() {
    RelearnException::check(my_rank.is_initialized(), "MPIWrapper::get_my_rank_str: MPIWrapper is not initialized");
    return my_rank_str;
}

void MPIWrapper::wait_all_tokens(const std::vector<AsyncToken>& tokens) {
    const int size = static_cast<int>(tokens.size());

    std::vector<MPI_Request> requests(size);

    for (auto i = 0; i < size; i++) {
        const auto& request = tokens[i];
        MPI_Request translated_token = translation_map[request];
        translation_map.erase(request);
        requests[i] = translated_token;
    }

    std::vector<MPI_Status> statuses(size);
    const int error_code = MPI_Waitall(size, requests.data(), statuses.data());

    if (error_code != 0) {
        std::stringstream ss{};
        ss << "I'm " << my_rank << ", i have " << size << " tokens and my errors are:\n";
        for (const auto& status : statuses) {
            ss << status.MPI_ERROR << ' ' << status.MPI_SOURCE << ' ' << status.MPI_TAG << '\n';
        }

        std::cout << ss.str();
        fflush(stdout);
    }

    RelearnException::check(error_code == 0, "MPIWrapper::wait_all_tokens: Error code received: {}", error_code);
}

void MPIWrapper::register_custom_function() {
    minsummax = std::make_unique<MPI_Op>();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    MPI_Op_create(reinterpret_cast<MPI_User_function*>(MPIUserDefinedOperation::min_sum_max), 1, minsummax.get());
}

void MPIWrapper::free_custom_function() {
    MPI_Op_free(minsummax.get());
    minsummax.reset();
}

void MPIWrapper::lock_window(MPIWindow::Window window_type, const MPIRank rank, const MPI_Locktype lock_type) {
    RelearnException::check(rank.is_initialized(), "MPIWrapper::lock_window: rank was: {}", rank);
    const auto lock_type_int = translate_lock_type(lock_type);

    const auto& window = MPIWindow::mpi_windows[window_type]; // NOLINT(readability-qualified-auto, llvm-qualified-auto)
    const int error_code = MPI_Win_lock(lock_type_int, rank.get_rank(), MPI_MODE_NOCHECK, window->window);
    RelearnException::check(error_code == 0, "MPIWrapper::lock_window: Error code received: {}", error_code);
}

void MPIWrapper::unlock_window(MPIWindow::Window window_type, const MPIRank rank) {
    RelearnException::check(rank.is_initialized(), "MPIWrapper::unlock_window: rank was: {}", rank);
    const auto& window = MPIWindow::mpi_windows[window_type]; // NOLINT(readability-qualified-auto, llvm-qualified-auto)
    const int error_code = MPI_Win_unlock(rank.get_rank(), window->window);
    RelearnException::check(error_code == 0, "MPIWrapper::unlock_window: Error code received: {}", error_code);
}

void MPIWrapper::lock_window_all(MPIWindow::Window window_type) {
    const auto& window = MPIWindow::mpi_windows[window_type]; // NOLINT(readability-qualified-auto, llvm-qualified-auto)
    const int error_code = MPI_Win_lock_all(MPI_MODE_NOCHECK, window->window);
    RelearnException::check(error_code == MPI_SUCCESS, "MPIWrapper::unlock_window: Error code received: {}", error_code);
}

void MPIWrapper::unlock_window_all(MPIWindow::Window window_type) {
    const auto& window = MPIWindow::mpi_windows[window_type]; // NOLINT(readability-qualified-auto, llvm-qualified-auto)
    const int error_code = MPI_Win_unlock_all(window->window);
    RelearnException::check(error_code == MPI_SUCCESS, "MPIWrapper::unlock_window: Error code received: {}", error_code);
}

void MPIWrapper::sync_window(MPIWindow::Window window_type) {
    const auto& window = MPIWindow::mpi_windows[window_type]; // NOLINT(readability-qualified-auto, llvm-qualified-auto)
    const int error_code = MPI_Win_sync(window->window);
    RelearnException::check(error_code == MPI_SUCCESS, "MPIWrapper::unlock_window: Error code received: {}", error_code);
}

void MPIWrapper::finalize() {
    barrier();
    free_custom_function();

    for (auto i = 0U; i < MPIWindow::num_windows; i++) {
        auto& window = MPIWindow::mpi_windows[i];
        if (window != nullptr && window->initialized) {

            const int error_code_1 = MPI_Win_free(&window->window);
            RelearnException::check(error_code_1 == 0, "MPIWrapper::finalize: Error code received: {}", error_code_1);
            const int error_code_2 = MPI_Free_mem(window->my_base_pointer);
            RelearnException::check(error_code_2 == 0, "MPIWrapper::finalize: Error code received: {}", error_code_2);

            window->my_base_pointer = nullptr;
            MPIWindow::mpi_windows[i].release();
        }
    }

    const int error_code = MPI_Finalize();
    RelearnException::check(error_code == 0, "MPIWrapper::finalize: Error code received: {}", error_code);
}

// This combination function assumes that it's called with the correct MPI datatype
void MPIUserDefinedOperation::min_sum_max(const void* invec, void* inoutvec, const int* const len, [[maybe_unused]] void* dtype) {
    const auto real_length = *len / 3;

    // NOLINTNEXTLINE
    const auto* in = reinterpret_cast<const double*>(invec);
    // NOLINTNEXTLINE
    auto* inout = reinterpret_cast<double*>(inoutvec);

    for (int i = 0; i < real_length; i++) {
        // NOLINTNEXTLINE
        inout[3 * i] = std::min(in[3 * i], inout[3 * i]);
        // NOLINTNEXTLINE
        inout[3 * i + 1] += in[3 * i + 1];
        // NOLINTNEXTLINE
        inout[3 * i + 2] = std::max(in[3 * i + 2], inout[3 * i + 2]);
    }
}

#endif
