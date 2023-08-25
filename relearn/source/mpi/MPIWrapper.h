#pragma once

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
#include "util/ranges/Functional.hpp"

#if !RELEARN_MPI_FOUND
#include "MPINoWrapper.h"

using MPIWrapper = MPINoWrapper;
#else // #if MPI_FOUND

#include "io/LogFiles.h"
#include "mpi/CommunicationMap.h"
#include "util/MemoryHolder.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <vector>
#include <mpi.h>

#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>

template <typename T>
class OctreeNode;
class RelearnTest;

/**
 * This enum allows a type safe choice of locking types for memory windows
 */
enum class MPI_Locktype {
    Exclusive,
    Shared,
};

namespace MPIUserDefinedOperation {
/**
 * @brief Provides a custom reduction function for MPI that simultaneously computes the min, sum, and max of multiple values.
 * @param invec A double* (cast to int* because of MPI) with a tuple of data to reduce.
 *      Size must be at least *len / sizeof(double) / 3
 * @param inoutvec A double* (cast to int* because of MPI) with a tuple of data to reduce.
 *      Size must be at least *len / sizeof(double) / 3.
 *      Is also used as return value.
 * @param len The length of a tuple of data. Is only accessed hat *len.
 * @param dtype Unused
 */
void min_sum_max(const void* invec, void* inoutvec, const int* len, void* dtype);
} // namespace MPIUserDefinedOperation

/**
 * This class provides a static interface to every kind of MPI functionality that should be called from other classes.
 * It wraps functionality in a C++ type safe manner.
 * The first call must be MPIWrapper::init(...) and the last one MPIWrapper::finalize(), not calling any of those inbetween.
 */

/**
 * Represents one rma window for mpi
 */
struct RMAWindow {
    MPI_Win window{};
    std::uint64_t size{};
    std::vector<int64_t> base_pointers{};
    void* my_base_pointer{};

    bool initialized = false;

    RMAWindow() = delete;

    RMAWindow(void* ptr)
        : my_base_pointer(ptr) {
    }

    RMAWindow(const RMAWindow& other) = delete;
    RMAWindow(RMAWindow&& other) = delete;

    RMAWindow& operator=(const RMAWindow& other) = delete;
    RMAWindow& operator=(RMAWindow&& other) = default;
};

class MPIWindow {
public:
    /**
     * enum for the available rma windows
     */
    enum Window {
        Octree = 0U,
        FireHistory = 1U,
    };
    constexpr static size_t num_windows = 2;

    /**
     * Array that stores the rma windows based on the Window enum
     */
    inline static std::array<std::unique_ptr<RMAWindow>, num_windows> mpi_windows{};
};

class MPIWrapper {
    friend class RelearnTest;

public:
    /**
     * This enum serves as a marker for the function that should be used in reductions.
     * ReduceFunction::None is not supported and always triggers a RelearnException.
     */
    enum class ReduceFunction : char {
        Min = 0,
        Max = 1,
        Sum = 2,
        None = 3,
        MinSumMax = 100
    };

    using AsyncToken = size_t;

    /**
     * @brief Initializes the local MPI implementation via MPI_Init_Thread;
     *      initializes the global variables and the custom functions. Must be called before any other call to a member function.
     * @param argc Is passed to MPI_Init_Thread
     * @param argv Is passed to MPI_Init_Thread
     */
    static void init(int argc, char** argv);

    /**
     * @brief Initializes the shared RMA memory. Must be called before any call involving OctreeNode*.
     */
    template <typename AdditionalCellAttributes>
    static void init_buffer_octree() {
        static bool is_initialized = false;
        if (is_initialized) {
            return;
        }

        is_initialized = true;

        const auto octree_node_size = sizeof(OctreeNode<AdditionalCellAttributes>);
        size_t max_num_objects = init_window<AdditionalCellAttributes>(Constants::mpi_alloc_mem);

        // NOLINTNEXTLINE
        auto* cast = reinterpret_cast<OctreeNode<AdditionalCellAttributes>*>(MPIWindow::mpi_windows[MPIWindow::Window::Octree]->my_base_pointer);

        std::span<OctreeNode<AdditionalCellAttributes>> span{ cast, max_num_objects };
        MemoryHolder<AdditionalCellAttributes>::init(span);

        LogFiles::print_message_rank(MPIRank::root_rank(), "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
    }

    /**
     * @brief The calling MPI rank halts until all MPI ranks reach the method.
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void barrier();

    /**
     * @brief Reduces a value for every MPI rank with a reduction function such that the root_rank has the final result
     * @param value The local value that should be reduced
     * @param function The reduction function, should be associative and commutative
     * @param root_rank The MPI rank that shall hold the final result
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The result of the reduction; A dummy value on every other MPI rank
     */
    [[nodiscard]] static double reduce(double value, ReduceFunction function, MPIRank root_rank);

    /**
     * @brief Reduces a value for every MPI rank with a reduction function such that every rank has the final result
     * @param value The local value that should be reduced
     * @param function The reduction function, should be associative and commutative
     * @exception Throws a RelearnException if an MPI error occurs
     * @return The final result of the reduction
     */
    [[nodiscard]] static double all_reduce_double(double value, ReduceFunction function);

    /**
     * @brief Reduces a value for every MPI rank with a reduction function such that every rank has the final result
     * @param value The local value that should be reduced
     * @param function The reduction function, should be associative and commutative
     * @exception Throws a RelearnException if an MPI error occurs
     * @return The final result of the reduction
     */
    [[nodiscard]] static std::uint64_t all_reduce_uint64(std::uint64_t value, ReduceFunction function);

    /**
     * @brief Reduces multiple values for every MPI rank with a reduction function such that the root_rank has the final result. The reduction is performed componentwise
     * @param src The local array of values that shall be reduced
     * @param function The reduction function, should be associative and commutative
     * @param root_rank The MPI rank that shall hold the final result
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The results of the componentwise reduction; A dummy value on every other MPI rank
     */
    template <size_t size>
    [[nodiscard]] static std::array<double, size> reduce(const std::array<double, size>& src, const ReduceFunction function, const MPIRank root_rank) {
        RelearnException::check(root_rank.is_initialized(), "MPIWrapper::reduce: root_rank was negative");

        std::array<double, size> dst{ 0.0 };
        reduce_double(src.data(), dst.data(), size, function, root_rank.get_rank());

        return dst;
    }

    /**
     * @brief Reduces multiple values for every MPI rank with a reduction function such that the root_rank has the final result. The reduction is performed componentwise
     * @param src The local array of values that shall be reduced
     * @param function The reduction function, should be associative and commutative
     * @param root_rank The MPI rank that shall hold the final result
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The results of the componentwise reduction; A dummy value on every other MPI rank
     */
    template <size_t size>
    [[nodiscard]] static std::array<int64_t, size> reduce(const std::array<int64_t, size>& src, const ReduceFunction function, const MPIRank root_rank) {
        RelearnException::check(root_rank.is_initialized(), "MPIWrapper::reduce: root_rank was negative");

        std::array<int64_t, size> dst{ 0 };
        reduce_int64(src.data(), dst.data(), size, function, root_rank.get_rank());

        return dst;
    }

    /**
     * @brief Exchanges one size_t between every pair for MPI ranks
     * @param src The values that shall be sent to the other MPI ranks. MPI rank i receives src[i]
     * @return The values that were transmitted by the other MPI ranks. MPI rank i sent <return>[i]
     */
    static std::vector<size_t> all_to_all(const std::vector<size_t>& src);

    /**
     * @brief Gathers one value for each MPI rank into a vector on all MPI ranks
     * @param own_data The local value that shall be sent to all MPI ranks
     * @return The data from all MPI ranks. The value of MPI rank i is in results[i]
     * @exception Throws a RelearnException if an MPI error occurs
     */
    template <typename T>
    [[nodiscard]] static std::vector<T> all_gather(T own_data) {
        std::vector<T> results(get_num_ranks());
        all_gather(&own_data, results.data(), sizeof(T));
        return results;
    }

    /**
     * @brief Gathers multiple values for each MPI rank into the provided buffer on all MPI ranks
     * @param buffer The buffer to which the data will be written. The values of MPI rank i are in ptr[count * i + {0, 1, ..., count - 1}]
     * @param count The number of local values that shall be gathered
     * @exception Throws a RelearnException if an MPI error occurs or if count <= 0
     */
    template <typename T>
    static void all_gather_inline(std::span<T> buffer) {
        all_gather_inl(buffer.data(), static_cast<int>(buffer.size_bytes()));
    }

    /**
     * @brief Exchanges vectors of data with all MPI ranks
     * @tparam T The type that should be exchanged
     * @param values The values that should be exchanged. values[i] should be send to MPI rank i
     * @exception Throws a RelearnException if values.size() does not match the number of MPI ranks
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template <typename T>
    [[nodiscard]] static std::vector<std::vector<T>> exchange_values(const std::vector<std::vector<T>>& values) {
        RelearnException::check(values.size() == num_ranks,
            "MPIWrapper::exchange_values: There are too many values: {} for the number of ranks {}!", values.size(), num_ranks);

        const auto request_sizes = values
            | ranges::views::transform(ranges::size)
            | ranges::to_vector;

        std::vector<size_t> response_sizes = all_to_all(request_sizes);

        std::vector<std::vector<T>> retrieved_data(num_ranks);
        for (auto rank = 0; rank < num_ranks; rank++) {
            retrieved_data[rank].resize(response_sizes[rank]);
        }

        std::vector<AsyncToken> async_tokens{};
        for (const auto rank : MPIRank::range(num_ranks) | ranges::views::filter(not_equal_to(my_rank))) {
            const auto token = async_receive(std::span{ retrieved_data[rank.get_rank()] }, rank.get_rank());
            async_tokens.emplace_back(token);
        }

        for (const auto rank : MPIRank::range(num_ranks) | ranges::views::filter(not_equal_to(my_rank))) {
            const auto token = async_send(std::span{ values[rank.get_rank()] }, rank.get_rank());
            async_tokens.emplace_back(token);
        }

        wait_all_tokens(async_tokens);
        return retrieved_data;
    }

    /**
     * @brief Exchanges data with all MPI ranks
     * @tparam RequestType The type that should be exchanged
     * @param outgoing_requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template <typename RequestType>
    [[nodiscard]] static CommunicationMap<RequestType> exchange_requests(const CommunicationMap<RequestType>& outgoing_requests) {
        const auto number_ranks = get_num_ranks();
        const auto my_rank = get_my_rank();

        const auto& number_requests_outgoing = outgoing_requests.get_request_sizes_vector();
        const auto& number_requests_incoming = all_to_all(number_requests_outgoing);

        const auto size_hint = outgoing_requests.size();
        CommunicationMap<RequestType> incoming_requests(number_ranks, size_hint);
        incoming_requests.resize(number_requests_incoming);

        std::vector<AsyncToken> async_tokens{};

        for (const auto rank : MPIRank::range(number_ranks) | ranges::views::filter([&incoming_requests](const auto& rank) { return incoming_requests.contains(rank); })) {
            auto* buffer = incoming_requests.get_data(rank);
            const auto size = incoming_requests.size(rank);

            const auto token = async_receive(incoming_requests.get_span(rank), rank.get_rank());
            async_tokens.emplace_back(token);
        }

        for (const auto rank : MPIRank::range(number_ranks) | ranges::views::filter([&outgoing_requests](const auto& rank) { return outgoing_requests.contains(rank); })) {
            const auto* buffer = outgoing_requests.get_data(rank);
            const auto size = outgoing_requests.size(rank);

            const auto token = async_send(outgoing_requests.get_span(rank), rank.get_rank());
            async_tokens.emplace_back(token);
        }

        // Wait for all sends and receives to complete
        wait_all_tokens(async_tokens);

        return incoming_requests;
    }

    /**
     * @brief Downloads an OctreeNode on another MPI rank
     * @param dst The local node which shall be the copy of the remote node
     * @param target_rank The other MPI rank
     * @param src The pointer to the remote node, must be inside the remote's memory window
     * @param number_elements The number of elements to download
     * @exception Throws a RelearnException if an MPI error occurs, if number_elements <= 0, or if target_rank < 0
     */
    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const MPIRank target_rank, const OctreeNode<AdditionalCellAttributes>* src, const int number_elements) {
        RelearnException::check(number_elements > 0, "MPIWrapper::download_octree_node: number_elements is not positive");
        RelearnException::check(target_rank.is_initialized(), "MPIWrapper::download_octree_node: target_rank is not initialized");

        const auto& base_ptrs = get_base_pointers(MPIWindow::Window::Octree);
        RelearnException::check(target_rank.get_rank() < base_ptrs.size(), "MPIWrapper::download_octree_node: target_rank is larger than the pointers");
        const auto displacement = int64_t(src) - base_ptrs[target_rank.get_rank()];

        RelearnException::check(displacement >= 0, "MPIWrapper::download_octree_node: displacement is too small: {:X} - {:X}", int64_t(src), base_ptrs[target_rank.get_rank()]);

        get(MPIWindow::Window::Octree, dst, sizeof(OctreeNode<AdditionalCellAttributes>), target_rank.get_rank(), displacement, number_elements);
    }

    /**
     * @brief Downloads an OctreeNode on another MPI rank
     * @param dst The local node which shall be the copy of the remote node
     * @param target_rank The other MPI rank
     * @param offset The offset in the remote's memory window
     * @param number_elements The number of elements to download
     * @exception Throws a RelearnException if an MPI error occurs, if number_elements <= 0, if offset < 0, or if target_rank < 0
     */
    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const MPIRank target_rank, const std::uint64_t offset, const int number_elements) {
        RelearnException::check(number_elements > 0, "MPIWrapper::download_octree_node: number_elements is not positive");
        RelearnException::check(target_rank.is_initialized(), "MPIWrapper::download_octree_node: target_rank is not initialized");

        const auto& base_ptrs = get_base_pointers(MPIWindow::Window::Octree);
        RelearnException::check(target_rank.get_rank() < base_ptrs.size(), "MPIWrapper::download_octree_node: target_rank is larger than the pointers");

        get(MPIWindow::Window::Octree, dst, sizeof(OctreeNode<AdditionalCellAttributes>), target_rank.get_rank(), offset, number_elements);
    }

    /**
     * @brief Returns the number of MPI ranks
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The number of MPI ranks
     */
    [[nodiscard]] static int get_num_ranks();

    /**
     * @brief Returns the current MPI rank's id
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The current MPI rank's id
     */
    [[nodiscard]] static MPIRank get_my_rank();

    /**
     * @brief Returns the current MPI rank's id as string
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The current MPI rank's id as string
     */
    [[nodiscard]] static std::string get_my_rank_str();

    /**
     * @brief Locks the memory window on another MPI rank with the desired read/write protections
     * @param window RMA window that shall be locked
     * @param rank The other MPI rank
     * @param lock_type The type of locking
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void lock_window(MPIWindow::Window window, MPIRank rank, MPI_Locktype lock_type);

    /**
     * @brief Unlocks the memory window on another MPI rank
     * @param window RMA window that shall be unlocked
     * @param The other MPI rank
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void unlock_window(MPIWindow::Window window, MPIRank rank);

    /**
     * @brief Locks the memory window on all other MPI rank with shared protections
     * @param window RMA window that shall be locked
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void lock_window_all(MPIWindow::Window window);

    /**
     * @brief Unlocks the memory window on all other MPI rank with shared protections
     * @param window RMA window that shall be unlocked
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void unlock_window_all(MPIWindow::Window window);

    /**
     * @brief Syncs the memory window
     * @param window RMA window that shall be synced
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void sync_window(MPIWindow::Window window_type);

    /**
     * Downloads data from the rma window
     * @tparam T Type of the data
     * @param window_type Window type
     * @param target_rank Rank of the rma window
     * @param index Start index that shall be copied
     * @param number_elements Number of elements that shall be copied
     * @return Copied vector from the rma window
     */
    template <typename T>
    static std::vector<T> get_from_window(MPIWindow::Window window_type, int target_rank, uint64_t index, size_t number_elements) {
        const auto size = sizeof(T);
        const auto displacement = index * size;
        std::vector<T> buffer;
        buffer.resize(number_elements);
        lock_window(window_type, MPIRank{ target_rank }, MPI_Locktype::Shared);
        get(window_type, buffer.data(), size, target_rank, displacement, number_elements);
        unlock_window(window_type, MPIRank{ target_rank });
        return buffer;
    }

    /**
     * @brief Returns an approximation of how many bytes were sent.
     *      E.g., it only counts reduce once, so this is an underapproximation.
     * @return The number of bytes sent
     */
    static uint64_t get_number_bytes_sent() noexcept {
        return bytes_sent.load(std::memory_order::relaxed);
    }

    /**
     * @brief Returns an approximation of how many bytes were received.
     *      E.g., it only counts reduce on the root rank, so this is an underapproximation.
     * @return The number of bytes received
     */
    static std::uint64_t get_number_bytes_received() noexcept {
        return bytes_received.load(std::memory_order::relaxed);
    }

    /**
     * @brief Returns the number of bytes accessed remotely in windows
     * @return The number of bytes remotely accessed
     */
    static std::uint64_t get_number_bytes_remote_accessed() noexcept {
        return bytes_remote.load(std::memory_order::relaxed);
    }

    /**
     * @brief Starts measuring the communication that is sent, received, or remotely accessed
     */
    static void start_measuring_communication() noexcept {
        measure_communication.test_and_set(std::memory_order::relaxed);
    }

    /**
     * @brief Stops measuring the communication that is sent, received, or remotely accessed
     */
    static void stop_measureing_communication() noexcept {
        measure_communication.clear(std::memory_order::relaxed);
    }

    /**
     * @brief Finalizes the local MPI implementation.
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void finalize();

    /**
     * Creates and initializes a new rma window
     * @param window The rma window that shall be created
     * @param size Size of the window
     * @param number_ranks Number of ranks
     */
    template <typename T>
    static void create_rma_window(MPIWindow::Window window_type, std::uint64_t number_elements, size_t number_ranks) {
        RelearnException::check(MPIWindow::mpi_windows[window_type] == nullptr, "MPIWrapper::create_rma_window: Window {} is already created", window_type);

        void* ptr = nullptr;

        const auto size = number_elements * sizeof(T);

        if (const auto error_code = MPI_Alloc_mem(size, MPI_INFO_NULL, &ptr); error_code != 0) {
            RelearnException::fail("Allocating the shared memory returned the error: {}", error_code);
        }

        auto window = std::make_unique<RMAWindow>(ptr);

        window->size = size;

        const int error_code_2 = MPI_Win_create(window->my_base_pointer, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window->window);
        RelearnException::check(error_code_2 == MPI_SUCCESS, "MPI_RMA_MemAllocator::init: Error code received: {}", error_code_2);

        auto base_ptr = reinterpret_cast<int64_t>(window->my_base_pointer);
        std::vector<int64_t> base_pointers{};
        base_pointers.resize(number_ranks);

        const int error_code_3 = MPI_Allgather(&base_ptr, 1, MPI_AINT, base_pointers.data(), 1, MPI_AINT, MPI_COMM_WORLD);
        RelearnException::check(error_code_3 == MPI_SUCCESS, "MPI_RMA_MemAllocator::init: Error code received: {}", error_code_3);

        window->base_pointers = base_pointers;

        window->initialized = true;
        MPIWindow::mpi_windows[window_type] = std::move(window);
    }

    /**
     * @brief Stores an element of elements in a rma window
     * @param window_type Window type
     * @param Start index in the window
     * @param element The element to copy in the rma window
     */
    template <typename T>
    static void set_in_window(MPIWindow::Window window_type, uint64_t index, const T& element) {
        auto* base_ptr = static_cast<T*>(MPIWindow::mpi_windows[window_type]->my_base_pointer);
        auto* ptr = base_ptr + index;
        lock_window(window_type, my_rank, MPI_Locktype::Exclusive);
        *ptr = element;
        unlock_window(window_type, my_rank);
    }

    /**
     * @brief Stores a vector of elements in a rma window
     * @param window_type Window type
     * @param Start index in the window
     * @param vector The vector to copy in the rma window
     */
    template <typename T>
    static void set_in_window(MPIWindow::Window window_type, uint64_t index, const std::vector<T>& vector) {
        const auto size = sizeof(T);
        auto* base_ptr = static_cast<T*>(MPIWindow::mpi_windows[window_type]->my_base_pointer);
        auto* ptr = base_ptr + index;
        lock_window(window_type, my_rank, MPI_Locktype::Exclusive);
        std::copy(vector.begin(), vector.end(), ptr);
        unlock_window(window_type, my_rank);
    }

private:
    MPIWrapper() = default;

    static void add_to_sent(std::uint64_t number_bytes) {
        if (measure_communication.test(std::memory_order::relaxed)) {
            bytes_sent.fetch_add(number_bytes, std::memory_order::relaxed);
        }
    }

    static void add_to_received(std::uint64_t number_bytes) {
        if (measure_communication.test(std::memory_order::relaxed)) {
            bytes_received.fetch_add(number_bytes, std::memory_order::relaxed);
        }
    }

    static void add_to_remotely_accessed(std::uint64_t number_bytes) {
        if (measure_communication.test(std::memory_order::relaxed)) {
            bytes_remote.fetch_add(number_bytes, std::memory_order::relaxed);
        }
    }

    template <typename AdditionalCellAttributes>
    [[nodiscard]] static size_t init_window(size_t size_requested) {

        const auto octree_node_size = sizeof(OctreeNode<AdditionalCellAttributes>);

        // Number of objects "size_requested" Bytes correspond to
        const auto max_num_objects = size_requested / octree_node_size;

        // Store size of MPI_COMM_WORLD
        int my_num_ranks = -1;
        const int error_code_1 = MPI_Comm_size(MPI_COMM_WORLD, &my_num_ranks);
        RelearnException::check(error_code_1 == 0, "MPI_RMA_MemAllocator::init: Error code received: {}", error_code_1);

        const auto num_ranks = static_cast<size_t>(my_num_ranks);

        // Set window's displacement unit
        create_rma_window<OctreeNode<AdditionalCellAttributes>>(MPIWindow::Window::Octree, max_num_objects, num_ranks);

        return max_num_objects;
    }

    static void init_globals();

    static void register_custom_function();

    static void free_custom_function();

    static void all_gather(const void* own_data, void* buffer, int size);

    static void all_gather_inl(void* ptr, int count);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank);

    // NOLINTNEXTLINE
    [[nodiscard]] static AsyncToken async_s(const void* buffer, int count, int rank);

    // NOLINTNEXTLINE
    [[nodiscard]] static AsyncToken async_recv(void* buffer, int count, int rank);

    [[nodiscard]] static int translate_lock_type(MPI_Locktype lock_type);

    static void get(MPIWindow::Window window, void* origin, size_t size, int target_rank, std::uint64_t displacement, int number_elements);

    static void reduce_int64(const int64_t* src, int64_t* dst, size_t size, ReduceFunction function, int root_rank);

    static void reduce_double(const double* src, double* dst, size_t size, ReduceFunction function, int root_rank);

    /**
     * @brief Returns the base addresses of the memory windows of all memory windows.
     * @param window RMA window that we want the base pointers from
     * @return The base addresses of the memory windows. The base address for MPI rank i
     *      is found at <return>[i]
     */
    [[nodiscard]] static const std::vector<int64_t>& get_base_pointers(MPIWindow::Window window_type) noexcept {
        return MPIWindow::mpi_windows[window_type]->base_pointers;
    }

    /**
     * @brief Sends data to another MPI rank asynchronously
     * @param buffer The buffer that shall be sent to the other MPI rank
     * @param rank The other MPI rank that shall receive the data
     * @param token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    [[nodiscard]] static AsyncToken async_send(std::span<T> buffer, const int rank) {
        return async_s(buffer.data(), static_cast<int>(buffer.size_bytes()), rank);
    }

    /**
     * @brief Receives data from another MPI rank asynchronously
     * @param buffer The buffer where the data shall be written to
     * @param rank The other MPI rank that shall send the data
     * @param token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    [[nodiscard]] static AsyncToken async_receive(std::span<T> buffer, const int rank) {
        return async_recv(buffer.data(), static_cast<int>(buffer.size_bytes()), rank);
    }

    /**
     * @brief Waits for all supplied tokens
     * @param The tokens to be waited on
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void wait_all_tokens(const std::vector<AsyncToken>& tokens);

    static inline int num_ranks{ 0 }; // Number of ranks in MPI_COMM_WORLD
    static inline MPIRank my_rank{ MPIRank::uninitialized_rank() }; // My rank in MPI_COMM_WORLD

    static inline int thread_level_provided{ -1 }; // Thread level provided by MPI

    // NOLINTNEXTLINE
    static inline std::string my_rank_str{ "-1" };

    static inline std::atomic<std::uint64_t> bytes_sent{ 0 };
    static inline std::atomic<std::uint64_t> bytes_received{ 0 };
    static inline std::atomic<std::uint64_t> bytes_remote{ 0 };
    static inline std::atomic_flag measure_communication{};
};

#endif
