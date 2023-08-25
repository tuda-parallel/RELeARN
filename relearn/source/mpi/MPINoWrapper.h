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

#if !RELEARN_MPI_FOUND

#include "io/LogFiles.h"
#include "mpi/CommunicationMap.h"
#include "util/MemoryHolder.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"

#include <array>
#include <any>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

using MPI_Request = int;

constexpr inline auto MPI_LOCK_EXCLUSIVE = 0;
constexpr inline auto MPI_LOCK_SHARED = 1;

template <typename T>
class OctreeNode;
class RelearnTest;

enum class MPI_Locktype : int {
    Exclusive = MPI_LOCK_EXCLUSIVE,
    Shared = MPI_LOCK_SHARED,
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
    inline static std::array<std::any, num_windows> mpi_windows{};
};

class MPINoWrapper {
    friend class RelearnTest;

public:
    enum class ReduceFunction : char {
        Min = 0,
        Max = 1,
        Sum = 2,
        None = 3,
        MinSumMax = 100
    };

    using AsyncToken = MPI_Request;

    static void init(int argc, char** argv);

    template <typename AdditionalCellAttributes>
    static void init_buffer_octree() {
        const auto octree_node_size = sizeof(OctreeNode<AdditionalCellAttributes>);
        const auto max_num_objects = Constants::mpi_alloc_mem / octree_node_size;

        create_rma_window<OctreeNode<AdditionalCellAttributes>>(MPIWindow::Window::Octree, max_num_objects, 1);
        auto& data = MPIWindow::mpi_windows[MPIWindow::Window::Octree];
        auto& base_ptr = std::any_cast<std::vector<OctreeNode<AdditionalCellAttributes>>&>(data);

        std::span<OctreeNode<AdditionalCellAttributes>> span{ base_ptr.data(), max_num_objects };
        MemoryHolder<AdditionalCellAttributes>::init(span);

        LogFiles::print_message_rank(MPIRank::root_rank(), "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
    }

    static void barrier();

    [[nodiscard]] static double reduce(double value, ReduceFunction function, MPIRank root_rank);

    [[nodiscard]] static double all_reduce_double(double value, ReduceFunction function);

    [[nodiscard]] static uint64_t all_reduce_uint64(uint64_t value, ReduceFunction function);

    [[nodiscard]] static std::vector<size_t> all_to_all(const std::vector<size_t>& src);

    template <typename T, size_t size>
    [[nodiscard]] static std::array<T, size> reduce(const std::array<T, size>& src, ReduceFunction function, MPIRank root_rank) {
        RelearnException::check(root_rank.is_initialized(), "In MPIWrapper::reduce, root_rank was negative");

        std::array<T, size> dst{};
        reduce(src.data(), dst.data(), src.size() * sizeof(T), function, root_rank.get_rank());

        return dst;
    }

    template <typename RequestType>
    [[nodiscard]] static CommunicationMap<RequestType> exchange_requests(const CommunicationMap<RequestType>& outgoing_requests) {
        return outgoing_requests;
    }

    template <typename T>
    static std::vector<T> all_gather(T own_data) {
        std::vector<T> results(1);
        all_gather(&own_data, results.data(), sizeof(T));
        return results;
    }

    template <typename T>
    static void all_gather_inline(std::span<T> buffer) {
    }

    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const MPIRank target_rank, const OctreeNode<AdditionalCellAttributes>* src, const int number_elements) {
        for (auto i = 0; i < number_elements; i++) {
            dst[i] = src[i];
        }
    }

    template <typename T>
    static void create_rma_window(MPIWindow::Window window_type, std::uint64_t num_elements, size_t number_ranks) {
        RelearnException::check(!MPIWindow::mpi_windows[window_type].has_value(), "MPIWrapper::create_rma_window: Window {} is already created", window_type);

        std::vector<T> vector{};
        vector.resize(num_elements);
        MPIWindow::mpi_windows[window_type] = std::move(vector);
    }

    template <typename T>
    static std::vector<T> get_from_window(MPIWindow::Window window_type, int target_rank, uint64_t index, size_t number_elements) {

        const auto& begin_it = std::any_cast<std::vector<T>&>(MPIWindow::mpi_windows[window_type]).begin();

        return std::vector<T>(begin_it + index, begin_it + index + number_elements);
    }

    template <typename T>
    static void set_in_window(MPIWindow::Window window_type, uint64_t index, const T& element) {
        std::any_cast<std::vector<T>&>(MPIWindow::mpi_windows[window_type])[index] = element;
    }

    template <typename T>
    static void set_in_window(MPIWindow::Window window_type, uint64_t index, const std::vector<T>& element) {
        const auto& begin_it = std::any_cast<std::vector<T>&>(MPIWindow::mpi_windows[window_type]).begin();
        std::copy(element.begin(), element.end(), begin_it + index);
    }

    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const MPIRank target_rank, const uint64_t offset, const int number_elements) {
        RelearnException::fail("MPINoWrapper::download_octree_node: Cannot perform the offset version without MPI.");
    }

    [[nodiscard]] static int get_num_ranks();

    [[nodiscard]] static MPIRank get_my_rank();

    [[nodiscard]] static std::string get_my_rank_str();

    template <typename T>
    static std::vector<std::vector<T>> exchange_values(const std::vector<std::vector<T>>& values) {
        RelearnException::check(values.size() == 1 && values[0].empty(), "MPINoWrapper::exchange_values: There were values!");
        std::vector<std::vector<T>> return_value(1, std::vector<T>(0));
        return return_value;
    }

    static void lock_window(MPIWindow::Window window, MPIRank rank, MPI_Locktype lock_type);

    static void unlock_window(MPIWindow::Window window, MPIRank rank);

    static void lock_window_all(MPIWindow::Window window);

    static void unlock_window_all(MPIWindow::Window window);

    static void sync_window(MPIWindow::Window window_type);

    static void start_measuring_communication() noexcept {};

    static void stop_measureing_communication() noexcept {};

    static uint64_t get_number_bytes_sent() noexcept {
        return 0;
    }

    static uint64_t get_number_bytes_received() noexcept {
        return 0;
    }

    static uint64_t get_number_bytes_remote_accessed() noexcept {
        return 0;
    }

    static void finalize();

private:
    MPINoWrapper() = default;

    static inline std::string my_rank_str{ '0' };

    static void all_gather(const void* own_data, void* buffer, int size);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank);
};

#endif
