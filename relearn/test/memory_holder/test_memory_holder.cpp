/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_memory_holder.h"

#include "adapter/random/RandomAdapter.h"

#include "algorithm/Cells.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"
#include "util/shuffle/shuffle.h"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, NaiveCell>;
TYPED_TEST_SUITE(MemoryHolderTest, test_types);

TYPED_TEST(MemoryHolderTest, testInit) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);
    ASSERT_EQ(MH::get_size(), 1024);
    ASSERT_EQ(MH::get_current_memory().size(), span_memory.size());
    ASSERT_EQ(MH::get_current_memory().data(), span_memory.data());
    ASSERT_EQ(MH::get_current_filling(), 0);

    MH::make_all_available();
    ASSERT_EQ(MH::get_size(), 1024);
    ASSERT_EQ(MH::get_current_memory().size(), span_memory.size());
    ASSERT_EQ(MH::get_current_memory().data(), span_memory.data());
    ASSERT_EQ(MH::get_current_filling(), 0);

    std::vector<OctreeNode<AdditionalCellAttributes>> memory2(6 * 1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory2(memory2);

    MH::init(span_memory2);
    ASSERT_EQ(MH::get_size(), 6 * 1024);
    ASSERT_EQ(MH::get_current_memory().size(), span_memory2.size());
    ASSERT_EQ(MH::get_current_memory().data(), span_memory2.data());
    ASSERT_EQ(MH::get_current_filling(), 0);
}

TYPED_TEST(MemoryHolderTest, testGetAvailableException) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        ASSERT_THROW(auto val = MH::get_available(nullptr, i), RelearnException);
    }

    OctreeNode<AdditionalCellAttributes> root{};

    for (auto i = Constants::number_oct; i < Constants::number_oct * 1000; i++) {
        ASSERT_THROW(auto val = MH::get_available(&root, i), RelearnException);
    }

    std::vector<OctreeNode<AdditionalCellAttributes>> memory2(Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory2(memory2);

    MH::init(span_memory2);

    std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct> arr{};
    for (auto i = 0U; i < Constants::number_oct; i++) {
        arr[i] = MH::get_available(&root, i);
    }

    for (auto& node : memory) {
        for (auto i = 0U; i < Constants::number_oct; i++) {
            ASSERT_THROW(auto val = MH::get_available(&node, i), RelearnException);
        }
    }
}

TYPED_TEST(MemoryHolderTest, testGetAvailable) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    OctreeNode<AdditionalCellAttributes> root{};

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto* ptr = MH::get_available(&root, i);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, i);

        root.set_child(ptr, i);
    }

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root.get_child(child_idx);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = MH::get_available(child, i);
            auto dist = std::distance(span_memory.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + child_idx * Constants::number_oct + i);

            child->set_child(ptr, i);
        }
    }

    std::vector<OctreeNode<AdditionalCellAttributes>> memory2(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory2(memory2);

    MH::init(span_memory2);

    OctreeNode<AdditionalCellAttributes> root2{};

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto* ptr = MH::get_available(&root2, i);
        auto dist = std::distance(span_memory2.data(), ptr);

        ASSERT_EQ(dist, i);

        root2.set_child(ptr, i);
    }

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root2.get_child(child_idx);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = MH::get_available(child, i);
            auto dist = std::distance(span_memory2.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + child_idx * Constants::number_oct + i);

            child->set_child(ptr, i);
        }
    }
}

TYPED_TEST(MemoryHolderTest, testGetAvailableDisorganized) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    OctreeNode<AdditionalCellAttributes> root{};

    const std::vector<unsigned int> indices = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];

        auto* ptr = MH::get_available(&root, child_index);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, child_index);

        root.set_child(ptr, child_index);
    }

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];

        auto* child = root.get_child(child_index);

        const std::vector<unsigned int> indices_child = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

        for (auto j = 0U; j < Constants::number_oct; j++) {
            auto child_child_index = indices_child[j];

            auto* ptr = MH::get_available(child, child_child_index);
            auto dist = std::distance(span_memory.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + i * Constants::number_oct + child_child_index);

            child->set_child(ptr, child_child_index);
        }
    }
}

TYPED_TEST(MemoryHolderTest, testGetAvailableFull) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    const auto number_objects = RandomAdapter::get_random_integer(Constants::number_oct, Constants::number_oct * 1024, this->mt);

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(number_objects, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    std::vector<OctreeNode<AdditionalCellAttributes>> parents(1024, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < 1024U; i++) {
        const auto current_filling_expected = i * Constants::number_oct;

        ASSERT_EQ(current_filling_expected, MH::get_current_filling());
        ASSERT_EQ(number_objects, MH::get_size());

        for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
            if (current_filling_expected + Constants::number_oct > number_objects) {
                ASSERT_THROW(auto val = MH::get_available(&parents[i], child_idx), RelearnException);
            } else {
                ASSERT_NO_THROW(auto val = MH::get_available(&parents[i], child_idx));
            }
        }

        ASSERT_EQ(current_filling_expected + Constants::number_oct, MH::get_current_filling());
    }
}

TYPED_TEST(MemoryHolderTest, testMakeAvailable) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    const auto number_objects = RandomAdapter::get_random_integer(Constants::number_oct, Constants::number_oct * 1024, this->mt);
    const auto number_requesting_objects = RandomAdapter::get_random_integer(1U, 1024U, this->mt);

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(number_objects, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    std::vector<OctreeNode<AdditionalCellAttributes>> parents(number_requesting_objects, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < number_requesting_objects; i++) {
        if (i * Constants::number_oct + Constants::number_oct > number_objects) {
            ASSERT_THROW(auto val = MH::get_available(&parents[i], 0), RelearnException);
            continue;
        }

        auto val = MH::get_available(&parents[i], 0);
    }

    MH::make_all_available();

    ASSERT_EQ(MH::get_current_filling(), 0);
    ASSERT_EQ(MH::get_size(), number_objects);

    for (auto i = 0U; i < number_requesting_objects; i++) {
        ASSERT_THROW(auto val = MH::get_offset_from_parent(&parents[i]), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetOffsetException) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    ASSERT_THROW(auto val = MH::get_offset_from_parent(nullptr), RelearnException);

    for (auto i = 0U; i < 1024U; i++) {
        auto* ptr = &memory[i];
        ASSERT_THROW(auto val = MH::get_offset_from_parent(ptr), RelearnException);
    }

    std::vector<OctreeNode<AdditionalCellAttributes>> other_memory(1024, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < 1024U; i++) {
        auto* ptr = &other_memory[i];
        ASSERT_THROW(auto val = MH::get_offset_from_parent(ptr), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetOffset) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    OctreeNode<AdditionalCellAttributes> root{};

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto* ptr = MH::get_available(&root, i);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, i);

        root.set_child(ptr, i);
    }

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root.get_child(child_idx);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = MH::get_available(child, i);
            auto dist = std::distance(span_memory.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + child_idx * Constants::number_oct + i);

            child->set_child(ptr, i);
        }
    }

    ASSERT_EQ(0, MH::get_offset_from_parent(&root));

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root.get_child(child_idx);

        const auto offset_index = Constants::number_oct + child_idx * Constants::number_oct;

        ASSERT_EQ(offset_index * sizeof(OctreeNode<AdditionalCellAttributes>), MH::get_offset_from_parent(child));
    }
}

TYPED_TEST(MemoryHolderTest, testGetOffsetDisorganized) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    OctreeNode<AdditionalCellAttributes> root{};

    const std::vector<unsigned int> indices = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];
        auto* ptr = MH::get_available(&root, child_index);
        root.set_child(ptr, child_index);
    }

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];

        auto* child = root.get_child(child_index);

        const std::vector<unsigned int> indices_child = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

        for (auto j = 0U; j < Constants::number_oct; j++) {
            auto child_child_index = indices_child[j];
            auto* ptr = MH::get_available(child, child_child_index);
            child->set_child(ptr, child_child_index);
        }
    }

    ASSERT_EQ(0, MH::get_offset_from_parent(&root));

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];
        const auto offset = MH::get_offset_from_parent(root.get_child(child_index));
        const auto expected_offset = (i + 1) * Constants::number_oct * sizeof(OctreeNode<AdditionalCellAttributes>);

        ASSERT_EQ(expected_offset, offset);
    }
}

TYPED_TEST(MemoryHolderTest, testGetNodeFromOffsetException) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    for (auto i = std::uint64_t(0); i < 1024 * 10ULL; i++) {
        ASSERT_THROW(auto val = MH::get_node_from_offset(i), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetNodeFromOffset) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    const auto number_objects = RandomAdapter::get_random_integer(Constants::number_oct, Constants::number_oct * 1024, this->mt);
    const auto number_requesting_objects = number_objects / Constants::number_oct;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(number_objects, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    std::vector<OctreeNode<AdditionalCellAttributes>> parents(number_requesting_objects, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < number_requesting_objects; i++) {
        auto* ptr = &parents[i];
        ASSERT_NO_THROW(auto val = MH::get_available(ptr, 0));
    }

    for (std::uint64_t offset = 0; offset < number_objects; offset++) {
        if (offset >= number_requesting_objects * Constants::number_oct) {
            ASSERT_THROW(auto val = MH::get_node_from_offset(offset), RelearnException);
            continue;
        }

        auto* ptr = MH::get_node_from_offset(offset);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, offset);
    }
}

TYPED_TEST(MemoryHolderTest, testGetParentFromOffsetException) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;

    std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});
    std::span<OctreeNode<AdditionalCellAttributes>> span_memory(memory);

    MH::init(span_memory);

    for (auto i = std::uint64_t(0); i < 1024 * 10ULL; i++) {
        ASSERT_THROW(auto val = MH::get_parent_from_offset(i), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetParentFromOffset) {
    using AdditionalCellAttributes = TypeParam;
    using MH = MemoryHolder<AdditionalCellAttributes>;
    using Node = OctreeNode<AdditionalCellAttributes>;

    std::vector<Node> memory(128 * Constants::number_oct, Node{});
    std::span<Node> span_memory(memory);

    MH::init(span_memory);

    std::vector<Node> nodes(128, Node{});
    std::vector<std::pair<Node*, Node*>> relations{};

    for (auto& node : nodes) {
        for (auto i = 0; i < Constants::number_oct; i++) {
            auto* ptr = MH::get_available(&node, i);
            relations.emplace_back(&node, ptr);
        }
    }

    for (const auto [parent, child] : relations) {
        auto offset = std::distance(memory.data(), child);
        if (offset % Constants::number_oct != 0) {
            ASSERT_THROW(auto saved_parent = MH::get_parent_from_offset(offset), RelearnException);
            continue;
        }

        auto saved_parent = MH::get_parent_from_offset(offset);
        ASSERT_EQ(parent, saved_parent);
    }
}
