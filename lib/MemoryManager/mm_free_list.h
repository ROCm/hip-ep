#ifndef MM_FREE_LIST_H
#define MM_FREE_LIST_H

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace mm {

/*
 * Lock-free LIFO (stack) free list using atomic CAS.
 *
 * Each node stores a block index. Push/pop are O(1) and wait-free
 * for the common uncontended case.
 *
 * Intended for KV block pool per-format free lists where allocation
 * latency must be < 100 ns.
 */
class FreeList {
public:
    FreeList() = default;
    ~FreeList();

    /* Initialize with capacity. Pre-allocates node storage. */
    void init(uint32_t capacity);

    /* Push a block index onto the free list. Returns true on success. */
    bool push(uint32_t block_index);

    /* Pop a block index from the free list. Returns UINT32_MAX if empty. */
    uint32_t pop();

    /* Check if the free list is empty. */
    bool empty() const;

    /* Number of items currently in the free list. */
    uint32_t count() const;

    /* Release all storage. */
    void shutdown();

private:
    struct Node {
        uint32_t block_index;
        uint32_t next; /* index into nodes_ array, UINT32_MAX = end */
    };

    Node *nodes_ = nullptr;
    uint32_t capacity_ = 0;

    /* Head of the free list (index into nodes_). UINT32_MAX = empty. */
    std::atomic<uint32_t> head_{UINT32_MAX};

    /* Free node pool for recycling Node structs */
    std::atomic<uint32_t> free_nodes_head_{UINT32_MAX};

    /* Count of items in the list */
    std::atomic<uint32_t> count_{0};

    /* Next node index to allocate from the pre-allocated array */
    std::atomic<uint32_t> next_node_{0};

    uint32_t alloc_node();
    void free_node(uint32_t idx);
};

} // namespace mm

#endif /* MM_FREE_LIST_H */
