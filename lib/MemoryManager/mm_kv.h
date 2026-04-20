#ifndef MM_KV_H
#define MM_KV_H

#include "mm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocate a paged KV cache block.
 *
 * Blocks are fixed-size (block_size_tokens * bytes_per_token * num_kv_heads * head_dim * 2).
 * Allocated from a per-format lock-free free list. O(1) via atomic CAS.
 *
 * Returns MM_INVALID_BLOCK on failure (pool exhausted).
 */
mm_kv_block_t mm_kv_alloc_block(mm_kv_block_desc_t desc, mm_stream_t stream);

/*
 * Free a KV cache block. Decrements ref_count; returns to free list when
 * ref_count reaches 0. Double-free is a safe no-op.
 */
void mm_kv_free_block(mm_kv_block_t block, mm_stream_t stream);

/*
 * Build a block table: array of device pointers for paged attention.
 * Caller provides an array of block handles; returns a host-allocated
 * array of void* pointing to each block's GPU memory.
 *
 * The returned pointer must be freed by the caller via free().
 * Returns NULL on failure.
 */
void **mm_kv_get_block_table(const mm_kv_block_t *blocks, uint32_t num_blocks,
                             mm_device_t device);

/*
 * Query the format descriptor of a block.
 * Returns a zero-initialized desc if block is invalid.
 */
mm_kv_block_desc_t mm_kv_get_format(mm_kv_block_t block);

/*
 * Copy-on-write fork: increments ref_count on the source block and
 * returns a new handle pointing to the same physical memory.
 *
 * Used for beam search and speculative decoding where multiple
 * hypotheses share a prefix.
 *
 * Returns MM_INVALID_BLOCK on failure.
 */
mm_kv_block_t mm_kv_fork_block(mm_kv_block_t source);

/*
 * Get the GPU pointer for a KV block (for direct kernel access).
 * Returns NULL if block is invalid.
 */
void *mm_kv_get_block_ptr(mm_kv_block_t block);

/*
 * Get the number of free blocks available for a given format.
 */
uint32_t mm_kv_free_block_count(mm_kv_format_t format);

/*
 * Get the total number of blocks allocated for a given format.
 */
uint32_t mm_kv_total_block_count(mm_kv_format_t format);

/*
 * Initialize the KV block pool. Called by mm_init().
 * Allocates a slab of total_budget bytes and subdivides into blocks.
 */
int mm_kv_pool_init(size_t total_budget, uint32_t block_size_tokens,
                    mm_kv_format_t default_format);

/*
 * Shutdown the KV block pool. Called by mm_shutdown().
 */
void mm_kv_pool_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* MM_KV_H */
