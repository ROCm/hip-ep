#include "mm_kv.h"
#include "mm_block_table.h"
#include "mm_free_list.h"
#include "mm_hal.h"
#include "mm_internal.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace mm {

/*
 * Internal representation of a KV block.
 * Stored in a flat array indexed by block handle - 1.
 */
struct KvBlock {
    void              *gpu_ptr = nullptr;  /* Physical GPU memory */
    mm_kv_block_desc_t desc = {};          /* Format descriptor */
    uint32_t           ref_count = 0;      /* CoW reference count (mutex-protected) */
    bool               active = false;     /* In use (not on free list) */
    size_t             size_bytes = 0;     /* Block size in bytes */
};

/*
 * KV Block Pool: manages a slab of GPU memory subdivided into
 * fixed-size blocks. Per-format free lists for O(1) alloc/free.
 */
class KvBlockPool {
public:
    KvBlockPool() = default;
    ~KvBlockPool() { shutdown(); }

    int init(const mm_hal_t *hal, int device_id,
             size_t total_budget, uint32_t block_size_tokens,
             mm_kv_format_t default_format);

    mm_kv_block_t alloc_block(mm_kv_block_desc_t desc);
    void free_block(mm_kv_block_t handle);
    mm_kv_block_t fork_block(mm_kv_block_t handle);

    KvBlock *lookup(mm_kv_block_t handle);

    uint32_t free_count(mm_kv_format_t format) const;
    uint32_t total_count() const;

    void shutdown();

private:
    const mm_hal_t *hal_ = nullptr;
    int device_id_ = 0;

    /* Slab: single contiguous GPU allocation split into blocks */
    void *slab_base_ = nullptr;
    size_t slab_size_ = 0;

    /* Block metadata */
    std::vector<KvBlock> blocks_;
    uint32_t total_blocks_ = 0;
    size_t block_size_bytes_ = 0;
    uint32_t block_size_tokens_ = 0;

    /* Per-format free lists */
    FreeList free_list_; /* Single format for now (Phase 2 uses FP16 only) */
    mm_kv_format_t default_format_ = MM_KV_FMT_FP16;

    /* Handle generation (1-based, offset from a base to avoid collisions
       with pool handles) */
    static constexpr uint64_t kKvHandleBase = 0x4B560000; /* "KV" prefix */

    std::mutex mutex_; /* Protects block metadata writes */

    size_t compute_block_size(const mm_kv_block_desc_t &desc) const;
};

size_t KvBlockPool::compute_block_size(const mm_kv_block_desc_t &desc) const {
    /* Block stores K and V for block_size_tokens tokens.
       Size = block_size_tokens * num_kv_heads * head_dim * bytes_per_element * 2 (K+V) */
    size_t bytes_per_elem = 2; /* FP16 default */
    switch (desc.format) {
    case MM_KV_FMT_FP16:         bytes_per_elem = 2; break;
    case MM_KV_FMT_FP8_E4M3:    bytes_per_elem = 1; break;
    case MM_KV_FMT_INT4:         bytes_per_elem = 1; break; /* approximate */
    case MM_KV_FMT_TURBOQUANT_4: bytes_per_elem = 1; break;
    case MM_KV_FMT_TURBOQUANT_3: bytes_per_elem = 1; break;
    case MM_KV_FMT_TURBOQUANT_2: bytes_per_elem = 1; break;
    }

    if (desc.bytes_per_token > 0) {
        /* Use explicit bytes_per_token if provided */
        return (size_t)desc.block_size_tokens * desc.bytes_per_token * 2; /* K+V */
    }

    return (size_t)desc.block_size_tokens * desc.num_kv_heads *
           desc.head_dim * bytes_per_elem * 2; /* K+V */
}

int KvBlockPool::init(const mm_hal_t *hal, int device_id,
                      size_t total_budget, uint32_t block_size_tokens,
                      mm_kv_format_t default_format) {
    if (!hal || total_budget == 0)
        return MM_ERROR_INVALID_ARG;

    hal_ = hal;
    device_id_ = device_id;
    block_size_tokens_ = block_size_tokens;
    default_format_ = default_format;

    /* Use a default block size for pool initialization.
       Typical: 16 tokens * 4 kv_heads * 128 head_dim * 2 bytes * 2 (K+V) = 32KB */
    block_size_bytes_ = (size_t)block_size_tokens * 4 * 128 * 2 * 2;
    if (block_size_bytes_ < 256)
        block_size_bytes_ = 256;

    /* Align block size to 256 bytes */
    block_size_bytes_ = (block_size_bytes_ + 255) & ~(size_t)255;

    /* Calculate number of blocks that fit in budget */
    total_blocks_ = (uint32_t)(total_budget / block_size_bytes_);
    if (total_blocks_ == 0)
        return MM_ERROR_INVALID_ARG;

    slab_size_ = (size_t)total_blocks_ * block_size_bytes_;

    /* Allocate slab */
    slab_base_ = hal_->raw_alloc(device_id_, slab_size_, 256);
    if (!slab_base_)
        return MM_ERROR_OUT_OF_MEMORY;

    /* Initialize block metadata */
    blocks_.resize(total_blocks_);
    for (uint32_t i = 0; i < total_blocks_; ++i) {
        blocks_[i].gpu_ptr = static_cast<char *>(slab_base_) +
                             (size_t)i * block_size_bytes_;
        blocks_[i].desc = {};
        blocks_[i].desc.format = default_format_;
        blocks_[i].desc.block_size_tokens = block_size_tokens_;
        blocks_[i].ref_count = 0;
        blocks_[i].active = false;
        blocks_[i].size_bytes = block_size_bytes_;
    }

    /* Initialize free list and push all blocks */
    free_list_.init(total_blocks_);
    for (uint32_t i = 0; i < total_blocks_; ++i) {
        free_list_.push(i);
    }

    return MM_OK;
}

mm_kv_block_t KvBlockPool::alloc_block(mm_kv_block_desc_t desc) {
    uint32_t idx = free_list_.pop();
    if (idx == UINT32_MAX)
        return MM_INVALID_BLOCK; /* Pool exhausted */

    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[idx].desc = desc;
    blocks_[idx].desc.block_size_tokens = block_size_tokens_;
    blocks_[idx].ref_count = 1;
    blocks_[idx].active = true;
    blocks_[idx].size_bytes = compute_block_size(desc);
    if (blocks_[idx].size_bytes > block_size_bytes_)
        blocks_[idx].size_bytes = block_size_bytes_;

    return static_cast<mm_kv_block_t>(kKvHandleBase + idx + 1);
}

void KvBlockPool::free_block(mm_kv_block_t handle) {
    if (handle == MM_INVALID_BLOCK || handle <= kKvHandleBase)
        return;

    uint32_t idx = (uint32_t)(handle - kKvHandleBase - 1);
    if (idx >= total_blocks_)
        return;

    /* Decrement ref count; only return to free list when it hits 0 */
    std::lock_guard<std::mutex> lock(mutex_);
    if (blocks_[idx].ref_count == 0)
        return; /* Double free — safe no-op */
    blocks_[idx].ref_count--;
    if (blocks_[idx].ref_count == 0 && blocks_[idx].active) {
        blocks_[idx].active = false;
        free_list_.push(idx);
    }
}

mm_kv_block_t KvBlockPool::fork_block(mm_kv_block_t handle) {
    if (handle == MM_INVALID_BLOCK || handle <= kKvHandleBase)
        return MM_INVALID_BLOCK;

    uint32_t idx = (uint32_t)(handle - kKvHandleBase - 1);
    if (idx >= total_blocks_ || !blocks_[idx].active)
        return MM_INVALID_BLOCK;

    /* CoW: increment ref count, return same handle */
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[idx].ref_count++;
    return handle;
}

KvBlock *KvBlockPool::lookup(mm_kv_block_t handle) {
    if (handle == MM_INVALID_BLOCK || handle <= kKvHandleBase)
        return nullptr;

    uint32_t idx = (uint32_t)(handle - kKvHandleBase - 1);
    if (idx >= total_blocks_ || !blocks_[idx].active)
        return nullptr;

    return &blocks_[idx];
}

uint32_t KvBlockPool::free_count(mm_kv_format_t /*format*/) const {
    return free_list_.count();
}

uint32_t KvBlockPool::total_count() const {
    return total_blocks_;
}

void KvBlockPool::shutdown() {
    if (slab_base_ && hal_) {
        hal_->raw_free(device_id_, slab_base_);
        slab_base_ = nullptr;
    }
    free_list_.shutdown();
    blocks_.clear();
    total_blocks_ = 0;
    slab_size_ = 0;
}

/* Global KV pool accessor (stored in MmState, see mm_config.cpp) */
static KvBlockPool *get_kv_pool() {
    auto *s = mm_get_state();
    if (!s || !s->initialized.load(std::memory_order_acquire))
        return nullptr;
    return s->kv_pool;
}

} // namespace mm

/* ---- C API Implementation ---- */

extern "C" {

mm_kv_block_t mm_kv_alloc_block(mm_kv_block_desc_t desc,
                                mm_stream_t /*stream*/) {
    auto *pool = mm::get_kv_pool();
    if (!pool)
        return MM_INVALID_BLOCK;
    return pool->alloc_block(desc);
}

void mm_kv_free_block(mm_kv_block_t block, mm_stream_t /*stream*/) {
    auto *pool = mm::get_kv_pool();
    if (pool)
        pool->free_block(block);
}

void **mm_kv_get_block_table(const mm_kv_block_t *blocks, uint32_t num_blocks,
                             mm_device_t /*device*/) {
    auto *pool = mm::get_kv_pool();
    if (!pool || !blocks || num_blocks == 0)
        return nullptr;

    void **table = static_cast<void **>(malloc(sizeof(void *) * num_blocks));
    if (!table)
        return nullptr;

    for (uint32_t i = 0; i < num_blocks; ++i) {
        auto *blk = pool->lookup(blocks[i]);
        table[i] = blk ? blk->gpu_ptr : nullptr;
    }

    return table;
}

mm_kv_block_desc_t mm_kv_get_format(mm_kv_block_t block) {
    mm_kv_block_desc_t desc = {};
    auto *pool = mm::get_kv_pool();
    if (!pool)
        return desc;

    auto *blk = pool->lookup(block);
    if (blk)
        desc = blk->desc;
    return desc;
}

mm_kv_block_t mm_kv_fork_block(mm_kv_block_t source) {
    auto *pool = mm::get_kv_pool();
    if (!pool)
        return MM_INVALID_BLOCK;
    return pool->fork_block(source);
}

void *mm_kv_get_block_ptr(mm_kv_block_t block) {
    auto *pool = mm::get_kv_pool();
    if (!pool)
        return nullptr;

    auto *blk = pool->lookup(block);
    return blk ? blk->gpu_ptr : nullptr;
}

uint32_t mm_kv_free_block_count(mm_kv_format_t format) {
    auto *pool = mm::get_kv_pool();
    return pool ? pool->free_count(format) : 0;
}

uint32_t mm_kv_total_block_count(mm_kv_format_t /*format*/) {
    auto *pool = mm::get_kv_pool();
    return pool ? pool->total_count() : 0;
}

int mm_kv_pool_init(size_t total_budget, uint32_t block_size_tokens,
                    mm_kv_format_t default_format) {
    auto *s = mm::mm_get_state();
    if (!s || !s->hal)
        return MM_ERROR_NOT_INIT;
    if (total_budget == 0)
        return MM_OK; /* No KV cache requested */

    s->kv_pool = new mm::KvBlockPool();
    int err = s->kv_pool->init(s->hal, 0, total_budget,
                               block_size_tokens, default_format);
    if (err != MM_OK) {
        delete s->kv_pool;
        s->kv_pool = nullptr;
        return err;
    }
    return MM_OK;
}

void mm_kv_pool_shutdown(void) {
    auto *s = mm::mm_get_state();
    if (!s)
        return;
    if (s->kv_pool) {
        s->kv_pool->shutdown();
        delete s->kv_pool;
        s->kv_pool = nullptr;
    }
}

} /* extern "C" */
