#ifndef MM_INTERNAL_H
#define MM_INTERNAL_H

#include "mm_handle_table.h"
#include "mm_hal.h"
#include "mm_types.h"

#include <atomic>
#include <mutex>
#include <vector>

namespace mm {

/* Forward declarations */
class ArenaAllocator;
class StaticPoolManager;
class KvBlockPool;

/* Configuration with defaults applied */
struct MmConfig {
    size_t         gpu_memory_limit       = 0; /* 0 = auto-detect */
    float          kv_cache_fraction      = 0.90f;
    uint32_t       kv_block_size_tokens   = 16;
    mm_tier_t      max_tier               = MM_TIER_DRAM;
    bool           enable_prefix_caching  = true;
    bool           enable_defrag          = true;
    uint32_t       num_size_classes       = 8;
    float          high_watermark         = 0.90f;
    float          critical_watermark     = 0.95f;
    mm_kv_format_t default_kv_format      = MM_KV_FMT_FP16;
    mm_kv_format_t pressure_kv_format     = MM_KV_FMT_TURBOQUANT_4;
    mm_kv_format_t critical_kv_format     = MM_KV_FMT_TURBOQUANT_3;
    bool           enable_inline_quant    = false;
    bool           enable_adaptive_transcode = true;
};

/* Singleton state for the memory manager */
struct MmState {
    std::atomic<bool> initialized{false};
    MmConfig config;
    const mm_hal_t *hal = nullptr;

    /* Handle registry for all allocations */
    HandleTable handles;

    /* Sub-allocators (owned, created during init) */
    ArenaAllocator *arena = nullptr;
    KvBlockPool *kv_pool = nullptr;

    /* Active pools (created via mm_create_pool) */
    std::mutex pools_mutex;
    std::vector<StaticPoolManager *> pools;

    /* Memory budget */
    size_t total_gpu_memory = 0;
    size_t weight_pool_reserved = 0;
    size_t kv_cache_reserved = 0;
    size_t activation_reserved = 0;
};

/* Access the global singleton. Never returns nullptr after mm_init(). */
MmState *mm_get_state();

} // namespace mm

#endif /* MM_INTERNAL_H */
