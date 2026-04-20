#include "mm_runtime_bridge.h"
#include "mm_internal.h"

#include <cstdio>

/* Bridge state (workspace handle tracking) */
static mm_handle_t g_workspace_handle = MM_INVALID_HANDLE;
static void *g_workspace_ptr = nullptr;
static size_t g_workspace_size = 0;
static mm_stream_t g_stream = 0;

extern "C" {

int mm_bridge_init(size_t gpu_memory_limit, mm_stream_t hip_stream) {
    g_stream = hip_stream;

    /* Register the appropriate HAL backend */
#ifdef MM_HAS_HIP
    int err = mm_hal_register(mm_hal_hip_get());
#else
    int err = mm_hal_register(mm_hal_host_get());
#endif
    if (err != MM_OK) {
        std::fprintf(stderr, "[MM Bridge] HAL registration failed: %d\n", err);
        return err;
    }

    /* Initialize MM with configuration */
    mm_config_t config = {};
    config.gpu_memory_limit = gpu_memory_limit;
    config.kv_cache_fraction = 0.0f; /* Phase 1: no KV cache reservation yet */
    config.kv_block_size_tokens = 16;
    config.max_tier = MM_TIER_DRAM;
    config.enable_prefix_caching = false; /* Phase 3 */
    config.enable_defrag = false;
    config.num_size_classes = 8;
    config.high_watermark = 0.90f;
    config.critical_watermark = 0.95f;
    config.default_kv_format = MM_KV_FMT_FP16;
    config.pressure_kv_format = MM_KV_FMT_TURBOQUANT_4;
    config.critical_kv_format = MM_KV_FMT_TURBOQUANT_3;
    config.enable_inline_quant = false;
    config.enable_adaptive_transcode = false;

    err = mm_init(&config);
    if (err != MM_OK) {
        std::fprintf(stderr, "[MM Bridge] mm_init failed: %d\n", err);
        return err;
    }

    return MM_OK;
}

void mm_bridge_shutdown(void) {
    /* Free workspace if allocated */
    if (g_workspace_handle != MM_INVALID_HANDLE) {
        mm_free(g_workspace_handle, g_stream);
        g_workspace_handle = MM_INVALID_HANDLE;
        g_workspace_ptr = nullptr;
        g_workspace_size = 0;
    }

    mm_shutdown();
}

mm_pool_t mm_bridge_pool_init(size_t pool_size,
                              const size_t *buffer_offsets,
                              size_t num_buffers) {
    if (pool_size == 0)
        return MM_INVALID_POOL;

    /* Build the static plan from the compiler-generated offsets */
    auto *entries = new mm_buffer_entry_t[num_buffers];
    for (size_t i = 0; i < num_buffers; ++i) {
        entries[i].tensor_id = (uint32_t)i;
        entries[i].offset = buffer_offsets[i];
        entries[i].size = 0; /* Size not tracked per-entry in current compiler */
        entries[i].alignment = 256;
    }

    mm_static_plan_t plan = {};
    plan.total_size = pool_size;
    plan.mem_class = MM_CLASS_WEIGHT;
    plan.device = MM_DEVICE_GPU_0;
    plan.num_entries = (uint32_t)num_buffers;
    plan.entries = entries;

    mm_pool_t pool = mm_create_pool(&plan);

    delete[] entries;

    if (pool == MM_INVALID_POOL) {
        std::fprintf(stderr,
                     "[MM Bridge] mm_create_pool failed for %zu bytes\n",
                     pool_size);
    }

    return pool;
}

void *mm_bridge_pool_get_buffer(mm_pool_t pool, size_t index) {
    return mm_pool_get_ptr(pool, (uint32_t)index);
}

void *mm_bridge_pool_get_base(mm_pool_t pool) {
    /* Return the pointer for tensor_id 0 (the base of the pool) */
    return mm_pool_get_ptr(pool, 0);
}

void mm_bridge_pool_destroy(mm_pool_t pool) {
    mm_destroy_pool(pool);
}

void *mm_bridge_tensor_alloc(size_t size_bytes) {
    if (size_bytes == 0)
        return nullptr;

    mm_alloc_hints_t hints = {};
    hints.mem_class = MM_CLASS_ACTIVATION;
    hints.lifetime = MM_LIFETIME_STEP;
    hints.alignment = 256;

    mm_handle_t handle = mm_alloc(size_bytes, hints, g_stream);
    if (handle == MM_INVALID_HANDLE)
        return nullptr;

    return mm_get_ptr(handle, MM_DEVICE_GPU_0);
}

void mm_bridge_tensor_release(void *ptr, size_t size_bytes) {
    (void)ptr;
    (void)size_bytes;
    /* Arena allocations are freed in bulk via reset.
       For Phase 1, individual frees are no-ops within the arena. */
}

void *mm_bridge_workspace_ensure(size_t needed_size) {
    if (needed_size == 0)
        return g_workspace_ptr;

    if (g_workspace_size >= needed_size)
        return g_workspace_ptr;

    /* Free old workspace */
    if (g_workspace_handle != MM_INVALID_HANDLE) {
        mm_free(g_workspace_handle, g_stream);
        g_workspace_handle = MM_INVALID_HANDLE;
        g_workspace_ptr = nullptr;
        g_workspace_size = 0;
    }

    /* Allocate new, larger workspace */
    mm_alloc_hints_t hints = {};
    hints.mem_class = MM_CLASS_SCRATCH;
    hints.lifetime = MM_LIFETIME_REQUEST;
    hints.alignment = 256;

    g_workspace_handle = mm_alloc(needed_size, hints, g_stream);
    if (g_workspace_handle == MM_INVALID_HANDLE) {
        std::fprintf(stderr,
                     "[MM Bridge] workspace alloc failed for %zu bytes\n",
                     needed_size);
        return nullptr;
    }

    g_workspace_ptr = mm_get_ptr(g_workspace_handle, MM_DEVICE_GPU_0);
    g_workspace_size = needed_size;
    return g_workspace_ptr;
}

void *mm_bridge_workspace_get(void) {
    return g_workspace_ptr;
}

size_t mm_bridge_workspace_size(void) {
    return g_workspace_size;
}

int mm_bridge_is_initialized(void) {
    auto *s = mm::mm_get_state();
    return s->initialized.load(std::memory_order_acquire) ? 1 : 0;
}

} /* extern "C" */
