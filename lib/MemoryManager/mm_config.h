#ifndef MM_CONFIG_H
#define MM_CONFIG_H

#include "mm_types.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Core memory limits */
    size_t         gpu_memory_limit;      /* 0 = auto-detect */
    float          kv_cache_fraction;     /* [0,1], default 0.90 */
    uint32_t       kv_block_size_tokens;  /* default 16 */
    mm_tier_t      max_tier;              /* deepest tier to use */

    /* Feature toggles */
    bool           enable_prefix_caching;
    bool           enable_defrag;
    uint32_t       num_size_classes;      /* arena size classes, default 8 */

    /* Pressure watermarks */
    float          high_watermark;        /* 0.90: trigger background eviction */
    float          critical_watermark;    /* 0.95: trigger sync eviction */

    /* Adaptive compression */
    mm_kv_format_t default_kv_format;
    mm_kv_format_t pressure_kv_format;
    mm_kv_format_t critical_kv_format;
    bool           enable_inline_quant;
    bool           enable_adaptive_transcode;
} mm_config_t;

/*
 * Initialize the memory manager. Must be called before any other mm_* call.
 * Pass NULL for default configuration.
 * Returns MM_OK on success, negative error code on failure.
 */
int mm_init(const mm_config_t *config);

/*
 * Shutdown and release all resources. Safe to call even if not initialized.
 */
void mm_shutdown(void);

/*
 * Dump internal state for debugging. Pass NULL for stdout.
 */
void mm_dump_state(FILE *output);

#ifdef __cplusplus
}
#endif

#endif /* MM_CONFIG_H */
