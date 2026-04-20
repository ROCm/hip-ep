#include "mm_config.h"
#include "mm_arena.h"
#include "mm_internal.h"
#include "mm_kv.h"

#include <cstdio>

namespace mm {

static MmState g_state;

MmState *mm_get_state() { return &g_state; }

static int validate_config(const mm_config_t *config) {
  if (config->kv_cache_fraction < 0.0f || config->kv_cache_fraction > 1.0f)
    return MM_ERROR_INVALID_ARG;
  if (config->high_watermark < 0.0f || config->high_watermark > 1.0f)
    return MM_ERROR_INVALID_ARG;
  if (config->critical_watermark < config->high_watermark)
    return MM_ERROR_INVALID_ARG;
  if (config->num_size_classes < 1 || config->num_size_classes > 16)
    return MM_ERROR_INVALID_ARG;
  return MM_OK;
}

} // namespace mm

extern "C" {

int mm_init(const mm_config_t *config) {
  auto *s = mm::mm_get_state();

  if (s->initialized.load(std::memory_order_acquire))
    return MM_ERROR_ALREADY_INIT;

  /* Apply config (or defaults) */
  if (config) {
    int err = mm::validate_config(config);
    if (err != MM_OK)
      return err;

    s->config.gpu_memory_limit = config->gpu_memory_limit;
    s->config.kv_cache_fraction = config->kv_cache_fraction;
    s->config.kv_block_size_tokens = config->kv_block_size_tokens;
    s->config.max_tier = config->max_tier;
    s->config.enable_prefix_caching = config->enable_prefix_caching;
    s->config.enable_defrag = config->enable_defrag;
    s->config.num_size_classes = config->num_size_classes;
    s->config.high_watermark = config->high_watermark;
    s->config.critical_watermark = config->critical_watermark;
    s->config.default_kv_format = config->default_kv_format;
    s->config.pressure_kv_format = config->pressure_kv_format;
    s->config.critical_kv_format = config->critical_kv_format;
    s->config.enable_inline_quant = config->enable_inline_quant;
    s->config.enable_adaptive_transcode = config->enable_adaptive_transcode;
  }
  /* else: defaults from MmConfig struct are fine */

  /* Get HAL — must have been registered before init */
  s->hal = mm_hal_get();
  if (!s->hal)
    return MM_ERROR_NOT_INIT;

  /* Query GPU memory budget */
  if (s->config.gpu_memory_limit == 0) {
    s->total_gpu_memory = s->hal->get_total_memory(0);
  } else {
    s->total_gpu_memory = s->config.gpu_memory_limit;
  }

  /* Budget: activations get whatever is not reserved for KV cache.
     Weight pool is allocated separately via mm_create_pool(). */
  s->kv_cache_reserved =
      (size_t)(s->total_gpu_memory * s->config.kv_cache_fraction);
  s->activation_reserved = s->total_gpu_memory - s->kv_cache_reserved;

  /* Initialize arena allocator for activations + scratch */
  s->arena = new mm::ArenaAllocator();
  int err = s->arena->init(s->hal, 0, s->activation_reserved,
                           s->config.num_size_classes);
  if (err != MM_OK) {
    delete s->arena;
    s->arena = nullptr;
    return err;
  }

  /* Initialize KV cache block pool (if budget allows) */
  if (s->kv_cache_reserved > 0) {
    int kv_err =
        mm_kv_pool_init(s->kv_cache_reserved, s->config.kv_block_size_tokens,
                        s->config.default_kv_format);
    if (kv_err != MM_OK) {
      /* Non-fatal: KV pool init failure doesn't block basic operation */
      std::fprintf(stderr, "[MM] KV pool init failed (%d), paged KV disabled\n",
                   kv_err);
    }
  }

  s->initialized.store(true, std::memory_order_release);
  return MM_OK;
}

void mm_shutdown(void) {
  auto *s = mm::mm_get_state();

  if (!s->initialized.load(std::memory_order_acquire))
    return;

  /* Destroy all active pools */
  {
    std::lock_guard<std::mutex> lock(s->pools_mutex);
    for (auto *pool : s->pools) {
      if (pool)
        delete pool;
    }
    s->pools.clear();
  }

  /* Shutdown KV block pool */
  mm_kv_pool_shutdown();

  /* Shutdown arena */
  if (s->arena) {
    s->arena->shutdown();
    delete s->arena;
    s->arena = nullptr;
  }

  /* Clear handle table */
  s->handles.clear();

  /* Reset state */
  s->hal = nullptr;
  s->total_gpu_memory = 0;
  s->weight_pool_reserved = 0;
  s->kv_cache_reserved = 0;
  s->activation_reserved = 0;
  s->config = mm::MmConfig{};

  s->initialized.store(false, std::memory_order_release);
}

void mm_dump_state(FILE *output) {
  auto *s = mm::mm_get_state();
  FILE *f = output ? output : stdout;

  if (!s->initialized.load(std::memory_order_acquire)) {
    std::fprintf(f, "Memory Manager: NOT INITIALIZED\n");
    return;
  }

  std::fprintf(f, "=== Memory Manager State ===\n");
  std::fprintf(f, "Total GPU memory: %zu bytes\n", s->total_gpu_memory);
  std::fprintf(f, "KV cache reserved: %zu bytes\n", s->kv_cache_reserved);
  std::fprintf(f, "Activation reserved: %zu bytes\n", s->activation_reserved);
  std::fprintf(f, "Active handles: %zu\n", s->handles.active_count());

  if (s->arena) {
    std::fprintf(f, "Arena used: %zu / %zu bytes\n", s->arena->used_bytes(),
                 s->arena->total_bytes());
  }

  std::fprintf(f, "Active pools: %zu\n", s->pools.size());

  if (s->kv_pool) {
    uint32_t total = mm_kv_total_block_count(MM_KV_FMT_FP16);
    uint32_t free = mm_kv_free_block_count(MM_KV_FMT_FP16);
    std::fprintf(f, "KV blocks: %u allocated, %u free, %u total\n",
                 total - free, free, total);
  }

  std::fprintf(f, "============================\n");
}

} /* extern "C" */
