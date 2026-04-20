#ifndef MM_TYPES_H
#define MM_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque Handles ---- */

typedef uint64_t mm_handle_t;
typedef uint64_t mm_pool_t;
typedef uint64_t mm_stream_t;
typedef uint64_t mm_kv_block_t;
typedef uint64_t mm_fence_t;

#define MM_INVALID_HANDLE ((mm_handle_t)0)
#define MM_INVALID_POOL ((mm_pool_t)0)
#define MM_INVALID_BLOCK ((mm_kv_block_t)0)

/* ---- Enums ---- */

typedef enum {
  MM_CLASS_WEIGHT = 0,
  MM_CLASS_ACTIVATION = 1,
  MM_CLASS_KV_CACHE = 2,
  MM_CLASS_SCRATCH = 3,
} mm_memory_class_t;

typedef enum {
  MM_LIFETIME_STATIC = 0,
  MM_LIFETIME_REQUEST = 1,
  MM_LIFETIME_STEP = 2,
  MM_LIFETIME_TRANSIENT = 3,
} mm_lifetime_t;

typedef enum {
  MM_ACCESS_SEQUENTIAL = 0,
  MM_ACCESS_RANDOM = 1,
  MM_ACCESS_WRITE_ONCE = 2,
  MM_ACCESS_READ_MOSTLY = 3,
} mm_access_pattern_t;

typedef enum {
  MM_DEVICE_GPU_0 = 0,
  MM_DEVICE_GPU_1 = 1,
  MM_DEVICE_GPU_2 = 2,
  MM_DEVICE_GPU_3 = 3,
  MM_DEVICE_NPU = 16,
  MM_DEVICE_CPU = 32,
  MM_DEVICE_ANY = 255,
} mm_device_t;

typedef enum {
  MM_TIER_HBM = 0,
  MM_TIER_DRAM = 1,
  MM_TIER_SSD = 2,
  MM_TIER_NETWORK = 3,
} mm_tier_t;

typedef enum {
  MM_KV_FMT_FP16 = 0,
  MM_KV_FMT_FP8_E4M3 = 1,
  MM_KV_FMT_INT4 = 2,
  MM_KV_FMT_TURBOQUANT_4 = 3,
  MM_KV_FMT_TURBOQUANT_3 = 4,
  MM_KV_FMT_TURBOQUANT_2 = 5,
} mm_kv_format_t;

typedef enum {
  MM_COPY_HOST_TO_DEVICE = 0,
  MM_COPY_DEVICE_TO_HOST = 1,
  MM_COPY_DEVICE_TO_DEVICE = 2,
  MM_COPY_HOST_TO_HOST = 3,
} mm_copy_kind_t;

/* ---- Error Codes ---- */

#define MM_OK 0
#define MM_ERROR_NOT_INIT -1
#define MM_ERROR_ALREADY_INIT -2
#define MM_ERROR_INVALID_ARG -3
#define MM_ERROR_OUT_OF_MEMORY -4
#define MM_ERROR_HAL_FAILURE -5

/* ---- Structs ---- */

typedef struct {
  mm_memory_class_t mem_class;
  mm_lifetime_t lifetime;
  mm_access_pattern_t access_pattern;
  mm_device_t device_affinity;
  size_t alignment;
  bool shareable;
  size_t size_hint_max;
} mm_alloc_hints_t;

typedef struct {
  mm_tier_t current_tier;
  mm_memory_class_t mem_class;
  size_t size;
  uint32_t ref_count;
  mm_device_t resident_device;
} mm_alloc_info_t;

typedef struct {
  uint32_t tensor_id;
  size_t offset;
  size_t size;
  size_t alignment;
} mm_buffer_entry_t;

typedef struct {
  size_t total_size;
  mm_memory_class_t mem_class;
  mm_device_t device;
  uint32_t num_entries;
  mm_buffer_entry_t *entries;
} mm_static_plan_t;

typedef struct {
  mm_kv_format_t format;
  uint32_t block_size_tokens;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t num_layers;
  size_t bytes_per_token;
  bool has_qjl_residual;
  uint32_t polar_bits;
} mm_kv_block_desc_t;

#ifdef __cplusplus
}
#endif

#endif /* MM_TYPES_H */
