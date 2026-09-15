/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_INIT_CONFIG_ABI_H
#define HIP_INIT_CONFIG_ABI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Inference-init config passed through inference_init as void*, owned by the
// EP for the duration of hipdnn_ep_state_init_with_fs.
//
// Enumeration only: init copies the entries into RuntimeState, which outlives
// this and answers every later lookup.
typedef struct hipdnn_ep_init_config {
  void *self; // borrowed EP context; the runtime never frees it
  size_t (*provider_option_count)(void *self);
  void (*provider_option_at)(void *self, size_t index, const char **key,
                             const char **value);
} hipdnn_ep_init_config;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIP_INIT_CONFIG_ABI_H
