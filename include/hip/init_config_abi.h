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

// Session-init config passed through inference_init_v2 as void*, owned by the
// EP for the duration of hipdnn_ep_state_init_v2.
//
// Enumeration only: init copies the entries into RuntimeState, which outlives
// this and answers every later lookup. Pure C by necessity -- it crosses from
// the MSVC-built EP into Clang-compiled, JIT-loaded model code, so no C++ types
// and no vtables.
typedef struct hipdnn_ep_init_config {
  void *self; // borrowed EP context; the runtime never frees it
  size_t (*provider_option_count)(void *self);
  // Sets both outputs to null when `index` is out of range; otherwise they
  // stay valid for the duration of the init call.
  void (*provider_option_at)(void *self, size_t index, const char **key,
                             const char **value);
} hipdnn_ep_init_config;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIP_INIT_CONFIG_ABI_H
