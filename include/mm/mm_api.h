/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstddef>
#include <cstdio>

#include "mm/mm_config.h"
#include "mm/mm_kv.h"
#include "mm/mm_types.h"

namespace mm {

Status init(const Config *config = nullptr);
void shutdown();
int is_initialized();

handle_t alloc(std::size_t size_bytes, const AllocHints *hints);
Status free(handle_t handle);
void *get_ptr(handle_t handle);
Status query(handle_t handle, AllocInfo *info);

void dump_state(std::FILE *output);
int debug_enabled();
MetricsSnapshot metrics_snapshot();
void metrics_reset();

} // namespace mm
