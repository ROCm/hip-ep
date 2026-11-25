/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <hip/hip_runtime.h>
#include "gqa_paged_attention.h"

// Direct production-wrapper UT context. The adapter supplies a real HIP stream,
// sticky device status, grow-only scratch and the existing OpState lifetime.
// It does not construct an EP, execute a graph or enable operator profiling.
struct UtPagedRuntime;
UtPagedRuntime *utPagedRuntimeCreate();
void utPagedRuntimeDestroy(UtPagedRuntime *);
void *utPagedRuntimeStream(UtPagedRuntime *);
int utPagedRuntimeCall(UtPagedRuntime *, const hip_paged_attention_f16_desc &,
                       const int32_t *metadata, int64_t metadataCount);
int utPagedRuntimeStatus(UtPagedRuntime *);
