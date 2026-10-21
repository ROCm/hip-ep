// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include "hip_custom_kernels.h"

// Each entry points to one expert. Zero points contain two nibbles per byte.
struct RoutedMatrixTable {
  const uint8_t *weights[128];
  const void *scales[128];
  const uint8_t *zero_points[128];
};

// Grouped prefill stores a header, two row maps, and three words per tile.
inline size_t routed_grouped_scratch_bytes(int tokens, int k, int experts) {
  const size_t pairs = size_t(tokens) * k;
  return (1 + pairs * 2 + (pairs / 16 + experts) * 3) * sizeof(int32_t);
}

extern "C" HIP_KERNEL_API int hip_routed_experts(
    void *stream, const void *input, const int64_t *indices,
    const float *gates, const RoutedMatrixTable *up,
    const RoutedMatrixTable *down, void *activation, void *slots, void *output,
    int tokens, int hidden, int intermediate, int experts, int k, int block_size,
    int mode = 0, void *quant = nullptr, float *quant_scales = nullptr);
