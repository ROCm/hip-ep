/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

static int random_normal_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

static float f32_from_bits(int64_t bits) {
  uint32_t u = static_cast<uint32_t>(bits);
  float value = 0.0f;
  std::memcpy(&value, &u, sizeof(value));
  return value;
}

int wrap_random_normal_like(RuntimeState *state, void *output, int64_t rank,
                            const int64_t *shape, int64_t mean_bits,
                            int64_t scale_bits, int64_t seed_bits,
                            int64_t has_seed, int64_t data_type) {
  OP_PROFILE(
      "random_normal_like",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:%s", (long long)rank,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !output || (rank > 0 && !shape)) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_random_normal_like: null argument\n");
    return -1;
  }

  int hip_dtype = random_normal_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_random_normal_like: unsupported data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  uint64_t seed = static_cast<uint64_t>(seed_bits);
  if (!has_seed) {
    seed = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }

  float mean = f32_from_bits(mean_bits);
  float scale = f32_from_bits(scale_bits);
  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_random_normal_like: rank=%lld, has_seed=%lld, dtype=%s "
      "-> hip_random_normal_like\n",
      (long long)rank, (long long)has_seed, hipdnn_ep_datatype_name(data_type));

  return hip_random_normal_like(stream, output, rank, shape, mean, scale, seed,
                                hip_dtype);
}
