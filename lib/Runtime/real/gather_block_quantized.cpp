/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime stub for `com.microsoft.GatherBlockQuantized`.
//
// The full implementation needs:
//   * sub-byte unpack (int4 / uint4: 2 elements per byte, low nibble first)
//   * block-wise dequantize: out_scaled = (q - zp) * scale, where the
//     scale/zp index maps quantize_axis position p -> (p / block_size)
//   * gather along gather_axis using `indices`
//   * default zp = 0 for int4/uint4, 2^(bits-1) for uint8 when zero_points
//     is absent
//
// None of that exists yet — the function body deliberately prints its
// arguments and then throws so any model that exercises the op fails loudly
// (instead of silently passing because of CPU EP fallback or a bit-pattern
// match of zero-init garbage).

#include "../hipdnn_ep_runtime.h"

#include <cstdio>
#include <stdexcept>

extern "C" int wrap_gather_block_quantized(
    RuntimeState *state,
    const void *data, const void *indices, const void *scales,
    const void *zero_points, void *output,
    const int64_t *data_shape, int64_t data_rank,
    const int64_t *indices_shape, int64_t indices_rank,
    const int64_t *scales_shape, int64_t scales_rank,
    const int64_t *output_shape, int64_t output_rank,
    int64_t bits, int64_t block_size, int64_t gather_axis,
    int64_t quantize_axis,
    int64_t data_dtype, int64_t indices_dtype, int64_t scales_dtype) {
  (void)state;
  (void)data;
  (void)indices;
  (void)scales;
  (void)zero_points;
  (void)output;
  (void)data_shape;
  (void)indices_shape;
  (void)scales_shape;
  (void)output_shape;

  fprintf(stderr,
          "[gather_block_quantized] STUB CALLED — not yet implemented.\n"
          "  data_rank=%lld indices_rank=%lld scales_rank=%lld "
          "output_rank=%lld\n"
          "  bits=%lld block_size=%lld gather_axis=%lld quantize_axis=%lld\n"
          "  data_dtype=%s(%lld) indices_dtype=%s(%lld) scales_dtype=%s(%lld)\n"
          "  zero_points=%s\n",
          (long long)data_rank, (long long)indices_rank,
          (long long)scales_rank, (long long)output_rank, (long long)bits,
          (long long)block_size, (long long)gather_axis,
          (long long)quantize_axis, hipdnn_ep_datatype_name(data_dtype),
          (long long)data_dtype, hipdnn_ep_datatype_name(indices_dtype),
          (long long)indices_dtype, hipdnn_ep_datatype_name(scales_dtype),
          (long long)scales_dtype, zero_points ? "yes" : "null");
  fflush(stderr);

  throw std::runtime_error(
      "wrap_gather_block_quantized: runtime not implemented "
      "(com.microsoft.GatherBlockQuantized stub). See lib/Runtime/real/"
      "gather_block_quantized.cpp.");
}
