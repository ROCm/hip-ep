/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime wrapper for `com.microsoft.GatherBlockQuantized`.
//
// Glue between the lowering ABI (`wrap_gather_block_quantized`, all i64
// scalars + raw shape pointers) and the GPU kernel
// (`hip_gather_block_quantized` in lib/Runtime/Kernels) which only
// wants normalised axes, a packed enum dtype, and a few derived flags.
//
// Responsibilities of this wrapper:
//   * stream extraction from the opaque RuntimeState
//   * axis normalisation (the lowering forwards the raw ONNX attribute,
//     which may be negative)
//   * dtype mapping HIPDNN_EP_DATATYPE_* -> HIP_DTYPE_* and signedness
//     derivation from the data tensor element type
//   * default zp computation per the ONNX spec:
//       - signed int4/int8 storage ......... default 0
//       - unsigned uint4 storage ........... default 8
//       - unsigned uint8 storage ........... default 128
//   * spec compliance: uint8 uses gather_axis 0 and the last quantize axis
//   * size-zero output fast-path (no kernel launch)

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "gather_block_quantized_utils.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

namespace {

// Map the runtime's HIPDNN_EP_DATATYPE_* enum to the kernel-side
// hip_dtype_t. Returns -1 for unsupported types so callers can produce
// a clear error before launch.
int map_to_hip_dtype(int64_t hipdnn_dtype) {
  switch (hipdnn_dtype) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

} // namespace

extern "C" int wrap_gather_block_quantized(
    RuntimeState *state, const void *data, const void *indices,
    const void *scales, const void *zero_points, void *output,
    const int64_t *data_shape, int64_t data_rank, const int64_t *indices_shape,
    int64_t indices_rank, const int64_t *scales_shape, int64_t scales_rank,
    const int64_t *output_shape, int64_t output_rank, int64_t bits,
    int64_t block_size, int64_t gather_axis, int64_t quantize_axis,
    int64_t data_dtype, int64_t indices_dtype, int64_t scales_dtype) {
  OP_PROFILE(
      "gather_block_quantized",
      [&] {
        char b[128];
        snprintf(b, sizeof(b), "bits=%lld bs=%lld dr=%lld qr=%lld",
                 (long long)bits, (long long)block_size, (long long)data_rank,
                 (long long)indices_rank);
        return std::string(b);
      },
      state);

  if (!state || !data || !indices || !scales || !output || !data_shape ||
      !indices_shape || !scales_shape || !output_shape) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: null required argument\n");
    return -1;
  }
  if (bits != 4 && bits != 8) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: bits must be 4 or 8, got "
            "%lld\n",
            (long long)bits);
    return -1;
  }
  if (data_dtype != HIPDNN_EP_DATATYPE_INT8 &&
      data_dtype != HIPDNN_EP_DATATYPE_UINT8) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: data dtype must be i8 or "
            "ui8, got %s (%lld)\n",
            hipdnn_ep_datatype_name(data_dtype), (long long)data_dtype);
    return -1;
  }

  // Normalise potentially-negative axes against `data_rank`. The conversion
  // (lib/Conversion/OnnxToHip/GatherBlockQuantizedConversion.cpp) forwards
  // the raw ONNX attribute, which the spec lets be negative.
  int gather_axis_n = static_cast<int>(gather_axis);
  if (gather_axis_n < 0)
    gather_axis_n += static_cast<int>(data_rank);
  int quantize_axis_n = static_cast<int>(quantize_axis);
  if (quantize_axis_n < 0)
    quantize_axis_n += static_cast<int>(data_rank);
  if (gather_axis_n < 0 || gather_axis_n >= data_rank || quantize_axis_n < 0 ||
      quantize_axis_n >= data_rank) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: axis out of range "
            "(data_rank=%lld gather_axis=%lld quantize_axis=%lld)\n",
            (long long)data_rank, (long long)gather_axis,
            (long long)quantize_axis);
    return -1;
  }

  bool is_signed_data = data_dtype == HIPDNN_EP_DATATYPE_INT8;
  if (!hipdnn_ep::gbq::supportsUint8Axes(
          static_cast<int>(bits), !is_signed_data, static_cast<int>(data_rank),
          gather_axis_n, quantize_axis_n)) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: ONNX spec requires "
            "uint8 data to use gather_axis == 0 and the last dimension as "
            "quantize_axis (got gather_axis=%d, quantize_axis=%d, rank=%lld)\n",
            gather_axis_n, quantize_axis_n, (long long)data_rank);
    return -1;
  }

  // Default zero point when zero_points is omitted (ONNX spec):
  // "If zero_points is not provided, the default value is 0 for signed
  //  types and 2^(bits-1) for unsigned types." That gives the four-way
  // table:
  //   bits == 4 + int8  storage (int4)  .. 0
  //   bits == 4 + uint8 storage (uint4)  .. 8   (= 2^(bits-1))
  //   bits == 8 + int8  storage (int8)   .. 0
  //   bits == 8 + uint8 storage (uint8)  .. 128 (= 2^(bits-1))
  // The previous version hard-coded 0 for bits == 4, which silently
  // produces values shifted by +(8 * scale) on every uint4-packed weight
  // (canonical site: Qwen3.5 vision encoder's pos_embed.weight_Q4 — the
  // model dequantises to roughly +0.65/+0.73 in place of the correct
  // values, which cluster around 0).
  int default_zp = hipdnn_ep::gbq::defaultZeroPoint(
      static_cast<int>(bits), /*unsignedStorage=*/!is_signed_data);

  // Sub-byte storage: when bits == 4 and the data tensor element type is
  // uint8 / int8 (no native int4/uint4 in the EP type system), ONNX stores
  // two logical 4-bit values per byte, packed along quantize_axis with the
  // low nibble being the lower logical index. The memref-derived shape we
  // get from the lowering is therefore the BYTE shape (e.g. [2304, 576] for
  // a 2304x1152 logical tensor with quantize_axis=1). The kernel's coord
  // arithmetic assumes `data_shape` is the LOGICAL shape — read_quant does
  // the sub-byte unpack from a logical element index — so we materialise a
  // local shape array with quantize_axis doubled and pass that to the kernel.
  // The scales / zero_points tensors are not packed and keep their original
  // shape (the per-block scale axis already divides the logical element count).
  int64_t logical_data_shape_buf[8];
  if (data_rank >
      static_cast<int64_t>(sizeof(logical_data_shape_buf) / sizeof(int64_t))) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: data_rank=%lld exceeds "
            "max supported rank %zu\n",
            (long long)data_rank,
            sizeof(logical_data_shape_buf) / sizeof(int64_t));
    return -1;
  }
  for (int64_t i = 0; i < data_rank; ++i)
    logical_data_shape_buf[i] = data_shape[i];
  // Every bits=4 HIP operand uses byte-packed storage: native ONNX int4/uint4
  // constants are legalized to i8 before this wrapper, and uint8 fallback
  // models arrive packed already. Shape compatibility was verified in the HIP
  // dialect, so the logical extent is always twice the byte extent, including
  // a partially filled final quantization block.
  if (bits == 4)
    logical_data_shape_buf[quantize_axis_n] *= 2;
  const int64_t *logical_data_shape = logical_data_shape_buf;

  int hip_out_dtype = map_to_hip_dtype(scales_dtype);
  if (hip_out_dtype != HIP_DTYPE_FLOAT32 &&
      hip_out_dtype != HIP_DTYPE_FLOAT16 &&
      hip_out_dtype != HIP_DTYPE_BFLOAT16) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: unsupported scales/output "
            "dtype %s (%lld); kernel supports fp32/fp16/bf16\n",
            hipdnn_ep_datatype_name(scales_dtype), (long long)scales_dtype);
    return -1;
  }

  if (indices_dtype != HIPDNN_EP_DATATYPE_INT32 &&
      indices_dtype != HIPDNN_EP_DATATYPE_INT64) {
    fprintf(stderr,
            "[REAL] wrap_gather_block_quantized: indices dtype must be "
            "i32 or i64, got %s (%lld)\n",
            hipdnn_ep_datatype_name(indices_dtype), (long long)indices_dtype);
    return -1;
  }
  int indices_is_int64 = (indices_dtype == HIPDNN_EP_DATATYPE_INT64) ? 1 : 0;

  // Empty-output fast-path. Skip the launch entirely; matches ORT semantics
  // (no work, no error).
  int64_t total = 1;
  for (int64_t i = 0; i < output_rank; ++i)
    total *= output_shape[i];
  if (total <= 0)
    return 0;

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gather_block_quantized: data_dtype=%s indices_dtype=%s "
      "scales_dtype=%s bits=%lld block_size=%lld gather_axis=%d "
      "quantize_axis=%d signed=%d default_zp=%d has_zp=%d "
      "data_rank=%lld indices_rank=%lld out_rank=%lld total=%lld\n",
      hipdnn_ep_datatype_name(data_dtype),
      hipdnn_ep_datatype_name(indices_dtype),
      hipdnn_ep_datatype_name(scales_dtype), (long long)bits,
      (long long)block_size, gather_axis_n, quantize_axis_n,
      (int)is_signed_data, default_zp, zero_points ? 1 : 0,
      (long long)data_rank, (long long)indices_rank, (long long)output_rank,
      (long long)total);

  return hip_gather_block_quantized(
      stream, data, indices, scales, zero_points, output, logical_data_shape,
      static_cast<int>(data_rank), indices_shape,
      static_cast<int>(indices_rank), scales_shape,
      static_cast<int>(scales_rank), output_shape,
      static_cast<int>(output_rank), static_cast<int>(bits),
      static_cast<int>(block_size), gather_axis_n, quantize_axis_n, default_zp,
      is_signed_data ? 1 : 0, indices_is_int64, hip_out_dtype);
}
