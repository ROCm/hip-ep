/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"
#include "gather_block_quantized_utils.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

// =============================================================================
// com.microsoft.GatherBlockQuantized
// =============================================================================
//
// One thread per output element. The output rank is
//   q + (r - 1)   where q = indices_rank, r = data_rank
// and the output coord layout follows the ONNX spec:
//   output_shape = data.shape[0:gather_axis]
//               ++ indices.shape
//               ++ data.shape[gather_axis+1:]
// i.e. position p in the output coord maps to:
//   p in [0,                gather_axis)        -> data dim p
//   p in [gather_axis,      gather_axis + q)    -> indices dim (p - gather_axis)
//   p in [gather_axis + q,  q + r - 1)          -> data dim (p - q + 1)
//
// Per-thread work:
//   1. Decompose the linear output index into out_coord (row-major).
//   2. Slice indices coords out of out_coord[gather_axis : gather_axis+q]
//      and fold them into the indices linear index; read gather_index
//      (i32 or i64).
//   3. Splice gather_index into the data coord at gather_axis to recover
//      the original data coord; build the data linear element index AND
//      the scales / zero_points linear index in the same pass (the only
//      difference is dim `quantize_axis` is divided by `block_size`).
//   4. Read the quantized value (sub-byte unpack for bits == 4).
//   5. Read zero_point (sub-byte unpack for bits == 4) or fall back to
//      `default_zp` when `zero_points` is null.
//   6. Read the per-block scale, dequantize via (q - zp) * scale in fp32,
//      and write the result in the output dtype.
//
// Bounds-checking on `gather_index`: ONNX leaves the behaviour undefined
// for out-of-range indices but ORT silently clamps / writes 0. We follow
// the writes-0 path (cheaper than an abort) so the kernel never reads
// past the data buffer.

#define HIP_GBQ_MAX_RANK 8

namespace {

struct GbqDimsND {
  int64_t v[HIP_GBQ_MAX_RANK];
};

// ---- per-output-dtype load/store helpers ---------------------------------

template <typename OutT> struct DTypeIO;

template <> struct DTypeIO<float> {
  __device__ static inline float load(float x) { return x; }
  __device__ static inline float store(float x) { return x; }
};

template <> struct DTypeIO<__half> {
  __device__ static inline float load(__half x) { return __half2float(x); }
  __device__ static inline __half store(float x) { return __float2half(x); }
};

template <> struct DTypeIO<__hip_bfloat16> {
  __device__ static inline float load(__hip_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ static inline __hip_bfloat16 store(float x) {
    return __float2bfloat16(x);
  }
};

// ---- packed-quantized value extraction -----------------------------------

// Read one element from the packed quantized byte buffer. The interpretation
// (signed vs unsigned, 4 vs 8 bit) is controlled by `bits` and `is_signed`.
// `lin_elem` is the LOGICAL element index (post-unpack), not a byte offset.
__device__ inline int read_quant(const uint8_t *__restrict__ buf,
                                 int64_t lin_elem, int bits, bool is_signed) {
  if (bits == 4) {
    int64_t byte_idx = lin_elem >> 1;
    uint8_t byte = buf[byte_idx];
    return hipdnn_ep::gbq::decodeStorageByte(
        byte, bits, is_signed, static_cast<int>(lin_elem));
  }
  return hipdnn_ep::gbq::decodeStorageByte(buf[lin_elem], bits, is_signed,
                                          /*logicalIndex=*/0);
}

// ---- main kernel ---------------------------------------------------------

template <typename OutT>
__global__ void gather_block_quantized_kernel(
    const uint8_t *__restrict__ data,
    const void *__restrict__ indices,        // int32 or int64
    const OutT *__restrict__ scales,
    const uint8_t *__restrict__ zero_points, // nullable
    OutT *__restrict__ output,
    int64_t total_output_elements,
    int data_rank, int indices_rank, int output_rank,
    int gather_axis, int quantize_axis,
    int bits, int block_size,
    int default_zp, int is_signed_data, int indices_is_int64,
    GbqDimsND data_shape, GbqDimsND indices_shape,
    GbqDimsND scales_shape, GbqDimsND output_shape) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_output_elements) {
    return;
  }

  // 1. Decompose the linear output index into per-axis coords (row-major).
  int64_t out_coord[HIP_GBQ_MAX_RANK];
  {
    int64_t rem = tid;
    for (int d = output_rank - 1; d >= 0; --d) {
      int64_t dim = output_shape.v[d];
      out_coord[d] = rem % dim;
      rem /= dim;
    }
  }

  // 2. Pick out the indices coord slice from the middle of out_coord
  //    (positions [gather_axis, gather_axis + indices_rank)), then read
  //    gather_index. ONNX Tind ∈ {int32, int64}.
  int64_t indices_lin = 0;
  for (int d = 0; d < indices_rank; ++d) {
    indices_lin =
        indices_lin * indices_shape.v[d] + out_coord[gather_axis + d];
  }
  int64_t gather_index;
  if (indices_is_int64) {
    gather_index = static_cast<const int64_t *>(indices)[indices_lin];
  } else {
    gather_index = static_cast<int64_t>(
        static_cast<const int32_t *>(indices)[indices_lin]);
  }
  int64_t gather_axis_size = data_shape.v[gather_axis];
  if (gather_index < 0) {
    gather_index += gather_axis_size;
  }
  // Out-of-range indices -> write 0 and bail. Mirrors hip_gather's policy
  // and avoids unbounded reads on user-error indices tensors.
  if (gather_index < 0 || gather_index >= gather_axis_size) {
    output[tid] = DTypeIO<OutT>::store(0.0f);
    return;
  }

  // 3. Walk data dims left-to-right, building both:
  //      data_elem_lin  - row-major linear index in `data` at the
  //                        per-element granularity (sub-byte unpack
  //                        happens later in read_quant)
  //      scales_lin     - row-major linear index in `scales` /
  //                        `zero_points`; identical to data_elem_lin
  //                        except the quantize_axis coord is divided
  //                        by block_size.
  int64_t data_elem_lin = 0;
  int64_t scales_lin = 0;
  for (int d = 0; d < data_rank; ++d) {
    int64_t coord;
    if (d == gather_axis) {
      // The gather dim was replaced by `indices.shape` in the output;
      // it doesn't appear in out_coord.
      coord = gather_index;
    } else if (d < gather_axis) {
      // Data dims before gather_axis sit in out_coord at the same
      // positions [0, gather_axis).
      coord = out_coord[d];
    } else {
      // d > gather_axis: data dims after gather_axis are pushed back by
      // (indices_rank - 1) in the output (gather_axis itself was
      // expanded into indices_rank dims).
      coord = out_coord[d + indices_rank - 1];
    }
    data_elem_lin = data_elem_lin * data_shape.v[d] + coord;
    int64_t sc_coord =
        (d == quantize_axis) ? (coord / block_size) : coord;
    scales_lin = scales_lin * scales_shape.v[d] + sc_coord;
  }

  // 4. Quantized value (sub-byte unpack for bits == 4).
  int qval = read_quant(data, data_elem_lin, bits, is_signed_data != 0);

  // 5. Zero point. Same packing as data; defaults to the caller-supplied
  //    `default_zp` when zero_points is absent.
  int zp_val;
  if (zero_points != nullptr) {
    zp_val =
        read_quant(zero_points, scales_lin, bits, is_signed_data != 0);
  } else {
    zp_val = default_zp;
  }

  // 6. (q - zp) * scale, computed in fp32 for stability across the three
  //    scale dtypes (fp16/fp32/bf16). The output cast back to OutT applies
  //    standard IEEE round-to-nearest-even via __float2half / bf16 helpers.
  float diff = static_cast<float>(qval - zp_val);
  float scale = DTypeIO<OutT>::load(scales[scales_lin]);
  output[tid] = DTypeIO<OutT>::store(diff * scale);
}

template <typename OutT>
static int launch_gbq(hipStream_t stream, const uint8_t *data,
                      const void *indices, const OutT *scales,
                      const uint8_t *zero_points, OutT *output,
                      int64_t total, int data_rank, int indices_rank,
                      int output_rank, int gather_axis, int quantize_axis,
                      int bits, int block_size, int default_zp,
                      int is_signed_data, int indices_is_int64,
                      const GbqDimsND &data_shape,
                      const GbqDimsND &indices_shape,
                      const GbqDimsND &scales_shape,
                      const GbqDimsND &output_shape) {
  constexpr int kBlockSize = 256;
  int64_t grid_long = (total + kBlockSize - 1) / kBlockSize;
  // 32-bit grid dim is plenty (tested: 2^31 * 256 = 5.5e11 elems).
  if (grid_long > static_cast<int64_t>(0x7FFFFFFF)) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: total=%lld too large "
            "for one launch (grid_x overflow)\n",
            (long long)total);
    return -1;
  }
  int grid_size = static_cast<int>(grid_long);

  hipLaunchKernelGGL(gather_block_quantized_kernel<OutT>, dim3(grid_size),
                     dim3(kBlockSize), 0, stream, data, indices, scales,
                     zero_points, output, total, data_rank, indices_rank,
                     output_rank, gather_axis, quantize_axis, bits, block_size,
                     default_zp, is_signed_data, indices_is_int64, data_shape,
                     indices_shape, scales_shape, output_shape);
  return 0;
}

} // namespace

extern "C" int hip_gather_block_quantized(
    void *stream,
    const void *data,
    const void *indices,
    const void *scales,
    const void *zero_points,
    void *output,
    const int64_t *data_shape, int data_rank,
    const int64_t *indices_shape, int indices_rank,
    const int64_t *scales_shape, int scales_rank,
    const int64_t *output_shape, int output_rank,
    int bits, int block_size, int gather_axis, int quantize_axis,
    int default_zp, int is_signed_data, int indices_is_int64,
    int out_dtype) {
  // ---- Validation. We reject ranks > MAX_RANK loudly rather than risk
  //      a silently truncated coord-decomposition loop.
  if (data_rank <= 0 || data_rank > HIP_GBQ_MAX_RANK ||
      indices_rank < 0 || indices_rank > HIP_GBQ_MAX_RANK ||
      scales_rank != data_rank ||
      output_rank != indices_rank + data_rank - 1 ||
      output_rank > HIP_GBQ_MAX_RANK) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: bad ranks "
            "(max=%d) data=%d indices=%d scales=%d output=%d\n",
            HIP_GBQ_MAX_RANK, data_rank, indices_rank, scales_rank,
            output_rank);
    return -1;
  }
  if (gather_axis < 0 || gather_axis >= data_rank ||
      quantize_axis < 0 || quantize_axis >= data_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: axis out of range "
            "(data_rank=%d gather_axis=%d quantize_axis=%d). The host wrapper "
            "is responsible for normalizing negatives.\n",
            data_rank, gather_axis, quantize_axis);
    return -1;
  }
  if (bits != 4 && bits != 8) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: unsupported bits=%d "
            "(only 4 and 8 are supported)\n",
            bits);
    return -1;
  }
  if (block_size <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: bad block_size=%d\n",
            block_size);
    return -1;
  }
  // Spec: scales.shape[quantize_axis] * block_size >= data.shape[quantize_axis].
  // Defense in depth: block_size must evenly tile data dim (or pad to next
  // block) so that coord / block_size never indexes past the scales tensor.
  // We allow the standard "ceil-div" case where data dim < scales dim *
  // block_size as well as exact division.
  if (scales_shape[quantize_axis] <= 0 ||
      static_cast<int64_t>(scales_shape[quantize_axis]) * block_size <
          data_shape[quantize_axis]) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: scales dim along "
            "quantize_axis (%lld) * block_size (%d) < data dim (%lld); "
            "scales tensor underflows the block grid\n",
            (long long)scales_shape[quantize_axis], block_size,
            (long long)data_shape[quantize_axis]);
    return -1;
  }

  // ---- Pack shapes into fixed-size structs for value-passing.
  int64_t total = 1;
  GbqDimsND data_nd = {}, indices_nd = {}, scales_nd = {}, output_nd = {};
  for (int i = 0; i < data_rank; ++i) {
    data_nd.v[i] = data_shape[i];
    scales_nd.v[i] = scales_shape[i];
  }
  for (int i = 0; i < indices_rank; ++i) {
    indices_nd.v[i] = indices_shape[i];
  }
  for (int i = 0; i < output_rank; ++i) {
    output_nd.v[i] = output_shape[i];
    total *= output_shape[i];
  }
  if (total <= 0) {
    return 0;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_gather_block_quantized: out_dtype=%d, total=%lld, "
      "data_rank=%d, indices_rank=%d, output_rank=%d, gather_axis=%d, "
      "quantize_axis=%d, bits=%d, block_size=%d, signed=%d, idx_is_i64=%d, "
      "default_zp=%d, has_zp=%d\n",
      out_dtype, (long long)total, data_rank, indices_rank, output_rank,
      gather_axis, quantize_axis, bits, block_size, is_signed_data,
      indices_is_int64, default_zp, zero_points != nullptr ? 1 : 0);

  int rc = 0;
  switch (out_dtype) {
  case HIP_DTYPE_FLOAT32:
    rc = launch_gbq<float>(
        hip_stream, static_cast<const uint8_t *>(data), indices,
        static_cast<const float *>(scales),
        static_cast<const uint8_t *>(zero_points),
        static_cast<float *>(output), total, data_rank, indices_rank,
        output_rank, gather_axis, quantize_axis, bits, block_size, default_zp,
        is_signed_data, indices_is_int64, data_nd, indices_nd, scales_nd,
        output_nd);
    break;
  case HIP_DTYPE_FLOAT16:
    rc = launch_gbq<__half>(
        hip_stream, static_cast<const uint8_t *>(data), indices,
        static_cast<const __half *>(scales),
        static_cast<const uint8_t *>(zero_points),
        static_cast<__half *>(output), total, data_rank, indices_rank,
        output_rank, gather_axis, quantize_axis, bits, block_size, default_zp,
        is_signed_data, indices_is_int64, data_nd, indices_nd, scales_nd,
        output_nd);
    break;
  case HIP_DTYPE_BFLOAT16:
    rc = launch_gbq<__hip_bfloat16>(
        hip_stream, static_cast<const uint8_t *>(data), indices,
        static_cast<const __hip_bfloat16 *>(scales),
        static_cast<const uint8_t *>(zero_points),
        static_cast<__hip_bfloat16 *>(output), total, data_rank, indices_rank,
        output_rank, gather_axis, quantize_axis, bits, block_size, default_zp,
        is_signed_data, indices_is_int64, data_nd, indices_nd, scales_nd,
        output_nd);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized: unsupported "
            "out_dtype=%d (expected FLOAT32 / FLOAT16 / BFLOAT16)\n",
            out_dtype);
    return -1;
  }
  if (rc != 0) {
    return rc;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_block_quantized launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
