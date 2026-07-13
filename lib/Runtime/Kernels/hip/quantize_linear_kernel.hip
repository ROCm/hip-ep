/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kMaxRank = 8;

__device__ int64_t param_numel(const int64_t *shape, int rank) {
  if (rank == 0)
    return 1;
  int64_t n = 1;
  for (int i = 0; i < rank; ++i)
    n *= shape[i];
  return n;
}

__device__ void unravel_index(int64_t flat, const int64_t *shape, int rank,
                              int64_t *coords) {
  for (int d = rank - 1; d >= 0; --d) {
    coords[d] = flat % shape[d];
    flat /= shape[d];
  }
}

__device__ int64_t param_linear_index(const int64_t *coords, int x_rank,
                                      const int64_t *param_shape,
                                      int param_rank, int axis,
                                      int block_size) {
  if (param_rank == 0 || param_numel(param_shape, param_rank) == 1)
    return 0;

  int ax = axis;
  if (ax < 0)
    ax += x_rank;

  if (param_rank == 1 && block_size == 0)
    return coords[ax];

  int64_t idx = 0;
  int64_t stride = 1;
  for (int d = param_rank - 1; d >= 0; --d) {
    int64_t c;
    if (block_size > 0 && d == ax)
      c = coords[d] / block_size;
    else if (param_shape[d] == 1)
      c = 0;
    else
      c = coords[d];
    idx += c * stride;
    stride *= param_shape[d];
  }
  return idx;
}

template <typename T> __device__ float to_float(T v) { return static_cast<float>(v); }

template <> __device__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <> __device__ float to_float<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ T from_float(float v);

template <> __device__ float from_float<float>(float v) { return v; }

template <> __device__ __half from_float<__half>(float v) {
  return __float2half(v);
}

template <> __device__ __hip_bfloat16 from_float<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ float round_even(float v) { return nearbyintf(v); }

__device__ int8_t saturate_i8(float v) {
  v = round_even(v);
  if (v > 127.f)
    return 127;
  if (v < -128.f)
    return -128;
  return static_cast<int8_t>(v);
}

__device__ uint8_t saturate_u8(float v) {
  v = round_even(v);
  if (v > 255.f)
    return 255;
  if (v < 0.f)
    return 0;
  return static_cast<uint8_t>(v);
}

__device__ int16_t saturate_i16(float v) {
  v = round_even(v);
  if (v > 32767.f)
    return 32767;
  if (v < -32768.f)
    return -32768;
  return static_cast<int16_t>(v);
}

__device__ uint16_t saturate_u16(float v) {
  v = round_even(v);
  if (v > 65535.f)
    return 65535;
  if (v < 0.f)
    return 0;
  return static_cast<uint16_t>(v);
}

template <typename OutT, typename ScaleT, typename InputT, bool InputUnsigned>
__global__ void dequantize_linear_kernel(
    const InputT *x, const ScaleT *scale, const InputT *zero_point, OutT *y,
    int64_t total, int64_t x_shape[kMaxRank], int x_rank,
    int64_t scale_shape[kMaxRank], int scale_rank, int axis, int block_size,
    int default_zp) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int64_t coords[kMaxRank] = {};
  unravel_index(tid, x_shape, x_rank, coords);
  int64_t pidx = param_linear_index(coords, x_rank, scale_shape, scale_rank,
                                   axis, block_size);

  float xval;
  if constexpr (InputUnsigned)
    xval = static_cast<float>(static_cast<uint8_t>(x[tid]));
  else
    xval = static_cast<float>(x[tid]);

  float zpval = 0.f;
  if (zero_point)
    zpval = to_float<InputT>(zero_point[pidx]);
  else
    zpval = static_cast<float>(default_zp);

  float scaleval = to_float<ScaleT>(scale[pidx]);
  y[tid] = from_float<OutT>((xval - zpval) * scaleval);
}

template <typename OutT, typename ScaleT, typename InputT, bool OutUnsigned>
__global__ void quantize_linear_kernel(
    const InputT *x, const ScaleT *scale, const OutT *zero_point, OutT *y,
    int64_t total, int64_t x_shape[kMaxRank], int x_rank,
    int64_t scale_shape[kMaxRank], int scale_rank, int axis, int block_size,
    int default_zp) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int64_t coords[kMaxRank] = {};
  unravel_index(tid, x_shape, x_rank, coords);
  int64_t pidx = param_linear_index(coords, x_rank, scale_shape, scale_rank,
                                   axis, block_size);

  float xval = to_float<InputT>(x[tid]);
  float scaleval = to_float<ScaleT>(scale[pidx]);
  if (scaleval == 0.f)
    scaleval = 1.f;

  float zpval = 0.f;
  if (zero_point) {
    if constexpr (OutUnsigned)
      zpval = static_cast<float>(static_cast<uint8_t>(zero_point[pidx]));
    else
      zpval = static_cast<float>(zero_point[pidx]);
  } else {
    zpval = static_cast<float>(default_zp);
  }

  float q = xval / scaleval + zpval;
  if constexpr (std::is_same<OutT, int8_t>::value)
    y[tid] = saturate_i8(q);
  else if constexpr (std::is_same<OutT, uint8_t>::value)
    y[tid] = saturate_u8(q);
  else if constexpr (std::is_same<OutT, int16_t>::value)
    y[tid] = saturate_i16(q);
  else if constexpr (std::is_same<OutT, uint16_t>::value)
    y[tid] = saturate_u16(q);
  else
    y[tid] = static_cast<OutT>(round_even(q));
}

#define GQA_RETURN_LAUNCH_STATUS()                                             \
  do {                                                                         \
    hipError_t _st = hipGetLastError();                                        \
    return (_st == hipSuccess) ? 0 : static_cast<int>(_st);                    \
  } while (0)

} // namespace

extern "C" int hip_dequantize_linear(
    void *stream, const void *x, const void *scale, const void *zero_point,
    void *output, const int64_t *x_shape, int x_rank,
    const int64_t *scale_shape, int scale_rank, int axis, int block_size,
    int x_dtype, int scale_dtype, int out_dtype) {
  if (!x || !scale || !output || !x_shape || !scale_shape)
    return -1;
  if (x_rank < 0 || x_rank > kMaxRank || scale_rank < 0 || scale_rank > kMaxRank)
    return -1;

  int64_t total = 1;
  for (int i = 0; i < x_rank; ++i)
    total *= x_shape[i];
  if (total <= 0)
    return 0;

  int64_t x_shape_h[kMaxRank] = {};
  int64_t scale_shape_h[kMaxRank] = {};
  for (int i = 0; i < x_rank; ++i)
    x_shape_h[i] = x_shape[i];
  for (int i = 0; i < scale_rank; ++i)
    scale_shape_h[i] = scale_shape[i];

  const int block = 256;
  int grid = static_cast<int>((total + block - 1) / block);

  (void)hipGetLastError();

  // x_dtype: 5=int8, 7=uint8, 3=int32; scale/out: 0=f32, 1=f16, 2=bf16
  if (x_dtype == 5 && scale_dtype == 0 && out_dtype == 0) {
    dequantize_linear_kernel<float, float, int8_t, false>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const int8_t *>(x), static_cast<const float *>(scale),
            static_cast<const int8_t *>(zero_point),
            static_cast<float *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else if (x_dtype == 7 && scale_dtype == 0 && out_dtype == 0) {
    dequantize_linear_kernel<float, float, uint8_t, true>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const uint8_t *>(x), static_cast<const float *>(scale),
            static_cast<const uint8_t *>(zero_point),
            static_cast<float *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else if (x_dtype == 5 && scale_dtype == 0 && out_dtype == 1) {
    dequantize_linear_kernel<__half, float, int8_t, false>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const int8_t *>(x), static_cast<const float *>(scale),
            static_cast<const int8_t *>(zero_point),
            static_cast<__half *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else if (x_dtype == 7 && scale_dtype == 0 && out_dtype == 1) {
    dequantize_linear_kernel<__half, float, uint8_t, true>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const uint8_t *>(x), static_cast<const float *>(scale),
            static_cast<const uint8_t *>(zero_point),
            static_cast<__half *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else if (x_dtype == 3 && scale_dtype == 0 && out_dtype == 0) {
    dequantize_linear_kernel<float, float, int32_t, false>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const int32_t *>(x), static_cast<const float *>(scale),
            nullptr, static_cast<float *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else {
    return -1;
  }

  GQA_RETURN_LAUNCH_STATUS();
}

extern "C" int hip_quantize_linear(
    void *stream, const void *x, const void *scale, const void *zero_point,
    void *output, const int64_t *x_shape, int x_rank,
    const int64_t *scale_shape, int scale_rank, int axis, int block_size,
    int x_dtype, int scale_dtype, int out_dtype) {
  if (!x || !scale || !output || !x_shape || !scale_shape)
    return -1;
  if (x_rank < 0 || x_rank > kMaxRank || scale_rank < 0 || scale_rank > kMaxRank)
    return -1;

  int64_t total = 1;
  for (int i = 0; i < x_rank; ++i)
    total *= x_shape[i];
  if (total <= 0)
    return 0;

  int64_t x_shape_h[kMaxRank] = {};
  int64_t scale_shape_h[kMaxRank] = {};
  for (int i = 0; i < x_rank; ++i)
    x_shape_h[i] = x_shape[i];
  for (int i = 0; i < scale_rank; ++i)
    scale_shape_h[i] = scale_shape[i];

  const int block = 256;
  int grid = static_cast<int>((total + block - 1) / block);

  (void)hipGetLastError();

  if (x_dtype == 0 && scale_dtype == 0 && out_dtype == 7) {
    quantize_linear_kernel<uint8_t, float, float, true>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const float *>(x), static_cast<const float *>(scale),
            static_cast<const uint8_t *>(zero_point),
            static_cast<uint8_t *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else if (x_dtype == 0 && scale_dtype == 0 && out_dtype == 5) {
    quantize_linear_kernel<int8_t, float, float, false>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const float *>(x), static_cast<const float *>(scale),
            static_cast<const int8_t *>(zero_point),
            static_cast<int8_t *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else if (x_dtype == 1 && scale_dtype == 0 && out_dtype == 7) {
    quantize_linear_kernel<uint8_t, float, __half, true>
        <<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const __half *>(x), static_cast<const float *>(scale),
            static_cast<const uint8_t *>(zero_point),
            static_cast<uint8_t *>(output), total, x_shape_h, x_rank,
            scale_shape_h, scale_rank, axis, block_size, 0);
  } else {
    return -1;
  }

  GQA_RETURN_LAUNCH_STATUS();
}
