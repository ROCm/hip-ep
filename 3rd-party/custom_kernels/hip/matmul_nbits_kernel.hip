/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int kBlockDim = 16;

// Fused dequant-matmul kernel for int4 packed weights.
//
// Y[batch, m, n] = sum_k( A[batch,m,k] * dequant(B[n,k]) ) + bias[n]
//
// Dequantization per block: dequant = (q - zero_point) * scale
// Default zero_point = 8 for 4-bit (2^(bits-1)).
//
// Each thread computes one output element, accumulating in fp32 for precision.
// The dequantization is fused into the dot-product loop — no temporary buffer.
template <typename T>
__global__ void matmul_nbits_kernel(
    const T* __restrict__ A,
    const uint8_t* __restrict__ B,
    const T* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks = (K + block_size - 1) / block_size;
  int64_t blob_size = block_size / 2;

  float acc = 0.0f;
  const T* a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * k_blocks * blob_size;
  const T* s_row = scales + n * k_blocks;

  for (int64_t blk = 0; blk < k_blocks; blk++) {
    float scale_val = static_cast<float>(s_row[blk]);
    float zp_val = (zp != nullptr)
                       ? static_cast<float>(zp[n * k_blocks + blk])
                       : 8.0f;

    int64_t k_start = blk * block_size;
    int64_t k_end = k_start + block_size;
    if (k_end > K) k_end = K;
    const uint8_t* b_block = b_row + blk * blob_size;

    for (int64_t k = k_start; k < k_end; k++) {
      int64_t in_blk = k - k_start;
      int64_t byte_off = in_blk / 2;
      int nibble = static_cast<int>(in_blk & 1);
      uint8_t packed = b_block[byte_off];
      float qval = (nibble == 0) ? static_cast<float>(packed & 0xF)
                                 : static_cast<float>(packed >> 4);

      acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
    }
  }

  if (bias) acc += static_cast<float>(bias[n]);
  output[batch * M * N + m * N + n] = static_cast<T>(acc);
}

extern "C" int hip_matmul_nbits(
    void* stream,
    const void* A,
    const void* B,
    const void* scales,
    const void* zero_points,
    const void* bias,
    void* output,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes) {
  if (bits != 4) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only bits=4 supported, "
            "got %lld\n",
            (long long)bits);
    return -1;
  }

  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return 0;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_matmul_nbits: M=%lld N=%lld K=%lld batch=%lld "
      "bits=%lld bs=%lld elem=%lld zp=%s bias=%s\n",
      (long long)M, (long long)N, (long long)K, (long long)batch_count,
      (long long)bits, (long long)block_size, (long long)element_size_bytes,
      zero_points ? "yes" : "null", bias ? "yes" : "null");

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  dim3 block(kBlockDim, kBlockDim);
  dim3 grid(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
            static_cast<unsigned>((M + kBlockDim - 1) / kBlockDim),
            static_cast<unsigned>(batch_count));

  if (element_size_bytes == 2) {
    hipLaunchKernelGGL(
        matmul_nbits_kernel<__half>, grid, block, 0, hip_stream,
        static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
        static_cast<const __half*>(scales),
        static_cast<const uint8_t*>(zero_points),
        static_cast<const __half*>(bias), static_cast<__half*>(output),
        M, N, K, block_size);
  } else if (element_size_bytes == 4) {
    hipLaunchKernelGGL(
        matmul_nbits_kernel<float>, grid, block, 0, hip_stream,
        static_cast<const float*>(A), static_cast<const uint8_t*>(B),
        static_cast<const float*>(scales),
        static_cast<const uint8_t*>(zero_points),
        static_cast<const float*>(bias), static_cast<float*>(output),
        M, N, K, block_size);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
