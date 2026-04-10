/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Unit test for matmul_nbits GEMV kernel (M <= 4 decode path).
 * Compares GPU output against a CPU reference implementation.
 */

#include "hip_custom_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

// Host-side FP16 <-> FP32 conversion via bit manipulation.
// Avoids __truncsfhf2/__extendhfsf2 link errors when MSVC links hipcc objects.
static inline float to_float(__half h) {
  uint16_t bits;
  memcpy(&bits, &h, 2);
  uint32_t sign = (bits >> 15) & 1;
  uint32_t exp  = (bits >> 10) & 0x1F;
  uint32_t mant = bits & 0x3FF;
  float result;
  if (exp == 0) {
    // subnormal or zero
    result = ldexpf((float)mant / 1024.0f, -14);
  } else if (exp == 31) {
    // inf or nan
    uint32_t f = (sign << 31) | 0x7F800000 | (mant << 13);
    memcpy(&result, &f, 4);
    return result;
  } else {
    result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
  }
  return sign ? -result : result;
}
static inline __half make_half(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 31) & 1;
  int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
  uint32_t mant = bits & 0x7FFFFF;
  uint16_t h;
  if (exp > 15) {
    h = (uint16_t)((sign << 15) | 0x7C00); // inf
  } else if (exp < -14) {
    h = (uint16_t)(sign << 15); // zero (flush subnormals)
  } else {
    h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
  }
  __half result;
  memcpy(&result, &h, 2);
  return result;
}

// CPU reference: row-major int4 dequant + matmul
static void cpu_matmul_nbits_ref(
    const __half* A, const uint8_t* B, const __half* scales,
    const uint8_t* zp, const __half* bias, float* output,
    int M, int N, int K, int block_size) {
  int k_blocks = (K + block_size - 1) / block_size;
  int blob_size = block_size / 2;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      const uint8_t* b_row = B + n * k_blocks * blob_size;
      const __half* s_row = scales + n * k_blocks;

      for (int blk = 0; blk < k_blocks; blk++) {
        float s = to_float(s_row[blk]);
        float z = zp ? (float)zp[n * k_blocks + blk] : 8.0f;
        int k_start = blk * block_size;
        int k_end = k_start + block_size;
        if (k_end > K) k_end = K;
        const uint8_t* b_block = b_row + blk * blob_size;

        for (int k = k_start; k < k_end; k++) {
          int in_blk = k - k_start;
          int byte_off = in_blk / 2;
          int nibble = in_blk & 1;
          uint8_t packed = b_block[byte_off];
          float qval = nibble ? (float)(packed >> 4) : (float)(packed & 0xF);
          acc += to_float(A[m * K + k]) * (qval - z) * s;
        }
      }
      if (bias) acc += to_float(bias[n]);
      output[m * N + n] = acc;
    }
  }
}

struct TestCase {
  const char* name;
  int M, N, K, block_size;
  bool use_zp, use_bias;
};

static bool run_test(const TestCase& tc) {
  printf("  %-50s ... ", tc.name);

  int k_blocks = (tc.K + tc.block_size - 1) / tc.block_size;
  int blob_size = tc.block_size / 2;

  // Allocate host buffers
  size_t a_size = tc.M * tc.K;
  size_t b_size = tc.N * k_blocks * blob_size;
  size_t s_size = tc.N * k_blocks;
  size_t o_size = tc.M * tc.N;

  std::vector<__half> h_A(a_size);
  std::vector<uint8_t> h_B(b_size);
  std::vector<__half> h_scales(s_size);
  std::vector<uint8_t> h_zp(tc.use_zp ? s_size : 0);
  std::vector<__half> h_bias(tc.use_bias ? tc.N : 0);
  std::vector<__half> h_output(o_size, make_half(0.0f));
  std::vector<float> ref_output(o_size, 0.0f);

  // Fill with random data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  for (auto& v : h_A) v = make_half(dist(rng));
  for (auto& v : h_B) v = (uint8_t)byte_dist(rng);
  for (auto& v : h_scales) v = make_half(dist(rng) * 0.1f);  // small scales
  for (auto& v : h_zp) v = (uint8_t)(rng() % 16);  // 0-15
  for (auto& v : h_bias) v = make_half(dist(rng) * 0.5f);

  // CPU reference
  cpu_matmul_nbits_ref(
      h_A.data(), h_B.data(), h_scales.data(),
      tc.use_zp ? h_zp.data() : nullptr,
      tc.use_bias ? h_bias.data() : nullptr,
      ref_output.data(), tc.M, tc.N, tc.K, tc.block_size);

  // GPU buffers
  void *d_A, *d_B, *d_scales, *d_zp = nullptr, *d_bias = nullptr, *d_output;
  hipMalloc(&d_A, a_size * sizeof(__half));
  hipMalloc(&d_B, b_size);
  hipMalloc(&d_scales, s_size * sizeof(__half));
  if (tc.use_zp) hipMalloc(&d_zp, s_size);
  if (tc.use_bias) hipMalloc(&d_bias, tc.N * sizeof(__half));
  hipMalloc(&d_output, o_size * sizeof(__half));

  hipMemcpy(d_A, h_A.data(), a_size * sizeof(__half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B.data(), b_size, hipMemcpyHostToDevice);
  hipMemcpy(d_scales, h_scales.data(), s_size * sizeof(__half), hipMemcpyHostToDevice);
  if (tc.use_zp) hipMemcpy(d_zp, h_zp.data(), s_size, hipMemcpyHostToDevice);
  if (tc.use_bias) hipMemcpy(d_bias, h_bias.data(), tc.N * sizeof(__half), hipMemcpyHostToDevice);
  hipMemset(d_output, 0, o_size * sizeof(__half));

  // Run GPU kernel
  hipStream_t stream;
  hipStreamCreate(&stream);

  int rc = hip_matmul_nbits(
      stream, d_A, d_B, d_scales, d_zp, d_bias, d_output,
      tc.M, tc.N, tc.K, /*batch_count=*/1,
      /*bits=*/4, tc.block_size, /*element_size_bytes=*/2);

  hipStreamSynchronize(stream);

  if (rc != 0) {
    printf("FAIL (kernel returned %d)\n", rc);
    hipStreamDestroy(stream);
    hipFree(d_A); hipFree(d_B); hipFree(d_scales);
    if (d_zp) hipFree(d_zp); if (d_bias) hipFree(d_bias);
    hipFree(d_output);
    return false;
  }

  // Copy output back
  hipMemcpy(h_output.data(), d_output, o_size * sizeof(__half), hipMemcpyDeviceToHost);

  // Compare
  float max_abs_err = 0, max_rel_err = 0;
  int mismatch_count = 0;
  for (int i = 0; i < (int)o_size; i++) {
    float gpu_val = to_float(h_output[i]);
    float ref_val = ref_output[i];
    float abs_err = fabsf(gpu_val - ref_val);
    float rel_err = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;
    if (abs_err > max_abs_err) max_abs_err = abs_err;
    if (rel_err > max_rel_err) max_rel_err = rel_err;
    // FP16 tolerance: allow ~0.1% relative error or small absolute error
    if (abs_err > 0.5f && rel_err > 0.01f) {
      if (mismatch_count < 3) {
        fprintf(stderr, "    Mismatch at [%d]: gpu=%.6f ref=%.6f abs=%.6f rel=%.6f\n",
                i, gpu_val, ref_val, abs_err, rel_err);
      }
      mismatch_count++;
    }
  }

  // Cleanup
  hipStreamDestroy(stream);
  hipFree(d_A); hipFree(d_B); hipFree(d_scales);
  if (d_zp) hipFree(d_zp); if (d_bias) hipFree(d_bias);
  hipFree(d_output);

  if (mismatch_count > 0) {
    printf("FAIL (%d mismatches, max_abs=%.6f max_rel=%.6f)\n",
           mismatch_count, max_abs_err, max_rel_err);
    return false;
  }

  printf("PASS (max_abs=%.6f max_rel=%.6f)\n", max_abs_err, max_rel_err);
  return true;
}

int main() {
  printf("=== MatMulNBits GEMV Kernel Tests ===\n\n");

  TestCase tests[] = {
    {"M=1 N=4096 K=4096 bs=128 (QKV proj)",    1, 4096, 4096, 128, false, false},
    {"M=1 N=14336 K=4096 bs=128 (MLP gate)",    1, 14336, 4096, 128, false, false},
    {"M=1 N=4096 K=14336 bs=128 (MLP down)",    1, 4096, 14336, 128, false, false},
    {"M=1 N=4096 K=4096 bs=128 +zp",            1, 4096, 4096, 128, true, false},
    {"M=1 N=4096 K=4096 bs=128 +bias",          1, 4096, 4096, 128, false, true},
    {"M=1 N=4096 K=4096 bs=128 +zp+bias",       1, 4096, 4096, 128, true, true},
    {"M=4 N=4096 K=4096 bs=128 (small batch)",  4, 4096, 4096, 128, false, false},
    {"M=1 N=256 K=512 bs=32 (small)",            1, 256, 512, 32, false, false},
    {"M=1 N=1024 K=2048 bs=64 (medium)",         1, 1024, 2048, 64, false, false},
    {"M=2 N=4096 K=4096 bs=128 (batch=2)",       2, 4096, 4096, 128, false, false},
  };

  int passed = 0, failed = 0;
  for (const auto& tc : tests) {
    if (run_test(tc)) passed++;
    else failed++;
  }

  printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
