/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Unit test: MatMulNBits Vulkan compute shader vs CPU reference.
 *
 * For each test case:
 *   1. Generate deterministic pseudo-random inputs (A, B, scales, zp, bias)
 *   2. Compute CPU reference in float32 (gold standard)
 *   3. Run vulkan_matmul_nbits()
 *   4. Compare: max absolute error, RMSE, percentage of elements within tolerance
 *
 * No HIP dependency — this test is standalone Vulkan + CPU.
 */

#include "vulkan_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

// ===== FP16 conversion helpers =====

static uint16_t float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp > 15) return (uint16_t)(sign | 0x7C00);          // inf
    if (exp < -14) {
        mant |= 0x800000;
        int shift = -1 - exp;
        if (shift > 24) return (uint16_t)sign;
        mant >>= shift;
        return (uint16_t)(sign | (mant >> 13));
    }
    return (uint16_t)(sign | ((exp + 15) << 10) | (mant >> 13));
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

// ===== Simple LCG PRNG =====
static uint32_t g_seed = 42;
static uint32_t rng() { g_seed = g_seed * 1103515245u + 12345u; return g_seed; }
static float    rng_float(float lo, float hi) {
    return lo + (hi - lo) * ((rng() & 0xFFFF) / 65535.0f);
}

// ===== CPU reference implementation =====

static void cpu_matmul_nbits_ref(
    const uint16_t *A_fp16, const uint8_t *B_packed,
    const uint16_t *scales_fp16, const uint8_t *zp_u8,
    const uint16_t *bias_fp16, float *output_f32,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count, int64_t block_size) {

    int64_t k_blocks = (K + block_size - 1) / block_size;

    for (int64_t b = 0; b < batch_count; b++) {
        for (int64_t m = 0; m < M; m++) {
            for (int64_t n = 0; n < N; n++) {
                double acc = 0.0;
                for (int64_t blk = 0; blk < k_blocks; blk++) {
                    float scale = half_to_float(scales_fp16[n * k_blocks + blk]);
                    float zp_val = zp_u8 ? (float)zp_u8[n * k_blocks + blk] : 0.0f;

                    int64_t k_start = blk * block_size;
                    int64_t k_end   = k_start + block_size;
                    if (k_end > K) k_end = K;
                    int64_t blob_off = blk * (block_size / 2);

                    for (int64_t k = k_start; k < k_end; k++) {
                        int64_t in_blk = k - k_start;
                        uint8_t packed = B_packed[n * (K / 2) + blob_off + in_blk / 2];
                        float qval = ((in_blk & 1) == 0) ? (float)(packed & 0xF)
                                                          : (float)(packed >> 4);
                        float a_val = half_to_float(A_fp16[b * M * K + m * K + k]);
                        acc += (double)a_val * (double)(qval - zp_val) * (double)scale;
                    }
                }
                if (bias_fp16) {
                    acc += (double)half_to_float(bias_fp16[n]);
                }
                output_f32[b * M * N + m * N + n] = (float)acc;
            }
        }
    }
}

// ===== Test runner =====

struct TestCase {
    const char *name;
    int64_t M, N, K, block_size, batch_count;
    bool has_zp;
    bool has_bias;
};

static int run_test(const TestCase &tc) {
    g_seed = 42; // reset RNG for reproducibility

    int64_t k_blocks = (tc.K + tc.block_size - 1) / tc.block_size;

    // Allocate inputs
    size_t a_count     = (size_t)(tc.batch_count * tc.M * tc.K);
    size_t b_count     = (size_t)(tc.N * (tc.K / 2));
    size_t s_count     = (size_t)(tc.N * k_blocks);
    size_t out_count   = (size_t)(tc.batch_count * tc.M * tc.N);

    std::vector<uint16_t> A_fp16(a_count);
    std::vector<uint8_t>  B_packed(b_count);
    std::vector<uint16_t> scales_fp16(s_count);
    std::vector<uint8_t>  zp_u8(tc.has_zp ? s_count : 0);
    std::vector<uint16_t> bias_fp16(tc.has_bias ? (size_t)tc.N : 0);
    std::vector<uint16_t> out_vk(out_count, 0);
    std::vector<float>    out_ref(out_count, 0.0f);

    // Fill random data
    for (auto &v : A_fp16)     v = float_to_half(rng_float(-1.0f, 1.0f));
    for (auto &v : B_packed)   v = (uint8_t)(rng() & 0xFF);
    for (auto &v : scales_fp16) v = float_to_half(rng_float(0.001f, 0.1f));
    for (auto &v : zp_u8)      v = (uint8_t)(rng() % 16);
    for (auto &v : bias_fp16)  v = float_to_half(rng_float(-0.5f, 0.5f));

    // CPU reference
    cpu_matmul_nbits_ref(
        A_fp16.data(), B_packed.data(), scales_fp16.data(),
        tc.has_zp ? zp_u8.data() : nullptr,
        tc.has_bias ? bias_fp16.data() : nullptr,
        out_ref.data(),
        tc.M, tc.N, tc.K, tc.batch_count, tc.block_size);

    // Vulkan
    auto t0 = std::chrono::high_resolution_clock::now();

    int rc = vulkan_matmul_nbits(
        nullptr,
        A_fp16.data(), B_packed.data(), scales_fp16.data(),
        tc.has_zp ? zp_u8.data() : nullptr,
        tc.has_bias ? bias_fp16.data() : nullptr,
        out_vk.data(),
        tc.M, tc.N, tc.K, tc.batch_count,
        4, tc.block_size, 2);

    auto t1 = std::chrono::high_resolution_clock::now();
    double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gpu_ms = vulkan_get_last_kernel_time_ms();

    if (rc != 0) {
        fprintf(stderr, "  FAIL [%s]: vulkan_matmul_nbits returned %d\n",
                tc.name, rc);
        return 1;
    }

    // Compare
    double max_abs_err = 0.0;
    double sum_sq_err  = 0.0;
    int    bad_count   = 0;
    double tolerance   = 0.15; // FP16 accumulation tolerance

    for (size_t i = 0; i < out_count; i++) {
        float vk_val  = half_to_float(out_vk[i]);
        float ref_val = out_ref[i];
        double err    = fabs((double)vk_val - (double)ref_val);
        max_abs_err   = std::max(max_abs_err, err);
        sum_sq_err   += err * err;

        // Relative tolerance for larger values
        double rel_tol = std::max(tolerance, (double)fabs(ref_val) * 0.02);
        if (err > rel_tol) bad_count++;
    }

    double rmse = sqrt(sum_sq_err / (double)out_count);
    double pct_good = 100.0 * (1.0 - (double)bad_count / (double)out_count);

    // Compute GFLOPS (2*M*N*K FLOPs for matmul)
    double flops = 2.0 * (double)tc.M * (double)tc.N * (double)tc.K * (double)tc.batch_count;
    double gflops_gpu = (gpu_ms > 0.0) ? (flops / (gpu_ms * 1e6)) : 0.0;

    bool pass = (pct_good >= 99.0) && (rmse < 1.0);

    printf("  %s [%-16s]: M=%-3lld N=%-6lld K=%-5lld bs=%-3lld | "
           "gpu=%.3fms e2e=%.1fms | "
           "%.1f GFLOPS/s | "
           "max_err=%.4f rmse=%.6f good=%.1f%%\n",
           pass ? "PASS" : "FAIL", tc.name,
           (long long)tc.M, (long long)tc.N, (long long)tc.K,
           (long long)tc.block_size,
           gpu_ms, e2e_ms,
           gflops_gpu,
           max_abs_err, rmse, pct_good);

    return pass ? 0 : 1;
}

// ===== Main =====

int main() {
    printf("MatMulNBits Vulkan Unit Tests\n");
    printf("=============================\n\n");

    TestCase tests[] = {
        // Decode shapes (M=1) — most performance-critical
        {"decode_qkvo",     1,   4096,  4096, 128, 1, false, false},
        {"decode_up_gate",  1,  14336,  4096, 128, 1, false, false},
        {"decode_down",     1,   4096, 14336, 128, 1, false, false},
        {"decode_lm_head",  1, 128256,  4096, 128, 1, false, false},
        {"decode_gqa_kv",   1,   1024,  4096, 128, 1, false, false},

        // With zero_points
        {"decode_zp",       1,   4096,  4096, 128, 1, true,  false},
        {"decode_zp_bias",  1,   4096,  4096, 128, 1, true,  true},

        // With bias
        {"decode_bias",     1,   4096,  4096, 128, 1, false, true},

        // Prefill shapes (M>1)
        {"prefill_128",   128,   4096,  4096, 128, 1, false, false},
        {"prefill_32",     32,   4096,  4096, 128, 1, false, false},

        // Batched
        {"batched",         1,   4096,  4096, 128, 2, false, false},

        // Existing test shape (block_size=32)
        {"existing_bs32",   1,   5120,  2880,  32, 1, false, false},

        // Edge cases
        {"small",           1,     16,   128, 128, 1, false, false},
        {"tiny",            1,      1,   128, 128, 1, false, false},
        {"bs32_small",      1,    256,   256,  32, 1, true,  true},

        // Medium sizes for performance measurement
        {"medium",          1,   8192,  8192, 128, 1, false, false},
    };

    int total = sizeof(tests) / sizeof(tests[0]);
    int failures = 0;

    for (int i = 0; i < total; i++) {
        failures += run_test(tests[i]);
    }

    printf("\n=============================\n");
    printf("Results: %d/%d passed, %d failed\n", total - failures, total, failures);

    // ===== GPU-pointer mode tests =====
    printf("\nGPU-Pointer Mode (zero-copy via VK_EXT_external_memory_host)\n");
    printf("============================================================\n\n");

    // Test a subset of shapes in GPU-pointer mode
    // We allocate page-aligned host memory (simulating hipMalloc on UMA)
    // and call vulkan_matmul_nbits_gpu() which imports it zero-copy.
    size_t alignment = vulkan_get_host_pointer_alignment();
    if (alignment == 0) alignment = 4096;

    TestCase gpu_tests[] = {
        {"gpu_qkvo",       1,   4096,  4096, 128, 1, false, false},
        {"gpu_up_gate",    1,  14336,  4096, 128, 1, false, false},
        {"gpu_down",       1,   4096, 14336, 128, 1, false, false},
        {"gpu_lm_head",    1, 128256,  4096, 128, 1, false, false},
        {"gpu_medium",     1,   8192,  8192, 128, 1, false, false},
    };

    int gpu_total = sizeof(gpu_tests) / sizeof(gpu_tests[0]);
    int gpu_failures = 0;

    for (int i = 0; i < gpu_total; i++) {
        const TestCase &tc = gpu_tests[i];
        g_seed = 42;

        int64_t k_blocks = (tc.K + tc.block_size - 1) / tc.block_size;
        size_t a_bytes   = (size_t)(tc.batch_count * tc.M * tc.K * 2);
        size_t b_bytes   = (size_t)(tc.N * (tc.K / 2));
        size_t s_bytes   = (size_t)(tc.N * k_blocks * 2);
        size_t out_bytes = (size_t)(tc.batch_count * tc.M * tc.N * 2);
        size_t out_count = (size_t)(tc.batch_count * tc.M * tc.N);

        // Allocate page-aligned memory (simulating UMA hipMalloc)
        size_t a_alloc = (a_bytes + alignment - 1) & ~(alignment - 1);
        size_t b_alloc = (b_bytes + alignment - 1) & ~(alignment - 1);
        size_t s_alloc = (s_bytes + alignment - 1) & ~(alignment - 1);
        size_t o_alloc = (out_bytes + alignment - 1) & ~(alignment - 1);

#ifdef _WIN32
        void *a_gpu = _aligned_malloc(a_alloc + alignment, alignment);
        void *b_gpu = _aligned_malloc(b_alloc + alignment, alignment);
        void *s_gpu = _aligned_malloc(s_alloc + alignment, alignment);
        void *o_gpu = _aligned_malloc(o_alloc + alignment, alignment);
#else
        void *a_gpu, *b_gpu, *s_gpu, *o_gpu;
        posix_memalign(&a_gpu, alignment, a_alloc + alignment);
        posix_memalign(&b_gpu, alignment, b_alloc + alignment);
        posix_memalign(&s_gpu, alignment, s_alloc + alignment);
        posix_memalign(&o_gpu, alignment, o_alloc + alignment);
#endif

        // Fill with random data
        uint16_t *a_fp16 = (uint16_t *)a_gpu;
        uint8_t  *b_pack = (uint8_t *)b_gpu;
        uint16_t *scales = (uint16_t *)s_gpu;
        uint16_t *out_vk = (uint16_t *)o_gpu;

        for (size_t j = 0; j < a_bytes / 2; j++) a_fp16[j] = float_to_half(rng_float(-1.0f, 1.0f));
        for (size_t j = 0; j < b_bytes; j++) b_pack[j] = (uint8_t)(rng() & 0xFF);
        for (size_t j = 0; j < s_bytes / 2; j++) scales[j] = float_to_half(rng_float(0.001f, 0.1f));
        memset(o_gpu, 0, out_bytes);

        // CPU reference
        std::vector<float> out_ref(out_count, 0.0f);
        cpu_matmul_nbits_ref(a_fp16, b_pack, scales, nullptr, nullptr,
                             out_ref.data(), tc.M, tc.N, tc.K,
                             tc.batch_count, tc.block_size);

        // GPU-pointer dispatch (zero-copy)
        auto t0 = std::chrono::high_resolution_clock::now();
        int rc = vulkan_matmul_nbits_gpu(a_gpu, b_gpu, s_gpu, nullptr, nullptr, o_gpu,
                                          tc.M, tc.N, tc.K, tc.batch_count,
                                          4, tc.block_size, 2);
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gpu_ms = vulkan_get_last_kernel_time_ms();

        if (rc != 0) {
            printf("  FAIL [%-16s]: vulkan_matmul_nbits_gpu returned %d\n", tc.name, rc);
            gpu_failures++;
        } else {
            double max_err = 0.0, sum_sq = 0.0;
            for (size_t j = 0; j < out_count; j++) {
                double err = fabs((double)half_to_float(out_vk[j]) - (double)out_ref[j]);
                max_err = std::max(max_err, err);
                sum_sq += err * err;
            }
            double rmse = sqrt(sum_sq / (double)out_count);
            double flops = 2.0 * tc.M * tc.N * tc.K * tc.batch_count;
            double gflops = (gpu_ms > 0) ? (flops / (gpu_ms * 1e6)) : 0;
            bool pass = (rmse < 1.0);

            printf("  %s [%-16s]: N=%-6lld K=%-5lld | gpu=%.3fms e2e=%.1fms | "
                   "%.1f GFLOPS/s | max_err=%.4f rmse=%.6f\n",
                   pass ? "PASS" : "FAIL", tc.name,
                   (long long)tc.N, (long long)tc.K,
                   gpu_ms, e2e_ms, gflops, max_err, rmse);
            if (!pass) gpu_failures++;
        }

#ifdef _WIN32
        _aligned_free(a_gpu); _aligned_free(b_gpu);
        _aligned_free(s_gpu); _aligned_free(o_gpu);
#else
        free(a_gpu); free(b_gpu); free(s_gpu); free(o_gpu);
#endif
    }

    printf("\nGPU-pointer results: %d/%d passed, %d failed\n",
           gpu_total - gpu_failures, gpu_total, gpu_failures);

    failures += gpu_failures;

    printf("\n=============================\n");
    printf("TOTAL: %d/%d passed, %d failed\n",
           (total + gpu_total) - failures, total + gpu_total, failures);

    vulkan_context_destroy();

    return failures > 0 ? 1 : 0;
}
