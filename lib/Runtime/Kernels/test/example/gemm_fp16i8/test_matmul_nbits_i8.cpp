/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ============================================================
// custom_kernels MatMulNBits bits=8 (uint8, unpacked) Verification + Benchmark
//
// Exercises the hip_matmul_nbits() API for the bits=8 quantized-weight
// kernel: B is one byte per weight (NOT bit-packed, unlike bits=2/3/4).
//
//   C[M×N] = A[M×K] × dequant(B)^T
//
// Public API tensors -- all ROW-MAJOR:
//   A       : FP16 or FP32 [batch, M, K]  (element_size_bytes selects dtype)
//   B       : uint8 [N, K]                 one byte per weight, no packing
//   scales  : FP16  [N, num_groups_k]
//   zeros   : uint8 [N, num_groups_k]      (optional, one byte per group;
//                                            default zp=128 when null)
//   C       : FP16 or FP32 [batch, M, N]   (same dtype as A)
//
// Three internal dispatch paths (selected automatically by shape):
//   - WMMA  (prefill): batch=1, K%32==0, M>=16, block_size>=32 & %32==0
//   - GEMV  (decode) : not WMMA-eligible, block_size>=16
//   - naive          : fallback (e.g. block_size<16)
//
// The fp32 (element_size_bytes==4) instantiation rounds A through fp16 at
// the load boundary (no separate cast kernel/scratch buffer), accumulates
// in fp32, and stores raw fp32 -- see testFp32Shape() below for how that
// maps onto the on-disk fp16 reference.
//
// Workflow:
//   1) python gen_matmul_nbits_i8_data.py MxKxN --group-size GS --dir data
//   2) build and run the test executable
// ============================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_custom_kernels.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cstdlib>

static float half_to_float(__half h)
{
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t f;
    if(exp == 0)
    {
        if(mant == 0)
            f = sign << 31;
        else
        {
            exp = 1;
            while(!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    }
    else if(exp == 31)
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    else
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

#define HIP_CHECK(call)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t err = (call);                                            \
        if(err != hipSuccess)                                               \
        {                                                                   \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \""                           \
                      << hipGetErrorString(err) << "\"" << std::endl;       \
            exit(1);                                                        \
        }                                                                   \
    } while(0)

template <typename T>
static bool readBin(const std::string& path, std::vector<T>& data, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if(!f.is_open())
        return false;
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return f.good();
}

// ============================================================
// Benchmark helpers
// ============================================================
static int calibrateIters(double warmup_ms, int warmup_count,
                          double target_ms = 1500.0, int lo = 5, int hi = 200)
{
    double per_iter = warmup_ms / warmup_count;
    if(per_iter <= 0) return lo;
    return std::max(lo, std::min(hi, static_cast<int>(target_ms / per_iter)));
}

constexpr int NROUNDS = 5;

struct MeasureResult {
    double median_ms;
    double min_ms;
    double max_ms;
};

static MeasureResult measureMedian(hipStream_t stream, int niters,
                                   const std::function<void()>& launch)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    std::vector<float> round_ms(NROUNDS);
    for(int r = 0; r < NROUNDS; r++)
    {
        hipEventRecord(ev0, stream);
        for(int i = 0; i < niters; i++)
            launch();
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&round_ms[r], ev0, ev1);
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    std::sort(round_ms.begin(), round_ms.end());
    return {round_ms[NROUNDS / 2], round_ms[0], round_ms[NROUNDS - 1]};
}

// ============================================================
// Per-kernel benchmark + verify
// ============================================================
struct KernelStat {
    std::string label;
    bool launch_ok  = false;
    double avg_ms   = 0.0;
    double gflops   = 0.0;
    double bw_gbs   = 0.0;
    double min_ms   = 0.0;
    double max_ms   = 0.0;
    bool   has_ref  = false;
    int    errors   = -1;
    int    total    = 0;
    float  max_diff = 0.0f;
    float  max_rdiff = 0.0f;
};

static int verifyAgainstRef(const std::vector<__half>& h_C,
                            const std::vector<__half>& h_C_ref, size_t countC,
                            float& max_diff_out, float& max_rdiff_out)
{
    int errors = 0;
    float max_diff = 0.0f, max_rdiff = 0.0f;
    for(size_t i = 0; i < countC; i++)
    {
        float gpu_val = half_to_float(h_C[i]);
        float ref_val = half_to_float(h_C_ref[i]);
        float diff    = std::fabs(gpu_val - ref_val);
        float rdiff   = (std::fabs(ref_val) > 1e-6f) ? diff / std::fabs(ref_val) : diff;
        if(diff > max_diff) max_diff = diff;
        if(rdiff > max_rdiff) max_rdiff = rdiff;
        float tol = std::fabs(ref_val) * 0.05f + 0.1f;
        if(diff > tol) errors++;
    }
    max_diff_out = max_diff;
    max_rdiff_out = max_rdiff;
    return errors;
}

static KernelStat benchmarkAndVerify(
    const std::string& label,
    hipStream_t stream,
    const std::function<int()>& launch_checked,
    const std::function<void()>& launch,
    int M, int N, int K,
    double mem_bytes,
    __half* d_C, size_t countC,
    const std::vector<__half>& h_C_ref, bool has_ref)
{
    KernelStat st;
    st.label   = label;
    st.has_ref = has_ref;

    std::cout << "\n  --- " << label << " ---" << std::endl;

    constexpr int PRE_WARMUP = 500;
    std::cout << "  Pre-warmup (" << PRE_WARMUP << " iters)..." << std::flush;
    auto pw0 = std::chrono::steady_clock::now();
    for(int w = 0; w < PRE_WARMUP; w++)
        launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    double pw_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - pw0).count() / 1000.0;
    std::cout << " done (" << std::fixed << std::setprecision(3)
              << pw_ms / PRE_WARMUP << " ms/iter)" << std::endl;

    std::cout << "  Warmup..." << std::flush;
    auto tw0 = std::chrono::steady_clock::now();
    int status = 0;
    for(int w = 0; w < 3; w++)
    {
        status = launch_checked();
        if(status != 0) break;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double warmup_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tw0).count() / 1000.0;

    if(status != 0)
    {
        std::cout << " FAILED (status=" << status << ")" << std::endl;
        return st;
    }

    int niters = calibrateIters(warmup_ms, 3);
    std::cout << " OK (" << std::fixed << std::setprecision(2) << warmup_ms
              << " ms), iters=" << niters << std::endl;

    std::cout << "  Benchmarking (" << NROUNDS << " rounds x " << niters
              << " iters)..." << std::flush;
    auto mr = measureMedian(stream, niters, launch);
    std::cout << " done" << std::endl;

    st.launch_ok = true;
    st.avg_ms    = mr.median_ms / niters;
    st.min_ms    = mr.min_ms / niters;
    st.max_ms    = mr.max_ms / niters;
    st.gflops    = (2.0 * M * N * K) / (st.avg_ms * 1e6);
    st.bw_gbs    = mem_bytes * niters / (mr.median_ms * 1e6);

    std::cout << "  Median: " << std::setprecision(6) << st.avg_ms << " ms, "
              << std::setprecision(2) << st.gflops << " GFLOPS, "
              << st.bw_gbs << " GB/s   (range " << st.min_ms << " ~ "
              << st.max_ms << " ms)" << std::endl;

    if(has_ref)
    {
        std::vector<__half> h_C(countC);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, countC * sizeof(__half),
                            hipMemcpyDeviceToHost));
        float max_diff = 0.0f, max_rdiff = 0.0f;
        int errors = verifyAgainstRef(h_C, h_C_ref, countC, max_diff, max_rdiff);

        st.errors    = errors;
        st.total     = static_cast<int>(countC);
        st.max_diff  = max_diff;
        st.max_rdiff = max_rdiff;

        std::cout << "  Verify: " << (st.total - errors) << "/" << st.total
                  << " OK, max_abs_diff=" << std::setprecision(4) << max_diff
                  << ", max_rel_diff=" << (max_rdiff * 100.0f) << "%   "
                  << (errors == 0 ? "PASS" : "FAIL") << std::endl;
    }

    return st;
}

// ============================================================
// FP32 A / FP32 output benchmark + verify
//
// Mirrors benchmarkAndVerify above, but d_C is a float* buffer
// (element_size_bytes==4) instead of __half*. The reference stays the
// on-disk fp16 C_ref -- upcast per-element with half_to_float() before
// comparing.
// ============================================================
static KernelStat benchmarkAndVerifyFp32(
    const std::string& label,
    hipStream_t stream,
    const std::function<int()>& launch_checked,
    const std::function<void()>& launch,
    int M, int N, int K,
    double mem_bytes,
    float* d_C, size_t countC,
    const std::vector<__half>& h_C_ref, bool has_ref)
{
    KernelStat st;
    st.label   = label;
    st.has_ref = has_ref;

    std::cout << "\n  --- " << label << " ---" << std::endl;

    constexpr int PRE_WARMUP = 500;
    std::cout << "  Pre-warmup (" << PRE_WARMUP << " iters)..." << std::flush;
    auto pw0 = std::chrono::steady_clock::now();
    for(int w = 0; w < PRE_WARMUP; w++)
        launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    double pw_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - pw0).count() / 1000.0;
    std::cout << " done (" << std::fixed << std::setprecision(3)
              << pw_ms / PRE_WARMUP << " ms/iter)" << std::endl;

    std::cout << "  Warmup..." << std::flush;
    auto tw0 = std::chrono::steady_clock::now();
    int status = 0;
    for(int w = 0; w < 3; w++)
    {
        status = launch_checked();
        if(status != 0) break;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double warmup_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tw0).count() / 1000.0;

    if(status != 0)
    {
        std::cout << " FAILED (status=" << status << ")" << std::endl;
        return st;
    }

    int niters = calibrateIters(warmup_ms, 3);
    std::cout << " OK (" << std::fixed << std::setprecision(2) << warmup_ms
              << " ms), iters=" << niters << std::endl;

    std::cout << "  Benchmarking (" << NROUNDS << " rounds x " << niters
              << " iters)..." << std::flush;
    auto mr = measureMedian(stream, niters, launch);
    std::cout << " done" << std::endl;

    st.launch_ok = true;
    st.avg_ms    = mr.median_ms / niters;
    st.min_ms    = mr.min_ms / niters;
    st.max_ms    = mr.max_ms / niters;
    st.gflops    = (2.0 * M * N * K) / (st.avg_ms * 1e6);
    st.bw_gbs    = mem_bytes * niters / (mr.median_ms * 1e6);

    std::cout << "  Median: " << std::setprecision(6) << st.avg_ms << " ms, "
              << std::setprecision(2) << st.gflops << " GFLOPS, "
              << st.bw_gbs << " GB/s   (range " << st.min_ms << " ~ "
              << st.max_ms << " ms)" << std::endl;

    if(has_ref)
    {
        std::vector<float> h_C(countC);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, countC * sizeof(float),
                            hipMemcpyDeviceToHost));
        int errors = 0;
        float max_diff = 0.0f, max_rdiff = 0.0f;
        for(size_t i = 0; i < countC; i++)
        {
            float gpu_val = h_C[i];
            float ref_val = half_to_float(h_C_ref[i]);
            float diff    = std::fabs(gpu_val - ref_val);
            float rdiff   = (std::fabs(ref_val) > 1e-6f) ? diff / std::fabs(ref_val) : diff;
            if(diff > max_diff) max_diff = diff;
            if(rdiff > max_rdiff) max_rdiff = rdiff;
            float tol = std::fabs(ref_val) * 0.05f + 0.1f;
            if(diff > tol) errors++;
        }

        st.errors    = errors;
        st.total     = static_cast<int>(countC);
        st.max_diff  = max_diff;
        st.max_rdiff = max_rdiff;

        std::cout << "  Verify: " << (st.total - errors) << "/" << st.total
                  << " OK, max_abs_diff=" << std::setprecision(4) << max_diff
                  << ", max_rel_diff=" << (max_rdiff * 100.0f) << "%   "
                  << (errors == 0 ? "PASS" : "FAIL") << std::endl;
    }

    return st;
}

// ============================================================
// FP16 A / FP16 output single-shape test (bits=8)
// ============================================================
bool testShape(int M, int N, int K, int group_size,
              const std::string& data_dir, bool use_zeros)
{
    int num_groups_k = (K + group_size - 1) / group_size;

    std::cout << "\n=== MatMulNBits i8 (fp16)  M=" << M << " N=" << N
              << " K=" << K << " group_size=" << group_size
              << (use_zeros ? "" : " (no zeros)") << " ===" << std::endl;

    size_t countA = static_cast<size_t>(M) * K;
    size_t countC = static_cast<size_t>(M) * N;
    size_t countB = static_cast<size_t>(N) * K;
    size_t countS = static_cast<size_t>(N) * num_groups_k;

    std::vector<__half>  h_A;
    std::vector<uint8_t> h_B;
    std::vector<__half>  h_S;
    std::vector<uint8_t> h_Z;
    std::vector<__half>  h_Cref;

    bool ok = readBin(data_dir + "/matmul_nbits_i8_A.bin", h_A, countA)
           && readBin(data_dir + "/matmul_nbits_i8_B.bin", h_B, countB)
           && readBin(data_dir + "/matmul_nbits_i8_scales.bin", h_S, countS);
    if(use_zeros)
        ok = ok && readBin(data_dir + "/matmul_nbits_i8_zeros_u8.bin", h_Z, countS);
    bool has_ref = readBin(data_dir + "/matmul_nbits_i8_C_ref.bin", h_Cref, countC);

    if(!ok)
    {
        std::cerr << "  ERROR: failed to read i8 input data from " << data_dir << "/" << std::endl;
        std::cerr << "  Run: python gen_matmul_nbits_i8_data.py " << M << "x" << K << "x" << N
                  << " --group-size " << group_size
                  << (use_zeros ? "" : " --no-zeros") << " --dir " << data_dir << std::endl;
        return false;
    }
    std::cout << "  Loaded A + i8 weights from " << data_dir << "/" << std::endl;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    __half*  d_A = nullptr;
    __half*  d_C = nullptr;
    uint8_t* d_B = nullptr;
    __half*  d_S = nullptr;
    uint8_t* d_Z = nullptr;

    HIP_CHECK(hipMalloc(&d_A, countA * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C, countC * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B, countB));
    HIP_CHECK(hipMalloc(&d_S, countS * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), countA * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, countC * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), countB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_S, h_S.data(), countS * sizeof(__half), hipMemcpyHostToDevice));
    if(use_zeros)
    {
        HIP_CHECK(hipMalloc(&d_Z, countS));
        HIP_CHECK(hipMemcpy(d_Z, h_Z.data(), countS, hipMemcpyHostToDevice));
    }

    auto launch_checked = [&]() -> int {
        return hip_matmul_nbits(
            stream, d_A, d_B, d_S,
            use_zeros ? d_Z : nullptr,
            nullptr,           // bias
            d_C,
            M, N, K,
            1,                 // batch_count
            8,                 // bits
            group_size,        // block_size
            2,                 // element_size_bytes (fp16)
            1,                 // zp_elem_size (uint8, one byte per group)
            nullptr,           // pre_unpacked_zp_u8 (direct convention)
            nullptr);          // pre_unpacked_zp_fp16 (unused for bits=8)
    };
    auto launch = [&]() { launch_checked(); };

    double mem_bytes = static_cast<double>(countA) * 2 + static_cast<double>(countB)
                      + static_cast<double>(countS) * 2
                      + (use_zeros ? static_cast<double>(countS) : 0.0)
                      + static_cast<double>(countC) * 2;

    KernelStat st = benchmarkAndVerify(
        "i8", stream, launch_checked, launch, M, N, K,
        mem_bytes, d_C, countC, h_Cref, has_ref);

    bool pass = st.launch_ok && (!st.has_ref || st.errors == 0);

    hipFree(d_A);
    hipFree(d_C);
    hipFree(d_B);
    hipFree(d_S);
    if(d_Z) hipFree(d_Z);
    hipStreamDestroy(stream);

    return pass;
}

// ============================================================
// FP32 A / FP32 output single-shape test (bits=8)
//
// Reuses the same on-disk fp16 A/B/scales/zeros as the fp16 test above --
// A is upcast host-side (exact fp16->fp32), so the kernel's internal
// fp32->fp16 downcast reproduces that exact fp16 A, making the existing
// fp16 i8 C_ref the correct ground truth here too, just compared through
// float containers (mirrors gemm_fp16u2's testFp32Shape()).
// ============================================================
bool testFp32Shape(int M, int N, int K, int group_size,
                   const std::string& data_dir, bool use_zeros)
{
    int num_groups_k = (K + group_size - 1) / group_size;

    std::cout << "\n=== MatMulNBits i8 (fp32)  M=" << M << " N=" << N
              << " K=" << K << " group_size=" << group_size
              << (use_zeros ? "" : " (no zeros)") << " ===" << std::endl;

    size_t countA = static_cast<size_t>(M) * K;
    size_t countC = static_cast<size_t>(M) * N;
    size_t countB = static_cast<size_t>(N) * K;
    size_t countS = static_cast<size_t>(N) * num_groups_k;

    std::vector<__half> h_A_fp16;
    if(!readBin(data_dir + "/matmul_nbits_i8_A.bin", h_A_fp16, countA))
    {
        std::cerr << "  ERROR: cannot read " << data_dir << "/matmul_nbits_i8_A.bin" << std::endl;
        return false;
    }
    std::vector<float> h_A(countA);
    for(size_t i = 0; i < countA; i++)
        h_A[i] = half_to_float(h_A_fp16[i]);

    std::vector<uint8_t> h_B;
    std::vector<__half>  h_S;
    std::vector<uint8_t> h_Z;
    std::vector<__half>  h_Cref;
    bool ok = readBin(data_dir + "/matmul_nbits_i8_B.bin", h_B, countB)
           && readBin(data_dir + "/matmul_nbits_i8_scales.bin", h_S, countS);
    if(use_zeros)
        ok = ok && readBin(data_dir + "/matmul_nbits_i8_zeros_u8.bin", h_Z, countS);
    bool has_ref = readBin(data_dir + "/matmul_nbits_i8_C_ref.bin", h_Cref, countC);

    if(!ok)
    {
        std::cerr << "  ERROR: failed to read i8 input data from " << data_dir << "/" << std::endl;
        return false;
    }
    std::cout << "  Loaded A (fp32-upcast) + i8 weights from " << data_dir << "/" << std::endl;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    float*   d_A = nullptr;
    float*   d_C = nullptr;
    uint8_t* d_B = nullptr;
    __half*  d_S = nullptr;
    uint8_t* d_Z = nullptr;

    HIP_CHECK(hipMalloc(&d_A, countA * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, countC * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, countB));
    HIP_CHECK(hipMalloc(&d_S, countS * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), countA * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, countC * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), countB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_S, h_S.data(), countS * sizeof(__half), hipMemcpyHostToDevice));
    if(use_zeros)
    {
        HIP_CHECK(hipMalloc(&d_Z, countS));
        HIP_CHECK(hipMemcpy(d_Z, h_Z.data(), countS, hipMemcpyHostToDevice));
    }

    auto launch_checked = [&]() -> int {
        return hip_matmul_nbits(
            stream, d_A, d_B, d_S,
            use_zeros ? d_Z : nullptr,
            nullptr,           // bias
            d_C,
            M, N, K,
            1,                 // batch_count
            8,                 // bits
            group_size,        // block_size
            4,                 // element_size_bytes (fp32)
            1,                 // zp_elem_size (uint8, one byte per group)
            nullptr,           // pre_unpacked_zp_u8 (direct convention)
            nullptr);          // pre_unpacked_zp_fp16 (unused for bits=8)
    };
    auto launch = [&]() { launch_checked(); };

    double mem_bytes = static_cast<double>(countA) * 4 + static_cast<double>(countB)
                      + static_cast<double>(countS) * 2
                      + (use_zeros ? static_cast<double>(countS) : 0.0)
                      + static_cast<double>(countC) * 4;

    KernelStat st = benchmarkAndVerifyFp32(
        "i8(fp32)", stream, launch_checked, launch, M, N, K,
        mem_bytes, d_C, countC, h_Cref, has_ref);

    bool pass = st.launch_ok && (!st.has_ref || st.errors == 0);

    hipFree(d_A);
    hipFree(d_C);
    hipFree(d_B);
    hipFree(d_S);
    if(d_Z) hipFree(d_Z);
    hipStreamDestroy(stream);

    return pass;
}

// ============================================================

int main(int argc, char* argv[])
{
    std::cout << "custom_kernels MatMulNBits bits=8 (uint8, unpacked) Verification" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (arch: " << prop.gcnArchName << ")" << std::endl;

    int M = 128, N = 128, K = 128, gs = 128;
    std::string data_dir = "data";
    bool use_zeros = true;
    bool fp32_mode = false;

    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--no-zeros")
            use_zeros = false;
        else if(std::string(argv[i]) == "--fp32")
            fp32_mode = true;
    }

    if(argc >= 2 && std::string(argv[1]) != "--no-zeros" && std::string(argv[1]) != "--fp32")
    {
        if(sscanf(argv[1], "%dx%dx%d", &M, &K, &N) != 3)
        {
            std::cerr << "Usage: " << argv[0]
                      << " [MxKxN] [group_size] [data_dir] [--no-zeros] [--fp32]" << std::endl;
            return 1;
        }
    }
    if(argc >= 3 && std::string(argv[2]) != "--no-zeros" && std::string(argv[2]) != "--fp32") gs = atoi(argv[2]);
    if(argc >= 4 && std::string(argv[3]) != "--no-zeros" && std::string(argv[3]) != "--fp32") data_dir = argv[3];

    std::cout << "Data dir: " << data_dir << std::endl;
    if(!use_zeros)
        std::cout << "Zero points: disabled (--no-zeros, default zp=128)" << std::endl;
    if(fp32_mode)
        std::cout << "Mode: FP32 A / FP32 output (element_size_bytes=4, bits=8)" << std::endl;

    bool all_pass = fp32_mode
        ? testFp32Shape(M, N, K, gs, data_dir, use_zeros)
        : testShape(M, N, K, gs, data_dir, use_zeros);

    std::cout << "\n==========================================================================" << std::endl;
    std::cout << "Overall: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << std::endl;

    return all_pass ? 0 : 1;
}
