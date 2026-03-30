// ============================================================
// custom_kernels MatMulNBits (WMMA Fused GEMM + Dequantization) Verification
//
// Tests the hip_matmul_nbits() API which performs:
//   C[M×N] = A[M×K] × dequant(B_packed[N×K/2])^T
//
// When the WMMA fast path is active (M%128==0, N%128==0, K%32==0):
//   A:        FP16 col-major (M × K), stride lda = M
//   B_packed: uint4 packed (N × K/2), each byte = 2 values (low nibble first)
//   scales:   FP16 (N × num_groups_k), per-column per-group
//   zeros:    FP16 (N × num_groups_k), per-column per-group zero point
//             (optional, nullptr to skip)
//   C:        FP16 col-major (M × N), stride ldc = M
//
// Workflow:
//   1) python gen_matmul_nbits_data.py MxKxN --group-size GS --dir data
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
#include <thread>

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
                          double target_ms = 2000.0, int lo = 5, int hi = 200)
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

    std::cout << "\n  [debug] per-round raw (ms): ";
    for(int r = 0; r < NROUNDS; r++)
        std::cout << std::fixed << std::setprecision(2) << round_ms[r] << " ";
    std::cout << "  (per-iter: ";
    for(int r = 0; r < NROUNDS; r++)
        std::cout << std::fixed << std::setprecision(3) << round_ms[r] / niters << " ";
    std::cout << ")" << std::endl;

    std::sort(round_ms.begin(), round_ms.end());
    return {round_ms[NROUNDS / 2], round_ms[0], round_ms[NROUNDS - 1]};
}

bool test_matmul_nbits(int M, int N, int K, int group_size,
                       const std::string& data_dir, bool use_zeros)
{
    int num_groups_k = (K + group_size - 1) / group_size;

    std::cout << "\n=== Test MatMulNBits (WMMA) M=" << M << " N=" << N << " K=" << K
              << " group_size=" << group_size
              << (use_zeros ? "" : " (no zeros)") << " ===" << std::endl;

    std::string fA = data_dir + "/matmul_nbits_A.bin";
    std::string fB = data_dir + "/matmul_nbits_B_packed.bin";
    std::string fS = data_dir + "/matmul_nbits_scales.bin";
    std::string fZ = data_dir + "/matmul_nbits_zeros.bin";
    std::string fC = data_dir + "/matmul_nbits_C_ref.bin";

    std::vector<__half>  h_A;
    std::vector<uint8_t> h_B_packed;
    std::vector<__half>  h_scales;
    std::vector<__half>  h_zeros;

    size_t countA = static_cast<size_t>(M) * K;
    size_t countB = static_cast<size_t>(N) * K / 2;
    size_t countS = static_cast<size_t>(N) * num_groups_k;
    size_t countZ = static_cast<size_t>(N) * num_groups_k;
    size_t countC = static_cast<size_t>(M) * N;

    bool data_ok = readBin(fA, h_A, countA) &&
                   readBin(fB, h_B_packed, countB) &&
                   readBin(fS, h_scales, countS);
    if(use_zeros)
        data_ok = data_ok && readBin(fZ, h_zeros, countZ);

    if(!data_ok)
    {
        std::cerr << "  ERROR: Failed to read input data files from " << data_dir << "/" << std::endl;
        std::cerr << "  Run: python gen_matmul_nbits_data.py "
                  << M << "x" << K << "x" << N
                  << " --group-size " << group_size
                  << (use_zeros ? "" : " --no-zeros")
                  << " --dir " << data_dir << std::endl;
        return false;
    }
    std::cout << "  Loaded input data from " << data_dir << "/" << std::endl;

    std::vector<__half> h_C_ref;
    bool has_ref = readBin(fC, h_C_ref, countC);
    if(!has_ref)
        std::cout << "  WARNING: No Python reference file (" << fC << "), skipping verification." << std::endl;

    __half*  d_A        = nullptr;
    uint8_t* d_B_packed = nullptr;
    __half*  d_scales   = nullptr;
    __half*  d_zeros    = nullptr;
    __half*  d_C        = nullptr;

    size_t size_A      = countA * sizeof(__half);
    size_t size_B      = countB;
    size_t size_scales = countS * sizeof(__half);
    size_t size_zeros  = countZ * sizeof(__half);
    size_t size_C      = countC * sizeof(__half);

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B_packed, size_B));
    HIP_CHECK(hipMalloc(&d_scales, size_scales));
    if(use_zeros)
    {
        HIP_CHECK(hipMalloc(&d_zeros, size_zeros));
    }
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_packed, h_B_packed.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, h_scales.data(), size_scales, hipMemcpyHostToDevice));
    if(use_zeros)
    {
        HIP_CHECK(hipMemcpy(d_zeros, h_zeros.data(), size_zeros, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemset(d_C, 0, size_C));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto launch_kernel = [&]() {
        hip_matmul_nbits(
            stream,
            d_A,
            d_B_packed,
            d_scales,
            use_zeros ? d_zeros : nullptr,
            nullptr,   // no bias
            d_C,
            M, N, K,
            1,         // batch_count
            4,         // bits
            group_size,// block_size
            2);        // element_size_bytes (fp16)
    };

    constexpr int PRE_WARMUP = 5000;
    std::cout << "  Pre-warmup (" << PRE_WARMUP << " iters)..." << std::flush;
    auto pw0 = std::chrono::steady_clock::now();
    for(int w = 0; w < PRE_WARMUP; w++)
        launch_kernel();
    HIP_CHECK(hipStreamSynchronize(stream));
    double pw_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - pw0).count() / 1000.0;
    std::cout << " done (" << std::fixed << std::setprecision(0) << pw_ms << " ms, "
              << std::setprecision(3) << pw_ms / PRE_WARMUP << " ms/iter)" << std::endl;

    std::cout << "  Warmup..." << std::flush;
    auto tw0 = std::chrono::steady_clock::now();
    int status = 0;
    for(int w = 0; w < 3; w++)
    {
        status = hip_matmul_nbits(
            stream,
            d_A,
            d_B_packed,
            d_scales,
            use_zeros ? d_zeros : nullptr,
            nullptr,
            d_C,
            M, N, K,
            1, 4, group_size, 2);
        if(status != 0) break;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    auto tw1 = std::chrono::steady_clock::now();
    double warmup_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(tw1 - tw0).count() / 1000.0;

    if(status != 0)
    {
        std::cout << " FAILED (status=" << status << ")" << std::endl;
        hipFree(d_A);
        hipFree(d_B_packed);
        hipFree(d_scales);
        if(d_zeros) hipFree(d_zeros);
        hipFree(d_C);
        hipStreamDestroy(stream);
        return false;
    }

    int niters = calibrateIters(warmup_ms, 3);
    std::cout << " OK (" << std::fixed << std::setprecision(2) << warmup_ms
              << " ms), iters=" << niters << std::endl;

    std::cout << "  Benchmarking (" << NROUNDS << " rounds x " << niters << " iters)..."
              << std::flush;
    auto mr = measureMedian(stream, niters, launch_kernel);

    double avg_ms    = mr.median_ms / niters;
    double gflops    = (2.0 * M * N * K) / (avg_ms * 1e6);
    double mem_bytes = static_cast<double>(countA) * 2 + static_cast<double>(countB)
                     + static_cast<double>(countS) * 2
                     + (use_zeros ? static_cast<double>(countZ) * 2 : 0.0)
                     + static_cast<double>(countC) * 2;
    double bw_gbs    = mem_bytes * niters / (mr.median_ms * 1e6);
    double range_pct = (mr.max_ms - mr.min_ms) / mr.median_ms * 100.0;

    std::cout << " done" << std::endl;
    std::cout << "\n  === Performance ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Median: " << avg_ms << " ms, "
              << gflops << " GFLOPS, "
              << bw_gbs << " GB/s" << std::endl;
    std::cout << "  Range:  " << mr.min_ms / niters << " ~ " << mr.max_ms / niters
              << " ms  (jitter " << std::setprecision(1) << range_pct << "%)" << std::endl;

    std::vector<__half> h_C(countC);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    bool pass = true;

    if(has_ref)
    {
        int errors      = 0;
        int total        = M * N;
        float max_diff  = 0.0f;
        float max_rdiff = 0.0f;

        for(int i = 0; i < total; i++)
        {
            float gpu_val = half_to_float(h_C[i]);
            float ref_val = half_to_float(h_C_ref[i]);
            float diff    = std::fabs(gpu_val - ref_val);
            float rdiff   = (std::fabs(ref_val) > 1e-6f) ? diff / std::fabs(ref_val) : diff;

            if(diff > max_diff) max_diff = diff;
            if(rdiff > max_rdiff) max_rdiff = rdiff;

            float tol = std::fabs(ref_val) * 0.05f + 0.1f;
            if(diff > tol)
                errors++;
        }

        std::cout << "\n  === GPU vs Python Reference ===" << std::endl;
        std::cout << "  Verified " << total << " elements, " << errors << " errors" << std::endl;
        std::cout << "  Max abs diff: " << max_diff << ", max rel diff: "
                  << std::fixed << std::setprecision(4) << (max_rdiff * 100.0f) << "%" << std::endl;

        std::cout << "  Sample C values (GPU vs Python):" << std::endl;
        for(int i = 0; i < 5 && i < total; i++)
        {
            float gpu_val = half_to_float(h_C[i]);
            float ref_val = half_to_float(h_C_ref[i]);
            std::cout << "    [" << i << "] GPU=" << std::setprecision(6) << gpu_val
                      << "  Ref=" << ref_val
                      << "  diff=" << std::fabs(gpu_val - ref_val) << std::endl;
        }

        pass = (errors == 0);
        std::cout << "  Result: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }
    else
    {
        std::cout << "  GPU output sample:" << std::endl;
        int total = M * N;
        for(int i = 0; i < 5 && i < total; i++)
            std::cout << "    [" << i << "] = " << half_to_float(h_C[i]) << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B_packed);
    hipFree(d_scales);
    if(d_zeros) hipFree(d_zeros);
    hipFree(d_C);
    hipStreamDestroy(stream);

    return pass;
}

int main(int argc, char* argv[])
{
    std::cout << "custom_kernels MatMulNBits (WMMA Fused GEMM + Dequantization) Verification" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (arch: " << prop.gcnArchName << ")" << std::endl;

    std::string arch(prop.gcnArchName);
    if(arch.find("gfx11") == std::string::npos && arch.find("gfx12") == std::string::npos)
    {
        std::cerr << "WARNING: WMMA fast path requires RDNA3+ (gfx11xx/gfx12xx). "
                  << "Current arch: " << arch << std::endl;
    }

    int M = 128, N = 128, K = 128, gs = 128;
    std::string data_dir = "data";
    bool use_zeros = true;

    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--no-zeros")
            use_zeros = false;
    }

    if(argc >= 2 && std::string(argv[1]) != "--no-zeros")
    {
        if(sscanf(argv[1], "%dx%dx%d", &M, &K, &N) != 3)
        {
            std::cerr << "Usage: " << argv[0]
                      << " [MxKxN] [group_size] [data_dir] [--no-zeros]" << std::endl;
            return 1;
        }
    }
    if(argc >= 3 && std::string(argv[2]) != "--no-zeros") gs = atoi(argv[2]);
    if(argc >= 4 && std::string(argv[3]) != "--no-zeros") data_dir = argv[3];

    std::cout << "Data dir: " << data_dir << std::endl;
    if(!use_zeros)
        std::cout << "Zero points: disabled (--no-zeros)" << std::endl;

    bool all_pass = true;
    all_pass &= test_matmul_nbits(M, N, K, gs, data_dir, use_zeros);

    std::cout << "\n==========================================================================" << std::endl;
    std::cout << "Overall: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << std::endl;

    return all_pass ? 0 : 1;
}
