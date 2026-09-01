/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ============================================================
// custom_kernels MatMulNBits Verification
//
// Tests the hip_matmul_nbits() API which performs:
//   C[M×N] = A[M×K] × dequant(B_packed[N×K/2])^T
//
// Public API — all tensors are ROW-MAJOR per ONNX convention:
//   A:        FP16 row-major [batch, M, K]
//   B_packed: uint4 packed [N, K/2], each byte = 2 values (low nibble first)
//   scales:   FP16 [N, num_groups_k], per-column per-group
//   zeros:    FP16 [N, num_groups_k], per-column per-group zero point
//             (optional, nullptr to skip)
//   C:        FP16 row-major [batch, M, N]
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
#include <cstdlib>
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

// ============================================================
// True-data mode: parse shape.json from a real-data folder
// ============================================================
struct TestFiles {
    std::string a     = "matmul_nbits_A.bin";
    std::string b     = "matmul_nbits_B_packed.bin";
    std::string s     = "matmul_nbits_scales.bin";
    std::string z     = "matmul_nbits_zeros.bin";
    std::string c_ref = "matmul_nbits_C_ref.bin";
};

struct ShapeConfig {
    bool   no_zeros   = true;
    int    batch_size = 1;
    int    M = 0, K = 0, N = 0;
    int    block_size = 128;
    TestFiles files;
};

static std::string slurpFile(const std::string& path)
{
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static int jsonInt(const std::string& s, const std::string& key)
{
    std::string needle = "\"" + key + "\"";
    auto pos = s.find(needle);
    if(pos == std::string::npos) return 0;
    pos = s.find(':', pos + needle.size());
    while(pos < s.size() && s[pos] != '-' && !std::isdigit(static_cast<unsigned char>(s[pos])))
        pos++;
    return std::stoi(s.substr(pos));
}

static bool jsonBool(const std::string& s, const std::string& key)
{
    std::string needle = "\"" + key + "\"";
    auto pos = s.find(needle);
    if(pos == std::string::npos) return false;
    pos = s.find(':', pos + needle.size());
    auto end = s.find_first_of(",}", pos);
    return s.substr(pos, end - pos).find("true") != std::string::npos;
}

static std::string jsonNestedStr(const std::string& s,
                                 const std::string& objKey,
                                 const std::string& field)
{
    auto opos = s.find("\"" + objKey + "\"");
    if(opos == std::string::npos) return "";
    auto brace = s.find('{', opos);
    if(brace == std::string::npos) return "";
    int depth = 1;
    auto bend = brace + 1;
    while(bend < s.size() && depth > 0)
    {
        if(s[bend] == '{') depth++;
        if(s[bend] == '}') depth--;
        bend++;
    }
    auto obj = s.substr(brace, bend - brace);
    auto fpos = obj.find("\"" + field + "\"");
    if(fpos == std::string::npos) return "";
    auto colon = obj.find(':', fpos);
    auto q1 = obj.find('"', colon);
    auto q2 = obj.find('"', q1 + 1);
    return obj.substr(q1 + 1, q2 - q1 - 1);
}

static std::vector<int> jsonIntArray(const std::string& s, const std::string& key)
{
    std::vector<int> result;
    std::string needle = "\"" + key + "\"";
    auto pos = s.find(needle);
    if(pos == std::string::npos) return result;
    auto bracket = s.find('[', pos);
    if(bracket == std::string::npos) return result;
    auto end_bracket = s.find(']', bracket);
    if(end_bracket == std::string::npos) return result;
    std::string arr = s.substr(bracket + 1, end_bracket - bracket - 1);
    size_t i = 0;
    while(i < arr.size())
    {
        while(i < arr.size() && !std::isdigit(static_cast<unsigned char>(arr[i])) && arr[i] != '-')
            i++;
        if(i >= arr.size()) break;
        result.push_back(std::stoi(arr.substr(i)));
        while(i < arr.size() && (std::isdigit(static_cast<unsigned char>(arr[i])) || arr[i] == '-'))
            i++;
    }
    return result;
}

static std::vector<int> jsonNestedIntArray(const std::string& s,
                                           const std::string& objKey,
                                           const std::string& arrKey)
{
    auto opos = s.find("\"" + objKey + "\"");
    if(opos == std::string::npos) return {};
    auto brace = s.find('{', opos);
    if(brace == std::string::npos) return {};
    int depth = 1;
    auto bend = brace + 1;
    while(bend < s.size() && depth > 0)
    {
        if(s[bend] == '{') depth++;
        if(s[bend] == '}') depth--;
        bend++;
    }
    return jsonIntArray(s.substr(brace, bend - brace), arrKey);
}

static ShapeConfig parseShapeJson(const std::string& path)
{
    auto json = slurpFile(path);
    if(json.empty())
    {
        std::cerr << "ERROR: cannot read " << path << std::endl;
        exit(1);
    }
    ShapeConfig cfg;
    cfg.no_zeros   = jsonBool(json, "no_zeros");
    cfg.batch_size = jsonInt(json, "batch_size");
    cfg.M          = jsonInt(json, "M");
    cfg.K          = jsonInt(json, "K");
    cfg.N          = jsonInt(json, "N");
    cfg.block_size = jsonInt(json, "block_size");

    cfg.files.a     = jsonNestedStr(json, "ifm_disc",    "file_name");
    cfg.files.b     = jsonNestedStr(json, "wts_disc",    "file_name");
    cfg.files.s     = jsonNestedStr(json, "scales_disc", "file_name");
    cfg.files.c_ref = jsonNestedStr(json, "output_disc", "file_name");
    cfg.files.z     = "";

    std::cout << "  Parsed shape.json: M=" << cfg.M << " K=" << cfg.K
              << " N=" << cfg.N << " block_size=" << cfg.block_size
              << " no_zeros=" << cfg.no_zeros
              << " batch=" << cfg.batch_size << std::endl;
    std::cout << "  Files: A=" << cfg.files.a
              << "  B=" << cfg.files.b
              << "  S=" << cfg.files.s
              << "  C_ref=" << cfg.files.c_ref << std::endl;
    return cfg;
}

// ============================================================
// Model sweep mode: parse model config and iterate all shapes
// ============================================================

struct ModelConfig {
    std::vector<int> M_array;
    std::vector<int> K_array;
    std::vector<int> N_array;
};

static ModelConfig parseModelConfig(const std::string& path)
{
    auto json = slurpFile(path);
    if(json.empty())
    {
        std::cerr << "ERROR: cannot read " << path << std::endl;
        exit(1);
    }
    ModelConfig cfg;
    cfg.M_array = jsonIntArray(json, "M_array");
    cfg.K_array = jsonNestedIntArray(json, "KN_pairs", "K");
    cfg.N_array = jsonNestedIntArray(json, "KN_pairs", "N");
    if(cfg.K_array.size() != cfg.N_array.size())
    {
        std::cerr << "ERROR: K and N arrays in KN_pairs must have the same length" << std::endl;
        exit(1);
    }
    std::cout << "  Model: " << cfg.M_array.size() << " M values x "
              << cfg.K_array.size() << " KN pairs = "
              << cfg.M_array.size() * cfg.K_array.size() << " shapes" << std::endl;
    std::cout << "  M_array: ";
    for(int m : cfg.M_array) std::cout << m << " ";
    std::cout << std::endl;
    for(size_t i = 0; i < cfg.K_array.size(); i++)
        std::cout << "  KN[" << i << "]: K=" << cfg.K_array[i]
                  << " N=" << cfg.N_array[i] << std::endl;
    return cfg;
}

// ============================================================

bool test_matmul_nbits(int M, int N, int K, int group_size,
                       const std::string& data_dir, bool use_zeros,
                       const TestFiles& tf = TestFiles{})
{
    int num_groups_k = (K + group_size - 1) / group_size;

    std::cout << "\n=== Test MatMulNBits M=" << M << " N=" << N << " K=" << K
              << " group_size=" << group_size
              << (use_zeros ? "" : " (no zeros)") << " ===" << std::endl;

    std::string fA = data_dir + "/" + tf.a;
    std::string fB = data_dir + "/" + tf.b;
    std::string fS = data_dir + "/" + tf.s;
    std::string fZ = data_dir + "/" + tf.z;
    std::string fC = data_dir + "/" + tf.c_ref;

    std::vector<__half>  h_A;
    std::vector<uint8_t> h_B_packed;
    std::vector<__half>  h_scales;
    std::vector<__half>  h_zeros;

    size_t countA = static_cast<size_t>(M) * K;
    // B_packed rows are padded to num_groups_k * (group_size/2) bytes (ONNX
    // MatMulNBits blob layout -- the last group is padded to a full
    // group_size even when K % group_size != 0), NOT a plain K/2 bytes/row.
    size_t countB = static_cast<size_t>(N) * num_groups_k * (group_size / 2);
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
        std::cout << "  WARNING: No reference file (" << fC << "), skipping verification." << std::endl;

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
            2,         // element_size_bytes (fp16)
            2,         // zp_elem_size (fp16 zero_points, used as-is)
            nullptr,   // pre_unpacked_zp_u8 (unused, zp_elem_size==2)
            nullptr);  // pre_unpacked_zp_fp16 (unused, zero_points is already fp16)
    };

    constexpr int PRE_WARMUP = 2000;
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
            1, 4, group_size, 2, 2, nullptr, nullptr);
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
    std::cout << "  Median: "
              << std::setprecision(6) << avg_ms << " ms, "
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

// ============================================================
// Model sweep: pre-warmup once, then benchmark + verify all shapes
// ============================================================

struct ShapeResult {
    int    M, K, N;
    double median_ms;
    double gflops;
    double bw_gbs;
    bool   pass;
    int    errors;
    int    checked;
};

static int runModelSweep(const std::string& json_path, int group_size,
                         bool use_zeros, const std::string& data_root)
{
    std::cout << "\nModel sweep: " << json_path
              << "  (group_size=" << group_size
              << (use_zeros ? ", with zeros)" : ", no zeros)")
              << "\n  Data root: " << data_root << std::endl;

    auto model = parseModelConfig(json_path);
    if(model.M_array.empty() || model.K_array.empty())
    {
        std::cerr << "ERROR: empty model config" << std::endl;
        return 1;
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int total_shapes = (int)(model.K_array.size() * model.M_array.size());
    std::vector<ShapeResult> results;
    results.reserve(total_shapes);
    int shape_idx = 0;
    bool warmup_done = false;

    for(size_t ki = 0; ki < model.K_array.size(); ki++)
    {
        int K = model.K_array[ki];
        int N = model.N_array[ki];

        for(size_t mi = 0; mi < model.M_array.size(); mi++)
        {
            int M = model.M_array[mi];
            shape_idx++;

            int num_groups_k = (K + group_size - 1) / group_size;
            size_t countA = (size_t)M * K;
            // See test_matmul_nbits() above: B_packed rows are padded to
            // num_groups_k * (group_size/2) bytes, not a plain K/2.
            size_t countB = (size_t)N * num_groups_k * (group_size / 2);
            size_t countS = (size_t)N * num_groups_k;
            size_t countZ = (size_t)N * num_groups_k;
            size_t countC = (size_t)M * N;

            std::cout << "\n=== [" << shape_idx << "/" << total_shapes
                      << "] M=" << M << " K=" << K << " N=" << N
                      << " gs=" << group_size
                      << (use_zeros ? "" : " (no zeros)") << " ===" << std::endl;

            std::string shape_dir = data_root + "/"
                + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);

            std::vector<__half>  h_A;
            std::vector<uint8_t> h_B;
            std::vector<__half>  h_S, h_Z;

            bool data_ok = readBin(shape_dir + "/matmul_nbits_A.bin", h_A, countA)
                        && readBin(shape_dir + "/matmul_nbits_B_packed.bin", h_B, countB)
                        && readBin(shape_dir + "/matmul_nbits_scales.bin", h_S, countS);
            if(use_zeros)
                data_ok = data_ok && readBin(shape_dir + "/matmul_nbits_zeros.bin", h_Z, countZ);

            if(!data_ok)
            {
                std::cerr << "  ERROR: cannot read data from " << shape_dir << "/" << std::endl;
                std::cerr << "  Run: make gendata_model  to generate all data first" << std::endl;
                results.push_back({M, K, N, 0, 0, 0, false, -1, 0});
                continue;
            }
            std::cout << "  Loaded data from " << shape_dir << "/" << std::endl;

            std::vector<__half> h_C_ref;
            bool has_ref = readBin(shape_dir + "/matmul_nbits_C_ref.bin", h_C_ref, countC);
            if(!has_ref)
                std::cout << "  WARNING: no reference file, skipping verification" << std::endl;

            __half *dA, *dS, *dZ = nullptr, *dC;
            uint8_t *dB;
            HIP_CHECK(hipMalloc(&dA, countA * sizeof(__half)));
            HIP_CHECK(hipMalloc(&dB, countB));
            HIP_CHECK(hipMalloc(&dS, countS * sizeof(__half)));
            if(use_zeros) HIP_CHECK(hipMalloc(&dZ, countZ * sizeof(__half)));
            HIP_CHECK(hipMalloc(&dC, countC * sizeof(__half)));

            HIP_CHECK(hipMemcpy(dA, h_A.data(), countA * sizeof(__half), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dB, h_B.data(), countB, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dS, h_S.data(), countS * sizeof(__half), hipMemcpyHostToDevice));
            if(use_zeros)
                HIP_CHECK(hipMemcpy(dZ, h_Z.data(), countZ * sizeof(__half), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(dC, 0, countC * sizeof(__half)));

            auto launch = [&]() {
                hip_matmul_nbits(stream, dA, dB, dS,
                                 use_zeros ? dZ : nullptr, nullptr, dC,
                                 M, N, K, 1, 4, group_size, 2, 2, nullptr, nullptr);
            };

            // Pre-warmup (once, on first successfully loaded shape)
            if(!warmup_done)
            {
                constexpr int PRE_WARMUP = 200;
                std::cout << "  Pre-warmup (" << PRE_WARMUP << " iters)..." << std::flush;
                auto t0 = std::chrono::steady_clock::now();
                for(int w = 0; w < PRE_WARMUP; w++)
                    launch();
                HIP_CHECK(hipStreamSynchronize(stream));
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count() / 1000.0;
                std::cout << " done (" << std::fixed << std::setprecision(0)
                          << ms << " ms, "
                          << std::setprecision(3) << ms / PRE_WARMUP << " ms/iter)" << std::endl;
                HIP_CHECK(hipMemset(dC, 0, countC * sizeof(__half)));
                warmup_done = true;
            }

            // Calibration warmup (per shape)
            std::cout << "  Warmup..." << std::flush;
            auto tw0 = std::chrono::steady_clock::now();
            int status = 0;
            for(int w = 0; w < 3; w++)
            {
                status = hip_matmul_nbits(stream, dA, dB, dS,
                                          use_zeros ? dZ : nullptr, nullptr, dC,
                                          M, N, K, 1, 4, group_size, 2, 2, nullptr, nullptr);
                if(status != 0) break;
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            double warmup_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tw0).count() / 1000.0;

            if(status != 0)
            {
                std::cout << " FAILED (status=" << status << ")" << std::endl;
                results.push_back({M, K, N, 0, 0, 0, false, -1, 0});
                hipFree(dA); hipFree(dB); hipFree(dS);
                if(dZ) hipFree(dZ); hipFree(dC);
                continue;
            }

            int niters = calibrateIters(warmup_ms, 3);
            std::cout << " OK (" << std::fixed << std::setprecision(2)
                      << warmup_ms << " ms), iters=" << niters << std::endl;

            std::cout << "  Benchmarking (" << NROUNDS << " rounds x "
                      << niters << " iters)..." << std::flush;
            auto mr = measureMedian(stream, niters, launch);

            double avg_ms    = mr.median_ms / niters;
            double gflops    = (2.0 * M * N * K) / (avg_ms * 1e6);
            double mem_bytes = (double)countA * 2 + (double)countB
                             + (double)countS * 2
                             + (use_zeros ? (double)countZ * 2 : 0.0)
                             + (double)countC * 2;
            double bw_gbs    = mem_bytes * niters / (mr.median_ms * 1e6);

            std::cout << " done" << std::endl;
            std::cout << "\n  === Performance ===" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Median: " << avg_ms << " ms, "
                      << gflops << " GFLOPS, " << bw_gbs << " GB/s" << std::endl;

            // Download and verify against Python reference
            std::vector<__half> h_C(countC);
            HIP_CHECK(hipMemcpy(h_C.data(), dC, countC * sizeof(__half), hipMemcpyDeviceToHost));

            bool pass = true;
            int errors = 0;
            int total = M * N;

            if(has_ref)
            {
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
                    if(diff > tol) errors++;
                }

                pass = (errors == 0);
                std::cout << "\n  === GPU vs Python Reference ===" << std::endl;
                std::cout << "  Verified " << total << " elements, " << errors << " errors" << std::endl;
                std::cout << "  Max abs diff: " << std::setprecision(4) << max_diff
                          << ", max rel diff: " << (max_rdiff * 100.0f) << "%" << std::endl;
                std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
            }
            else
            {
                std::cout << "  Verify: SKIPPED (no reference)" << std::endl;
            }

            results.push_back({M, K, N, avg_ms, gflops, bw_gbs,
                               pass, errors, total});

            hipFree(dA); hipFree(dB); hipFree(dS);
            if(dZ) hipFree(dZ); hipFree(dC);
        }
    }

    hipStreamDestroy(stream);

    // ---- Summary table ----
    std::cout << "\n=========================================================================="
              << std::endl;
    std::cout << "Model Sweep Summary: " << json_path << std::endl;
    std::cout << "=========================================================================="
              << std::endl;
    std::cout << std::right
              << std::setw(4)  << "#"    << " | "
              << std::setw(5)  << "M"    << " | "
              << std::setw(5)  << "K"    << " | "
              << std::setw(7)  << "N"    << " | "
              << std::setw(11) << "Median(ms)"  << " | "
              << std::setw(9)  << "GFLOPS"  << " | "
              << std::setw(8)  << "GB/s"    << " | "
              << "Status" << std::endl;
    std::cout << "-----+-------+-------+---------+-------------+-----------+----------+--------"
              << std::endl;

    int pass_count = 0, fail_count = 0;
    for(size_t i = 0; i < results.size(); i++)
    {
        auto& r = results[i];
        if(r.pass) pass_count++; else fail_count++;
        std::cout << std::right << std::setw(4) << (i + 1) << " | "
                  << std::setw(5) << r.M << " | "
                  << std::setw(5) << r.K << " | "
                  << std::setw(7) << r.N << " | "
                  << std::setw(11) << std::fixed << std::setprecision(6) << r.median_ms << " | "
                  << std::setw(9)  << std::setprecision(2) << r.gflops << " | "
                  << std::setw(8)  << std::setprecision(2) << r.bw_gbs << " | "
                  << (r.pass ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << "=========================================================================="
              << std::endl;
    std::cout << "Total: " << results.size() << " shapes, "
              << pass_count << " passed, " << fail_count << " failed" << std::endl;
    std::cout << "=========================================================================="
              << std::endl;

    return (fail_count == 0) ? 0 : 1;
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

    // --- Check for --true-data mode ---
    std::string true_data_folder;
    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--true-data" && i + 1 < argc)
        {
            true_data_folder = argv[++i];
            break;
        }
    }

    if(!true_data_folder.empty())
    {
        std::string json_path = true_data_folder + "/shape.json";
        std::cout << "True-data mode: " << json_path << std::endl;

        auto cfg = parseShapeJson(json_path);
        bool use_zeros = !cfg.no_zeros;

        bool all_pass = test_matmul_nbits(
            cfg.M, cfg.N, cfg.K, cfg.block_size,
            true_data_folder, use_zeros, cfg.files);

        std::cout << "\n==========================================================================" << std::endl;
        std::cout << "Overall: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << std::endl;
        return all_pass ? 0 : 1;
    }

    // --- Check for --model mode ---
    std::string model_json;
    std::string model_data_root = "data_model";
    int model_gs = 128;
    bool model_use_zeros = true;
    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--model" && i + 1 < argc)
            model_json = argv[++i];
        else if(std::string(argv[i]) == "--data-root" && i + 1 < argc)
            model_data_root = argv[++i];
        else if(std::string(argv[i]) == "--group-size" && i + 1 < argc)
            model_gs = std::atoi(argv[++i]);
        else if(std::string(argv[i]) == "--no-zeros")
            model_use_zeros = false;
    }

    if(!model_json.empty())
        return runModelSweep(model_json, model_gs, model_use_zeros, model_data_root);

    // --- Random-data mode (original) ---
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
            std::cerr << "       " << argv[0]
                      << " --true-data <folder>" << std::endl;
            std::cerr << "       " << argv[0]
                      << " --model <json> [--data-root DIR] [--group-size GS] [--no-zeros]" << std::endl;
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
