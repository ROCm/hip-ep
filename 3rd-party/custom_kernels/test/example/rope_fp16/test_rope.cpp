// ============================================================
// custom_kernels RoPE (Rotary Positional Embedding) Verification
//
// Tests hip_rope_forward() and hip_rope_forward_bak() (old) against
// NumPy reference and compares their performance side by side.
//
// Modes:
//   Single: test_rope.exe --dir data [--bench-iters 100]
//   Multi:  test_rope.exe --multi data_0 data_1 ... [--bench-iters 100]
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
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs) { std::cerr << "Cannot open " << path << std::endl; return false; }
    data.resize(count);
    ifs.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    if((size_t)ifs.gcount() != count * sizeof(T))
    {
        std::cerr << "Short read: " << path << " expected " << count * sizeof(T)
                  << " got " << ifs.gcount() << std::endl;
        return false;
    }
    return true;
}

struct MetaInfo {
    int batch = 0, seq_len = 0, num_heads = 0, head_dim = 0;
    int rotary_dim = 0, max_seq_len = 0;
    int interleaved = 0;
};

static MetaInfo readMeta(const std::string& dir)
{
    MetaInfo m;
    std::ifstream ifs(dir + "/rope_meta.txt");
    if(!ifs) {
        std::cerr << "Cannot open " << dir << "/rope_meta.txt" << std::endl;
        exit(1);
    }
    std::string line;
    while(std::getline(ifs, line)) {
        auto eq = line.find('=');
        if(eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        int val = std::stoi(line.substr(eq + 1));
        if(key == "batch") m.batch = val;
        else if(key == "seq_len") m.seq_len = val;
        else if(key == "num_heads") m.num_heads = val;
        else if(key == "head_dim") m.head_dim = val;
        else if(key == "rotary_dim") m.rotary_dim = val;
        else if(key == "max_seq_len") m.max_seq_len = val;
        else if(key == "interleaved") m.interleaved = val;
    }
    return m;
}

struct VerifyResult {
    size_t errors = 0;
    float max_abs_diff = 0.0f;
};

static VerifyResult doVerify(const __half* gpu_out, const __half* ref,
                             size_t count, float atol = 5e-2f, float rtol = 1e-2f)
{
    VerifyResult r;
    for(size_t i = 0; i < count; i++)
    {
        float g = half_to_float(gpu_out[i]);
        float rv = half_to_float(ref[i]);
        float abs_d = std::fabs(g - rv);
        float rel_d = (std::fabs(rv) > 1e-6f) ? (abs_d / std::fabs(rv)) : 0.0f;
        if(abs_d > atol && rel_d > rtol) r.errors++;
        if(abs_d > r.max_abs_diff) r.max_abs_diff = abs_d;
    }
    return r;
}

typedef int (*rope_fn)(void*, const void*, const void*, const void*,
                       const void*, void*,
                       int64_t, int64_t, int64_t, int64_t,
                       int64_t, int64_t, int64_t, int64_t);

struct KernelPerf {
    bool pass;
    float max_abs_diff;
    double latency_us;
    double throughput_gbs;
};

static KernelPerf benchOneKernel(
    rope_fn fn,
    __half* d_input, int64_t* d_pos, __half* d_cos, __half* d_sin, __half* d_output,
    const __half* h_ref,
    int B, int S, int H, int D, int RD, int MS, int interleaved,
    size_t input_count, int bench_iters)
{
    KernelPerf kp;

    fn(nullptr, d_input, d_pos, d_cos, d_sin, d_output,
       B, S, H, D, RD, MS, interleaved, 2);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<__half> h_output(input_count);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, input_count * sizeof(__half), hipMemcpyDeviceToHost));

    VerifyResult vr = doVerify(h_output.data(), h_ref, input_count);
    kp.pass = (vr.errors == 0);
    kp.max_abs_diff = vr.max_abs_diff;

    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < bench_iters; i++) {
        fn(nullptr, d_input, d_pos, d_cos, d_sin, d_output,
           B, S, H, D, RD, MS, interleaved, 2);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    kp.latency_us = us / bench_iters;
    kp.throughput_gbs = input_count * 2.0 * 2.0 / (kp.latency_us * 1e-6) / 1e9;

    return kp;
}

struct TestResult {
    int B, S, H, D, RD;
    std::string mode;
    size_t elements;
    KernelPerf opt;
    KernelPerf old;
};

static TestResult runOneTest(const std::string& data_dir, int bench_iters)
{
    MetaInfo meta = readMeta(data_dir);
    const int B  = meta.batch;
    const int S  = meta.seq_len;
    const int H  = meta.num_heads;
    const int D  = meta.head_dim;
    const int RD = meta.rotary_dim;
    const int MS = meta.max_seq_len;
    const int interleaved = meta.interleaved;

    TestResult res;
    res.B = B; res.S = S; res.H = H; res.D = D; res.RD = RD;
    res.mode = interleaved ? "interleaved" : "half-rot";

    const size_t input_count = (size_t)B * S * H * D;
    const size_t pos_count   = (size_t)B * S;
    const size_t cache_count = (size_t)MS * (RD / 2);
    res.elements = input_count;

    std::vector<__half> h_input, h_cos, h_sin, h_ref;
    std::vector<int64_t> h_pos;

    if(!readBin(data_dir + "/rope_input.bin", h_input, input_count) ||
       !readBin(data_dir + "/rope_position_ids.bin", h_pos, pos_count) ||
       !readBin(data_dir + "/rope_cos_cache.bin", h_cos, cache_count) ||
       !readBin(data_dir + "/rope_sin_cache.bin", h_sin, cache_count) ||
       !readBin(data_dir + "/rope_output_ref.bin", h_ref, input_count)) {
        res.opt = {false, -1, 0, 0};
        res.old = {false, -1, 0, 0};
        return res;
    }

    __half *d_input, *d_cos, *d_sin, *d_output;
    int64_t *d_pos;

    HIP_CHECK(hipMalloc(&d_input,  input_count * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_pos,    pos_count * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_cos,    cache_count * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_sin,    cache_count * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_output, input_count * sizeof(__half)));

    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_count * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pos,   h_pos.data(),   pos_count * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cos,   h_cos.data(),   cache_count * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sin,   h_sin.data(),   cache_count * sizeof(__half), hipMemcpyHostToDevice));

    res.opt = benchOneKernel(
        hip_rope_forward,
        d_input, d_pos, d_cos, d_sin, d_output,
        h_ref.data(), B, S, H, D, RD, MS, interleaved,
        input_count, bench_iters);

    res.old = benchOneKernel(
        hip_rope_forward_bak,
        d_input, d_pos, d_cos, d_sin, d_output,
        h_ref.data(), B, S, H, D, RD, MS, interleaved,
        input_count, bench_iters);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_pos));
    HIP_CHECK(hipFree(d_cos));
    HIP_CHECK(hipFree(d_sin));
    HIP_CHECK(hipFree(d_output));

    return res;
}

static void printSummaryTable(const std::vector<TestResult>& results)
{
    const int W = 130;
    std::cout << "\n" << std::string(W, '=') << "\n"
              << "                              RoPE Kernel Performance Comparison (opt vs old)\n"
              << std::string(W, '=') << "\n";

    std::cout << std::left
              << std::setw(4)  << "#"
              << std::setw(3)  << "B"
              << std::setw(6)  << "S"
              << std::setw(4)  << "H"
              << std::setw(5)  << "D"
              << std::setw(4)  << "RD"
              << std::setw(13) << "Mode"
              << std::setw(11) << "Elements"
              << std::right
              << std::setw(10) << "opt(us)"
              << std::setw(10) << "old(us)"
              << std::setw(10) << "opt BW"
              << std::setw(10) << "old BW"
              << std::setw(9)  << "Speedup"
              << std::setw(7)  << "opt"
              << std::setw(7)  << "old"
              << "\n";

    std::cout << std::string(W, '-') << "\n";

    int pass_opt = 0, pass_old = 0;

    for(size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        double speedup = (r.opt.latency_us > 0 && r.old.latency_us > 0)
                         ? r.old.latency_us / r.opt.latency_us : 0.0;

        std::cout << std::left
                  << std::setw(4)  << (i + 1)
                  << std::setw(3)  << r.B
                  << std::setw(6)  << r.S
                  << std::setw(4)  << r.H
                  << std::setw(5)  << r.D
                  << std::setw(4)  << r.RD
                  << std::setw(13) << r.mode
                  << std::setw(11) << r.elements
                  << std::right << std::fixed
                  << std::setw(10) << std::setprecision(2) << r.opt.latency_us
                  << std::setw(10) << std::setprecision(2) << r.old.latency_us
                  << std::setw(9)  << std::setprecision(1) << r.opt.throughput_gbs << "G"
                  << std::setw(9)  << std::setprecision(1) << r.old.throughput_gbs << "G"
                  << std::setw(8)  << std::setprecision(2) << speedup << "x"
                  << std::setw(7)  << (r.opt.pass ? "PASS" : "FAIL")
                  << std::setw(7)  << (r.old.pass ? "PASS" : "FAIL")
                  << "\n";

        if(r.opt.pass) pass_opt++;
        if(r.old.pass) pass_old++;
    }

    std::cout << std::string(W, '-') << "\n"
              << "Total: " << results.size() << " shapes"
              << "  |  opt: " << pass_opt << " PASS"
              << "  |  old: " << pass_old << " PASS\n"
              << std::string(W, '=') << "\n";
}

int main(int argc, char** argv)
{
    std::string data_dir = "data";
    int bench_iters = 100;
    bool multi_mode = false;
    std::vector<std::string> multi_dirs;

    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if(arg == "--bench-iters" && i + 1 < argc) {
            bench_iters = std::stoi(argv[++i]);
        } else if(arg == "--multi") {
            multi_mode = true;
            for(int j = i + 1; j < argc; j++) {
                std::string a = argv[j];
                if(a.substr(0, 2) == "--") break;
                multi_dirs.push_back(a);
                i = j;
            }
        }
    }

    if(multi_mode && !multi_dirs.empty()) {
        std::vector<TestResult> results;
        for(size_t i = 0; i < multi_dirs.size(); i++) {
            std::cout << "[" << (i + 1) << "/" << multi_dirs.size() << "] "
                      << multi_dirs[i] << "..." << std::flush;
            TestResult r = runOneTest(multi_dirs[i], bench_iters);
            double speedup = (r.opt.latency_us > 0 && r.old.latency_us > 0)
                             ? r.old.latency_us / r.opt.latency_us : 0.0;
            results.push_back(r);
            std::cout << " opt=" << std::fixed << std::setprecision(2) << r.opt.latency_us
                      << "us  old=" << r.old.latency_us << "us  "
                      << std::setprecision(2) << speedup << "x\n";
        }
        printSummaryTable(results);
        for(const auto& r : results) {
            if(!r.opt.pass) return 1;
        }
        return 0;
    }

    // Single-test mode
    std::cout << "=== RoPE Test (single) ===\n";
    TestResult r = runOneTest(data_dir, bench_iters);
    double speedup = (r.opt.latency_us > 0 && r.old.latency_us > 0)
                     ? r.old.latency_us / r.opt.latency_us : 0.0;

    std::cout << "  B=" << r.B << " S=" << r.S << " H=" << r.H
              << " D=" << r.D << " RD=" << r.RD << " mode=" << r.mode << "\n"
              << "  Elements: " << r.elements << "\n\n"
              << "  [opt] latency: " << std::fixed << std::setprecision(2) << r.opt.latency_us
              << " us  BW: " << r.opt.throughput_gbs << " GB/s  " << (r.opt.pass ? "PASS" : "FAIL") << "\n"
              << "  [old] latency: " << r.old.latency_us
              << " us  BW: " << r.old.throughput_gbs << " GB/s  " << (r.old.pass ? "PASS" : "FAIL") << "\n"
              << "  Speedup: " << std::setprecision(2) << speedup << "x\n";

    return r.opt.pass ? 0 : 1;
}
