/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Offline sweep driver for the MatMulNBits LUT.
 *
 * Walks a shape list and touches each (path, shape, M) once, which is enough to
 * make the in-kernel autotuner run and log its winner. The winners are read off
 * stderr by scripts/update_lut.py. The default mode retains this autotune-log
 * path. Windows --requests mode instead pins candidate configs for fast LUT
 * ranking or pairs two libraries' production paths for regression screening.
 *
 * Run with HIPDNN_EP_DEBUG=1 so the in-kernel autotuner logs its winner. Each
 * run starts with an empty in-process tune map (the tuners no longer persist to
 * an on-disk cache), so every shape is measured fresh.
 *
 * Shape list is CSV with a header, columns K,N,block_size,has_zp (extra columns
 * ignored) -- i.e. what extract_shapes.py emits.
 *
 * Build it as "Regenerating" in ../README.md describes: for the release target
 * at the release -O2, so the sweep times the code custom_kernels_<arch>.dll runs
 * (gfx11-generic or gfx1170; the table is still per chip, update_lut.py --arch),
 * and linked with tools/empty_lut_data.cpp, so the shapes it measures do not hit
 * the shipped table.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

extern "C" int hip_matmul_nbits(
    void* stream, const void* A, const void* B, const void* scales,
    const void* zero_points, const void* bias, void* output,
    int64_t M, int64_t N, int64_t K, int64_t batch, int64_t bits,
    int64_t block_size, int64_t element_size_bytes, int64_t zp_elem_size,
    const void* pre_unpacked_zp_u8, const void* pre_unpacked_zp_fp16);

extern "C" size_t hip_matmul_nbits_u4_padrow_bytes(int N, int K);
extern "C" int hip_matmul_nbits_u4_prepack_padrow(
    void* stream, const void* B, void* dst, int N, int K);
extern "C" int hip_matmul_nbits_u4_wmma_stride(
    void* stream, int cfg, int M, int N, int K, int group_size,
    const void* A, int lda, const void* B, const void* scales,
    const void* zeros, void* C, int ldc, int b_row_bytes);
extern "C" void hip_matmul_nbits_u4_wmma_sweep_extended(int enable);
extern "C" int hip_matmul_nbits_dp4a(
    void* stream_v, const void* A, const void* B, const void* scales,
    const void* zp_u8, const void* bias, void* out,
    int64_t M, int64_t N, int64_t K, int64_t block_size,
    void* a_qb_scratch, void* a_scale_scratch);

#define HIP_OK(c)                                                             \
  do {                                                                        \
    hipError_t e_ = (c);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP %s at line %d\n", hipGetErrorString(e_),      \
                   __LINE__);                                                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static uint16_t F2H(float f) {
  _Float16 h = (_Float16)f;
  uint16_t o;
  std::memcpy(&o, &h, 2);
  return o;
}

// Mirrors shouldPadRow() in lib/Runtime/real/matmul_nbits.cpp. Only shapes the
// production gate would actually route through the padded layout are swept for
// it; measuring the rest would fill the table with rows nothing can reach.
static bool shouldPadRow(long long K) {
  const long long stride = K / 2;
  if (stride % 128 != 0) return false;
  long long a = stride / 128, b = 64;
  while (b) { long long t = b; b = a % b; a = t; }
  return a >= 8;
}

struct Shape {
  int k = 0, n = 0, gs = 0;
  bool zp = false;
  bool operator<(const Shape& o) const {
    return std::tie(k, n, gs, zp) < std::tie(o.k, o.n, o.gs, o.zp);
  }
};

static std::vector<Shape> loadShapes(const char* path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  std::string line;
  if (!std::getline(in, line)) return {};
  std::vector<std::string> head;
  {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) head.push_back(cell);
  }
  auto col = [&](const char* name) {
    for (size_t i = 0; i < head.size(); ++i)
      if (head[i] == name) return static_cast<int>(i);
    std::fprintf(stderr, "shape csv has no '%s' column\n", name);
    std::exit(1);
  };
  const int ck = col("K"), cn = col("N"), cb = col("block_size"),
            cz = col("has_zp");

  std::set<Shape> uniq;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    if (static_cast<int>(cells.size()) <= std::max(std::max(ck, cn),
                                                   std::max(cb, cz)))
      continue;
    Shape s;
    s.k = std::atoi(cells[ck].c_str());
    s.n = std::atoi(cells[cn].c_str());
    s.gs = std::atoi(cells[cb].c_str());
    s.zp = std::atoi(cells[cz].c_str()) != 0;
    // Drop only genuinely untunable shapes. K%32!=0 is kept: its decode GEMV
    // is still tunable (that kernel has a K%32 remainder phase, verified for
    // K=4304), it just has no WMMA prefill and no dp4a path. The sweep loop
    // measures decode only for those (see the k32 handling below).
    if (s.k <= 0 || s.n <= 0 || s.gs <= 0) continue;
    uniq.insert(s);
  }
  return {uniq.begin(), uniq.end()};
}

// Device buffers sized for the largest shape, reused across the sweep so the
// sweep measures tuning and not hipMalloc.
struct Buffers {
  uint16_t *a = nullptr, *sc = nullptr, *zpf = nullptr, *out = nullptr;
  uint8_t *b = nullptr, *zpu = nullptr, *bpad = nullptr;
  void *qb = nullptr, *qs = nullptr;
  size_t a_n = 0, sc_n = 0, out_n = 0, b_n = 0, bpad_n = 0, qb_n = 0, qs_n = 0;

  void grow(void** p, size_t* have, size_t need) {
    if (*have >= need) return;
    if (*p) HIP_OK(hipFree(*p));
    HIP_OK(hipMalloc(p, need));
    *have = need;
  }
};

// Pinned ranking and paired production screening share the ordinary LUT driver.
// Ranking values select configs only; formal acceptance uses the pinned UT.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

extern "C" { bool hipdnnKernelUtLaunchAudit = false; }

struct LoadedMatmulNbits {
  HMODULE library = nullptr;
  decltype(&hip_matmul_nbits) production = nullptr;
  decltype(&hip_matmul_nbits_u4_wmma_stride) configured = nullptr;
  using Padded = int (*)(void*, int, int, int, int, int, const void*, int,
                        const void*, const void*, const void*, void*, int);
  Padded padded = nullptr;
  bool* launch_audit = nullptr;
  using Flush = int (*)(void*);
  Flush flush = nullptr;
  explicit LoadedMatmulNbits(const char* path) {
    library = LoadLibraryA(path);
    if (!library) throw std::runtime_error("cannot load measurement library");
    production = reinterpret_cast<decltype(production)>(GetProcAddress(library, "hip_matmul_nbits"));
    configured = reinterpret_cast<decltype(configured)>(GetProcAddress(library, "hip_matmul_nbits_u4_wmma_stride"));
    padded = reinterpret_cast<Padded>(GetProcAddress(library, "hip_matmul_nbits_u4_wmma_padrow"));
    launch_audit = reinterpret_cast<bool*>(GetProcAddress(library, "hipdnnKernelUtLaunchAudit"));
    flush = reinterpret_cast<Flush>(GetProcAddress(library, "fflush"));
    if (!production || !configured || !padded || !launch_audit)
      throw std::runtime_error("measurement library lacks production exports");
  }
};

// Mirror the UT's cold working set, so retained accurate receipts remain
// compatible with fast ranking and short-M winners do not fit cache residency.
struct ExactColdWeights {
  unsigned char* allocation = nullptr;
  size_t b_offset = 0, scale_offset = 0, zero_offset = 0, stride = 0;
  int count = 1;
  size_t weight_bytes = 0;
  ExactColdWeights(const void* b, size_t bb, const void* scales, size_t sb,
                   const void* zeros, size_t zb) {
    weight_bytes = bb + sb + zb;
    count = std::min(4096, std::max(1, static_cast<int>(((size_t(256) << 20) + weight_bytes - 1) / weight_bytes)));
    scale_offset = (bb + 15) & ~size_t(15);
    zero_offset = (scale_offset + sb + 15) & ~size_t(15);
    stride = (zero_offset + zb + 255) & ~size_t(255);
    HIP_OK(hipMalloc(&allocation, stride * count));
    for (int i = 0; i < count; ++i) {
      HIP_OK(hipMemcpyAsync(allocation + i * stride, b, bb, hipMemcpyDeviceToDevice));
      HIP_OK(hipMemcpyAsync(allocation + i * stride + scale_offset, scales, sb, hipMemcpyDeviceToDevice));
      if (zb) HIP_OK(hipMemcpyAsync(allocation + i * stride + zero_offset, zeros, zb, hipMemcpyDeviceToDevice));
    }
    HIP_OK(hipDeviceSynchronize());
  }
  ~ExactColdWeights() { if (allocation) hipFree(allocation); }
  const void* part(int turn, size_t offset) const {
    return allocation + (turn % count) * stride + offset;
  }
};

struct ExactSweepRequest {
  std::string id;
  int m, n, k, gs;
  bool zeros, padded;
  std::vector<int> configs;
};

static std::vector<ExactSweepRequest> exactSweepRequests(const char* path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open exact requests");
  std::string line;
  std::getline(in, line);
  if (line != "id,m,n,k,group_size,zero_point,row_stride,configs")
    throw std::runtime_error("invalid exact request header");
  std::vector<ExactSweepRequest> requests;
  std::set<std::string> ids;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.back() == '\r') line.pop_back();
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) fields.push_back(item);
    if (fields.size() != 8) throw std::runtime_error("invalid exact request");
    ExactSweepRequest r{fields[0], std::stoi(fields[1]), std::stoi(fields[2]),
        std::stoi(fields[3]), std::stoi(fields[4]), fields[5] == "Asymmetric",
        fields[6] == "Padded", {}};
    if (!ids.insert(r.id).second || r.m < 8 || r.n < 1 || r.k < 1 || r.gs < 16
        || (r.gs & (r.gs - 1)) || (r.k % 32 && r.gs < 32)
        || (fields[5] != "Symmetric" && fields[5] != "Asymmetric")
        || (fields[6] != "Arrival" && fields[6] != "Padded")
        || (r.padded && (r.k % 32 || r.k % r.gs)))
      throw std::runtime_error("unsupported exact WMMA key");
    std::stringstream cs(fields[7]);
    while (std::getline(cs, item, ';')) {
      const int id = std::stoi(item);
      if (id < 0 || id >= 72) throw std::runtime_error("invalid config id");
      r.configs.push_back(id);
    }
    if (r.configs.empty() || std::set<int>(r.configs.begin(), r.configs.end()).size() != r.configs.size())
      throw std::runtime_error("empty or duplicate considered configs");
    requests.push_back(std::move(r));
  }
  return requests;
}

// A bounded median policy, without per-config thermal qualification. Three
// samples amortize event overhead; nonpositive/nonfinite samples are retried.
static double fastPinnedMedian(const std::function<void()>& launch,
                               std::vector<double>& samples) {
  hipEvent_t start, stop;
  HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&stop));
  launch(); launch(); HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipEventRecord(start)); launch(); HIP_OK(hipEventRecord(stop));
  HIP_OK(hipEventSynchronize(stop));
  float pilot = 0;
  HIP_OK(hipEventElapsedTime(&pilot, start, stop));
  const int count = std::isfinite(pilot) && pilot > 0
      ? std::min(200, std::max(1, static_cast<int>(std::ceil(1.0 / pilot)))) : 20;
  for (int attempt = 0; attempt < 6 && samples.size() < 3; ++attempt) {
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < count; ++i) launch();
    HIP_OK(hipEventRecord(stop)); HIP_OK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, start, stop));
    if (std::isfinite(ms) && ms > 0) samples.push_back(ms / count);
  }
  HIP_OK(hipEventDestroy(start)); HIP_OK(hipEventDestroy(stop));
  if (samples.size() != 3) throw std::runtime_error("fast measurement incomplete");
  auto ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  return ordered[1];
}

static float H2F(uint16_t value) {
  _Float16 h;
  std::memcpy(&h, &value, 2);
  return static_cast<float>(h);
}

static int runExactSweep(int argc, char** argv) {
  const char *requests_path = nullptr, *candidate_path = nullptr, *baseline_path = nullptr;
  std::string mode = "rank";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--requests" && i + 1 < argc) requests_path = argv[++i];
    else if (arg == "--candidate-library" && i + 1 < argc) candidate_path = argv[++i];
    else if (arg == "--baseline-library" && i + 1 < argc) baseline_path = argv[++i];
    else if (arg == "--measurement-purpose" && i + 1 < argc) mode = argv[++i];
    else throw std::runtime_error("unknown exact sweep argument");
  }
  if (!requests_path || !candidate_path || (mode != "rank" && mode != "screen")
      || (mode == "screen" && !baseline_path))
    throw std::runtime_error("exact sweep requires requests, libraries and rank/screen purpose");
  LoadedMatmulNbits candidate(candidate_path);
  std::unique_ptr<LoadedMatmulNbits> baseline;
  if (baseline_path) baseline.reset(new LoadedMatmulNbits(baseline_path));
  if (mode == "screen" && (!candidate.flush || !baseline->flush))
    throw std::runtime_error("screen libraries must export their CRT audit flush");
  hipDeviceProp_t props;
  HIP_OK(hipGetDeviceProperties(&props, 0));
  if (std::string(props.gcnArchName).substr(0, 7) != "gfx1151")
    throw std::runtime_error("exact measurement requires the declared gfx1151 device");
  const auto requests = exactSweepRequests(requests_path);
  std::printf("FAST_IDENTITY arch=%s purpose=%s requests=%zu statistic=median3 warmup=2 target_block_ms=1 cache=rotated256MiB_max4096\n", props.gcnArchName, mode.c_str(), requests.size());
  Buffers buffers;
  size_t out_bytes = 0, zeros_bytes = 0;
  bool settled = false;
  auto last_loaded = std::chrono::steady_clock::now();
  const auto begin = std::chrono::steady_clock::now();
  for (size_t index = 0; index < requests.size(); ++index) {
    const auto& r = requests[index];
    const int kp = (r.k + 31) & ~31;
    // Config ranking supplies the prepared A layout. Production screening
    // supplies the real arrival stride and lets the wrapper pad a K tail.
    const int lda = mode == "rank" ? kp : r.k;
    const int groups = (r.k + r.gs - 1) / r.gs;
    const int row_bytes = groups * (r.gs / 2);
    std::vector<uint16_t> a(static_cast<size_t>(r.m) * lda, 0);
    std::vector<uint8_t> b(static_cast<size_t>(r.n) * row_bytes, 0);
    std::vector<uint16_t> scales(static_cast<size_t>(r.n) * groups);
    std::vector<uint16_t> zeros(scales.size());
    for (int m = 0; m < r.m; ++m)
      for (int k = 0; k < r.k; ++k)
        a[static_cast<size_t>(m) * lda + k] = F2H(float((m * 3 + k * 7) % 23 - 11) / 32);
    for (int n = 0; n < r.n; ++n) {
      for (int k = 0; k < r.k; ++k)
        b[static_cast<size_t>(n) * row_bytes + k / 2] |= uint8_t(((n * 5 + k * 3) % 16) << (4 * (k & 1)));
      for (int g = 0; g < groups; ++g) {
        scales[static_cast<size_t>(n) * groups + g] = F2H(.01f + .001f * ((n + g) % 7));
        zeros[static_cast<size_t>(n) * groups + g] = F2H(float(7 + (n + g) % 3));
      }
    }
    buffers.grow((void**)&buffers.a, &buffers.a_n, a.size() * 2);
    buffers.grow((void**)&buffers.b, &buffers.b_n, b.size());
    buffers.grow((void**)&buffers.sc, &buffers.sc_n, scales.size() * 2);
    buffers.grow((void**)&buffers.zpf, &zeros_bytes, zeros.size() * 2);
    buffers.grow((void**)&buffers.out, &out_bytes, static_cast<size_t>(r.m) * r.n * 2);
    HIP_OK(hipMemcpy(buffers.a, a.data(), a.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buffers.b, b.data(), b.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buffers.sc, scales.data(), scales.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buffers.zpf, zeros.data(), zeros.size() * 2, hipMemcpyHostToDevice));
    if (r.padded && mode == "rank") {
      const size_t need = hip_matmul_nbits_u4_padrow_bytes(r.n, r.k);
      buffers.grow((void**)&buffers.bpad, &buffers.bpad_n, need);
      if (hip_matmul_nbits_u4_prepack_padrow(nullptr, buffers.b, buffers.bpad, r.n, r.k))
        throw std::runtime_error("padded input preparation failed");
      HIP_OK(hipDeviceSynchronize());
    }
    const void* cold_source_b = r.padded && mode == "rank" ? buffers.bpad : buffers.b;
    const size_t cold_b_bytes = r.padded && mode == "rank"
        ? static_cast<size_t>(r.n) * (r.k / 2 + 128) : b.size();
    ExactColdWeights cold(cold_source_b, cold_b_bytes, buffers.sc, scales.size() * 2,
                          buffers.zpf, r.zeros ? zeros.size() * 2 : 0);
    int turn = 0;
    auto launch = [&](LoadedMatmulNbits& api, int config) {
      const void* weight_b = cold.part(turn, 0);
      const void* weight_scales = cold.part(turn, cold.scale_offset);
      const void* weight_zeros = r.zeros ? cold.part(turn, cold.zero_offset) : nullptr;
      ++turn;
      int rc;
      if (config >= 0)
        rc = api.configured(nullptr, config, r.m, r.n, kp, r.gs, buffers.a, kp,
            weight_b, weight_scales, weight_zeros, buffers.out,
            r.n, r.padded ? r.k / 2 + 128 : 0);
      else if (r.padded)
        rc = api.padded(nullptr, -1, r.m, r.n, r.k, r.gs, buffers.a, lda,
                        weight_b, weight_scales, weight_zeros, buffers.out, r.n);
      else
        rc = api.production(nullptr, buffers.a, weight_b, weight_scales, weight_zeros, nullptr,
            buffers.out, r.m, r.n, r.k, 1, 4, r.gs, 2, 2, nullptr, nullptr);
      if (rc) throw std::runtime_error("production/configured launch failed");
      HIP_OK(hipGetLastError());
    };
    // CPU sampling spans tile boundaries, first/last rows and random interior
    // coordinates. Full CPU-reference leaf coverage remains a final gate.
    std::vector<std::pair<size_t, float>> reference;
    const int m_edges[] = {0, r.m - 1, 15, 16, 63, 64, 127, 128};
    const int n_edges[] = {0, r.n - 1, 15, 16, 63, 64, 127, 128};
    for (int sample = 0; sample < 64; ++sample) {
      const int m = sample < 8 ? std::min(r.m - 1, m_edges[sample])
                              : int((size_t(sample) * 997 + r.m - 1) % r.m);
      const int n = sample < 8 ? std::min(r.n - 1, n_edges[sample])
                              : int((size_t(sample) * 787 + r.n - 1) % r.n);
      float acc = 0;
      for (int k = 0; k < r.k; ++k) {
        const unsigned packed = b[static_cast<size_t>(n) * row_bytes + k / 2];
        const float z = r.zeros ? H2F(zeros[static_cast<size_t>(n) * groups + k / r.gs]) : 8;
        acc += H2F(a[static_cast<size_t>(m) * lda + k]) * (float((packed >> (4 * (k & 1))) & 15) - z)
             * H2F(scales[static_cast<size_t>(n) * groups + k / r.gs]);
      }
      reference.emplace_back(static_cast<size_t>(m) * r.n + n, acc);
    }
    auto verify = [&] {
      HIP_OK(hipDeviceSynchronize());
      for (const auto& sample : reference) {
        uint16_t result;
        HIP_OK(hipMemcpy(&result, buffers.out + sample.first, 2, hipMemcpyDeviceToHost));
        const float value = H2F(result);
        if (!std::isfinite(value) || std::fabs(value - sample.second) > .1f + .05f * std::fabs(sample.second))
          throw std::runtime_error("sampled CPU correctness failed");
      }
    };
    const bool input_gap = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_loaded).count() > .1;
    // Fill copies through untimed reads, as the standard UT does.
    const int read_settle = std::min(cold.count, static_cast<int>(((size_t(128) << 20) + cold.weight_bytes - 1) / cold.weight_bytes));
    for (int i = 0; i < read_settle; ++i)
      launch(candidate, mode == "rank" ? r.configs.front() : -1);
    HIP_OK(hipDeviceSynchronize());
    if (!settled || input_gap) {
      const auto load_begin = std::chrono::steady_clock::now();
      do { launch(candidate, mode == "rank" ? r.configs.front() : -1); HIP_OK(hipDeviceSynchronize()); }
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin).count() < (settled ? .06 : 1));
      settled = true;
    }
    if (mode == "rank") {
      for (int config : r.configs) {
        std::printf("FAST_CONFIG_BEGIN id=%s config=%d\n", r.id.c_str(), config);
        std::fflush(stdout);
        *candidate.launch_audit = true;
        launch(candidate, config); HIP_OK(hipDeviceSynchronize());
        *candidate.launch_audit = false;
        std::vector<double> samples;
        const double median = fastPinnedMedian([&]{launch(candidate, config);}, samples);
        verify();
        std::printf("FAST_CONFIG id=%s config=%d median_ms=%.9f samples=%.9f;%.9f;%.9f cpu_samples=64 correctness=PASS\n",
                    r.id.c_str(), config, median, samples[0], samples[1], samples[2]);
      }
    } else {
      for (int side_index = 0; side_index < 2; ++side_index) {
        const bool is_candidate = ((index + side_index) % 2) != 0;
        auto& api = is_candidate ? candidate : *baseline;
        std::printf("FAST_PRODUCTION_BEGIN id=%s side=%s\n", r.id.c_str(), is_candidate ? "candidate" : "baseline");
        std::fflush(stdout);
        *api.launch_audit = true;
        launch(api, -1); HIP_OK(hipDeviceSynchronize());
        *api.launch_audit = false;
        // Each DLL owns a CRT buffer. Flush its untimed audit before the
        // driver writes the result so two library streams remain unambiguous.
        if (api.flush(nullptr)) throw std::runtime_error("library audit flush failed");
        std::vector<double> samples;
        const double median = fastPinnedMedian([&]{launch(api, -1);}, samples);
        verify();
        std::printf("FAST_PRODUCTION id=%s side=%s median_ms=%.9f samples=%.9f;%.9f;%.9f cpu_samples=64 correctness=PASS\n",
                    r.id.c_str(), is_candidate ? "candidate" : "baseline", median, samples[0], samples[1], samples[2]);
      }
    }
    last_loaded = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::printf("FAST_ROW_COMPLETE id=%s completed=%zu required=%zu elapsed_seconds=%.3f\n", r.id.c_str(), index + 1, requests.size(), elapsed);
    std::fflush(stdout);
    // Exit at a complete row so the controller can optimize and resume. Never
    // silently truncate coverage or continue a projected multi-day schedule.
    if (index >= 7 && elapsed / (index + 1) * requests.size() > 8 * 3600)
      return 8;
  }
  return 0;
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--requests") {
      try { return runExactSweep(argc, argv); }
      catch (const std::exception& error) {
        std::fprintf(stderr, "EXACT_SWEEP_INCOMPLETE %s\n", error.what()); return 7;
      }
    }
  }
#endif
  const char* shapes_path = nullptr;
  std::vector<int> m_list = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                             8192, 16384};
  bool do_prefill = true, do_decode = true, do_dp4a = true, do_padrow = true;
  // Also time the extended WMMA entries (WMMA_EXT_CONFIGS). Only for a device
  // whose table is meant to name them.
  bool extended = false;
  int limit = 0;
  /* Skip (M, N) pairs whose C alone would exceed this. Sized to pass every real
   * prefill shape below big_n at the full M ladder -- the widest is N=32768 x
   * M=16384 = 512 Mi halfs -- while still capping the lm_head columns, which
   * big_n already forces to M=1. A 32 Mi cap dropped real up/down-proj shapes
   * (N=17408 K=5120 at M=2048/4096); 256 Mi still drops those at M=16384. */
  long long max_out_elems = 512ll << 20;  // 512 Mi halfs = 1 GB
  /* Columns at or above this are the lm_head / vocab projection, whose B runs
   * to 600 MB and whose WMMA sweep is ~680 dispatches over it -- tens of
   * minutes per M.
   *
   * Most exports only ever hand it one token: 14 of the 18 models with a vocab
   * projection declare logits as [batch, 1, vocab], because a Gather selects
   * the last position first. For those the prefill M ladder measures a shape
   * that cannot occur, so it is skipped and M=1 is swept instead.
   *
   * Four exports do not (gemma-4-12B-it-kquant, gemma-4-E2B, gemma-4-E4B,
   * Nemotron-3-Super-120B declare [batch, sequence_len, vocab]), so for those
   * the wide-M rows are real. Pass --big-n 0 to disable the skip entirely, or
   * list those shapes in a separate CSV and sweep it without the cap. */
  int big_n = 65536;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--shapes" && i + 1 < argc) shapes_path = argv[++i];
    else if (a == "--m" && i + 1 < argc) {
      m_list.clear();
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ',')) m_list.push_back(std::atoi(tok.c_str()));
    } else if (a == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
    else if (a == "--max-out-elems" && i + 1 < argc)
      max_out_elems = std::atoll(argv[++i]);
    else if (a == "--big-n" && i + 1 < argc) big_n = std::atoi(argv[++i]);
    else if (a == "--no-prefill") do_prefill = false;
    else if (a == "--no-decode") do_decode = false;
    else if (a == "--no-dp4a") do_dp4a = false;
    else if (a == "--no-padrow") do_padrow = false;
    else if (a == "--extended") extended = true;
    else {
      std::fprintf(stderr,
                   "usage: %s --shapes shapes.csv [--m 1,8,128] [--limit N]\n"
                   "          [--no-prefill] [--no-decode] [--no-dp4a] [--no-padrow]\n"
                   "          [--extended]\n",
                   argv[0]);
      return 1;
    }
  }
  if (!shapes_path) {
    std::fprintf(stderr, "--shapes is required\n");
    return 1;
  }

  hipDeviceProp_t props;
  HIP_OK(hipGetDeviceProperties(&props, 0));
  std::vector<Shape> shapes = loadShapes(shapes_path);
  if (limit > 0 && static_cast<int>(shapes.size()) > limit)
    shapes.resize(limit);

  hip_matmul_nbits_u4_wmma_sweep_extended(extended ? 1 : 0);
  std::fprintf(stderr, "#SWEEP arch=%s shapes=%zu extended=%d\n",
               props.gcnArchName, shapes.size(), extended ? 1 : 0);

  Buffers buf;
  std::vector<uint16_t> h_a, h_sc, h_zpf;
  std::vector<uint8_t> h_b, h_zpu;

  int done = 0;
  for (const Shape& s : shapes) {
    const int ngk = (s.k + s.gs - 1) / s.gs;
    const size_t row = static_cast<size_t>(ngk) * (s.gs / 2);
    const size_t b_bytes = static_cast<size_t>(s.n) * row;

    const bool k32 = (s.k % 32 == 0);
    // K%32!=0 (e.g. K=4304): decode runs the fp GEMV on the real K, prefill
    // runs the WMMA K-padding path (dispatch pads K up to a multiple of 32).
    // dp4a is unavailable. The prefill/LUT key is the padded K, so its markers
    // below print k_pad rather than s.k.
    const int k_pad = k32 ? s.k : ((s.k + 31) & ~31);
    std::vector<int> ms;
    if (big_n > 0 && s.n >= big_n) {
      ms.push_back(1);
      std::fprintf(stderr, "#BIGN N=%d K=%d: M=1 only (vocab projection)\n",
                   s.n, s.k);
    } else {
      for (int m : m_list)
        if (static_cast<long long>(m) * s.n <= max_out_elems) ms.push_back(m);
    }
    if (ms.empty()) {
      std::fprintf(stderr, "#SKIP N=%d K=%d gs=%d zp=%d (output too large)\n",
                   s.n, s.k, s.gs, int(s.zp));
      continue;
    }
    if (ms.size() != m_list.size())
      std::fprintf(stderr, "#CAP N=%d K=%d: M capped at %d\n", s.n, s.k,
                   ms.back());
    const int max_m = *std::max_element(ms.begin(), ms.end());

    h_a.assign(static_cast<size_t>(max_m) * s.k, F2H(0.05f));
    for (size_t i = 0; i < h_a.size(); ++i)
      h_a[i] = F2H(float(int(i % 23) - 11) * 0.0625f);
    h_sc.assign(static_cast<size_t>(s.n) * ngk, 0);
    for (size_t i = 0; i < h_sc.size(); ++i)
      h_sc[i] = F2H(0.01f + 0.001f * float(i % 7));
    h_zpf.assign(h_sc.size(), F2H(8.0f));
    h_zpu.assign(h_sc.size(), 8);
    h_b.assign(b_bytes, 0);
    for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = uint8_t(i * 31);

    buf.grow((void**)&buf.a, &buf.a_n, h_a.size() * 2);
    buf.grow((void**)&buf.sc, &buf.sc_n, h_sc.size() * 2);
    buf.grow((void**)&buf.zpf, &buf.out_n, h_zpf.size() * 2);
    buf.grow((void**)&buf.zpu, &buf.qb_n, h_zpu.size());
    buf.grow((void**)&buf.b, &buf.b_n, b_bytes);
    {
      static size_t out_have = 0;
      buf.grow((void**)&buf.out, &out_have,
               static_cast<size_t>(max_m) * s.n * 2);
    }
    HIP_OK(hipMemcpy(buf.a, h_a.data(), h_a.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buf.sc, h_sc.data(), h_sc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buf.zpf, h_zpf.data(), h_zpf.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buf.zpu, h_zpu.data(), h_zpu.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(buf.b, h_b.data(), b_bytes, hipMemcpyHostToDevice));

    const void* zp_f = s.zp ? (const void*)buf.zpf : nullptr;
    const void* zp_u = s.zp ? (const void*)buf.zpu : nullptr;

    for (int m : ms) {
      // The marker is what update_lut.py keys the following autotune lines to.
      if (m == 1) {
        if (do_decode) {
          std::fprintf(stderr, "#SHAPE path=decode M=1 N=%d K=%d gs=%d zp=%d\n",
                       s.n, s.k, s.gs, int(s.zp));
          hip_matmul_nbits(nullptr, buf.a, buf.b, buf.sc, zp_f, nullptr,
                           buf.out, 1, s.n, s.k, 1, 4, s.gs, 2, 2, nullptr,
                           nullptr);
          HIP_OK(hipDeviceSynchronize());
        }
        if (do_dp4a && k32) {   // dp4a kernel requires K%32==0
          static size_t qb_have = 0, qs_have = 0;
          buf.grow(&buf.qb, &qb_have, static_cast<size_t>(s.k));
          buf.grow(&buf.qs, &qs_have, static_cast<size_t>(ngk) * sizeof(float));
          std::fprintf(stderr, "#SHAPE path=dp4a M=1 N=%d K=%d gs=%d zp=%d\n",
                       s.n, s.k, s.gs, int(s.zp));
          hip_matmul_nbits_dp4a(nullptr, buf.a, buf.b, buf.sc, zp_u, nullptr,
                                buf.out, 1, s.n, s.k, s.gs, buf.qb, buf.qs);
          HIP_OK(hipDeviceSynchronize());
        }
        continue;
      }
      if (!do_prefill) continue;

      // Marker prints k_pad: for K%32!=0 the dispatch pads to k_pad and the
      // WMMA tuner (and the runtime LUT query) key on the padded K. For k32
      // shapes k_pad == s.k, so this is unchanged.
      std::fprintf(stderr, "#SHAPE path=prefill M=%d N=%d K=%d gs=%d zp=%d\n",
                   m, s.n, k_pad, s.gs, int(s.zp));
      hip_matmul_nbits(nullptr, buf.a, buf.b, buf.sc, zp_f, nullptr, buf.out,
                       m, s.n, s.k, 1, 4, s.gs, 2, 2, nullptr, nullptr);
      HIP_OK(hipDeviceSynchronize());

      if (do_padrow && shouldPadRow(s.k) && (s.k % s.gs == 0)) {
        static size_t pad_have = 0;
        const size_t need = hip_matmul_nbits_u4_padrow_bytes(s.n, s.k);
        buf.grow((void**)&buf.bpad, &pad_have, need);
        if (hip_matmul_nbits_u4_prepack_padrow(nullptr, buf.b, buf.bpad, s.n,
                                               s.k) == 0) {
          HIP_OK(hipDeviceSynchronize());
          std::fprintf(stderr,
                       "#SHAPE path=prefill_padrow M=%d N=%d K=%d gs=%d zp=%d "
                       "pad=128\n",
                       m, s.n, s.k, s.gs, int(s.zp));
          hip_matmul_nbits_u4_wmma_stride(nullptr, -1, m, s.n, s.k, s.gs, buf.a,
                                          s.k, buf.bpad, buf.sc, zp_f, buf.out,
                                          s.n, s.k / 2 + 128);
          HIP_OK(hipDeviceSynchronize());
        }
      }
    }
    if (++done % 10 == 0)
      std::fprintf(stderr, "#PROGRESS %d/%zu\n", done, shapes.size());
  }
  std::fprintf(stderr, "#DONE %d shapes\n", done);
  return 0;
}
