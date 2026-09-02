/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Offline sweep driver for the MatMulNBits LUT.
 *
 * Walks a shape list and touches each (path, shape, M) once, which is enough to
 * make the in-kernel autotuner run and log its winner. The winners are read off
 * stderr by scripts/update_lut.py -- this tool deliberately does not pick or
 * print configs itself, so there is exactly one implementation of "which config
 * wins", the one that ships.
 *
 * Run with HIPDNN_EP_DEBUG=1 and delete the tune caches first, or shapes that
 * are already cached will be silently skipped:
 *     Remove-Item "$env:TEMP\morphizen_*cache*"
 *
 * Shape list is CSV with a header, columns K,N,block_size,has_zp (extra columns
 * ignored) -- i.e. what extract_shapes.py emits.
 *
 * Build (from the repo root):
 *   clang++ -x hip --offload-arch=gfx1151 -O3 -std=c++17 -w \
 *       -I lib/Runtime/Kernels/include \
 *       lib/Runtime/Kernels/hip/autotune/matmul_nbits/tools/mn_autotune_sweep.cpp \
 *       lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip -o mn_sweep.exe
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
extern "C" int hip_matmul_nbits_dp4a(
    void* stream_v, const void* A, const void* B, const void* scales,
    const void* zp_u8, const void* bias, void* out,
    int64_t N, int64_t K, int64_t block_size,
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

int main(int argc, char** argv) {
  const char* shapes_path = nullptr;
  std::vector<int> m_list = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  bool do_prefill = true, do_decode = true, do_dp4a = true, do_padrow = true;
  int limit = 0;
  /* Skip (M, N) pairs whose C alone would exceed this. Sized to pass every real
   * prefill shape below big_n at the full M ladder -- the widest is N~32768 x
   * M=4096 = 134 Mi halfs (268 MB) -- while still capping the lm_head columns,
   * which big_n already forces to M=1. The old 32 Mi cap wrongly dropped real
   * up/down-proj shapes at M=2048/4096 (e.g. N=17408 K=5120). */
  long long max_out_elems = 256ll << 20;  // 256 Mi halfs = 512 MB
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
    else {
      std::fprintf(stderr,
                   "usage: %s --shapes shapes.csv [--m 1,8,128] [--limit N]\n"
                   "          [--no-prefill] [--no-decode] [--no-dp4a] [--no-padrow]\n",
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

  std::fprintf(stderr, "#SWEEP arch=%s shapes=%zu\n", props.gcnArchName,
               shapes.size());

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
    if (s.n >= big_n) {
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
                                buf.out, s.n, s.k, s.gs, buf.qb, buf.qs);
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
                       "#SHAPE path=prefill_padrow M=%d N=%d K=%d gs=%d zp=%d\n",
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
