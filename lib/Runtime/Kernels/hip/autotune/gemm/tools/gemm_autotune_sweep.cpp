/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Offline sweep driver for the Gemm LUT (Stage A10, plan.md §5).
 *
 * Walks a shape list and touches each (phase, dtype, M, trans_b) once, which is
 * enough to make hip_gemm's in-kernel autotuner run and log its winner. This
 * tool deliberately does not pick or print configs itself -- it only prints the
 * `#SHAPE` marker update_lut.py keys its parsing to, then calls hip_gemm(). The
 * kernel's own tuner is the only place "which config wins" is decided, so the
 * table can never drift from what production actually runs.
 *
 * Run with HIPDNN_EP_DEBUG=1 (per-candidate + winner logging) and
 * HIPDNN_GEMM_AUTOTUNE_MODE=online (force the sweep; MODE=lookup would just
 * consult the table-in-progress and never time anything). update_lut.py sets
 * both env vars for you (see cmd_measure).
 *
 * Build (from the repo root, mirrors matmul_nbits_autotune_sweep.cpp):
 *   hipcc --offload-arch=gfx1151 -O3 -std=c++17 -w \
 *       -I lib/Runtime/Kernels/include \
 *       lib/Runtime/Kernels/hip/autotune/gemm/tools/gemm_autotune_sweep.cpp \
 *       lib/Runtime/Kernels/hip/gemm_kernel.hip \
 *       lib/Runtime/Kernels/hip/autotune/gemm/tools/gemm_autotune_stub.cpp \
 *       -o gemm_sweep.exe
 *
 * (links the flatbuffers-free stub, same reasoning as the matmul_nbits sweep:
 * this tool always runs in `online` mode, so HIPDNN_GEMM_AUTOTUNE_MODE=online
 * makes every LUT-lookup call site short-circuit to -1 before it would ever
 * call gemm_lut::resolve() for real, but the symbol still needs a definition
 * to link.)
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

extern "C" int hip_gemm(void *stream, const void *A, const void *B,
                        const void *C, void *Y, int M, int N, int K,
                        float alpha, float beta, int transA, int transB,
                        int type_code, int cDim0, int cDim1);

#define HIP_OK(c)                                                             \
  do {                                                                        \
    hipError_t e_ = (c);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP %s at line %d\n", hipGetErrorString(e_),      \
                   __LINE__);                                                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

/* type_code values, mirroring gemm_kernel.hip's kTypeFloat16/32/64/kTypeBFloat16. */
static constexpr int kTypeFloat16 = 0;
static constexpr int kTypeFloat32 = 1;
static constexpr int kTypeFloat64 = 2;
static constexpr int kTypeBFloat16 = 3;

static uint16_t F2H(float f) {
  _Float16 h = (_Float16)f;
  uint16_t o;
  std::memcpy(&o, &h, 2);
  return o;
}

/* Round-to-nearest-even truncation of fp32 to bf16 (upper 16 bits). Exact
 * rounding does not matter for a timing sweep (no correctness check here),
 * this just avoids an all-zero mantissa pattern that a truncating cast alone
 * would produce for the fixed small values below. */
static uint16_t F2BF16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

struct Shape {
  int n = 0, k = 0;
  std::vector<int> m_list; // per-shape M ladder from the CSV (empty = use the
                            // tool's default / --m override, see main()).
  bool operator<(const Shape &o) const {
    return std::tie(n, k) < std::tie(o.n, o.k);
  }
};

static std::vector<Shape> loadShapes(const char *path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  std::string line;
  if (!std::getline(in, line))
    return {};
  std::vector<std::string> head;
  {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
      head.push_back(cell);
  }
  auto col = [&](const char *name) {
    for (size_t i = 0; i < head.size(); ++i)
      if (head[i] == name)
        return static_cast<int>(i);
    std::fprintf(stderr, "shape csv has no '%s' column\n", name);
    std::exit(1);
  };
  const int cn = col("N"), ck = col("K");
  // m_list is optional (older/hand-written shape CSVs may omit it); when
  // absent every row falls back to the tool's default / --m override.
  int cm = -1;
  for (size_t i = 0; i < head.size(); ++i)
    if (head[i] == "m_list") { cm = static_cast<int>(i); break; }

  std::set<Shape> uniq;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
      cells.push_back(cell);
    if (static_cast<int>(cells.size()) <= std::max(cn, ck))
      continue;
    Shape s;
    s.n = std::atoi(cells[cn].c_str());
    s.k = std::atoi(cells[ck].c_str());
    if (s.n <= 0 || s.k <= 0)
      continue;
    // m_list is space-separated *within* its own comma cell (e.g.
    // "1 4 8 16 32 128"), so it survives the comma split above whole.
    if (cm >= 0 && cm < static_cast<int>(cells.size())) {
      std::stringstream ms(cells[cm]);
      std::string tok;
      while (ms >> tok) {
        const int v = std::atoi(tok.c_str());
        if (v > 0)
          s.m_list.push_back(v);
      }
    }
    uniq.insert(s);
  }
  return {uniq.begin(), uniq.end()};
}

enum class DType { F16, BF16, F32 };

static const char *dtypeName(DType d) {
  switch (d) {
  case DType::F16: return "f16";
  case DType::BF16: return "bf16";
  case DType::F32: return "f32";
  }
  return "?";
}
static int dtypeTypeCode(DType d) {
  switch (d) {
  case DType::F16: return kTypeFloat16;
  case DType::BF16: return kTypeBFloat16;
  case DType::F32: return kTypeFloat32;
  }
  return -1;
}
static int dtypeElemBytes(DType d) { return d == DType::F32 ? 4 : 2; }

static std::vector<DType> parseDtypes(const std::string &csv) {
  std::vector<DType> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok == "f16") out.push_back(DType::F16);
    else if (tok == "bf16") out.push_back(DType::BF16);
    else if (tok == "f32") out.push_back(DType::F32);
    else {
      std::fprintf(stderr, "unknown dtype '%s' (want f16,bf16,f32)\n",
                   tok.c_str());
      std::exit(1);
    }
  }
  return out;
}

enum class Phase { Wmma, GemvNt, GemvNn, TiledFma };

static std::vector<Phase> parsePhases(const std::string &csv) {
  std::vector<Phase> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok == "Wmma") out.push_back(Phase::Wmma);
    else if (tok == "GemvNt") out.push_back(Phase::GemvNt);
    else if (tok == "GemvNn") out.push_back(Phase::GemvNn);
    else if (tok == "TiledFma") out.push_back(Phase::TiledFma);
    else {
      std::fprintf(stderr,
                   "unknown phase '%s' (want Wmma,GemvNt,GemvNn,TiledFma)\n",
                   tok.c_str());
      std::exit(1);
    }
  }
  return out;
}

/* Device buffers sized for the largest shape touched, reused across the sweep
 * so the sweep measures tuning, not hipMalloc. */
struct Buffers {
  void *a = nullptr, *b = nullptr, *y = nullptr;
  size_t a_n = 0, b_n = 0, y_n = 0;

  void grow(void **p, size_t *have, size_t need_bytes) {
    if (*have >= need_bytes)
      return;
    if (*p)
      HIP_OK(hipFree(*p));
    HIP_OK(hipMalloc(p, need_bytes));
    *have = need_bytes;
  }
};

int main(int argc, char **argv) {
  const char *shapes_path = nullptr;
  std::vector<int> m_list = {1,   8,   16,   32,   64,  128,
                             256, 512, 1024, 2048, 4096};
  bool m_override = false; // --m given explicitly: overrides every shape's
                           // own CSV m_list (pilots / targeted re-sweeps).
                           // Without it, each shape uses its own m_list
                           // column (plan.md §4.6 -- vocab/moe/vis need a
                           // much shorter M ladder than "the rest", and
                           // applying one uniform list to every row wastes
                           // GPU time measuring M values those categories
                           // will never be queried at).
  std::string dtypes_csv = "f16,bf16,f32";
  std::string phases_csv = "Wmma,GemvNt,GemvNn,TiledFma";
  int limit = 0;
  double max_mem_frac = 0.6;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--shapes" && i + 1 < argc) shapes_path = argv[++i];
    else if (a == "--m" && i + 1 < argc) {
      m_list.clear();
      m_override = true;
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ','))
        m_list.push_back(std::atoi(tok.c_str()));
    } else if (a == "--dtypes" && i + 1 < argc) dtypes_csv = argv[++i];
    else if (a == "--phases" && i + 1 < argc) phases_csv = argv[++i];
    else if (a == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
    else if (a == "--max-mem-frac" && i + 1 < argc)
      max_mem_frac = std::atof(argv[++i]);
    else {
      std::fprintf(stderr,
                   "usage: %s --shapes shapes.csv [--m 1,8,128] "
                   "[--dtypes f16,bf16,f32] [--phases Wmma,GemvNt,GemvNn,"
                   "TiledFma] [--limit N] [--max-mem-frac 0.6]\n",
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
  // hip_gemm's entry point rejects a null stream outright (`if (!stream ...)
  // return -1;`), so every call below needs a real one -- see plan.md §5
  // Stage A10.
  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));
  std::vector<Shape> shapes = loadShapes(shapes_path);
  if (limit > 0 && static_cast<int>(shapes.size()) > limit)
    shapes.resize(limit);
  const std::vector<DType> dtypes = parseDtypes(dtypes_csv);
  const std::vector<Phase> phases = parsePhases(phases_csv);
  const size_t mem_cap =
      static_cast<size_t>(props.totalGlobalMem * max_mem_frac);

  std::fprintf(stderr, "#SWEEP arch=%s shapes=%zu dtypes=%s phases=%s\n",
               props.gcnArchName, shapes.size(), dtypes_csv.c_str(),
               phases_csv.c_str());

  Buffers buf;
  std::vector<uint8_t> h_a, h_b;

  int done = 0;
  for (const Shape &sh : shapes) {
    const std::vector<int> &eff_m_list =
        (!m_override && !sh.m_list.empty()) ? sh.m_list : m_list;
    for (DType dt : dtypes) {
      const int elem = dtypeElemBytes(dt);
      const int max_m =
          *std::max_element(eff_m_list.begin(), eff_m_list.end());
      const size_t a_bytes = static_cast<size_t>(max_m) * sh.k * elem;
      const size_t b_bytes = static_cast<size_t>(sh.n) * sh.k * elem;
      const size_t y_bytes = static_cast<size_t>(max_m) * sh.n * elem;
      if (a_bytes + b_bytes + y_bytes > mem_cap) {
        std::fprintf(stderr,
                     "#SKIP dtype=%s N=%d K=%d (needs %.2f GB > cap)\n",
                     dtypeName(dt), sh.n, sh.k,
                     (a_bytes + b_bytes + y_bytes) / 1e9);
        continue;
      }

      h_a.assign(a_bytes, 0);
      h_b.assign(b_bytes, 0);
      if (dt == DType::F32) {
        float *fa = reinterpret_cast<float *>(h_a.data());
        float *fb = reinterpret_cast<float *>(h_b.data());
        for (size_t i = 0; i < h_a.size() / 4; ++i)
          fa[i] = float(int(i % 23) - 11) * 0.0625f;
        for (size_t i = 0; i < h_b.size() / 4; ++i)
          fb[i] = 0.01f + 0.001f * float(i % 7);
      } else {
        uint16_t *ha = reinterpret_cast<uint16_t *>(h_a.data());
        uint16_t *hb = reinterpret_cast<uint16_t *>(h_b.data());
        auto conv = (dt == DType::F16) ? F2H : F2BF16;
        for (size_t i = 0; i < h_a.size() / 2; ++i)
          ha[i] = conv(float(int(i % 23) - 11) * 0.0625f);
        for (size_t i = 0; i < h_b.size() / 2; ++i)
          hb[i] = conv(0.01f + 0.001f * float(i % 7));
      }
      buf.grow(&buf.a, &buf.a_n, a_bytes);
      buf.grow(&buf.b, &buf.b_n, b_bytes);
      buf.grow(&buf.y, &buf.y_n, y_bytes);
      HIP_OK(hipMemcpy(buf.a, h_a.data(), a_bytes, hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(buf.b, h_b.data(), b_bytes, hipMemcpyHostToDevice));

      for (Phase ph : phases) {
        // TiledFma is only measured for fp32 this round (plan.md §5 Stage B);
        // Wmma only for fp16/bf16 (no WMMA instruction for fp32). GemvNt/GemvNn
        // (decode, M=1) apply to all three dtypes.
        if (ph == Phase::TiledFma && dt != DType::F32) continue;
        if (ph == Phase::Wmma && dt == DType::F32) continue;

        const char *phase_name = ph == Phase::Wmma        ? "Wmma"
                                 : ph == Phase::GemvNt     ? "GemvNt"
                                 : ph == Phase::GemvNn     ? "GemvNn"
                                                            : "TiledFma";
        if (ph == Phase::GemvNt || ph == Phase::GemvNn) {
          const int tb = (ph == Phase::GemvNt) ? 1 : 0;
          std::fprintf(stderr,
                       "#SHAPE phase=%s act=%s wts=%s out=%s tb=%d M=1 N=%d "
                       "K=%d\n",
                       phase_name, dtypeName(dt), dtypeName(dt), dtypeName(dt),
                       tb, sh.n, sh.k);
          const int rc = hip_gemm(stream, buf.a, buf.b, nullptr, buf.y, 1,
                                  sh.n, sh.k, 1.0f, 0.0f, 0, tb,
                                  dtypeTypeCode(dt), 1, 1);
          HIP_OK(hipDeviceSynchronize());
          if (rc != 0)
            std::fprintf(stderr, "#ERROR hip_gemm rc=%d phase=%s\n", rc,
                         phase_name);
          continue;
        }

        // Wmma / TiledFma: sweep the M ladder (skip M values whose Y would
        // exceed the same mem_cap headroom already spent on A/B).
        for (int m : eff_m_list) {
          if (m == 1) continue; // M=1 is the GEMV decode phases' job, not this one.
          const size_t y_this = static_cast<size_t>(m) * sh.n * elem;
          if (a_bytes + b_bytes + y_this > mem_cap) {
            std::fprintf(stderr,
                         "#SKIP phase=%s dtype=%s M=%d N=%d K=%d (Y too big)\n",
                         phase_name, dtypeName(dt), m, sh.n, sh.k);
            continue;
          }
          if (ph == Phase::Wmma) {
            std::fprintf(stderr,
                         "#SHAPE phase=Wmma act=%s wts=%s out=%s tb=1 M=%d "
                         "N=%d K=%d\n",
                         dtypeName(dt), dtypeName(dt), dtypeName(dt), m, sh.n,
                         sh.k);
            const int rc = hip_gemm(stream, buf.a, buf.b, nullptr, buf.y, m,
                                    sh.n, sh.k, 1.0f, 0.0f, 0, 1,
                                    dtypeTypeCode(dt), 1, 1);
            HIP_OK(hipDeviceSynchronize());
            if (rc != 0)
              std::fprintf(stderr, "#ERROR hip_gemm rc=%d phase=Wmma M=%d\n",
                           rc, m);
          } else { // TiledFma: both trans_b values (plan.md §5 Stage B)
            for (int tb : {0, 1}) {
              std::fprintf(stderr,
                           "#SHAPE phase=TiledFma act=%s wts=%s out=%s tb=%d "
                           "M=%d N=%d K=%d\n",
                           dtypeName(dt), dtypeName(dt), dtypeName(dt), tb, m,
                           sh.n, sh.k);
              const int rc = hip_gemm(stream, buf.a, buf.b, nullptr, buf.y, m,
                                      sh.n, sh.k, 1.0f, 0.0f, 0, tb,
                                      dtypeTypeCode(dt), 1, 1);
              HIP_OK(hipDeviceSynchronize());
              if (rc != 0)
                std::fprintf(stderr,
                             "#ERROR hip_gemm rc=%d phase=TiledFma M=%d tb=%d\n",
                             rc, m, tb);
            }
          }
        }
      }
    }
    if (++done % 10 == 0)
      std::fprintf(stderr, "#PROGRESS %d/%zu\n", done, shapes.size());
  }
  std::fprintf(stderr, "#DONE %d shapes\n", done);
  return 0;
}
