/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GQA offline autotune resolver.
//
// This is host code compiled into the per-arch custom_kernels_<arch> shared
// library (see lib/Runtime/Kernels/CMakeLists.txt), alongside gqa_kernel.hip.
// The measured table it reads is embedded in that same DLL, so the data symbol
// never crosses the JIT boundary; real/gqa.cpp (bitcode) reaches this resolver
// through the exported extern "C" shims at the bottom of the file. See the
// header for why the table and its consumer had to move together.
//
// The lookup is hierarchical, most-impactful axis first:
//
//   1. kernel identity -- phase, head_dim, kv_dtype -- matched exactly. These
//      are not distances: d64 and d128 are different kernel instantiations, so
//      borrowing across them is meaningless.
//   2. geometry -- (num_heads, kv_num_heads). The exact pair is preferred; if
//   it
//      was never measured, the search stays within the same heads-per-group
//      ratio (the axis the kernels template on), never crossing to a different
//      ratio. This is the axis with the largest effect on the config.
//   3. lengths -- seq_q, seq_kv, batch -- nearest measured point in weighted
//      log2 space. Distance 0 is an Exact hit.
//
// A geometry with no usable point falls to its (phase, head_dim) Fallback row;
// no table at all falls to the compiled-in heuristic. Both last resorts return
// a runnable config, so the op never fails for lack of a table.
//
// One distance function serves both the exact-geometry search (where the head
// terms are zero, so it reduces to lengths + batch) and the same-hpg fuzzy
// search (where the head terms pick the closest measured head pair). Keeping
// the tiering in the lookup rather than the schema is what lets a later
// experiment re-order or re-tier the axes without regenerating the table.

#include "gqa_autotune.h"

#include "gqa_autotune_generated.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if !defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
#include <hip/hip_runtime.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace hipdnn_ep {
namespace {

namespace fbs = hipdnn_ep::gqa_autotune::fbs;

constexpr uint32_t kSchemaVersion = 9;
// Bump when the config knobs or the meaning of a stored config changes. A table
// stamped with an older ABI is rejected outright rather than allowed to pick a
// config that no longer means what it meant when it was measured. v3 is the
// nearest-neighbour rewrite (v2 was the tier table in runtime.bc).
constexpr const char kKernelAbi[] = "gqa-v3";

bool logOn() {
  static const bool on = [] {
#ifdef _WIN32
    char buf[8];
    return GetEnvironmentVariableA("HIPDNN_GQA_LUT_LOG", buf, sizeof(buf)) >
               0 &&
           buf[0] >= '1';
#else
    const char *v = getenv("HIPDNN_GQA_LUT_LOG");
    return v && v[0] >= '1';
#endif
  }();
  return on;
}

// resolve() runs once per dispatch -- once per token per layer for decode -- so
// logging every call floods a served model with tens of thousands of identical
// lines. Emit each distinct line once per process instead: the diagnostic still
// shows every shape's resolution (exact/nearest/fallback/heuristic) without the
// per-token repetition.
bool logLineFirstSeen(const char *line) {
  static std::mutex m;
  static std::unordered_set<std::string> seen;
  std::lock_guard<std::mutex> lk(m);
  return seen.insert(line).second;
}

// ---------------------------------------------------------------------------
// Classification (the exact-match key)
// ---------------------------------------------------------------------------

fbs::GqaPhase decodePhase() { return fbs::GqaPhase::Decode; }

fbs::GqaPhase prefillPhase(GqaPrefillVariant v) {
  switch (v) {
  case GqaPrefillVariant::V5:
    return fbs::GqaPhase::PrefillV5;
  case GqaPrefillVariant::V7:
    return fbs::GqaPhase::PrefillV7;
  case GqaPrefillVariant::V8:
    return fbs::GqaPhase::PrefillV8;
  }
  return fbs::GqaPhase::PrefillV8;
}

fbs::GqaHeadDim headDimClass(int head_dim) {
  switch (head_dim) {
  case 64:
    return fbs::GqaHeadDim::D64;
  case 128:
    return fbs::GqaHeadDim::D128;
  case 256:
    return fbs::GqaHeadDim::D256;
  default:
    return fbs::GqaHeadDim::Any;
  }
}

fbs::GqaKvDtype kvDtypeClass(int kv_dtype) {
  return kv_dtype == 1 ? fbs::GqaKvDtype::Int8 : fbs::GqaKvDtype::Fp16;
}

unsigned headsPerGroup(int num_heads, int kv_num_heads) {
  if (kv_num_heads <= 0 || num_heads <= 0 || num_heads % kv_num_heads != 0)
    return 0;
  const int hpg = num_heads / kv_num_heads;
  return hpg > 16 ? 0 : static_cast<unsigned>(hpg);
}

// Kernel identity: phase, head_dim, kv_dtype. The coarsest exact-match key,
// shared by the geometry and hpg keys below.
uint32_t kernelKey(fbs::GqaPhase phase, fbs::GqaHeadDim head_dim,
                   fbs::GqaKvDtype kv_dtype) {
  return (static_cast<uint32_t>(phase) & 0x7u) |
         ((static_cast<uint32_t>(head_dim) & 0x3u) << 3) |
         ((static_cast<uint32_t>(kv_dtype) & 0x3u) << 5);
}

// Exact geometry group: kernel identity + the (num_heads, kv_num_heads) pair.
// The primary lookup; a point lands here under its own exact head counts.
uint64_t geomKey(fbs::GqaPhase phase, fbs::GqaHeadDim head_dim,
                 fbs::GqaKvDtype kv_dtype, unsigned num_heads,
                 unsigned kv_num_heads) {
  return static_cast<uint64_t>(kernelKey(phase, head_dim, kv_dtype)) |
         (static_cast<uint64_t>(num_heads & 0x1FFu) << 8) |
         (static_cast<uint64_t>(kv_num_heads & 0xFFu) << 17);
}

// Fuzzy geometry group: kernel identity + heads-per-group ratio. A geometry the
// table never measured falls here, staying within its own ratio.
uint32_t hpgKey(fbs::GqaPhase phase, fbs::GqaHeadDim head_dim,
                fbs::GqaKvDtype kv_dtype, unsigned hpg) {
  return kernelKey(phase, head_dim, kv_dtype) | ((hpg & 0x1Fu) << 8);
}

uint32_t fallbackKey(fbs::GqaPhase phase, fbs::GqaHeadDim head_dim) {
  return (static_cast<uint32_t>(phase) & 0x7u) |
         ((static_cast<uint32_t>(head_dim) & 0x3u) << 3);
}

// ---------------------------------------------------------------------------
// What a decode config needs to be runnable (kernel facts, not table policy)
// ---------------------------------------------------------------------------

// WMMA is templated for these (head_dim, heads-per-group) pairs only. A nearest
// point naming WMMA for a geometry outside this set is rejected and the search
// moves on -- the same self-correction matmul's validator does.
bool wmmaSupported(int head_dim, unsigned heads_per_group) {
  if (heads_per_group == 4)
    return head_dim == 64 || head_dim == 128;
  return heads_per_group == 8 && head_dim == 64;
}

int decodeEffectiveLen(const GqaDecodeRequest &request) {
  int effective_len = std::max(request.effective_skv, 1);
  if (request.local_window > 0)
    effective_len = std::min(effective_len, request.local_window);
  return effective_len;
}

// Splits that have work to do: the split-K decode gives each block a 16-key
// tile, so beyond this the extra blocks scan nothing. The stored split count is
// what the top of a shape's neighbourhood wants; the clamp is what makes the
// same point right at a shorter length.
int usefulSplits(const GqaDecodeRequest &request) {
  return std::max(1, (decodeEffectiveLen(request) + 15) / 16);
}

GqaDecodeConfig clampDecodeConfig(const GqaDecodeRequest &request,
                                  GqaDecodeConfig config) {
  config.splits = std::min(config.splits, usefulSplits(request));
  if (config.splits < 1)
    config.splits = 1;
  return config;
}

bool validDecodeConfig(const GqaDecodeRequest &request,
                       const GqaDecodeConfig &config) {
  if (config.splits < 1 || config.splits > request.max_splits ||
      config.splits > 64)
    return false;
  if (!config.use_wmma)
    return config.bkv == 16;
  if (!wmmaSupported(request.head_dim,
                     headsPerGroup(request.num_heads, request.kv_num_heads)))
    return false;
  // d128's kernel fixes the tile at 32 and ignores this placeholder; d64's
  // value is a real choice. Both accept only compiled tile heights.
  return config.bkv == 16 || config.bkv == 32;
}

// ---------------------------------------------------------------------------
// Heuristic last resort (used only when no table loaded at all)
// ---------------------------------------------------------------------------

// Blocks the part can have in flight per CU, fitted over 4722 measured decode
// shapes on gfx1151. See the split-count discussion below.
constexpr int kBlocksPerCu = 17;
constexpr int kAssumedCus = 20;

// llama.cpp's launch_fattn rule: the split count that fills the machine without
// opening another wave. Needs no measurement and follows the hardware, which is
// what a last resort should do.
int occupancySplits(const GqaDecodeRequest &request, int cus) {
  const int tiles_dst =
      std::max(1, request.batch) * std::max(1, request.num_heads);
  const int useful = usefulSplits(request);
  const int blocks_per_wave = std::max(1, cus * kBlocksPerCu);
  int best = 1, best_waves = 0;
  double best_efficiency = 0.0;
  for (int splits = 1; splits <= useful; ++splits) {
    const long long total = static_cast<long long>(tiles_dst) * splits;
    const int waves =
        static_cast<int>((total + blocks_per_wave - 1) / blocks_per_wave);
    const double efficiency = static_cast<double>(total) /
                              (static_cast<double>(waves) * blocks_per_wave);
    if (best_efficiency >= 0.95 && waves > best_waves)
      break;
    if (efficiency > best_efficiency) {
      best = splits;
      best_waves = waves;
      best_efficiency = efficiency;
    }
  }
  return best;
}

GqaDecodeConfig decodeHeuristic(const GqaDecodeRequest &request, int cus) {
  const int cap = std::max(1, std::min(request.max_splits, 64));
  return {/*use_wmma=*/false,
          /*splits=*/std::max(1, std::min(cap, occupancySplits(request, cus))),
          /*bkv=*/16};
}

GqaPrefillConfig prefillHeuristic(const GqaPrefillRequest &request) {
  switch (request.variant) {
  case GqaPrefillVariant::V5:
    return {/*m_tiles=*/1, /*bkv=*/32, /*nw=*/0, /*mt=*/0, /*nd=*/0};
  case GqaPrefillVariant::V7:
    return {/*m_tiles=*/0, /*bkv=*/32, /*nw=*/1, /*mt=*/1, /*nd=*/0};
  case GqaPrefillVariant::V8:
    return {/*m_tiles=*/0, /*bkv=*/32, /*nw=*/0, /*mt=*/1, /*nd=*/2};
  }
  return {0, 32, 0, 0, 0};
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

struct Answer {
  fbs::GqaConfigKind kind;
  // Decode payload.
  bool use_wmma;
  int bkv;
  // Prefill payload.
  GqaPrefillConfig prefill;
};

// A measured point, with the logs taken once at load rather than per lookup.
struct Point {
  float l_num_heads, l_kv_num_heads, l_batch, l_seq_q, l_seq_kv;
  uint32_t num_heads, kv_num_heads, batch, seq_q, seq_kv;
  uint16_t config;
  uint8_t splits; // decode split count; 0 on prefill
};

struct Table {
  std::vector<Answer> pool;
  // Primary: exact (kernel, num_heads, kv_num_heads) -> its measured points.
  std::unordered_map<uint64_t, std::vector<Point>> exact_geom;
  // Fuzzy: (kernel, heads-per-group) -> every point of that ratio, pooled over
  // head pairs, for a geometry the table never measured exactly.
  std::unordered_map<uint32_t, std::vector<Point>> by_hpg;
  // fallbackKey -> (config index, splits)
  std::unordered_map<uint32_t, std::pair<uint16_t, uint8_t>> fallbacks;
  float w_num_heads = 1.0f, w_kv_num_heads = 1.0f, w_batch = 1.0f;
  float w_seq_q = 1.0f, w_seq_kv = 1.0f;
  bool loaded = false;
  uint32_t points = 0;
  uint32_t invalid_points = 0;
  std::atomic<uint32_t> rejected{0};
};

std::string currentGpuArch() {
#if defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
  // No device to ask; the arch a table is checked against comes from the
  // environment, so a build machine can verify a shipped .fb without a GPU.
#ifdef _WIN32
  char buf[64];
  DWORD n = GetEnvironmentVariableA("HIPDNN_EP_GQA_ARCH", buf, sizeof(buf));
  return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
  const char *v = getenv("HIPDNN_EP_GQA_ARCH");
  return v ? std::string(v) : std::string();
#endif
#else
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess)
    return std::string();
  std::string arch(props.gcnArchName);
  // gcnArchName carries feature suffixes like "gfx1151:xnack-"; the table is
  // stamped with the bare arch.
  const size_t colon = arch.find(':');
  return colon == std::string::npos ? arch : arch.substr(0, colon);
#endif
}

int currentComputeUnits() {
#if defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
  return 0;
#else
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess)
    return 0;
  return props.multiProcessorCount;
#endif
}

// A point must classify every field it is keyed on, carry positive dims, and
// name a config of the kind its phase consumes.
bool pointConsistent(const fbs::GqaTunePoint &p, fbs::GqaConfigKind kind) {
  if (p.phase() == fbs::GqaPhase::Any || p.head_dim() == fbs::GqaHeadDim::Any ||
      p.kv_dtype() == fbs::GqaKvDtype::Any)
    return false;
  if (p.num_heads() == 0 || p.kv_num_heads() == 0 || p.seq_kv() == 0 ||
      p.seq_q() == 0 || p.batch() == 0)
    return false;
  const bool decode = p.phase() == fbs::GqaPhase::Decode;
  // Int8 KV is a decode-only path (prefill dequantizes to fp16 first).
  if (!decode && p.kv_dtype() != fbs::GqaKvDtype::Fp16)
    return false;
  return kind ==
         (decode ? fbs::GqaConfigKind::Decode : fbs::GqaConfigKind::Prefill);
}

bool compatible(const fbs::GqaAutotuneLut *lut) {
  if (lut->schema_version() != kSchemaVersion) {
    if (logOn())
      fprintf(stderr, "[gqa-lut] schema %u != %u, ignoring table\n",
              lut->schema_version(), kSchemaVersion);
    return false;
  }
  if (!lut->kernel_abi() || lut->kernel_abi()->str() != kKernelAbi) {
    if (logOn())
      fprintf(stderr, "[gqa-lut] kernel_abi mismatch, ignoring table\n");
    return false;
  }
  if (lut->gpu_arch() && !lut->gpu_arch()->str().empty()) {
    const std::string actual = currentGpuArch();
    if (actual.empty() || actual != lut->gpu_arch()->str()) {
      if (logOn())
        fprintf(stderr, "[gqa-lut] arch \"%s\" != device \"%s\", ignoring\n",
                lut->gpu_arch()->c_str(), actual.c_str());
      return false;
    }
  }
  return true;
}

void loadBuffer(Table &t, const unsigned char *data, size_t size) {
  if (!data || size == 0) {
    if (logOn())
      fprintf(stderr, "[gqa-lut] no embedded table (size 0); shapes will use "
                      "the heuristic\n");
    return;
  }
  flatbuffers::Verifier verifier(data, size);
  if (!fbs::VerifyGqaAutotuneLutBuffer(verifier)) {
    if (logOn())
      fprintf(stderr, "[gqa-lut] buffer failed verification\n");
    return;
  }
  const fbs::GqaAutotuneLut *lut = fbs::GetGqaAutotuneLut(data);
  if (!lut || !lut->points() || !lut->configs() || !compatible(lut)) {
    if (logOn())
      fprintf(stderr, "[gqa-lut] table present (%zu bytes) but unusable\n",
              size);
    return;
  }
  if (!(lut->weight_num_heads() > 0.0f) ||
      !(lut->weight_kv_num_heads() > 0.0f) || !(lut->weight_batch() > 0.0f) ||
      !(lut->weight_seq_q() > 0.0f) || !(lut->weight_seq_kv() > 0.0f)) {
    if (logOn())
      fprintf(stderr, "[gqa-lut] non-positive metric weight\n");
    return;
  }
  t.w_num_heads = lut->weight_num_heads();
  t.w_kv_num_heads = lut->weight_kv_num_heads();
  t.w_batch = lut->weight_batch();
  t.w_seq_q = lut->weight_seq_q();
  t.w_seq_kv = lut->weight_seq_kv();

  const auto *configs = lut->configs();
  t.pool.reserve(configs->size());
  for (const fbs::GqaTuneConfig *c : *configs) {
    Answer a{};
    a.kind = c->kind();
    if (a.kind == fbs::GqaConfigKind::Decode) {
      a.use_wmma = c->use_wmma() != 0;
      a.bkv = c->bkv();
    } else {
      a.prefill =
          GqaPrefillConfig{c->m_tiles(), c->bkv(), c->nw(), c->mt(), c->nd()};
    }
    t.pool.push_back(a);
  }

  for (const fbs::GqaTunePoint *p : *lut->points()) {
    const unsigned idx = p->config();
    if (idx >= t.pool.size() || !pointConsistent(*p, t.pool[idx].kind)) {
      ++t.invalid_points;
      continue;
    }
    Point point;
    point.num_heads = p->num_heads();
    point.kv_num_heads = p->kv_num_heads();
    point.batch = p->batch();
    point.seq_q = p->seq_q();
    point.seq_kv = p->seq_kv();
    point.l_num_heads = std::log2(static_cast<float>(point.num_heads));
    point.l_kv_num_heads = std::log2(static_cast<float>(point.kv_num_heads));
    point.l_batch = std::log2(static_cast<float>(point.batch));
    point.l_seq_q = std::log2(static_cast<float>(point.seq_q));
    point.l_seq_kv = std::log2(static_cast<float>(point.seq_kv));
    point.config = static_cast<uint16_t>(idx);
    point.splits = p->splits();
    t.exact_geom[geomKey(p->phase(), p->head_dim(), p->kv_dtype(),
                         point.num_heads, point.kv_num_heads)]
        .push_back(point);
    // Also index under the heads-per-group ratio for the fuzzy fallback. A
    // ratio the kernels are not templated for (hpg 0) has no fuzzy group; it
    // can still be found by exact geometry.
    const unsigned hpg = headsPerGroup(static_cast<int>(point.num_heads),
                                       static_cast<int>(point.kv_num_heads));
    if (hpg != 0)
      t.by_hpg[hpgKey(p->phase(), p->head_dim(), p->kv_dtype(), hpg)].push_back(
          point);
    ++t.points;
  }

  if (lut->fallbacks()) {
    for (const fbs::GqaTuneFallback *f : *lut->fallbacks()) {
      const unsigned idx = f->config();
      if (idx >= t.pool.size() || f->phase() == fbs::GqaPhase::Any ||
          f->head_dim() == fbs::GqaHeadDim::Any)
        continue;
      const bool decode = f->phase() == fbs::GqaPhase::Decode;
      if (t.pool[idx].kind !=
          (decode ? fbs::GqaConfigKind::Decode : fbs::GqaConfigKind::Prefill))
        continue;
      t.fallbacks.emplace(
          fallbackKey(f->phase(), f->head_dim()),
          std::make_pair(static_cast<uint16_t>(idx), f->splits()));
    }
  }

  t.loaded = true;
  if (logOn())
    fprintf(stderr,
            "[gqa-lut] loaded %u points in %zu geometries (%u rejected), "
            "for %s\n",
            t.points, t.exact_geom.size(), t.invalid_points,
            lut->gpu_arch() ? lut->gpu_arch()->c_str() : "?");
}

} // namespace
} // namespace hipdnn_ep

// Emitted by cmake/xxd.py from lut/<arch>.fb; see lib/Runtime/Kernels/
// CMakeLists.txt. Arch-neutral on purpose: each custom_kernels_<arch> DLL links
// exactly one such payload (its own arch's table, or an empty stub), so this
// one reference resolves whatever arch the DLL is built for.
extern "C" const unsigned char kGqaLutData[];
extern "C" const size_t kGqaLutData_size;

namespace hipdnn_ep {
namespace {

Table &table() {
  static Table *t = [] {
    auto *fresh = new Table();
    loadBuffer(*fresh, kGqaLutData, kGqaLutData_size);
    return fresh;
  }();
  return *t;
}

struct Policy {
  GqaAutotuneMode mode = GqaAutotuneMode::Lookup;
  int compute_units = 0;
};

const char *modeName(GqaAutotuneMode mode) {
  return mode == GqaAutotuneMode::Online ? "online (GPU benchmark, table "
                                           "bypassed)"
                                         : "lookup (offline table)";
}

constexpr const char *kDefaultModeSource = "the default";

struct ModeChoice {
  GqaAutotuneMode mode = GqaAutotuneMode::Lookup;
  const char *source = kDefaultModeSource;
};

// HIPDNN_GQA_AUTOTUNE_MODE outranks the gqa_autotune_mode provider option, which
// outranks the default. An unrecognised value says so rather than quietly
// handing the session to the lever below it: the point of the switch is to
// compare the two paths, so silently running the other one is the failure worth
// a message.
ModeChoice chooseMode(const char *provider_mode) {
#ifdef _WIN32
  char buf[16];
  DWORD n =
      GetEnvironmentVariableA("HIPDNN_GQA_AUTOTUNE_MODE", buf, sizeof(buf));
  std::string raw = (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : "";
#else
  const char *v = getenv("HIPDNN_GQA_AUTOTUNE_MODE");
  std::string raw = v ? std::string(v) : "";
#endif
  const char *source = "HIPDNN_GQA_AUTOTUNE_MODE";
  if (raw.empty()) {
    raw = provider_mode ? provider_mode : "";
    source = "gqa_autotune_mode (provider option)";
  }
  std::string mode;
  for (char c : raw)
    if (c > ' ')
      mode.push_back(c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c);
  if (mode == "lookup")
    return {GqaAutotuneMode::Lookup, source};
  if (mode == "online")
    return {GqaAutotuneMode::Online, source};
  if (!mode.empty() && logOn())
    fprintf(stderr,
            "[gqa-lut] unrecognised %s=\"%s\" (expected \"lookup\" or "
            "\"online\"); using the default, %s\n",
            source, raw.c_str(), modeName(GqaAutotuneMode::Lookup));
  return {GqaAutotuneMode::Lookup, kDefaultModeSource};
}

// The nearest point in the group, plus whether it was accepted by the decode
// validator. Returns the pool index and squared distance through out-params.
// `decode` requests validate+clamp the candidate; prefill accepts any point in
// its own phase group. On rejection the search walks outwards to the next
// point.
bool nearestUsable(const std::vector<Point> &pts, float qnh, float qkh,
                   float qb, float qsq, float qsk, const Table &t,
                   const GqaDecodeRequest *decode_req, size_t &chosen,
                   float &best_d2) {
  auto dist2 = [&](const Point &p) {
    const float d0 = t.w_num_heads * (qnh - p.l_num_heads);
    const float d1 = t.w_kv_num_heads * (qkh - p.l_kv_num_heads);
    const float d2b = t.w_batch * (qb - p.l_batch);
    const float d3 = t.w_seq_q * (qsq - p.l_seq_q);
    const float d4 = t.w_seq_kv * (qsk - p.l_seq_kv);
    return d0 * d0 + d1 * d1 + d2b * d2b + d3 * d3 + d4 * d4;
  };
  auto usable = [&](const Point &p) {
    if (!decode_req)
      return t.pool[p.config].kind == fbs::GqaConfigKind::Prefill;
    const Answer &a = t.pool[p.config];
    if (a.kind != fbs::GqaConfigKind::Decode)
      return false;
    const GqaDecodeConfig cfg = clampDecodeConfig(
        *decode_req, GqaDecodeConfig{a.use_wmma, p.splits, a.bkv});
    return validDecodeConfig(*decode_req, cfg);
  };

  size_t best = 0;
  float bd2 = dist2(pts[0]);
  for (size_t i = 1; i < pts.size(); ++i) {
    const float d2 = dist2(pts[i]);
    if (d2 < bd2) {
      bd2 = d2;
      best = i;
    }
  }
  if (usable(pts[best])) {
    chosen = best;
    best_d2 = bd2;
    return true;
  }
  // Nearest was rejected: order the rest and walk outwards.
  std::vector<std::pair<float, size_t>> order;
  order.reserve(pts.size());
  for (size_t i = 0; i < pts.size(); ++i)
    if (i != best)
      order.emplace_back(dist2(pts[i]), i);
  std::sort(order.begin(), order.end());
  for (const auto &cand : order) {
    if (usable(pts[cand.second])) {
      chosen = cand.second;
      best_d2 = cand.first;
      return true;
    }
  }
  return false;
}

} // namespace
} // namespace hipdnn_ep

// ---------------------------------------------------------------------------
// C-ABI (exported from custom_kernels_<arch>)
// ---------------------------------------------------------------------------

extern "C" {

void *hip_gqa_autotune_create(const char *provider_mode) {
  auto *p = new hipdnn_ep::Policy();
  const hipdnn_ep::ModeChoice choice = hipdnn_ep::chooseMode(provider_mode);
  p->mode = choice.mode;
  p->compute_units = hipdnn_ep::currentComputeUnits();
  // Says which lever decided, not just what the mode is: an override that did
  // not take otherwise looks identical to the default.
  if (hipdnn_ep::logOn())
    fprintf(stderr, "[gqa-lut] mode %s (from %s)\n",
            hipdnn_ep::modeName(p->mode), choice.source);
  // Force the table to load now so HIPDNN_GQA_LUT_LOG reports at session start,
  // not on the first dispatch.
  (void)hipdnn_ep::table();
  return p;
}

void hip_gqa_autotune_destroy(void *policy) {
  delete static_cast<hipdnn_ep::Policy *>(policy);
}

int hip_gqa_autotune_mode(const void *policy) {
  if (!policy)
    return static_cast<int>(hipdnn_ep::GqaAutotuneMode::Lookup);
  return static_cast<int>(static_cast<const hipdnn_ep::Policy *>(policy)->mode);
}

void hip_gqa_autotune_resolve_decode(void *policy,
                                     const hipdnn_ep::GqaDecodeRequest *req,
                                     hipdnn_ep::GqaDecodeResult *out) {
  using namespace hipdnn_ep;
  const int cus = policy ? static_cast<Policy *>(policy)->compute_units : 0;
  Table &t = table();

  const fbs::GqaHeadDim hd = headDimClass(req->head_dim);
  if (t.loaded && hd != fbs::GqaHeadDim::Any) {
    const fbs::GqaKvDtype dt = kvDtypeClass(req->kv_dtype);
    const int eff = decodeEffectiveLen(*req);
    const float qnh = std::log2(float(std::max(req->num_heads, 1)));
    const float qkh = std::log2(float(std::max(req->kv_num_heads, 1)));
    const float qb = std::log2(float(std::max(req->batch, 1)));
    const float qsq = 0.0f; // decode is single-query; log2(1) = 0
    const float qsk = std::log2(float(std::max(eff, 1)));

    // Geometry tier: exact (num_heads, kv_num_heads) first, then same-hpg.
    const std::vector<Point> *group = nullptr;
    const auto eit = t.exact_geom.find(
        geomKey(decodePhase(), hd, dt, unsigned(std::max(req->num_heads, 0)),
                unsigned(std::max(req->kv_num_heads, 0))));
    if (eit != t.exact_geom.end()) {
      group = &eit->second;
    } else {
      const unsigned hpg = headsPerGroup(req->num_heads, req->kv_num_heads);
      if (hpg != 0) {
        const auto hit = t.by_hpg.find(hpgKey(decodePhase(), hd, dt, hpg));
        if (hit != t.by_hpg.end())
          group = &hit->second;
      }
    }
    if (group) {
      size_t chosen = 0;
      float best_d2 = 0.0f;
      if (nearestUsable(*group, qnh, qkh, qb, qsq, qsk, t, req, chosen,
                        best_d2)) {
        const Point &p = (*group)[chosen];
        const Answer &a = t.pool[p.config];
        const GqaDecodeConfig cfg = clampDecodeConfig(
            *req, GqaDecodeConfig{a.use_wmma, p.splits, a.bkv});
        const bool exact = best_d2 == 0.0f;
        out->config = cfg;
        out->source = exact ? GqaTuneSource::Exact : GqaTuneSource::Nearest;
        out->distance = std::sqrt(best_d2);
        if (logOn()) {
          char buf[192];
          snprintf(buf, sizeof(buf),
                   "[gqa-lut] decode %s H=%d G=%d d=%d skv=%d -> wmma=%d "
                   "splits=%d bkv=%d d=%.3f",
                   exact ? "exact" : "nearest", req->num_heads,
                   req->kv_num_heads, req->head_dim, eff, cfg.use_wmma,
                   cfg.splits, cfg.bkv, out->distance);
          if (logLineFirstSeen(buf))
            fprintf(stderr, "%s\n", buf);
        }
        return;
      }
      t.rejected.fetch_add(1, std::memory_order_relaxed);
    }

    const auto fit = t.fallbacks.find(fallbackKey(decodePhase(), hd));
    if (fit != t.fallbacks.end()) {
      const Answer &a = t.pool[fit->second.first];
      if (a.kind == fbs::GqaConfigKind::Decode) {
        const GqaDecodeConfig cfg = clampDecodeConfig(
            *req, GqaDecodeConfig{a.use_wmma, fit->second.second, a.bkv});
        if (validDecodeConfig(*req, cfg)) {
          out->config = cfg;
          out->source = GqaTuneSource::Fallback;
          out->distance = 0.0f;
          if (logOn()) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "[gqa-lut] decode fallback H=%d G=%d d=%d skv=%d -> "
                     "wmma=%d splits=%d bkv=%d",
                     req->num_heads, req->kv_num_heads, req->head_dim, eff,
                     cfg.use_wmma, cfg.splits, cfg.bkv);
            if (logLineFirstSeen(buf))
              fprintf(stderr, "%s\n", buf);
          }
          return;
        }
      }
    }
  }

  out->config = decodeHeuristic(*req, cus > 0 ? cus : kAssumedCus);
  out->source = GqaTuneSource::Heuristic;
  out->distance = 0.0f;
  if (logOn()) {
    char buf[192];
    snprintf(buf, sizeof(buf),
             "[gqa-lut] decode heuristic H=%d G=%d d=%d skv=%d -> wmma=%d "
             "splits=%d bkv=%d",
             req->num_heads, req->kv_num_heads, req->head_dim,
             decodeEffectiveLen(*req), out->config.use_wmma, out->config.splits,
             out->config.bkv);
    if (logLineFirstSeen(buf))
      fprintf(stderr, "%s\n", buf);
  }
}

void hip_gqa_autotune_resolve_prefill(void *policy,
                                      const hipdnn_ep::GqaPrefillRequest *req,
                                      hipdnn_ep::GqaPrefillResult *out) {
  using namespace hipdnn_ep;
  (void)policy;
  Table &t = table();

  const fbs::GqaHeadDim hd = headDimClass(req->head_dim);
  if (t.loaded && hd != fbs::GqaHeadDim::Any) {
    const fbs::GqaPhase phase = prefillPhase(req->variant);
    const fbs::GqaKvDtype dt = fbs::GqaKvDtype::Fp16;
    const float qnh = std::log2(float(std::max(req->num_heads, 1)));
    const float qkh = std::log2(float(std::max(req->kv_num_heads, 1)));
    const float qb = std::log2(float(std::max(req->batch, 1)));
    const float qsq = std::log2(float(std::max(req->seq_q, 1)));
    const float qsk = std::log2(float(std::max(req->seq_kv, 1)));

    // Geometry tier: exact (num_heads, kv_num_heads) first, then same-hpg.
    const std::vector<Point> *group = nullptr;
    const auto eit = t.exact_geom.find(
        geomKey(phase, hd, dt, unsigned(std::max(req->num_heads, 0)),
                unsigned(std::max(req->kv_num_heads, 0))));
    if (eit != t.exact_geom.end()) {
      group = &eit->second;
    } else {
      const unsigned hpg = headsPerGroup(req->num_heads, req->kv_num_heads);
      if (hpg != 0) {
        const auto hit = t.by_hpg.find(hpgKey(phase, hd, dt, hpg));
        if (hit != t.by_hpg.end())
          group = &hit->second;
      }
    }
    if (group) {
      size_t chosen = 0;
      float best_d2 = 0.0f;
      if (nearestUsable(*group, qnh, qkh, qb, qsq, qsk, t,
                        /*decode_req=*/nullptr, chosen, best_d2)) {
        const Point &p = (*group)[chosen];
        const bool exact = best_d2 == 0.0f;
        out->config = t.pool[p.config].prefill;
        out->source = exact ? GqaTuneSource::Exact : GqaTuneSource::Nearest;
        out->distance = std::sqrt(best_d2);
        if (logOn()) {
          char buf[192];
          snprintf(buf, sizeof(buf),
                   "[gqa-lut] prefill %s H=%d G=%d d=%d sq=%d skv=%d d=%.3f",
                   exact ? "exact" : "nearest", req->num_heads,
                   req->kv_num_heads, req->head_dim, req->seq_q, req->seq_kv,
                   out->distance);
          if (logLineFirstSeen(buf))
            fprintf(stderr, "%s\n", buf);
        }
        return;
      }
    }

    const auto fit = t.fallbacks.find(fallbackKey(phase, hd));
    if (fit != t.fallbacks.end() &&
        t.pool[fit->second.first].kind == fbs::GqaConfigKind::Prefill) {
      out->config = t.pool[fit->second.first].prefill;
      out->source = GqaTuneSource::Fallback;
      out->distance = 0.0f;
      if (logOn()) {
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "[gqa-lut] prefill fallback H=%d G=%d d=%d sq=%d skv=%d",
                 req->num_heads, req->kv_num_heads, req->head_dim, req->seq_q,
                 req->seq_kv);
        if (logLineFirstSeen(buf))
          fprintf(stderr, "%s\n", buf);
      }
      return;
    }
  }

  out->config = prefillHeuristic(*req);
  out->source = GqaTuneSource::Heuristic;
  out->distance = 0.0f;
  if (logOn()) {
    char buf[192];
    snprintf(buf, sizeof(buf),
             "[gqa-lut] prefill heuristic H=%d G=%d d=%d sq=%d skv=%d",
             req->num_heads, req->kv_num_heads, req->head_dim, req->seq_q,
             req->seq_kv);
    if (logLineFirstSeen(buf))
      fprintf(stderr, "%s\n", buf);
  }
}

void hip_gqa_autotune_fallback_prefill(const hipdnn_ep::GqaPrefillRequest *req,
                                       hipdnn_ep::GqaPrefillConfig *out) {
  *out = hipdnn_ep::prefillHeuristic(*req);
}

int hip_gqa_autotune_table_loaded() {
  return hipdnn_ep::table().loaded ? 1 : 0;
}

uint32_t hip_gqa_autotune_point_count() { return hipdnn_ep::table().points; }

uint32_t hip_gqa_autotune_invalid_points() {
  return hipdnn_ep::table().invalid_points;
}

} // extern "C"
