/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The whole GQA autotune policy. Every tier is a row in the offline table, so
// nothing about which config to run is decided here: this file turns a request
// into keys, probes them in order, and checks that what came back can run.
//
// The keys are what changed in schema 5. A key used to carry the request's
// exact head counts and batch, which meant a model whose (num_heads,
// kv_num_heads) pair was never measured could only be answered by the
// last-resort row, and a batched deployment always was. It now carries
// heads-per-group and, optionally, a bucket of batch*num_heads. That set is
// closed -- flash_decode_geometry_ok admits heads-per-group in {1,2,3,4,5,8,16}
// at head_dim in {64,128,256} -- so a complete table has a row for every
// geometry that can reach it. Measurement puts the cost of pooling head counts
// at one percentage point of shapes within 5% of optimum and nothing on the
// worst case; see RdpCapture/ops_analyze/gqa/.
//
// This is host code compiled into the runtime bitcode, not into the per-arch
// kernel library -- it lives here because it belongs to the GQA autotune story,
// not because it is device code. lib/Runtime/CMakeLists.txt compiles it.
#include "gqa_autotune.h"

#include "gqa_autotune_generated.h"
// debug_log.h goes by explicit relative path: Kernels/include/ ships its own
// debug_log.h, so an unqualified include would resolve to the kernel-side one
// depending on -I order. runtime_types.h deliberately does NOT: real/ and mock/
// each ship one and the -I flag selects the right copy.
#include "../../../../debug_log.h"
#if !defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
#include "runtime_types.h"
#endif

#include "morphizen-foundation/file_io.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_ep {
namespace {

namespace fbs = mlir::hip;

// 2: bucket labels gained the midpoints between powers of two, decode split
//    counts are clamped to the splits that have work, and prefill v5 keys on
//    seq_kv like the other variants.
// 3: the last resort moved into the table as Fallback rows, so every tier is
// data. 4: rows keyed on heads-per-group, which answer a q:kv ratio the table
// knows at
//    other head counts.
// 5: one flat vector of 12-byte rows instead of nested tables of int32 (61
// bytes a
//    row); heads-per-group and a batch*num_heads bucket replace the exact head
//    counts and batch; the length-exact tier is gone, measured to answer
//    nothing a bucket row does not (a decode length is essentially never a
//    bucket label, and on the boundary shapes where it is, the bucket row
//    already lands on the optimum); max_seq is gone, measured to move every
//    candidate together; and a config is one named point of the launch space
//    instead of seven free ints, so an illegal combination cannot be spelled.
constexpr uint32_t kGqaLutSchemaVersion = 5;
constexpr size_t kMaxLutBytes = 64u * 1024u * 1024u;

// What a row resolves to once the config name has been expanded.
struct Answer {
  bool use_wmma = false;
  int splits = 0;
  int bkv = 16;
  GqaPrefillConfig prefill{};
};

// Compute units on this part, read once. The fallback split count is a function
// of it, which is what lets that fallback follow the hardware instead of being
// a number somebody picked on one GPU.
static int currentComputeUnits() {
#if defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
  return 0;
#else
  int device = 0;
  hipDeviceProp_t prop{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&prop, device) != hipSuccess)
    return 0;
  return prop.multiProcessorCount;
#endif
}

struct GqaAutotunePolicy {
  GqaAutotuneMode mode = GqaAutotuneMode::Lookup;
  // One map, keyed on the packed row key. The tier is part of the key, so the
  // probe order is a loop over four keys rather than a walk over four
  // containers.
  std::unordered_map<uint64_t, Answer> rows;
  std::atomic<uint64_t> invalid_entries{0};
  // Read once at creation; 0 means the query failed and the fallback assumes a
  // part this size.
  int compute_units = 0;
};

// ---- key encoding ----------------------------------------------------------

// The fields are all small enums; 33 bits of them fit in one word, so a lookup
// is one hash of one integer.
static uint64_t packKey(fbs::GqaTunePhase phase, fbs::GqaTuneTier tier,
                        fbs::GqaTuneKvDtype kv_dtype,
                        fbs::GqaTuneHeadDim head_dim, unsigned hpg,
                        fbs::GqaParClass par, fbs::GqaBatchClass batch,
                        fbs::GqaSeqBucket seq_q, fbs::GqaSeqBucket seq_kv,
                        fbs::GqaWindowClass window) {
  return (static_cast<uint64_t>(phase) & 0x3ull) |
         ((static_cast<uint64_t>(tier) & 0x3ull) << 2) |
         ((static_cast<uint64_t>(kv_dtype) & 0x3ull) << 4) |
         ((static_cast<uint64_t>(head_dim) & 0x3ull) << 6) |
         ((static_cast<uint64_t>(hpg) & 0x1Full) << 8) |
         ((static_cast<uint64_t>(par) & 0xFull) << 13) |
         ((static_cast<uint64_t>(seq_q) & 0x7Full) << 17) |
         ((static_cast<uint64_t>(seq_kv) & 0x7Full) << 24) |
         ((static_cast<uint64_t>(window) & 0xFull) << 31) |
         ((static_cast<uint64_t>(batch) & 0xFull) << 35);
}

static int normalizeWindow(int local_window) {
  return local_window > 0 ? local_window : 0;
}

static int nextPowerOfTwo(int value) {
  if (value <= 1)
    return 1;
  if (value > (INT_MAX / 2))
    return INT_MAX;
  uint32_t v = static_cast<uint32_t>(value - 1);
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return static_cast<int>(v + 1);
}

static int log2Exact(int power_of_two) {
  int k = 0;
  while ((1 << (k + 1)) <= power_of_two && k < 30)
    ++k;
  return k;
}

// Round a length up to a bucket label and name that label.
//
// The labels are four to the octave -- 1, 1.25, 1.5, 1.75 times each power of
// two
// -- so a length in (L, 2L] for L = 2^k lands on L + i*(L/4) for the smallest i
// that reaches it, and the row labelled S12288 answers every request in
// (10240, 12288]. The enum's values ascend with the label, which is what lets
// this be arithmetic instead of a table: the four labels of the octave above
// 2^k hold codes 4k-3 .. 4k.
//
// A length above the longest label saturates rather than missing. A 512 k
// context is not a shape anyone runs on this part, and answering it with the
// longest row measured is better than dropping it to the last resort.
static fbs::GqaSeqBucket seqBucket(int length) {
  // Below 5 the quarter steps are not integers, so the first four labels are
  // named directly. Their codes are their labels.
  if (length <= 1)
    return fbs::GqaSeqBucket::S1;
  if (length <= 4)
    return static_cast<fbs::GqaSeqBucket>(length);
  const int k =
      log2Exact(nextPowerOfTwo(length)) - 1; // length is in (2^k, 2^k+1]
  const int lower = 1 << k;
  const int step = lower >> 2;
  const int index = (length - lower + step - 1) / step; // 1..4
  const int code = 4 * k - 4 + index;
  const int top = static_cast<int>(fbs::GqaSeqBucket::S262144);
  return static_cast<fbs::GqaSeqBucket>(code < top ? code : top);
}

// The window, rounded up to a power of two. Windows are powers of two in every
// model that has one (128 on gpt-oss, 4096 on Mistral), and a row's label means
// "windows up to this", so a value in between rounds up like a length does.
static fbs::GqaWindowClass windowClass(int local_window) {
  if (normalizeWindow(local_window) == 0)
    return fbs::GqaWindowClass::NoWindow;
  const int k = log2Exact(nextPowerOfTwo(local_window));
  const int code = 2 + (k - 7); // W128 is 2
  const int lowest = static_cast<int>(fbs::GqaWindowClass::W128);
  const int top = static_cast<int>(fbs::GqaWindowClass::W65536);
  return static_cast<fbs::GqaWindowClass>(
      code < lowest ? lowest : (code > top ? top : code));
}

// batch * num_heads, rounded up to a power of two: how many independent work
// items the launch has.
static fbs::GqaParClass parClass(int batch, int num_heads) {
  const long long work = static_cast<long long>(batch > 0 ? batch : 1) *
                         static_cast<long long>(num_heads > 0 ? num_heads : 1);
  const int clamped = work > INT_MAX / 2 ? INT_MAX / 2 : static_cast<int>(work);
  const int code = 1 + log2Exact(nextPowerOfTwo(clamped));
  const int top = static_cast<int>(fbs::GqaParClass::P8192);
  return static_cast<fbs::GqaParClass>(code > top ? top : code);
}

// The batch size on its own, rounded up to a power of two. The parallelism
// class above is the product with the head count, and the product is not
// enough: two requests with the same product and different batches disagree
// about WMMA. See GqaBatchClass in the schema.
static fbs::GqaBatchClass batchClass(int batch) {
  const int code = 1 + log2Exact(nextPowerOfTwo(batch > 0 ? batch : 1));
  const int top = static_cast<int>(fbs::GqaBatchClass::B512);
  return static_cast<fbs::GqaBatchClass>(code > top ? top : code);
}

static fbs::GqaTuneHeadDim headDimClass(int head_dim) {
  switch (head_dim) {
  case 64:
    return fbs::GqaTuneHeadDim::D64;
  case 128:
    return fbs::GqaTuneHeadDim::D128;
  case 256:
    return fbs::GqaTuneHeadDim::D256;
  default:
    return fbs::GqaTuneHeadDim::Any;
  }
}

static fbs::GqaTuneKvDtype kvDtypeClass(int kv_dtype) {
  return kv_dtype == 1 ? fbs::GqaTuneKvDtype::Int8 : fbs::GqaTuneKvDtype::Fp16;
}

static unsigned headsPerGroup(int num_heads, int kv_num_heads) {
  if (kv_num_heads <= 0 || num_heads <= 0 || num_heads % kv_num_heads != 0)
    return 0;
  const int hpg = num_heads / kv_num_heads;
  return hpg > 16 ? 0 : static_cast<unsigned>(hpg);
}

// ---- config names ----------------------------------------------------------

// The knobs a named prefill config stands for, and the variant allowed to name
// it. A name that is not here is not a config: `nd = 4` with `bkv = 64` has no
// name, so no table can ask for it and the loader needs no per-knob rules.
static bool prefillKnobs(fbs::GqaTuneConfig name, GqaPrefillVariant *variant,
                         GqaPrefillConfig *out) {
  using C = fbs::GqaTuneConfig;
  const auto v5 = [&](int m_tiles, int bkv) {
    *variant = GqaPrefillVariant::V5;
    *out = GqaPrefillConfig{m_tiles, bkv, 0, 0, 0};
    return true;
  };
  const auto v7 = [&](int nw, int bkv, int mt) {
    *variant = GqaPrefillVariant::V7;
    *out = GqaPrefillConfig{0, bkv, nw, mt, 0};
    return true;
  };
  const auto v8 = [&](int nd, int mt, int bkv) {
    *variant = GqaPrefillVariant::V8;
    *out = GqaPrefillConfig{0, bkv, 0, mt, nd};
    return true;
  };
  switch (name) {
  case C::MT1_BKV32:
    return v5(1, 32);
  case C::MT1_BKV64:
    return v5(1, 64);
  case C::MT2_BKV32:
    return v5(2, 32);
  case C::MT2_BKV64:
    return v5(2, 64);
  case C::NW1_BKV32_MT1:
    return v7(1, 32, 1);
  case C::NW1_BKV32_MT2:
    return v7(1, 32, 2);
  case C::NW1_BKV64_MT1:
    return v7(1, 64, 1);
  case C::NW1_BKV64_MT2:
    return v7(1, 64, 2);
  case C::NW2_BKV32_MT1:
    return v7(2, 32, 1);
  case C::NW2_BKV32_MT2:
    return v7(2, 32, 2);
  case C::NW2_BKV64_MT1:
    return v7(2, 64, 1);
  case C::NW2_BKV64_MT2:
    return v7(2, 64, 2);
  case C::NW4_BKV32_MT1:
    return v7(4, 32, 1);
  case C::NW4_BKV32_MT2:
    return v7(4, 32, 2);
  case C::NW4_BKV64_MT1:
    return v7(4, 64, 1);
  case C::NW4_BKV64_MT2:
    return v7(4, 64, 2);
  case C::ND2_MT1_BKV32:
    return v8(2, 1, 32);
  case C::ND2_MT1_BKV64:
    return v8(2, 1, 64);
  case C::ND2_MT2_BKV32:
    return v8(2, 2, 32);
  case C::ND2_MT2_BKV64:
    return v8(2, 2, 64);
  case C::ND4_MT1_BKV32:
    return v8(4, 1, 32);
  case C::ND4_MT2_BKV32:
    return v8(4, 2, 32);
  default:
    return false;
  }
}

static fbs::GqaTunePhase phaseOf(GqaPrefillVariant variant) {
  switch (variant) {
  case GqaPrefillVariant::V5:
    return fbs::GqaTunePhase::PrefillV5;
  case GqaPrefillVariant::V7:
    return fbs::GqaTunePhase::PrefillV7;
  case GqaPrefillVariant::V8:
    return fbs::GqaTunePhase::PrefillV8;
  }
  return fbs::GqaTunePhase::PrefillV8;
}

// ---- what a config needs to be runnable ------------------------------------

static bool wmmaSupported(int head_dim, unsigned heads_per_group) {
  if (heads_per_group == 4)
    return head_dim == 64 || head_dim == 128;
  return heads_per_group == 8 && head_dim == 64;
}

static bool decodeWmmaConfig(fbs::GqaTuneConfig config) {
  using C = fbs::GqaTuneConfig;
  return config == C::Wmma || config == C::WmmaBkv16 ||
         config == C::WmmaBkv32;
}

static int decodeEffectiveLen(const GqaDecodeRequest &request) {
  int effective_len = std::max(request.effective_skv, 1);
  if (request.local_window > 0)
    effective_len = std::min(effective_len, request.local_window);
  return effective_len;
}

// Splits that have work to do: the split-K decode gives each block a 16-key
// tile, so beyond this the extra blocks scan nothing and only add reduce rows.
// Every other config source already respects it -- flashDecodeCandidateSplits
// clamps the tuner's ladder and decodeHeuristic clamps its own answer -- so a
// table row is held to the same rule.
//
// This is what lets one bucket row cover a whole interval at short context,
// where the measured optimum *is* this bound: on 64:8:64 the winner at
// 132/160/192/224/ 256 keys is exactly 9/10/12/14/16 splits. Without the clamp
// a row would have to hold the value its shortest length tolerates (8), which
// costs 17% at the top of the interval; with it, one row holds 16 and lands on
// the optimum throughout.
static int usefulSplits(const GqaDecodeRequest &request) {
  return std::max(1, (decodeEffectiveLen(request) + 15) / 16);
}

static GqaDecodeConfig clampDecodeConfig(const GqaDecodeRequest &request,
                                         GqaDecodeConfig config) {
  config.splits = std::min(config.splits, usefulSplits(request));
  if (config.splits < 1)
    config.splits = 1;
  return config;
}

static bool validDecodeConfig(const GqaDecodeRequest &request,
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
  // value is a real LUT choice. Both paths accept only compiled tile heights.
  return config.bkv == 16 || config.bkv == 32;
}

// Blocks the part can have in flight at once, per compute unit. Fitted over
// 4722 measured decode shapes as the single constant that best explains their
// optimal split counts; 17 is what came out on gfx1151, where occupancy per
// kernel instantiation actually ranges from about 8 to 40. The right value is
// the one hipOccupancyMaxActiveBlocksPerMultiprocessor reports for the kernel
// about to be launched, which is what llama.cpp queries -- but this file is
// host policy code with no device symbols in scope, so it uses the fitted
// constant and the CU count the driver reports.
constexpr int kBlocksPerCu = 17;
constexpr int kAssumedCus = 20;

// The split count that fills the machine without opening another wave.
//
// This is llama.cpp's rule, from launch_fattn in ggml/src/ggml-cuda/
// fattn-common.cuh: walk the candidate split counts, keep the one with the best
// ratio of blocks issued to blocks a wave can hold, and stop once that is good
// enough to not be worth another wave. It needs no measurement and it follows
// the hardware, which is exactly what a last resort should do.
//
// Measured against the fixed `min(8, useful)` it replaces, over every decode
// shape in the grid: median 91.0% of optimum to 96.5%, p10 48.9% to 81.4%,
// share within 10% of optimum 52% to 75%. It is *not* better than the table --
// the table is at 96% within 10% -- because occupancy is not the only term:
// every extra split re-reads the KV and adds a row to the reduce. That is why
// this is the fallback and the table is the policy.
static int occupancySplits(const GqaDecodeRequest &request, int cus) {
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

static GqaDecodeConfig decodeHeuristic(const GqaDecodeRequest &request,
                                       int cus) {
  const int cap = std::max(1, std::min(request.max_splits, 64));
  // WMMA is templated for some (head_dim, heads-per-group) pairs only, and this
  // path runs when nothing is known, so it stays on the kernel that always
  // exists.
  return {/*use_wmma=*/false,
          /*splits=*/std::max(1, std::min(cap, occupancySplits(request, cus))),
          /*bkv=*/16};
}

static GqaPrefillConfig prefillHeuristic(const GqaPrefillRequest &request) {
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

// ---- loading ---------------------------------------------------------------

// A row has to mean exactly what its tier says it means. Wildcarding a field
// the tier is supposed to key on would make the row answer far more than it was
// measured on, and setting a field the tier ignores would make it answer far
// less than it looks like it does; both are silent, so both are rejected.
static bool rowConsistent(const fbs::GqaTuneRow &row) {
  const bool decode = row.phase() == fbs::GqaTunePhase::Decode;
  const bool has_hpg = row.hpg() >= 1 && row.hpg() <= 16;
  const bool has_par = row.par() != fbs::GqaParClass::Any;
  const bool has_batch = row.batch() != fbs::GqaBatchClass::Any;
  const bool has_skv = row.seq_kv() != fbs::GqaSeqBucket::Any;
  const bool has_sq = row.seq_q() != fbs::GqaSeqBucket::Any;
  const bool has_window = row.window() != fbs::GqaWindowClass::Any;

  switch (row.tier()) {
  case fbs::GqaTuneTier::Fallback:
    // Nothing but phase, kv_dtype and head_dim, and head_dim may be Any too.
    return !has_hpg && !has_par && !has_batch && !has_skv && !has_sq &&
           !has_window;
  case fbs::GqaTuneTier::Length:
    // No geometry at all, so a decode row cannot name WMMA: it is templated for
    // some (head_dim, heads-per-group) pairs only, and this row will be handed
    // the pairs nobody measured.
    return !has_hpg && !has_par && !has_batch && has_skv && has_window &&
           has_sq == !decode && row.head_dim() != fbs::GqaTuneHeadDim::Any &&
           (!decode || !decodeWmmaConfig(row.config()));
  case fbs::GqaTuneTier::HeadGroup:
    return has_hpg && !has_par && !has_batch && has_skv && has_window &&
           has_sq == !decode && row.head_dim() != fbs::GqaTuneHeadDim::Any;
  case fbs::GqaTuneTier::Geometry:
    // `batch` is optional here and nowhere else: a Geometry row may or may not
    // name a batch, and pruning keeps the batch-specific one only where it
    // disagrees with the batch-agnostic one.
    return has_hpg && has_par && has_skv && has_window && has_sq == !decode &&
           row.head_dim() != fbs::GqaTuneHeadDim::Any;
  }
  return false;
}

static bool rowAnswer(const fbs::GqaTuneRow &row, Answer *out) {
  if (row.phase() == fbs::GqaTunePhase::Decode) {
    using C = fbs::GqaTuneConfig;
    if (row.config() != C::Scalar && !decodeWmmaConfig(row.config()))
      return false;
    if (row.splits() < 1 || row.splits() > 64)
      return false;
    out->use_wmma = decodeWmmaConfig(row.config());
    out->splits = row.splits();
    // Wmma predates the tunable d64 tile height and remains the name of d128's
    // fixed-BKV=32 implementation. It maps to 16 here because the kernel ignores
    // the knob at d128; the explicit names carry d64's real choice.
    out->bkv = row.config() == C::WmmaBkv32 ? 32 : 16;
    return true;
  }
  if (row.splits() != 0)
    return false;
  GqaPrefillVariant variant = GqaPrefillVariant::V8;
  if (!prefillKnobs(row.config(), &variant, &out->prefill))
    return false;
  // The config's own variant has to be the row's phase, or the row would hand
  // v7 knobs to the v8 dispatcher.
  return phaseOf(variant) == row.phase();
}

static std::string currentGpuArch() {
#if defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
  return {};
#else
  int device = 0;
  hipDeviceProp_t prop{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&prop, device) != hipSuccess)
    return {};
  // gcnArchName may carry a feature suffix (gfx1151:xnack-) that the LUT keys
  // do not. Scanned with strchr because std::string::find lowers to MSVC's
  // vectorized helpers, which have no definition in JIT-linked bitcode.
  const char *arch = prop.gcnArchName;
  const char *feature_suffix = std::strchr(arch, ':');
  if (feature_suffix)
    return std::string(arch, feature_suffix);
  return std::string(arch);
#endif
}

static bool compatibleLut(const fbs::GqaAutotuneLut *lut) {
  if (lut->schema_version() != kGqaLutSchemaVersion) {
    fprintf(
        stderr,
        "GQA LUT ignored: schema version %u is not supported (expected %u)\n",
        lut->schema_version(), kGqaLutSchemaVersion);
    return false;
  }
  if (!lut->kernel_abi() || lut->kernel_abi()->str() != kGqaKernelAbi) {
    fprintf(stderr, "GQA LUT ignored: kernel ABI mismatch\n");
    return false;
  }
  if (lut->gpu_arch() && !lut->gpu_arch()->str().empty()) {
    const std::string actual = currentGpuArch();
    if (actual.empty() || actual != lut->gpu_arch()->str()) {
      fprintf(stderr, "GQA LUT ignored: GPU arch is %s, LUT requires %s\n",
              actual.empty() ? "<unknown>" : actual.c_str(),
              lut->gpu_arch()->c_str());
      return false;
    }
  }
  if (lut->rocm_version() != 0) {
#if defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
    fprintf(stderr,
            "GQA LUT ignored: GPU-free validation requires rocm_version=0\n");
    return false;
#else
    int actual = 0;
    if (hipRuntimeGetVersion(&actual) != hipSuccess ||
        actual != lut->rocm_version()) {
      fprintf(stderr,
              "GQA LUT ignored: ROCm runtime version is %d, LUT requires %d\n",
              actual, lut->rocm_version());
      return false;
    }
#endif
  }
  return true;
}

static bool loadLutBuffer(GqaAutotunePolicy &policy, const uint8_t *data,
                          size_t size) {
  flatbuffers::Verifier verifier(data, size);
  if (!fbs::VerifyGqaAutotuneLutBuffer(verifier)) {
    fprintf(stderr, "GQA LUT ignored: invalid FlatBuffer\n");
    return false;
  }
  const auto *lut = fbs::GetGqaAutotuneLut(data);
  if (!compatibleLut(lut))
    return false;
  if (!lut->rows())
    return true;

  size_t loaded = 0;
  for (const fbs::GqaTuneRow *row : *lut->rows()) {
    Answer answer;
    if (!row || !rowConsistent(*row) || !rowAnswer(*row, &answer)) {
      policy.invalid_entries.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    const uint64_t key = packKey(
        row->phase(), row->tier(), row->kv_dtype(), row->head_dim(), row->hpg(),
        row->par(), row->batch(), row->seq_q(), row->seq_kv(), row->window());
    // insert_or_assign, so a duplicate key is the last row rather than the
    // first. validate_lut_json.py rejects duplicates offline, because either
    // way one of the two rows silently does nothing.
    policy.rows.insert_or_assign(key, answer);
    ++loaded;
  }
  // model_key says which measurement campaign produced these rows. It is the only
  // way to tell two tables apart once one is packed, so it is logged rather than
  // carried unread.
  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] loaded GQA LUT: %zu rows (%zu invalid), built from %s\n",
      loaded,
      static_cast<size_t>(
          policy.invalid_entries.load(std::memory_order_relaxed)),
      lut->model_key() ? lut->model_key()->c_str() : "(no model_key)");
  return true;
}

static void loadLutFromFileSystem(GqaAutotunePolicy &policy,
                                  morphizen::FileSystem *fs) {
  if (!fs)
    return;
  const std::string &configured_filename =
      hipdnn_ep::env_string("HIPDNN_GQA_LUT_FILE");
  const char *filename = configured_filename.empty()
                             ? kGqaAutotuneFilename
                             : configured_filename.c_str();

  auto reader = fs->create_reader_template(filename);
  if (!reader) {
    RUNTIME_DEBUG_LOG("[Runtime DEBUG] no GQA LUT found at %s\n", filename);
    return;
  }
  const size_t size = reader->size();
  if (size == 0 || size > kMaxLutBytes) {
    fprintf(stderr, "GQA LUT ignored: invalid file size %zu\n", size);
    return;
  }
  if (void *mapped = reader->mmap()) {
    (void)loadLutBuffer(policy, static_cast<const uint8_t *>(mapped), size);
    return;
  }

  std::vector<uint8_t> bytes(size);
  reader->rewind();
  size_t offset = 0;
  while (offset < size) {
    const size_t n = reader->fread(bytes.data() + offset, size - offset);
    if (n == 0)
      break;
    offset += n;
  }
  if (offset != size) {
    fprintf(stderr, "GQA LUT ignored: short read (%zu of %zu bytes)\n", offset,
            size);
    return;
  }
  (void)loadLutBuffer(policy, bytes.data(), bytes.size());
}

// ---- probing ---------------------------------------------------------------

// Probes per resolve: the parallelism classes from the request's down to P1,
// then HeadGroup, Length and the two Fallback forms. All of them are misses in
// the worst case, and a miss is one hash of one 64-bit word.
constexpr int kMaxProbes = 2 * 14 + 4;

// One tier's key, with the fields that tier wildcards set to Any.
struct Probe {
  fbs::GqaTuneTier tier;
  GqaTuneSource source;
  fbs::GqaTuneHeadDim head_dim;
  unsigned hpg;
  fbs::GqaParClass par;
  fbs::GqaBatchClass batch;
  fbs::GqaSeqBucket seq_q;
  fbs::GqaSeqBucket seq_kv;
  fbs::GqaWindowClass window;
};

// The tiers, in the order they are tried, for a request that resolved to these
// buckets. Ordered by how much each gives up, which is also the order of
// decreasing measurement support -- see GqaTuneTier in the schema.
//
// The parallelism class is walked *downward* before the tier is given up on,
// the way a length is rounded up to a label: both round the request to a key
// the table has, in the direction that is safe. Down is the safe direction here
// because a row measured at less parallelism holds more splits, and too many
// splits costs the reduce step while too few leaves the machine idle. Measured
// on 105 held-out shapes that had a lower class to walk to, against giving up
// on the tier and taking the pooled row: the median gives up a point (97.1
// to 96.0) and the tail improves a lot (p10 84.9 to 89.8, worst 66.5 to 78.7,
// share within 10% of optimum 81% to 90%). Walking *up* was measured too and is
// worse than either.
static int buildProbes(fbs::GqaTuneHeadDim head_dim, unsigned hpg,
                       fbs::GqaParClass par, fbs::GqaBatchClass batch,
                       fbs::GqaSeqBucket seq_q, fbs::GqaSeqBucket seq_kv,
                       fbs::GqaWindowClass window, Probe *out) {
  const fbs::GqaParClass any_par = fbs::GqaParClass::Any;
  const fbs::GqaBatchClass any_batch = fbs::GqaBatchClass::Any;
  const fbs::GqaSeqBucket any_seq = fbs::GqaSeqBucket::Any;
  int n = 0;
  // A geometry with no heads-per-group -- head counts that do not divide, or a
  // ratio the kernels are not templated for -- skips the tiers keyed on it
  // rather than probing them with a zero, which is a Length row's key.
  if (hpg != 0) {
    for (int p = static_cast<int>(par);
         p >= static_cast<int>(fbs::GqaParClass::P1); --p) {
      // At each parallelism class, the row measured at this batch first and the
      // one measured over all of them second. Two requests with the same number
      // of work items and different batches want different configs often enough
      // to be worth a key field -- 8:1:64 at batch 32 and 64:8:64 at batch 4
      // are both 256 work items and disagree about WMMA in 7 shapes of 7 -- and
      // rarely enough that most classes ship only the shared row.
      const fbs::GqaBatchClass batches[2] = {batch, any_batch};
      for (fbs::GqaBatchClass b : batches)
        out[n++] = Probe{fbs::GqaTuneTier::Geometry,
                         GqaTuneSource::Geometry,
                         head_dim,
                         hpg,
                         static_cast<fbs::GqaParClass>(p),
                         b,
                         seq_q,
                         seq_kv,
                         window};
    }
    out[n++] = Probe{fbs::GqaTuneTier::HeadGroup,
                     GqaTuneSource::HeadGroup,
                     head_dim,
                     hpg,
                     any_par,
                     any_batch,
                     seq_q,
                     seq_kv,
                     window};
  }
  out[n++] = Probe{fbs::GqaTuneTier::Length,
                   GqaTuneSource::Length,
                   head_dim,
                   0,
                   any_par,
                   any_batch,
                   seq_q,
                   seq_kv,
                   window};
  out[n++] = Probe{fbs::GqaTuneTier::Fallback,
                   GqaTuneSource::Fallback,
                   head_dim,
                   0,
                   any_par,
                   any_batch,
                   any_seq,
                   any_seq,
                   fbs::GqaWindowClass::Any};
  // A Fallback row may leave head_dim Any as well, for a head_dim nobody
  // measured. Only this tier is allowed to.
  out[n++] = Probe{fbs::GqaTuneTier::Fallback,
                   GqaTuneSource::Fallback,
                   fbs::GqaTuneHeadDim::Any,
                   0,
                   any_par,
                   any_batch,
                   any_seq,
                   any_seq,
                   fbs::GqaWindowClass::Any};
  return n;
}

// Probes a tier for the request's own KV dtype first, then for a dtype-agnostic
// row. Every row measured so far is dtype-agnostic: prefill dequantises an Int8
// cache to fp16 scratch before it dispatches, so its tuning cannot depend on
// the dtype, and decode's Int8 path has not been swept. A dtype-specific row,
// once measured, beats the shared one without a schema change.
static const Answer *find(const GqaAutotunePolicy &policy,
                          fbs::GqaTunePhase phase, fbs::GqaTuneKvDtype dtype,
                          const Probe &probe) {
  const fbs::GqaTuneKvDtype dtypes[2] = {dtype, fbs::GqaTuneKvDtype::Any};
  for (fbs::GqaTuneKvDtype dt : dtypes) {
    const auto it = policy.rows.find(
        packKey(phase, probe.tier, dt, probe.head_dim, probe.hpg, probe.par,
                probe.batch, probe.seq_q, probe.seq_kv, probe.window));
    if (it != policy.rows.end())
      return &it->second;
  }
  return nullptr;
}

} // namespace

void *gqa_autotune_create(morphizen::FileSystem *fs) {
  auto policy = std::make_unique<GqaAutotunePolicy>();
  policy->compute_units = currentComputeUnits();
  const std::string &mode = hipdnn_ep::env_string("HIPDNN_GQA_AUTOTUNE_MODE");
  if (mode == "online")
    policy->mode = GqaAutotuneMode::Online;
  loadLutFromFileSystem(*policy, fs);
  return policy.release();
}

void gqa_autotune_destroy(void *policy) {
  delete static_cast<GqaAutotunePolicy *>(policy);
}

GqaAutotuneMode gqa_autotune_mode(const void *policy) {
  if (!policy)
    return GqaAutotuneMode::Lookup;
  return static_cast<const GqaAutotunePolicy *>(policy)->mode;
}

GqaDecodeResult gqa_autotune_resolve_decode(void *opaque_policy,
                                            const GqaDecodeRequest &request) {
  auto *policy = static_cast<GqaAutotunePolicy *>(opaque_policy);
  if (policy) {
    // The parallelism walk is the long one: every class from the request's down
    // to P1, then the three tiers below and the head_dim-agnostic last resort.
    Probe probes[kMaxProbes];
    const int n = buildProbes(
        headDimClass(request.head_dim),
        headsPerGroup(request.num_heads, request.kv_num_heads),
        parClass(request.batch, request.num_heads), batchClass(request.batch),
        fbs::GqaSeqBucket::Any, seqBucket(decodeEffectiveLen(request)),
        windowClass(request.local_window), probes);
    for (int i = 0; i < n; ++i) {
      const Answer *answer = find(*policy, fbs::GqaTunePhase::Decode,
                                  kvDtypeClass(request.kv_dtype), probes[i]);
      if (!answer)
        continue;
      // A row holds the split count the top of its bucket wants; the clamp is
      // what makes the same row right lower down.
      const GqaDecodeConfig config = clampDecodeConfig(
          request,
          GqaDecodeConfig{answer->use_wmma, answer->splits, answer->bkv});
      if (!validDecodeConfig(request, config)) {
        policy->invalid_entries.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      return {config, probes[i].source};
    }
  }
  // Only reachable with no usable table at all -- no file, an arch or schema
  // mismatch, or a table shipped without Fallback rows. Keeping it means a
  // rejected LUT degrades instead of failing the op.
  return {decodeHeuristic(request, policy && policy->compute_units > 0
                                       ? policy->compute_units
                                       : kAssumedCus),
          GqaTuneSource::Heuristic};
}

GqaPrefillResult
gqa_autotune_resolve_prefill(void *opaque_policy,
                             const GqaPrefillRequest &request) {
  auto *policy = static_cast<GqaAutotunePolicy *>(opaque_policy);
  if (policy) {
    // local_window is v5's alone: window_ok in real/gqa.cpp admits a window to
    // the fused path at head_dim 64 only, so v7 and v8 never see one.
    const bool v5 = request.variant == GqaPrefillVariant::V5;
    // The parallelism walk is the long one: every class from the request's down
    // to P1, then the three tiers below and the head_dim-agnostic last resort.
    Probe probes[kMaxProbes];
    const int n = buildProbes(
        headDimClass(request.head_dim),
        headsPerGroup(request.num_heads, request.kv_num_heads),
        parClass(request.batch, request.num_heads), batchClass(request.batch),
        seqBucket(std::max(request.seq_q, 1)),
        seqBucket(std::max(request.seq_kv, 1)),
        v5 ? windowClass(request.local_window) : fbs::GqaWindowClass::NoWindow,
        probes);
    for (int i = 0; i < n; ++i) {
      const Answer *answer = find(*policy, phaseOf(request.variant),
                                  fbs::GqaTuneKvDtype::Fp16, probes[i]);
      if (!answer)
        continue;
      return {answer->prefill, probes[i].source};
    }
  }
  // See the note in resolve_decode: reachable only without a usable table.
  return {prefillHeuristic(request), GqaTuneSource::Heuristic};
}

GqaPrefillConfig
gqa_autotune_fallback_prefill(const GqaPrefillRequest &request) {
  return prefillHeuristic(request);
}

const char *gqa_tune_source_name(GqaTuneSource source) {
  switch (source) {
  case GqaTuneSource::Geometry:
    return "geometry";
  case GqaTuneSource::HeadGroup:
    return "head_group";
  case GqaTuneSource::Length:
    return "length";
  case GqaTuneSource::Fallback:
    return "fallback";
  case GqaTuneSource::Heuristic:
    return "heuristic";
  }
  return "unknown";
}

} // namespace hipdnn_ep
