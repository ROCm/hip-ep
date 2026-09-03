/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace morphizen {
class FileSystem;
}

namespace hipdnn_ep {

inline constexpr const char kGqaAutotuneFilename[] = "gqa_autotune.fb";
// PR #675 added BKV=32 as a separately tunable d64 WMMA decode candidate. A
// gqa-v1 row can only name "WMMA", which now means the incomplete BKV=16
// subset, so reject it instead of silently losing the new candidate.
inline constexpr const char kGqaKernelAbi[] = "gqa-v2";

// Where a config came from, in the order the policy tries them. Each name is a
// tier of the offline table, and the tier is part of a row's key.
//   Geometry          heads-per-group, a bucket of batch*num_heads, and bucketed
//                     lengths.
//   ExactHeadGroup    exact (num_heads, kv_num_heads) pair and bucketed lengths.
//                     Finest-grain matching for known H/G pairs (e.g. Llama H=32,G=8).
//   HeadGroup         heads-per-group and bucketed lengths, pooled over head
//                     counts. Fuzzy fallback for new models with known hpg.
//                     The tier that carries production traffic when no ExactHeadGroup
//                     row exists: the (head_dim, heads-per-group) pairs the fused path
//                     admits are a closed set of 21, so a complete table answers
//                     every geometry here.
//   Length            bucketed lengths only, pooled over geometries. What answers a
//                     heads-per-group with no row of its own, and most prefill shapes,
//                     since the prefill kernels are templated on head_dim alone.
//   Fallback          phase, kv_dtype and head_dim only, ranked on its worst case.
//   Heuristic         a config compiled into gqa_autotune.cpp, reached only when there
//                     is no usable table at all: no file, an arch/schema mismatch, or a
//                     table with no Fallback rows. A LUT that loads answers every shape.
//
// Every LUT tier is data, measured offline and reviewed as rows. Widening what
// the table answers means adding rows, not adding another matcher here -- a
// config derived at dispatch time is a config nobody reviewed. `Heuristic`
// exists so a rejected table degrades instead of failing the op, which is also
// why seeing it in a log means the table did not load rather than that a shape
// was missed.
//
// The tiers are not equally good, and the gap is worth logging for. Over the
// decode grid, with each tier used alone: ExactHeadGroup (TBD), Geometry ~99.5%,
// HeadGroup lands 98% of shapes within 5% of optimum with a worst case of 1.13x,
// Length 84% and 1.25x, Fallback 69% and 1.52x.
enum class GqaTuneSource : uint8_t {
  Geometry,
  ExactHeadGroup,
  HeadGroup,
  Length,
  Fallback,
  Heuristic,
};

enum class GqaAutotuneMode : uint8_t {
  Lookup,
  Online,
};

// `max_seq`, the KV cache capacity, is deliberately absent: sweeping it from
// seq_kv to 128k while holding the work fixed moves every candidate by the same
// factor (about +26% once the cache reaches 64k, a cache-set aliasing
// threshold) and never changes which one wins. Keying on it would multiply the
// table by the capacities a deployment might use and match none of them.
struct GqaDecodeRequest {
  int kv_dtype;
  int batch;
  int num_heads;
  int kv_num_heads;
  int head_dim;
  int effective_skv;
  int max_splits;
  int local_window;
};

struct GqaDecodeConfig {
  bool use_wmma;
  int splits;
  // d64 WMMA has BKV=16 and BKV=32 implementations. Scalar ignores this; d128
  // WMMA is fixed at BKV=32 in the kernel and uses the legacy Wmma row name.
  int bkv;
};

struct GqaDecodeResult {
  GqaDecodeConfig config;
  GqaTuneSource source;
};

enum class GqaPrefillVariant : uint8_t {
  V5,
  V7,
  V8,
};

struct GqaPrefillRequest {
  GqaPrefillVariant variant;
  int batch;
  int num_heads;
  int kv_num_heads;
  int head_dim;
  int seq_q;
  int seq_kv;
  int local_window;
};

struct GqaPrefillConfig {
  int m_tiles;
  int bkv;
  int nw;
  int mt;
  int nd;
};

struct GqaPrefillResult {
  GqaPrefillConfig config;
  GqaTuneSource source;
};

// Opaque session-owned policy. Missing or incompatible LUT files produce an
// empty policy rather than an initialization failure; lookup mode then uses the
// deterministic heuristic and never benchmarks on the GPU.
void *gqa_autotune_create(morphizen::FileSystem *fs);
void gqa_autotune_destroy(void *policy);

GqaAutotuneMode gqa_autotune_mode(const void *policy);

// Apply the session's gqa_autotune_mode provider option.
// HIPDNN_GQA_AUTOTUNE_MODE has higher priority, so this is a no-op when that
// variable was set. `mode` is parsed into the typed policy during the call and
// is never retained.
void gqa_autotune_apply_provider_mode(void *policy, const char *mode);

GqaDecodeResult gqa_autotune_resolve_decode(void *policy,
                                            const GqaDecodeRequest &request);
GqaPrefillResult gqa_autotune_resolve_prefill(void *policy,
                                              const GqaPrefillRequest &request);

// Tier 3 on its own: the config that is only known to run. Callers use this to
// recover after the kernel rejected a resolved config -- asking resolve_* again
// would just hand back another config from the same tiers that already failed.
GqaPrefillConfig
gqa_autotune_fallback_prefill(const GqaPrefillRequest &request);

const char *gqa_tune_source_name(GqaTuneSource source);

} // namespace hipdnn_ep
