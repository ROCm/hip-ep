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
inline constexpr const char kGqaKernelAbi[] = "gqa-v1";

enum class GqaTuneSource : uint8_t {
  Exact,
  Bucket,
  Heuristic,
};

enum class GqaAutotuneMode : uint8_t {
  Lookup,
  Online,
};

struct GqaDecodeRequest {
  int kv_dtype;
  int batch;
  int num_heads;
  int kv_num_heads;
  int head_dim;
  int effective_skv;
  bool exact_length_known;
  int max_seq;
  int max_splits;
  int local_window;
};

struct GqaDecodeConfig {
  bool use_wmma;
  int splits;
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
  int max_seq;
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
GqaDecodeResult gqa_autotune_resolve_decode(void *policy,
                                            const GqaDecodeRequest &request);
GqaPrefillResult
gqa_autotune_resolve_prefill(void *policy,
                             const GqaPrefillRequest &request);
const char *gqa_tune_source_name(GqaTuneSource source);

} // namespace hipdnn_ep
