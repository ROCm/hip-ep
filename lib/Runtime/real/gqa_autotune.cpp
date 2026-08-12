/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gqa_autotune.h"

#include "../debug_log.h"
#include "gqa_autotune_generated.h"
#if !defined(HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE)
#include "runtime_types.h"
#endif

#include "morphizen-foundation/file_io.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_ep {
namespace {

constexpr uint32_t kGqaLutSchemaVersion = 1;
constexpr size_t kMaxLutBytes = 64u * 1024u * 1024u;

struct DecodeKey {
  int kv_dtype;
  int batch;
  int num_heads;
  int kv_num_heads;
  int head_dim;
  int seq_kv;
  int max_seq;
  int local_window;

  bool operator==(const DecodeKey &other) const {
    return kv_dtype == other.kv_dtype && batch == other.batch &&
           num_heads == other.num_heads &&
           kv_num_heads == other.kv_num_heads &&
           head_dim == other.head_dim && seq_kv == other.seq_kv &&
           max_seq == other.max_seq && local_window == other.local_window;
  }
};

static size_t hashCombine(size_t seed, int value) {
  return seed ^ (std::hash<int>{}(value) + 0x9e3779b9u + (seed << 6) +
                 (seed >> 2));
}

struct DecodeKeyHash {
  size_t operator()(const DecodeKey &key) const {
    size_t h = 0;
    h = hashCombine(h, key.kv_dtype);
    h = hashCombine(h, key.batch);
    h = hashCombine(h, key.num_heads);
    h = hashCombine(h, key.kv_num_heads);
    h = hashCombine(h, key.head_dim);
    h = hashCombine(h, key.seq_kv);
    h = hashCombine(h, key.max_seq);
    return hashCombine(h, key.local_window);
  }
};

struct PrefillKey {
  int variant;
  int batch;
  int num_heads;
  int kv_num_heads;
  int head_dim;
  int seq_q;
  int seq_kv;
  int max_seq;
  int local_window;

  bool operator==(const PrefillKey &other) const {
    return variant == other.variant && batch == other.batch &&
           num_heads == other.num_heads &&
           kv_num_heads == other.kv_num_heads &&
           head_dim == other.head_dim && seq_q == other.seq_q &&
           seq_kv == other.seq_kv && max_seq == other.max_seq &&
           local_window == other.local_window;
  }
};

struct PrefillKeyHash {
  size_t operator()(const PrefillKey &key) const {
    size_t h = 0;
    h = hashCombine(h, key.variant);
    h = hashCombine(h, key.batch);
    h = hashCombine(h, key.num_heads);
    h = hashCombine(h, key.kv_num_heads);
    h = hashCombine(h, key.head_dim);
    h = hashCombine(h, key.seq_q);
    h = hashCombine(h, key.seq_kv);
    h = hashCombine(h, key.max_seq);
    return hashCombine(h, key.local_window);
  }
};

struct GqaAutotunePolicy {
  GqaAutotuneMode mode = GqaAutotuneMode::Lookup;
  std::unordered_map<DecodeKey, GqaDecodeConfig, DecodeKeyHash> decode_exact;
  std::unordered_map<DecodeKey, GqaDecodeConfig, DecodeKeyHash> decode_bucket;
  std::unordered_map<PrefillKey, GqaPrefillConfig, PrefillKeyHash> prefill_exact;
  std::unordered_map<PrefillKey, GqaPrefillConfig, PrefillKeyHash> prefill_bucket;
  std::atomic<uint64_t> exact_hits{0};
  std::atomic<uint64_t> bucket_hits{0};
  std::atomic<uint64_t> heuristic_misses{0};
  std::atomic<uint64_t> invalid_entries{0};
};

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
  if (v >= static_cast<uint32_t>(INT_MAX))
    return INT_MAX;
  return static_cast<int>(v + 1);
}

static bool wmmaSupported(int head_dim, int heads_per_group) {
  if (heads_per_group == 4)
    return head_dim == 64 || head_dim == 128;
  return heads_per_group == 8 && head_dim == 64;
}

static GqaDecodeConfig decodeHeuristic(const GqaDecodeRequest &request) {
  const int cap = std::max(1, std::min(request.max_splits, 64));
  int effective_len = std::max(request.effective_skv, 1);
  if (request.local_window > 0)
    effective_len = std::min(effective_len, request.local_window);
  const int useful_splits = std::max(1, (effective_len + 15) / 16);
  const int hpg = request.kv_num_heads > 0
                      ? request.num_heads / request.kv_num_heads
                      : 0;
  return {/*use_wmma=*/request.head_dim == 64 &&
                              wmmaSupported(request.head_dim, hpg),
          /*splits=*/std::min({8, cap, useful_splits})};
}

static bool validDecodeConfig(const GqaDecodeRequest &request,
                              const GqaDecodeConfig &config) {
  if (config.splits < 1 || config.splits > request.max_splits ||
      config.splits > 64)
    return false;
  if (!config.use_wmma)
    return true;
  if (request.kv_num_heads <= 0 ||
      request.num_heads % request.kv_num_heads != 0)
    return false;
  return wmmaSupported(request.head_dim,
                       request.num_heads / request.kv_num_heads);
}

static DecodeKey makeDecodeKey(const GqaDecodeRequest &request, bool bucket,
                               bool wildcard_max_seq = false) {
  int effective_len = std::max(request.effective_skv, 1);
  if (request.local_window > 0)
    effective_len = std::min(effective_len, request.local_window);
  return {request.kv_dtype,
          request.batch,
          request.num_heads,
          request.kv_num_heads,
          request.head_dim,
          bucket ? nextPowerOfTwo(effective_len) : effective_len,
          wildcard_max_seq ? 0 : request.max_seq,
          normalizeWindow(request.local_window)};
}

static int prefillVariantValue(GqaPrefillVariant variant) {
  return static_cast<int>(variant);
}

static PrefillKey makePrefillKey(const GqaPrefillRequest &request, bool bucket,
                                 bool wildcard_max_seq = false) {
  const bool v5 = request.variant == GqaPrefillVariant::V5;
  return {prefillVariantValue(request.variant),
          request.batch,
          request.num_heads,
          request.kv_num_heads,
          request.head_dim,
          bucket ? nextPowerOfTwo(std::max(request.seq_q, 1)) : request.seq_q,
          v5 ? 0
             : (bucket ? nextPowerOfTwo(std::max(request.seq_kv, 1))
                       : request.seq_kv),
          v5 || wildcard_max_seq ? 0 : request.max_seq,
          v5 ? normalizeWindow(request.local_window) : 0};
}

static GqaPrefillConfig
prefillHeuristic(const GqaPrefillRequest &request) {
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

static bool validPrefillConfig(const GqaPrefillRequest &request,
                               const GqaPrefillConfig &config) {
  if (config.bkv != 32 && config.bkv != 64)
    return false;
  switch (request.variant) {
  case GqaPrefillVariant::V5:
    return request.head_dim == 64 &&
           (config.m_tiles == 1 || config.m_tiles == 2);
  case GqaPrefillVariant::V7:
    return (request.head_dim == 128 || request.head_dim == 256) &&
           (config.nw == 1 || config.nw == 2 || config.nw == 4) &&
           (config.mt == 1 || config.mt == 2);
  case GqaPrefillVariant::V8:
    return request.head_dim == 256 &&
           (config.nd == 2 || config.nd == 4) &&
           (config.mt == 1 || config.mt == 2) &&
           (config.nd != 4 || config.bkv == 32);
  }
  return false;
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
  std::string arch(prop.gcnArchName);
  const size_t feature_suffix = arch.find(':');
  if (feature_suffix != std::string::npos)
    arch.resize(feature_suffix);
  return arch;
#endif
}

static bool compatibleLut(const mlir::hip::GqaAutotuneLut *lut) {
  if (lut->schema_version() != kGqaLutSchemaVersion) {
    fprintf(stderr,
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
  if (!mlir::hip::VerifyGqaAutotuneLutBuffer(verifier)) {
    fprintf(stderr, "GQA LUT ignored: invalid FlatBuffer\n");
    return false;
  }
  const auto *lut = mlir::hip::GetGqaAutotuneLut(data);
  if (!compatibleLut(lut))
    return false;
  if (!lut->entries())
    return true;

  size_t loaded = 0;
  for (const auto *entry : *lut->entries()) {
    if (!entry || !entry->key() || !entry->config())
      continue;
    const auto *key = entry->key();
    const auto *config = entry->config();
    if (key->batch() <= 0 || key->num_heads() <= 0 ||
        key->kv_num_heads() <= 0 || key->head_dim() <= 0) {
      policy.invalid_entries.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    if (key->phase() == mlir::hip::GqaTunePhase::Decode) {
      if (key->seq_kv() <= 0 || config->splits() <= 0 ||
          config->splits() > 64) {
        policy.invalid_entries.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      DecodeKey decode_key{
          static_cast<int>(key->kv_dtype()), key->batch(), key->num_heads(),
          key->kv_num_heads(), key->head_dim(), key->seq_kv(), key->max_seq(),
          normalizeWindow(key->local_window())};
      GqaDecodeConfig decode_config{config->use_wmma(), config->splits()};
      if (key->match() == mlir::hip::GqaTuneMatch::Exact)
        policy.decode_exact.insert_or_assign(decode_key, decode_config);
      else
        policy.decode_bucket.insert_or_assign(decode_key, decode_config);
    } else {
      int variant = -1;
      if (key->phase() == mlir::hip::GqaTunePhase::PrefillV5)
        variant = prefillVariantValue(GqaPrefillVariant::V5);
      else if (key->phase() == mlir::hip::GqaTunePhase::PrefillV7)
        variant = prefillVariantValue(GqaPrefillVariant::V7);
      else if (key->phase() == mlir::hip::GqaTunePhase::PrefillV8)
        variant = prefillVariantValue(GqaPrefillVariant::V8);
      const bool v5 =
          variant == prefillVariantValue(GqaPrefillVariant::V5);
      if (variant < 0 || key->seq_q() <= 0 ||
          (!v5 && key->seq_kv() <= 0)) {
        policy.invalid_entries.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      PrefillKey prefill_key{
          variant,
          key->batch(),
          key->num_heads(),
          key->kv_num_heads(),
          key->head_dim(),
          key->seq_q(),
          v5 ? 0 : key->seq_kv(),
          v5 ? 0 : key->max_seq(),
          v5 ? normalizeWindow(key->local_window()) : 0};
      GqaPrefillConfig prefill_config{
          config->m_tiles(), config->bkv(), config->nw(), config->mt(),
          config->nd()};
      if (key->match() == mlir::hip::GqaTuneMatch::Exact)
        policy.prefill_exact.insert_or_assign(prefill_key, prefill_config);
      else
        policy.prefill_bucket.insert_or_assign(prefill_key, prefill_config);
    }
    ++loaded;
  }
  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] loaded GQA LUT: %zu entries (%zu invalid)\n",
      loaded,
      static_cast<size_t>(
          policy.invalid_entries.load(std::memory_order_relaxed)));
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

static const GqaDecodeConfig *
findDecodeConfig(const std::unordered_map<DecodeKey, GqaDecodeConfig,
                                          DecodeKeyHash> &map,
                 const GqaDecodeRequest &request, bool bucket) {
  auto it = map.find(makeDecodeKey(request, bucket));
  if (it != map.end())
    return &it->second;
  // max_seq=0 is an explicit wildcard useful for architecture-level tables.
  it = map.find(makeDecodeKey(request, bucket, /*wildcard_max_seq=*/true));
  return it == map.end() ? nullptr : &it->second;
}

static const GqaPrefillConfig *
findPrefillConfig(const std::unordered_map<PrefillKey, GqaPrefillConfig,
                                          PrefillKeyHash> &map,
                  const GqaPrefillRequest &request, bool bucket) {
  auto it = map.find(makePrefillKey(request, bucket));
  if (it != map.end())
    return &it->second;
  it = map.find(
      makePrefillKey(request, bucket, /*wildcard_max_seq=*/true));
  return it == map.end() ? nullptr : &it->second;
}

} // namespace

void *gqa_autotune_create(morphizen::FileSystem *fs) {
  auto policy = std::make_unique<GqaAutotunePolicy>();
  const std::string &mode =
      hipdnn_ep::env_string("HIPDNN_GQA_AUTOTUNE_MODE");
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

GqaDecodeResult gqa_autotune_resolve_decode(
    void *opaque_policy, const GqaDecodeRequest &request) {
  auto *policy = static_cast<GqaAutotunePolicy *>(opaque_policy);
  if (policy) {
    if (request.exact_length_known) {
      if (const GqaDecodeConfig *config =
              findDecodeConfig(policy->decode_exact, request, false)) {
        if (validDecodeConfig(request, *config)) {
          policy->exact_hits.fetch_add(1, std::memory_order_relaxed);
          return {*config, GqaTuneSource::Exact};
        }
        policy->invalid_entries.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (const GqaDecodeConfig *config =
            findDecodeConfig(policy->decode_bucket, request, true)) {
      if (validDecodeConfig(request, *config)) {
        policy->bucket_hits.fetch_add(1, std::memory_order_relaxed);
        return {*config, GqaTuneSource::Bucket};
      }
      policy->invalid_entries.fetch_add(1, std::memory_order_relaxed);
    }
    policy->heuristic_misses.fetch_add(1, std::memory_order_relaxed);
  }
  return {decodeHeuristic(request), GqaTuneSource::Heuristic};
}

GqaPrefillResult gqa_autotune_resolve_prefill(
    void *opaque_policy, const GqaPrefillRequest &request) {
  auto *policy = static_cast<GqaAutotunePolicy *>(opaque_policy);
  if (policy) {
    if (const GqaPrefillConfig *config =
            findPrefillConfig(policy->prefill_exact, request, false)) {
      if (validPrefillConfig(request, *config)) {
        policy->exact_hits.fetch_add(1, std::memory_order_relaxed);
        return {*config, GqaTuneSource::Exact};
      }
      policy->invalid_entries.fetch_add(1, std::memory_order_relaxed);
    }
    if (const GqaPrefillConfig *config =
            findPrefillConfig(policy->prefill_bucket, request, true)) {
      if (validPrefillConfig(request, *config)) {
        policy->bucket_hits.fetch_add(1, std::memory_order_relaxed);
        return {*config, GqaTuneSource::Bucket};
      }
      policy->invalid_entries.fetch_add(1, std::memory_order_relaxed);
    }
    policy->heuristic_misses.fetch_add(1, std::memory_order_relaxed);
  }
  return {prefillHeuristic(request), GqaTuneSource::Heuristic};
}

const char *gqa_tune_source_name(GqaTuneSource source) {
  switch (source) {
  case GqaTuneSource::Exact:
    return "exact";
  case GqaTuneSource::Bucket:
    return "bucket";
  case GqaTuneSource::Heuristic:
    return "heuristic";
  }
  return "unknown";
}

} // namespace hipdnn_ep
