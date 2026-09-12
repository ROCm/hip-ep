/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_GQA_AUTOTUNE_H
#define HIPDNN_EP_GQA_AUTOTUNE_H

#include <cstdint>

// HIP_KERNEL_API: dllexport when building custom_kernels_<arch>, empty for the
// bitcode caller. The resolver and its table live in that DLL now (they used to
// be in runtime.bc); real/gqa.cpp, compiled to bitcode, calls the exported
// shims below the same way it already calls hip_gqa_flash_decode_configured.
// The only symbols that cross the JIT boundary are extern "C" *functions*,
// which DynamicLibrarySearchGenerator resolves reliably -- a data symbol (the
// old kGqaLutData array in runtime.bc) does not cross it safely on Windows,
// which is why the table moved into the DLL with its consumer.
//
// Defined here (guarded) rather than by including hip_custom_kernels.h so this
// header stays dependency-free: the mock runtime includes it (via
// runtime_state) without the kernel include path. hip_custom_kernels.h defines
// the same macro under the same guard, so including both in either order is
// fine.
#ifndef HIP_KERNEL_API
#if defined(_WIN32)
#if defined(HIP_CUSTOM_KERNELS_EXPORTS)
#define HIP_KERNEL_API __declspec(dllexport)
#else
#define HIP_KERNEL_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HIP_KERNEL_API __attribute__((visibility("default")))
#else
#define HIP_KERNEL_API
#endif
#endif // HIP_KERNEL_API

namespace hipdnn_ep {

// Where a resolved config came from. Reported so a caller or a test can tell a
// measured hit from the compiled-in last resort.
//
// Exact and Nearest are the same lookup: Exact is the case where the closest
// measured point sits at distance 0, i.e. this shape was measured. Fallback is
// the table's per-(phase, head_dim) last resort, used when the group has no
// usable point. Heuristic means no table loaded at all (arch/schema/abi
// mismatch, or an empty embed) and the config came from gqa_autotune.cpp -- a
// runnable config so the op never fails for lack of a table.
enum class GqaTuneSource : uint8_t {
  Exact = 0,
  Nearest = 1,
  Fallback = 2,
  Heuristic = 3,
};

enum class GqaAutotuneMode : uint8_t {
  Lookup,
  Online,
};

// KV cache capacity (max_seq) is deliberately not a request field: sweeping it
// while holding the work fixed moves every candidate by the same factor and
// never changes which one wins. The distance metric keys on the current lengths
// (effective_skv / seq_q / seq_kv), not the capacity.
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
  // WMMA is fixed at BKV=32 in the kernel.
  int bkv;
};

struct GqaDecodeResult {
  GqaDecodeConfig config;
  GqaTuneSource source;
  // Weighted log2 distance to the point that answered, in octaves. 0 on Exact,
  // meaningless on Fallback/Heuristic. Exposed so a log can show how far the
  // answer was extrapolated -- the signal for "add this shape to the sweep".
  float distance;
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
  float distance;
};

// Inline so both the DLL and the bitcode caller get it without a cross-boundary
// call -- it is only a label for a debug log.
inline const char *gqa_tune_source_name(GqaTuneSource source) {
  switch (source) {
  case GqaTuneSource::Exact:
    return "exact";
  case GqaTuneSource::Nearest:
    return "nearest";
  case GqaTuneSource::Fallback:
    return "fallback";
  case GqaTuneSource::Heuristic:
    return "heuristic";
  }
  return "unknown";
}

} // namespace hipdnn_ep

// ---------------------------------------------------------------------------
// C-ABI exported from custom_kernels_<arch> (see gqa_autotune.cpp).
//
// Declared at global extern "C" scope, matching hip_custom_kernels.h, so
// real/gqa.cpp calls these the same unqualified way it calls the other
// hip_gqa_* launchers. POD structs cross by pointer for ABI stability; both
// sides compile this one header with the same toolchain, so the layouts match.
// The bitcode caller sees HIP_KERNEL_API as empty (plain extern "C"); the DLL
// build sees dllexport.
// ---------------------------------------------------------------------------
extern "C" {

// Session-scoped policy: it holds the lookup/online mode and the
// device's CU count for the heuristic. The measured table itself is a
// process-wide static in the DLL, loaded once, independent of the policy.
HIP_KERNEL_API void *hip_gqa_autotune_create(const char *provider_mode);
HIP_KERNEL_API void hip_gqa_autotune_destroy(void *policy);

// Returns GqaAutotuneMode as an int (Lookup = 0, Online = 1).
HIP_KERNEL_API int hip_gqa_autotune_mode(const void *policy);

HIP_KERNEL_API void
hip_gqa_autotune_resolve_decode(void *policy,
                                const hipdnn_ep::GqaDecodeRequest *req,
                                hipdnn_ep::GqaDecodeResult *out);
HIP_KERNEL_API void
hip_gqa_autotune_resolve_prefill(void *policy,
                                 const hipdnn_ep::GqaPrefillRequest *req,
                                 hipdnn_ep::GqaPrefillResult *out);

// The heuristic prefill config on its own, for the caller's recovery path after
// the kernel rejected a resolved config: asking resolve again would hand back a
// config from the same neighbourhood that already failed.
HIP_KERNEL_API void
hip_gqa_autotune_fallback_prefill(const hipdnn_ep::GqaPrefillRequest *req,
                                  hipdnn_ep::GqaPrefillConfig *out);

// Diagnostics for the debug log: did a table load, how many points, how many
// were dropped as malformed or had a dangling config index.
HIP_KERNEL_API int hip_gqa_autotune_table_loaded();
HIP_KERNEL_API uint32_t hip_gqa_autotune_point_count();
HIP_KERNEL_API uint32_t hip_gqa_autotune_invalid_points();

} // extern "C"

#endif // HIPDNN_EP_GQA_AUTOTUNE_H
