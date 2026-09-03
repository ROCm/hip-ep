/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_ARTIFACT_ABI_H
#define HIP_ARTIFACT_ABI_H

// Single source of truth for the per-model artifact's C-ABI symbol names and
// the embedded-metadata global names. These names form the contract between
// the producer (GenerateInterface emits the functions / globals; CompilerDriver
// exports them from the native DLL) and every consumer that resolves them
// (InferenceState / LoadedArtifact in the EP, and the hip-test / hip-inspect
// tools). Defining them once here makes a rename a single compile-checked edit
// instead of a set of string literals that drift into silent runtime
// lookup failures.

namespace hipdnn::abi {

// C-ABI entry points emitted by GenerateInterface and resolved by consumers.
inline constexpr const char *kInferenceInit = "inference_init";
inline constexpr const char *kInferenceCompute = "inference_compute";
inline constexpr const char *kInferenceCleanup = "inference_cleanup";
inline constexpr const char *kInferenceGetMetadataJson =
    "inference_get_metadata_json";

// Optional runtime hooks (consumers probe for these; absent on older
// artifacts).
inline constexpr const char *kRuntimeBeginCompute =
    "hipdnn_ep_runtime_begin_compute";
inline constexpr const char *kSetOutputAllocator =
    "hipdnn_ep_set_output_allocator";
inline constexpr const char *kRuntimeFlushOpProfile =
    "hipdnn_ep_runtime_flush_op_profile";
inline constexpr const char *kRuntimeSetProviderOption =
    "hipdnn_ep_runtime_set_provider_option";
inline constexpr const char *kRuntimeGetProviderOption =
    "hipdnn_ep_runtime_get_provider_option";

// Per-op-state-slots C-ABI symbol names (see
// docs/design/op-state-slots-design.md). kOpStatesInitFn is emitted by
// GenerateOpStateInit and called from inference_init by GenerateInterface;
// kOpStatesAlloc is defined in lib/Runtime/op_state.cpp and called from that
// emitted init function. kOpStateSet is also defined there but is now called
// internally by each construct_<op> (via OpStateT::create) rather than emitted
// by the pass; it is kept here as the single source of truth for the symbol
// name. Centralizing these keeps the producer (passes) and consumer (runtime)
// from drifting into silent JIT symbol-lookup failures. (op_state_get / _set
// are resolved via the OpStateT runtime template, not emitted by a pass.)
inline constexpr const char *kOpStatesInitFn = "hipdnn_ep_op_states_init_fn";
inline constexpr const char *kOpStatesAlloc = "hipdnn_ep_op_states_alloc";
inline constexpr const char *kOpStateSet = "hipdnn_ep_op_state_set";

// Internal-linkage globals baked into the artifact by GenerateInterface.
//   kMetadataBlobGlobal: FlatBuffers HipModelMetaInfo consumed by
//                        inference_init (constant-weight upload) -- functional.
//   kMetadataJsonGlobal: human-readable mirror, surfaced via
//                        inference_get_metadata_json for the tools.
inline constexpr const char *kMetadataBlobGlobal = "__metadata_blob";
inline constexpr const char *kMetadataJsonGlobal = "__metadata_json";

} // namespace hipdnn::abi

#endif // HIP_ARTIFACT_ABI_H
