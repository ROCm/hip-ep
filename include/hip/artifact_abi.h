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

// Internal-linkage globals baked into the artifact by GenerateInterface.
//   kMetadataBlobGlobal: FlatBuffers HipModelMetaInfo consumed by
//                        inference_init (constant-weight upload) -- functional.
//   kMetadataJsonGlobal: human-readable mirror, surfaced via
//                        inference_get_metadata_json for the tools.
inline constexpr const char *kMetadataBlobGlobal = "__metadata_blob";
inline constexpr const char *kMetadataJsonGlobal = "__metadata_json";

} // namespace hipdnn::abi

#endif // HIP_ARTIFACT_ABI_H
