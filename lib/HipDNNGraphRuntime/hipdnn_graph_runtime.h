/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hipdnn_graph_runtime.h - C API for hipDNN graph execution ----------===//
//
// Public C API exported from hipdnn_graph_runtime.dll.
//
// Three consumers:
//   1. hip-compiler.dll (handle creation + singleton registry population)
//   2. The ORT EP (registry management + attach)
//   3. JITted per-model bitcode (hipdnn_graph_execute, resolved by
//      LlvmIrJit's process search generator against this DLL)
//
// All functions use C linkage and opaque pointers for cross-DLL safety.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_GRAPH_RUNTIME_H
#define HIPDNN_GRAPH_RUNTIME_H

#include <stdint.h>

#ifdef _WIN32
#ifdef HIPDNN_GRAPH_RUNTIME_EXPORTS
#define HIPDNN_GRAPH_RUNTIME_API __declspec(dllexport)
#else
#define HIPDNN_GRAPH_RUNTIME_API __declspec(dllimport)
#endif
#else
#define HIPDNN_GRAPH_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Create a hipDNN handle via hipdnnCreate(). Returns opaque handle as void*,
/// or nullptr on failure. Wraps hipdnn_backend so callers don't need to link
/// it.
HIPDNN_GRAPH_RUNTIME_API void *hipdnn_graph_create_handle();

/// Create a new graph registry. Returns opaque pointer.
HIPDNN_GRAPH_RUNTIME_API void *hipdnn_graph_registry_create();

/// Destroy the registry and all graphs it owns.
HIPDNN_GRAPH_RUNTIME_API void hipdnn_graph_registry_destroy(void *registry);

/// Store a compiled graph in the registry. Takes ownership of the graph.
/// @param registry  Opaque registry pointer from _create()
/// @param graph_id  Integer ID (matches IR attribute)
/// @param graph     Opaque pointer to hip::graph::HipDNNGraph (ownership
/// transferred)
HIPDNN_GRAPH_RUNTIME_API void
hipdnn_graph_registry_store(void *registry, int32_t graph_id, void *graph);

/// Set the process-level default graph registry. Used by the same-process
/// singleton pattern: hip-compiler.dll populates this during compilation,
/// the JITted per-model code reads it during inference via the
/// hipdnn_graph_execute fallback.
HIPDNN_GRAPH_RUNTIME_API void hipdnn_graph_set_default_registry(void *registry);

/// Set the process-level default hipDNN handle. Same singleton pattern.
HIPDNN_GRAPH_RUNTIME_API void hipdnn_graph_set_default_handle(void *handle);

/// Attach hipDNN handle and graph registry to a RuntimeState.
/// Called by EP after inference_init() returns.
/// @param state     Opaque RuntimeState* from inference_init()
/// @param handle    hipdnnHandle_t (cast to void*)
/// @param registry  Opaque registry pointer from _create()
HIPDNN_GRAPH_RUNTIME_API void
hipdnn_graph_runtime_attach(void *state, void *handle, void *registry);

/// Execute a compiled hipDNN graph. Called from the JITted per-model code.
/// Each compiled graph owns its GPU workspace (allocated during Compile()),
/// so no workspace management is needed from the caller.
/// Falls back to process-level defaults if state fields are nullptr.
/// @param state     RuntimeState* (provides handle and registry)
/// @param graph_id  Which graph to execute
/// @param num_io    Number of entries in uids[] and ptrs[]
/// @param uids      Tensor UIDs (from IR attributes)
/// @param ptrs      GPU memory pointers (from memref descriptors)
/// @return 0 on success, non-zero on failure
HIPDNN_GRAPH_RUNTIME_API int32_t hipdnn_graph_execute(
    void *state, int32_t graph_id, int32_t num_io, int64_t *uids, void **ptrs);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIPDNN_GRAPH_RUNTIME_H
