/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- runtime_state_internal.h - RuntimeState Internal Definition -------===//
//
// INTERNAL HEADER - Not part of public API
//
// This header defines the internal structure of RuntimeState.
// Only runtime implementation files should include this header.
//
// Public interface uses opaque pointer (see hipdnn_ep_runtime.h).
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
#define HIPDNN_EP_RUNTIME_STATE_INTERNAL_H

#include "runtime_types.h"

// Internal runtime state structure
// This struct is opaque to generated code (passed as void*)
struct RuntimeState {
  hipStream_t stream;
  miopenHandle_t miopen_handle;
  hipblasLtHandle_t hipblas_handle;

  // Single allocation holding all constants as one blob.
  // gpu_constants[i] points into gpu_constants_blob at the offset stored in
  // ConstantInfo, so only one allocation/copy is needed at init time.
  // On dGPU: hipMalloc (VRAM). On iGPU: hipHostMalloc (pinned system RAM,
  // GPU reads in-place, no hipMemcpy needed).
  void *gpu_constants_blob;
  bool constants_blob_is_host; // true = hipHostMalloc, false = hipMalloc
  void **gpu_constants;
  size_t num_constants;

  // Shared constants support: in OGA pipeline mode, prefill and decode models
  // share the same constants.bin. The second model reuses the first's blob
  // via a process-wide named shared memory descriptor with atomic ref count.
  bool constants_is_shared;       // true = reusing another model's blob
  void *shared_constants_mapping; // Win32 file mapping HANDLE
  void *shared_constants_view; // MapViewOfFile pointer (SharedConstantsMeta*)

  // Memory pooling support
  void *pool_base;        // Single large memory pool
  size_t pool_size;       // Total pool size in bytes
  size_t *buffer_offsets; // Offset for each buffer in the pool
  size_t num_buffers;     // Number of buffers in the pool

  // Shared workspace for operator temp buffers (MatMul GEMM ws, GQA pipeline).
  // Lazily grown via hipdnn_ep_state_ensure_workspace(); never shrinks.
  void *workspace;
  size_t workspace_size;

  // hipDNN graph execution support.
  // Set by EP via hipdnn_graph_runtime_attach() after inference_init().
  // hipdnn_handle: hipdnnHandle_t cast to void* (owned by EP, not cleaned up
  // here) hipdnn_graph_registry: opaque GraphRegistry* (owned by EP, not
  // cleaned up here)
  void *hipdnn_handle;
  void *hipdnn_graph_registry;

  // Unified Memory Manager state.
  // When mm_initialized is true, memory operations route through the MM API.
  // The legacy fields above (pool_base, workspace, etc.) remain for
  // backward compatibility with existing generated code.
  bool mm_initialized;

  // GPU-resident KV cache state.
  // Persistent GPU buffers for KV cache tensors, keyed by host pointer.
  // Eliminates H2D/D2H for KV cache when past_present_share_buffer=true.
  void *kv_cache_state;
};

#endif // HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
