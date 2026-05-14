/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Internal layout of RuntimeState. Runtime implementation files only.
// Public interface uses an opaque pointer (see hipdnn_ep_runtime.h).

#ifndef HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
#define HIPDNN_EP_RUNTIME_STATE_INTERNAL_H

#include "runtime_types.h"

// Op-module registry; defined in module_registry.cpp. Sits at the tail of
// RuntimeState so existing flat fields keep their offsets.
namespace hipdnn_ep {
struct ModuleRegistry;
} // namespace hipdnn_ep

// Opaque to generated code (passed as void*).
struct RuntimeState {
  hipStream_t stream;
  miopenHandle_t miopen_handle;
  hipblasLtHandle_t hipblas_handle;

  // Single allocation holding all constants as one blob.
  // gpu_constants[i] points into gpu_constants_blob at the offset stored in
  // ConstantInfo, so only one allocation/copy is needed at init time.
  // Always hipMalloc (VRAM) for both dGPU and iGPU.
  void *gpu_constants_blob;
  void **gpu_constants;
  size_t num_constants;

  // OGA pipeline shared constants cache: prefill and decode models share
  // the same constants blob via process-wide named shared memory + atomic
  // ref count. Set by try_attach_shared_constants when reusing another
  // model's blob; cleanup decrements ref_count and only the last
  // reference frees the GPU memory.
  bool constants_is_shared;
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

  // Per-operator profiling state (OpProfileState*, gated on HIPDNN_EP_PERF).
  void *op_profile;

  // Device-side error flag (0 = no error, non-zero = error code).
  int *device_error_flag;

  // hipDNN graph execution support. Both attached by the EP via
  // hipdnn_graph_runtime_attach() after inference_init(); owned by the EP
  // and not cleaned up here.
  void *hipdnn_handle;          // hipdnnHandle_t cast to void*
  void *hipdnn_graph_registry;  // opaque GraphRegistry*

  // Op-module registry. New ops add per-session state by registering
  // through HIPDNN_OP_MODULE rather than growing this struct.
  hipdnn_ep::ModuleRegistry *modules;
};

#endif // HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
