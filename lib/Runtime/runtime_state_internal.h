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

  // Host-mapped scratch buffer for tiny host-fed scalars routed away from the
  // GPU pool by hip-materialize-host-scalars. hipHostMalloc(hipHostMallocMapped):
  // host-writable AND GPU-readable. Grow-on-demand via
  // hipdnn_ep_get_host_scratch_base(); never shrinks. hipHostFree'd in cleanup.
  // Why: on gfx1151 the regular GPU pool is real device memory; host stores
  // into it SEGV. gfx1150 silently worked because hipMalloc returned UMA there.
  void *host_scratch_base;
  size_t host_scratch_size;

  // GQA GEMM descriptor cache (GqaGemmCache*) for the decomposed path.
  // Caches hipBLASLt descriptors + algorithms by GEMM shape.
  void *gqa_gemm_cache;

  // MatMulNBits asym-path zero_points unpack cache (ZpUnpackCache*).
  //
  // The asym AWQ path stores zero_points as packed nibbles [N, ceil(K/bs/2)].
  // Two unpacked layouts are needed (u8 [N, K/bs] for GEMV/naive, fp16 for
  // WMMA/col-major GEMV M>1), and naively the unpack kernel is launched on
  // every wrap_matmul_nbits call. For 8B asym decode this is ~225 redundant
  // launches per Compute(). Since zero_points points into the model constants
  // blob (stable for the session lifetime), we cache the unpacked buffer per
  // input pointer. Lazily created on first asym call. Owned by
  // lib/Runtime/real/matmul_nbits.cpp; freed in hipdnn_ep_state_cleanup via
  // hipdnn_ep_zp_unpack_cache_destroy.
  void *zp_unpack_cache;

  // Per-operator profiling state (OpProfileState*, gated on HIPDNN_EP_PERF).
  // Allocated in state_init, freed in state_cleanup.
  void *op_profile;

  // Device-side error flag used by kernels to report runtime-invalid inputs.
  // 0 = no error, non-zero = error code (currently -1).
  int *device_error_flag;

  // hipDNN graph execution support.
  // Set by EP via hipdnn_graph_runtime_attach() after inference_init().
  // hipdnn_handle: hipdnnHandle_t cast to void* (owned by EP, not cleaned up
  // here) hipdnn_graph_registry: opaque GraphRegistry* (owned by EP, not
  // cleaned up here)
  void *hipdnn_handle;
  void *hipdnn_graph_registry;

  // Per-Compute() cache for seqlens_k_val (decode hot path).
  //
  // Decode runs 32 GQA layers per token, all reading the same seqlens_k
  // from device memory. The decomposed-path readback in gqa.cpp issues a
  // hipMemcpyAsync(D2H) + hipStreamSynchronize per layer (31 redundant
  // pipeline stalls, ~30-45 ms/token on Strix Halo with the asym Llama
  // sliding-window path). The cache is on by default
  // (HIPDNN_EP_GQA_CACHE_SEQLENS=1, set to 0 to disable): the first GQA
  // in a forward pass populates the cache and the remaining 31 layers
  // reuse it.
  //
  // Invalidated by hipdnn_ep_runtime_begin_compute() at the start of each
  // Compute(), called from the EP-side MlirCustomOp::Compute() entry.
  // If the symbol is not exported (older model.dll), invalidation does
  // not happen and the cache is unsafe -- the EP logs a warning at
  // session creation and the user must set HIPDNN_EP_GQA_CACHE_SEQLENS=0.
  bool seqlens_k_cached_valid;
  int32_t seqlens_k_cached_val;
  const void *seqlens_k_cached_ptr;
};

#endif // HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
