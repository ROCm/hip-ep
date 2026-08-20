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
// For hipdnn_output_allocator_t (the output_allocator field below). Acyclic:
// hipdnn_ep_runtime.h only forward-declares RuntimeState; it does not include
// this internal header.
#include "hipdnn_ep_runtime.h"
// For `struct OpState` (the op_states array element type below). op_state.h is
// layout-agnostic (it only forward-declares RuntimeState), so including it here
// is acyclic.
#include "op_state.h"

#include <mutex>

struct RuntimePoolSite {
  int num_domains;
  void **pool_base;
  size_t *pool_size;
};

struct RuntimeHostScratchSite {
  void *base;
  size_t size;
};

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

  // Memory pooling support — module sites containing function-local domains.
  //
  // hip-pool-allocs assigns each top-level function a deterministic module
  // ordinal (site_id), then partitions that function's allocs into local
  // dominance domains. Runtime identity is the pair (site_id, domain_id), so
  // caller and outlined-helper domain zero never share a backing allocation.
  //
  // Both dimensions grow lazily without a compile-time cap and stabilize after
  // the first inference. No generated code caches addresses of table entries.
  int num_pool_sites;
  RuntimePoolSite *pool_sites;
  int legacy_pool_site_id; // Site initialized by hipdnn_ep_pool_init
  size_t *buffer_offsets;  // Offsets for static buffers in domain 0
  size_t num_buffers;      // Static buffer count in domain 0

  // Shared workspace for operator temp buffers (MatMul GEMM ws, GQA pipeline).
  // Lazily grown via hipdnn_ep_state_ensure_workspace(); never shrinks.
  void *workspace;
  size_t workspace_size;

  // Dedicated device metadata scratch for Slice. It is intentionally separate
  // from the general operator workspace so concurrent Slice host calls only
  // serialize each other, not unrelated operators. The mutex is heap-created
  // because RuntimeState itself is malloc-allocated and therefore does not run
  // C++ member constructors.
  void *slice_metadata_scratch;
  size_t slice_metadata_scratch_size;
  std::mutex *slice_metadata_mutex;

  // Per-function-site host-mapped scratch buffers for tiny host-fed scalars
  // routed away from the GPU pool by hip-materialize-host-scalars.
  // hipHostMalloc(hipHostMallocMapped): host-writable AND GPU-readable.
  // Grow-on-demand via hipdnn_ep_get_host_scratch_base(); never shrinks.
  // Distinct compiled functions have disjoint storage, so a helper cannot
  // invalidate its caller's live offset-zero scalar while growing. The table
  // is hipHostFree'd in cleanup.
  int num_host_scratch_sites;
  RuntimeHostScratchSite *host_scratch_sites;

  // Output allocator installed by the EP before inference_compute via
  // hipdnn_ep_set_output_allocator. hipdnn_ep_alloc_output forwards to
  // allocate(self, ...). Borrowed: `self` is EP-owned, never freed here.
  // allocate == nullptr means no allocator has been installed yet;
  // zero-initialized in initialize_state_handles.
  hipdnn_output_allocator_t output_allocator;

  // Per-session scratch buffer for wrap_qmoe transient device buffers
  // (expert_indices, expert_weights, gather_buf, fc1_buf, act_buf, fc2_buf,
  // token_ids, token_wts -- 8 sub-buffers laid out at fixed offsets).
  //
  // Why this exists: pre-cache wrap_qmoe issued 8 hipMalloc + 8 hipFree per
  // call, every layer, every inference. On 24-layer gpt-oss-20b that's 192
  // mallocs + 192 frees per token; HIP's hipMalloc takes ~50 us each on
  // Windows, so the storm cost ~10-12 ms/token and bottlenecked decode TPS to
  // roughly half the Vulkan baseline on the same hardware.
  //
  // Layout policy: one contiguous buffer sized to fit ALL sub-buffers for the
  // largest (num_tokens, hidden, inter, k, num_experts, elem) shape ever seen
  // by this session. Sub-buffer offsets recomputed per-call (cheap arithmetic);
  // the buffer itself grows on demand via hipdnn_ep_state_ensure_qmoe_scratch
  // and never shrinks (mirrors the `workspace` field's policy). Shared by all
  // qmoe instances in the session: safe because the HIP stream is serialised,
  // so the next qmoe launches only after the previous one's kernels finish.
  //
  // Pinned host mirror is needed for the 24-bytes-per-layer D2H readback of
  // expert routing decisions (still required at decode pre-Phase-2). hipHost-
  // Malloc'd once with hipHostMallocDefault; reused across calls without sync.
  void *qmoe_scratch;
  size_t qmoe_scratch_size;
  void *qmoe_host_scratch; // pinned host mirror for D2H of expert idx/weights
  size_t qmoe_host_scratch_size;

  // Per-session scratch buffer for the MIOpen convolution workspace
  // (wrap_miopenConvolutionForward, both 2D and the H=1 1D conv path).
  //
  // The MIOpen forward-convolution Find API selects an algorithm whose
  // workspace requirement is shape-dependent (winograd/gemm/etc). Whisper's
  // encoder front-end runs the same two Conv shapes every inference
  // (Cin=128/Cout=1280 K=3 s=1, Cin=1280/Cout=1280 K=3 s=2), so a per-call
  // hipMalloc/hipFree of the workspace would be wasted work after the
  // first call. Same grow-on-demand policy as qmoe_scratch above: lazily
  // allocated on first use, never shrinks, freed in
  // hipdnn_ep_state_cleanup. Single-buffer reuse is safe because the HIP
  // stream is serialised -- the next conv launches only after the previous
  // miopenConvolutionForward + bias add have consumed the workspace.
  void *conv_scratch;
  size_t conv_scratch_size;

  // Per-session scratch for the W4A8 dp4a matmul_nbits decode path
  // (hip_matmul_nbits_dp4a). One contiguous device buffer holding the
  // per-token quantized activation (int8, K bytes, nibble-deinterleaved) plus
  // the per-group activation scales (float, ceil(K/block_size)). Same
  // grow-on-demand / never-shrink policy as conv_scratch; lazily allocated on
  // first dp4a call, freed in hipdnn_ep_state_cleanup. Single-buffer reuse is
  // safe because the HIP stream is serialised (the next matmul_nbits only
  // launches after the previous dp4a gemv has consumed the quantized row).
  void *matmul_dp4a_scratch;
  size_t matmul_dp4a_scratch_size;

  // Per-session scratch for the linear-attention chunk-parallel gated_delta
  // prefill (hip_linear_attention_prefill_chunked). One contiguous device
  // buffer holding the per-(head,chunk) Uloc/W/rlast/alast tiles and the
  // chunk-start states for one window of the sequence, plus the cross-window
  // carry of the recurrent state. Sized by the window rather than by seq_len,
  // so it stops growing once the sequence exceeds one window. Same
  // grow-on-demand / never-shrink policy as conv_scratch; lazily allocated on
  // first prefill, freed in hipdnn_ep_state_cleanup. Single-buffer reuse is
  // safe because the HIP stream is serialised (the three prefill passes consume
  // it before the next linear-attention layer launches). Replaces a
  // process-static buffer so the footprint is bounded to the session and
  // released on teardown.
  void *la_scratch;
  size_t la_scratch_size;

  // NOTE: the GQA GEMM descriptor cache (GqaGemmCache) formerly lived here as
  // gqa_gemm_cache. It is now per-op-instance: each gqa instance owns one in
  // its GqaState op-state slot (see op_states below and
  // docs/design/op-state-slots-design.md), so concurrent sessions (and
  // distinct GQA layers) no longer share one descriptor map.

  // NOTE: the MultiHeadAttention GEMM descriptor cache (MhaGemmCache) formerly
  // lived here as mha_gemm_cache. It is now per-op-instance: each
  // multi_head_attention instance owns one in its MhaState op-state slot (see
  // op_states below and docs/design/op-state-slots-design.md).

  // NOTE: the CausalConvWithState MIOpen descriptor + algorithm cache
  // (CausalConvCache) formerly lived here as causal_conv_cache. It is now
  // per-op-instance: each causal_conv_with_state instance owns one in its
  // CausalConvState op-state slot (see op_states below and
  // docs/design/op-state-slots-design.md), so concurrent instances no longer
  // share one descriptor cache.

  // Asym zero_points unpack cache (ZpUnpackCache*) used by wrap_qmoe.
  //
  // The asym AWQ path stores zero_points as packed nibbles [N, ceil(K/bs/2)].
  // Two unpacked layouts are needed (u8 [N, K/bs] for GEMV/naive, fp16 for
  // WMMA/col-major GEMV M>1), and naively the unpack kernel is launched on
  // every call. Since zero_points points into the model constants blob (stable
  // for the session lifetime), we cache the unpacked buffer per input pointer.
  // Lazily created on first asym call. Freed in hipdnn_ep_state_cleanup via
  // hipdnn_ep_zp_unpack_cache_destroy.
  //
  // NOTE: matmul_nbits no longer uses this shared cache -- it owns a
  // per-instance ZpUnpackCache in its MatmulNbitsState op-state slot. This
  // field is therefore qmoe-owned.
  void *zp_unpack_cache;

  // Per-operator profiling state (OpProfileState*, gated on HIPDNN_EP_PERF).
  // Allocated in state_init, freed in state_cleanup.
  void *op_profile;

  // First host-observed operation error in the current Compute() call. This
  // preserves wrapper failures even if queuing the mirrored device flag fails.
  int host_error_status;

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
  // pipeline stalls, tens of ms per token on long-context decode). The cache
  // is on by default
  // (HIPDNN_EP_GQA_CACHE_SEQLENS=1, set to 0 to disable): the first GQA
  // in a forward pass populates the cache and the remaining 31 layers
  // reuse it.
  //
  // Invalidated by hipdnn_ep_runtime_begin_compute() at the start of each
  // Compute(), called from the EP-side MlirCustomOp::Compute() entry.
  // If the symbol is not exported (older per-model bitcode), invalidation
  // does not happen and the cache is unsafe -- the EP logs a warning at
  // session creation and the user must set HIPDNN_EP_GQA_CACHE_SEQLENS=0.
  bool seqlens_k_cached_valid;
  int32_t seqlens_k_cached_val;
  const void *seqlens_k_cached_ptr;

  // Independent loop-carrier bank blocks are checked out exclusively from a
  // synchronized best-fit cache and retained to the RuntimeState high-water.
  // Frames whose explicit destroy encountered a stream-sync failure remain
  // quarantined until a later successful graph sync proves their blocks safe
  // to recycle. These C++ implementation types stay opaque here.
  void *loop_bank_cache;
  void *quarantined_loop_frames;
  void *loop_frames_mutex;

  // Per-op state slots (see docs/design/op-state-slots-design.md). One entry
  // per stateful op instance, sized by --assign-op-state-slots and constructed
  // by the generated @hipdnn_ep_op_states_init_fn during inference_init. Each
  // entry's concrete type derives from OpState; cleanup walks the array calling
  // each object's `deletor`, so teardown needs no per-type knowledge. Managed
  // by hipdnn_ep_op_states_alloc / _set / _get in op_state.cpp.
  OpState **op_states;
  int num_op_states;
};

#endif // HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
