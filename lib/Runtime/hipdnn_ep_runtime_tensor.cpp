/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "debug_log.h"
#include "hip_cleanup.h"
#include "hipdnn_ep_runtime.h"
#include "op_profile.h"
#include "runtime_state_internal.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

// Fallback element size when tensor metadata is missing (covers fp32).
static constexpr size_t kDefaultElementSize = 4;
//===----------------------------------------------------------------------===//
// Per-Inference Performance Measurement (gated on HIPDNN_EP_PERF)
//===----------------------------------------------------------------------===//
// Records hipEvents on the GPU stream at phase boundaries to measure:
//   H2D  = time for all host-to-device async copies
//   Compute = time for main_graph GPU kernel dispatches
//   D2H  = time for all device-to-host async copies + stream sync
//
// Phase boundaries are detected by tracking the first call to each function
// type in the per-inference sequence:
//   prepare_input(0)  -> H2D start
//   prepare_output(0) -> H2D end / compute start
//   finalize_output(0)-> compute end / D2H start
//   stream_sync()     -> D2H end, then log results

struct PerfState {
  hipEvent_t h2d_start = nullptr;
  hipEvent_t h2d_end = nullptr;
  hipEvent_t d2h_start = nullptr;
  hipEvent_t d2h_end = nullptr;
  size_t h2d_bytes = 0;
  size_t h2d_count = 0;
  size_t d2h_bytes = 0;
  size_t d2h_count = 0;
  unsigned inference_num = 0;
  bool initialized = false;
};
static PerfState g_perf;

static void perf_ensure_events() {
  if (g_perf.initialized)
    return;
  if (hipEventCreate(&g_perf.h2d_start) != hipSuccess ||
      hipEventCreate(&g_perf.h2d_end) != hipSuccess ||
      hipEventCreate(&g_perf.d2h_start) != hipSuccess ||
      hipEventCreate(&g_perf.d2h_end) != hipSuccess) {
    fprintf(stderr, "[PERF] WARNING: hipEventCreate failed, disabling PERF\n");
    if (g_perf.h2d_start) {
      (void)hipEventDestroy(g_perf.h2d_start);
      g_perf.h2d_start = nullptr;
    }
    if (g_perf.h2d_end) {
      (void)hipEventDestroy(g_perf.h2d_end);
      g_perf.h2d_end = nullptr;
    }
    if (g_perf.d2h_start) {
      (void)hipEventDestroy(g_perf.d2h_start);
      g_perf.d2h_start = nullptr;
    }
    if (g_perf.d2h_end) {
      (void)hipEventDestroy(g_perf.d2h_end);
      g_perf.d2h_end = nullptr;
    }
    return;
  }
  g_perf.initialized = true;
}

//===----------------------------------------------------------------------===//
// GPU Tensor Buffer Pool
//===----------------------------------------------------------------------===//
// Reuses GPU allocations across inferences to eliminate per-call
// hipMalloc/hipFree overhead. Safe because tensor shapes (and therefore sizes)
// are fixed across inferences for a given model.

static std::unordered_map<size_t, std::vector<void *>> g_gpu_buffer_pool;

static void *pool_alloc(size_t size_bytes) {
  assert(size_bytes > 0 && "pool_alloc: size_bytes must be positive");
  auto it = g_gpu_buffer_pool.find(size_bytes);
  if (it != g_gpu_buffer_pool.end() && !it->second.empty()) {
    void *ptr = it->second.back();
    it->second.pop_back();
    return ptr;
  }
  void *ptr = nullptr;
  if (hipMalloc(&ptr, size_bytes) != hipSuccess)
    return nullptr;
  return ptr;
}

static void pool_release(void *ptr, size_t size_bytes) {
  assert(size_bytes > 0 && "pool_release: size_bytes must be positive");
  if (ptr)
    g_gpu_buffer_pool[size_bytes].push_back(ptr);
}

// Element size is read from tensor_t.element_size (set by EP caller)

// Helper function to check and log gcnArchName
static void check_gcnarch(const char *location) {
  if (!hipdnn_ep_debug_enabled())
    return;
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  if (err == hipSuccess) {
    fprintf(stderr, "[%s] gcnArchName='%s' (len=%zu)\n", location,
            prop.gcnArchName, strlen(prop.gcnArchName));
  } else {
    fprintf(stderr, "[%s] ERROR: hipGetDeviceProperties failed: %d\n", location,
            err);
  }
}

// Helper: Calculate total size in bytes for a tensor.
//
// Return value semantics:
//   * size_bytes > 0  -- regular non-empty tensor.
//   * size_bytes == 0 -- legitimate zero-element tensor (any shape[i] == 0),
//                        OR genuinely invalid input (rank>0 with null shape,
//                        any shape[i] < 0, or overflow). Callers that need
//                        to distinguish must inspect shape themselves.
//
// Zero-sized dims are legal under the ONNX spec and are produced routinely
// in practice (e.g. KV-cache `past_key_values.*.key/value` with shape
// [B, H, 0, D] on the first prefill step when past_sequence_length==0;
// OGA VLM decode supplies image_features as [0, hidden_size] every step
// after the prompt has been processed). The prepare_input / prepare_output
// path treats size_bytes==0 as a successful empty pass-through; this helper
// therefore must return 0 silently for zero-sized dims rather than logging
// "Invalid dimension" to stderr, which (a) is misleading because the input
// is correct, and (b) used to spam once per zero-dim tensor per inference
// step on every VLM model.
//
// Strictly-negative dims still indicate corrupt metadata and remain a
// reported error.
static size_t calculateTensorSize(const int64_t *shape, size_t rank,
                                  size_t element_size) {
  if (rank == 0) {
    return element_size; // Rank-0 scalar: 1 element
  }
  if (!shape) {
    return 0;
  }

  // Reject strictly-negative dims (corrupt metadata). Zero is allowed.
  for (size_t i = 0; i < rank; i++) {
    if (shape[i] < 0) {
      fprintf(stderr, "Invalid dimension at index %zu: %lld\n", i,
              (long long)shape[i]);
      return 0;
    }
  }

  // Multiply with overflow check. Skip the division guard when the current
  // dim is zero -- a zero factor cannot overflow, and SIZE_MAX/0 would be
  // undefined behavior. Once total_elements hits zero it stays there, which
  // is the correct answer for any zero-element tensor regardless of the
  // remaining dims.
  size_t total_elements = 1;
  for (size_t i = 0; i < rank; i++) {
    size_t dim = static_cast<size_t>(shape[i]);
    if (dim != 0 && total_elements > SIZE_MAX / dim) {
      fprintf(stderr, "Tensor size overflow at dimension %zu\n", i);
      return 0;
    }
    total_elements *= dim;
  }

  if (total_elements > SIZE_MAX / element_size) {
    fprintf(stderr, "Tensor size overflow when applying element size\n");
    return 0;
  }

  return total_elements * element_size;
}

// Prepare input tensor: parse, validate, allocate GPU buffer, H2D transfer
int hipdnn_ep_tensor_prepare_input(RuntimeState *state, span_t *inputs,
                                   size_t index, size_t expected_rank,
                                   TensorBuffer *out_buffer) {
  check_gcnarch("BEFORE prepare_input");

  // VERIFICATION: Struct sizes
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] === Struct Size Verification ===\n");
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] sizeof(TensorBuffer) = %zu\n",
                    sizeof(TensorBuffer));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, gpu_ptr) = %zu\n",
                    offsetof(TensorBuffer, gpu_ptr));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, host_ptr) = %zu\n",
                    offsetof(TensorBuffer, host_ptr));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, shape_ptr) = %zu\n",
                    offsetof(TensorBuffer, shape_ptr));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, rank) = %zu\n",
                    offsetof(TensorBuffer, rank));
  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] offsetof(TensorBuffer, size_bytes) = %zu\n",
      offsetof(TensorBuffer, size_bytes));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, is_pooled) = %zu\n",
                    offsetof(TensorBuffer, is_pooled));

  // VERIFICATION: tensor_t struct
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] sizeof(tensor_t) = %zu\n",
                    sizeof(tensor_t));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(tensor_t, data) = %zu\n",
                    offsetof(tensor_t, data));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(tensor_t, shape) = %zu\n",
                    offsetof(tensor_t, shape));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(tensor_t, rank) = %zu\n",
                    offsetof(tensor_t, rank));

  // Validate arguments
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!inputs) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null inputs\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!out_buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null out_buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // VERIFICATION: span_t access
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] inputs pointer = %p\n", (void *)inputs);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] inputs->data = %p\n",
                    (void *)inputs->data);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] inputs->count = %zu\n", inputs->count);

  // Validate index bounds
  if (index >= inputs->count) {
    fprintf(
        stderr,
        "hipdnn_ep_tensor_prepare_input: index %zu out of bounds (count=%zu)\n",
        index, inputs->count);
    return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
  }

  // Extract tensor from span
  tensor_t *tensor = &inputs->data[index];

  // DUMP: Raw memory of tensor_t struct
  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] tensor_t struct memory dump (address=%p):\n",
      (void *)tensor);
  auto *bytes = reinterpret_cast<unsigned char *>(tensor);
  for (size_t i = 0; i < sizeof(tensor_t); i++) {
    RUNTIME_DEBUG_LOG("  [%02zu] = 0x%02x\n", i, bytes[i]);
  }

  // DUMP: Field values
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] tensor->data = %p\n", tensor->data);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] tensor->shape = %p\n",
                    (void *)tensor->shape);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] tensor->rank = %zu\n", tensor->rank);

  // Validate field access doesn't corrupt memory (re-read test)
  void *data_before = tensor->data;
  int64_t *shape_before = tensor->shape;
  size_t rank_before = tensor->rank;

  // Re-read and compare
  if (tensor->data != data_before || tensor->shape != shape_before ||
      tensor->rank != rank_before) {
    fprintf(stderr, "[Runtime ERROR] Struct fields changed on re-read!\n");
    fprintf(stderr, "  data: %p -> %p\n", data_before, tensor->data);
    fprintf(stderr, "  shape: %p -> %p\n", (void *)shape_before,
            (void *)tensor->shape);
    fprintf(stderr, "  rank: %zu -> %zu\n", rank_before, tensor->rank);
    return HIPDNN_EP_ERR_NULL_POINTER; // Use generic error code
  }

  // Shape must be present (or rank==0 for scalars).
  if (!tensor->shape && tensor->rank != 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].shape is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate rank
  if (tensor->rank != expected_rank) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: rank mismatch at index %zu "
            "(expected %zu, got %zu)\n",
            index, expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Read element size from tensor struct (set by EP caller)
  size_t element_size = tensor->element_size;
  if (element_size == 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].element_size is 0, "
            "defaulting to %zu\n",
            index, kDefaultElementSize);
    element_size = kDefaultElementSize;
  }

  // Calculate buffer size
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);

  // Empty tensors (any dim == 0 -> size_bytes == 0) are legitimate inputs --
  // e.g. KV cache `past_key_values.*.key/value` with shape [B, H, 0, D] on
  // the prefill step (past_sequence_length == 0).  ORT supplies these with
  // `data == nullptr` because there is nothing to copy; we must accept that
  // and let downstream compiled kernels see an empty dim and iterate zero
  // times.  Skip the null-data check, skip pool_alloc + H2D, and return
  // SUCCESS with a zero-size buffer (gpu_ptr=nullptr is fine because
  // compiled MLIR kernels never dereference it when the corresponding dim
  // is zero).
  if (size_bytes == 0) {
    out_buffer->gpu_ptr = nullptr;
    out_buffer->host_ptr = tensor->data;
    out_buffer->shape_ptr = tensor->shape;
    out_buffer->rank = tensor->rank;
    out_buffer->size_bytes = 0;
    out_buffer->is_pooled = false;
    out_buffer->is_aliased = false;
    return HIPDNN_EP_SUCCESS;
  }

  // Non-empty: data must be non-null now.
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].data is null but "
            "size_bytes=%zu (rank=%zu)\n",
            index, size_bytes, tensor->rank);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_input[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu memory_type=%d\n",
      index, tensor->rank, element_size, size_bytes, tensor->memory_type);

  // Fast path: caller already placed `data` in GPU-accessible memory
  // (TENSOR_MEMORY_GPU), so we alias the buffer instead of pool_alloc + H2D.
  // This is the path that eliminates the per-decode 2 GB KV-cache H2D copy
  // when OGA's MorphiZenEP device interface allocated KV cache via our
  // hipHostMalloc(Mapped|Coherent) allocator (path A). The caller still owns
  // the buffer; finalize_output / free_input must skip pool_release in
  // this case (gated by TensorBuffer.is_aliased).
  //
  // Other memory_type values (CPU / FPGA / NPU) fall through to the legacy
  // host H2D path below — preserves behaviour for hip-test-dll,
  // hip-onnx-runner, and the OGA path-B-only configuration where KV cache
  // still lives in host RAM.
  const bool alias_caller_buffer = (tensor->memory_type == TENSOR_MEMORY_GPU);
  void *gpu_ptr = nullptr;
  if (alias_caller_buffer) {
    gpu_ptr = tensor->data;
  } else {
    // Allocate GPU buffer (pool reuses across inferences)
    gpu_ptr = pool_alloc(size_bytes);
    if (!gpu_ptr) {
      fprintf(stderr,
              "hipdnn_ep_tensor_prepare_input: failed to allocate %zu bytes\n",
              size_bytes);
      return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
    }
  }

  // PERF: at the start of each new inference, flush the previous inference's
  // timing breakdown (one line, easy to grep) and open a new window. We
  // piggy-back on prepare_input(0) instead of using a dedicated sync hook
  // because main_graph IR doesn't emit one.
  //
  // PERF mode trade-off (intentional, not a bug): the hipStreamSynchronize
  // below is *required* to make hipEventElapsedTime return valid per-phase
  // numbers (the H2D / Compute / D2H events are async-recorded on the GPU
  // stream and only become measurable once the stream has drained). The
  // cost is that each inference's stream is forced to fully serialise
  // before the next inference can submit work, which kills the natural
  // pipeline overlap between consecutive inferences and *artificially
  // lowers the measured TPS* compared to a normal run. So:
  //
  //   * HIPDNN_EP_PERF=1 -- accurate per-phase breakdown, sub-real TPS.
  //     Use when you need the H2D / Compute / D2H split.
  //   * HIPDNN_EP_DEBUG=1 -- log-only, no sync, real-throughput TPS.
  //     Use when you want traces but care about wall-clock perf.
  //
  // PERF intentionally no longer inherits from DEBUG (see
  // hipdnn_ep_perf_enabled() in debug_log.h) so adding debug printfs
  // doesn't silently re-impose the sync penalty.
  if (hipdnn_ep_perf_enabled() && index == 0) {
    perf_ensure_events();
    // Flush the previous inference's window (if any). inference_num > 0
    // means we've already record(h2d_start)'d at least once, so the events
    // are valid to query. Using inference_num instead of "h2d_count > 0"
    // keeps this working under Option A, where every input may take the
    // alias fast path and accumulate zero H2D bytes.
    if (g_perf.initialized && g_perf.inference_num > 0) {
      (void)hipStreamSynchronize(static_cast<hipStream_t>(state->stream));
      float h2d_ms = 0, compute_ms = 0, d2h_ms = 0;
      (void)hipEventElapsedTime(&h2d_ms, g_perf.h2d_start, g_perf.h2d_end);
      (void)hipEventElapsedTime(&compute_ms, g_perf.h2d_end, g_perf.d2h_start);
      (void)hipEventElapsedTime(&d2h_ms, g_perf.d2h_start, g_perf.d2h_end);
      const float total_ms = h2d_ms + compute_ms + d2h_ms;
      RUNTIME_PERF_LOG("[PERF] #%u: H2D %zut/%.1fMB/%.2fms | Compute %.2fms | "
                       "D2H %zut/%.1fMB/%.2fms | Total %.2fms\n",
                       g_perf.inference_num, g_perf.h2d_count,
                       g_perf.h2d_bytes / 1048576.0, h2d_ms, compute_ms,
                       g_perf.d2h_count, g_perf.d2h_bytes / 1048576.0, d2h_ms,
                       total_ms);
    }
    g_perf.h2d_bytes = 0;
    g_perf.h2d_count = 0;
    g_perf.d2h_bytes = 0;
    g_perf.d2h_count = 0;
    (void)hipEventRecord(g_perf.h2d_start,
                         static_cast<hipStream_t>(state->stream));
    op_profile_reset(static_cast<OpProfileState *>(state->op_profile));
    g_perf.inference_num++; // marks "window opened"; flush above guards on this
  }

  // H2D transfer (skipped on the alias fast path — caller's buffer is
  // already GPU-accessible, no copy needed).
  if (!alias_caller_buffer) {
    if (hipMemcpyAsync(gpu_ptr, tensor->data, size_bytes, hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(state->stream)) != hipSuccess) {
      fprintf(stderr, "hipdnn_ep_tensor_prepare_input: H2D transfer failed\n");
      HIP_CLEANUP(hipFree(gpu_ptr));
      return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
    }

    // PERF: accumulate H2D bytes (only the actual copy, not aliased buffers)
    if (hipdnn_ep_perf_enabled()) {
      g_perf.h2d_bytes += size_bytes;
      g_perf.h2d_count++;
    }
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false;
  out_buffer->is_aliased = alias_caller_buffer;

  check_gcnarch("AFTER prepare_input");
  return HIPDNN_EP_SUCCESS;
}

// Prepare output tensor: parse, validate, allocate GPU buffer (no H2D)
int hipdnn_ep_tensor_prepare_output(RuntimeState *state, span_t *outputs,
                                    size_t index, size_t expected_rank,
                                    TensorBuffer *out_buffer) {
  // Validate arguments
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!outputs) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null outputs\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!out_buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null out_buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate index bounds
  if (index >= outputs->count) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: index %zu out of bounds "
            "(count=%zu)\n",
            index, outputs->count);
    return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
  }

  // Extract tensor from span
  tensor_t *tensor = &outputs->data[index];

  // Shape must be present (or rank==0 for scalars).
  if (!tensor->shape && tensor->rank != 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].shape is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate rank
  if (tensor->rank != expected_rank) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: rank mismatch (expected %zu, got "
            "%zu)\n",
            expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Read element size from tensor struct (set by EP caller)
  size_t element_size = tensor->element_size;
  if (element_size == 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].element_size is 0, "
            "defaulting to %zu\n",
            index, kDefaultElementSize);
    element_size = kDefaultElementSize;
  }

  // Calculate buffer size
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);

  // Empty output tensors (any dim == 0) are legitimate -- e.g. an op
  // producing a slice with a zero-length dim.  Skip allocation and return
  // success with a zero-size buffer.  Compiled kernels never dereference
  // gpu_ptr when the corresponding dim is zero (the iteration range is
  // empty).
  if (size_bytes == 0) {
    out_buffer->gpu_ptr = nullptr;
    out_buffer->host_ptr = tensor->data;
    out_buffer->shape_ptr = tensor->shape;
    out_buffer->rank = tensor->rank;
    out_buffer->size_bytes = 0;
    out_buffer->is_pooled = false;
    out_buffer->is_aliased = false;
    return HIPDNN_EP_SUCCESS;
  }

  // Non-empty: data must be non-null now.
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].data is null but "
            "size_bytes=%zu (rank=%zu)\n",
            index, size_bytes, tensor->rank);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_output[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu memory_type=%d\n",
      index, tensor->rank, element_size, size_bytes, tensor->memory_type);

  // PERF: record H2D end on first output alloc (after all H2D copies queued)
  if (hipdnn_ep_perf_enabled() && index == 0 && g_perf.initialized) {
    (void)hipEventRecord(g_perf.h2d_end,
                         static_cast<hipStream_t>(state->stream));
  }

  // Fast path: caller's output OrtValue is in GPU-accessible memory, alias
  // it so the kernel writes directly into the caller's buffer and we can
  // skip both the pool_alloc here and the D2H copy in finalize_output. For
  // OGA path A this hits on present_key/present_value tensors that share
  // buffers with past_key/past_value (past_present_share_buffer=true).
  const bool alias_caller_buffer = (tensor->memory_type == TENSOR_MEMORY_GPU);
  void *gpu_ptr = nullptr;
  if (alias_caller_buffer) {
    gpu_ptr = tensor->data;
  } else {
    // Allocate GPU buffer (pool reuses across inferences)
    gpu_ptr = pool_alloc(size_bytes);
    if (!gpu_ptr) {
      fprintf(stderr,
              "hipdnn_ep_tensor_prepare_output: failed to allocate %zu bytes\n",
              size_bytes);
      return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
    }
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false;
  out_buffer->is_aliased = alias_caller_buffer;

  return HIPDNN_EP_SUCCESS;
}

// Finalize output tensor: D2H transfer, sync, release buffer
int hipdnn_ep_tensor_finalize_output(RuntimeState *state,
                                     TensorBuffer *buffer) {
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: null buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Empty output (zero-sized dim): nothing to copy, nothing to release.
  // prepare_output set gpu_ptr=nullptr and size_bytes=0 in this case.
  if (buffer->size_bytes == 0) {
    return HIPDNN_EP_SUCCESS;
  }

  int result = HIPDNN_EP_SUCCESS;

  // PERF: record D2H start on first output finalize (after all compute).
  // We always record, even on the alias fast path, so the [PERF] D2H window
  // is well-defined; aliased outputs just don't add bytes to the accumulator.
  if (hipdnn_ep_perf_enabled() && g_perf.d2h_count == 0 && g_perf.initialized) {
    (void)hipEventRecord(g_perf.d2h_start,
                         static_cast<hipStream_t>(state->stream));
  }

  // D2H transfer (async -- sync happens once after all outputs).
  // Skipped on the alias fast path: gpu_ptr already points into the caller's
  // GPU-accessible host_ptr (same physical pages on AMD APU mapped pinned
  // memory), so the kernel has already written the result; no copy needed.
  if (!buffer->is_aliased) {
    if (hipMemcpyAsync(buffer->host_ptr, buffer->gpu_ptr, buffer->size_bytes,
                       hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(state->stream)) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_tensor_finalize_output: D2H transfer failed\n");
      result = HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;
      // Continue to cleanup even on error (best-effort)
    }

    // PERF: accumulate D2H bytes (only the actual copies, not aliased)
    if (hipdnn_ep_perf_enabled()) {
      g_perf.d2h_bytes += buffer->size_bytes;
      g_perf.d2h_count++;
    }
  }

  // PERF: re-record d2h_end after every finalize_output (aliased or not).
  // The "real" last call wins; in-between records are cheap and let us avoid
  // having to know which finalize_output is the last one (the MLIR-emitted
  // main_graph does not signal end-of-inference back to us, see
  // prepare_input flush logic).
  if (hipdnn_ep_perf_enabled() && g_perf.initialized) {
    (void)hipEventRecord(g_perf.d2h_end,
                         static_cast<hipStream_t>(state->stream));
  }

  // Return buffer to pool only if we own it. Aliased buffers are owned by
  // the caller (e.g. OGA's KV cache OrtValue under path A); freeing them
  // would corrupt the caller's allocation.
  if (!buffer->is_aliased) {
    pool_release(buffer->gpu_ptr, buffer->size_bytes);
  }
  buffer->gpu_ptr = nullptr;
  buffer->is_aliased = false;

  return result;
}

// Synchronize GPU stream once (called after all finalize_output calls).
int hipdnn_ep_stream_sync(RuntimeState *state) {
  fprintf(stderr, "[stream_sync] enter state=%p\n", (void *)state);
  fflush(stderr);
  if (!state) {
    fprintf(stderr, "hipdnn_ep_stream_sync: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // PERF: record D2H end event (after all D2H copies queued)
  if (hipdnn_ep_perf_enabled() && g_perf.initialized) {
    (void)hipEventRecord(g_perf.d2h_end,
                         static_cast<hipStream_t>(state->stream));
  }

  if (hipStreamSynchronize(static_cast<hipStream_t>(state->stream)) !=
      hipSuccess) {
    fprintf(stderr, "hipdnn_ep_stream_sync: stream sync failed\n");
    return HIPDNN_EP_ERR_STREAM_SYNC_FAILED;
  }

  // PERF: compute and log timing breakdown
  if (hipdnn_ep_perf_enabled() && g_perf.initialized) {
    float h2d_ms = 0, compute_ms = 0, d2h_ms = 0;
    (void)hipEventElapsedTime(&h2d_ms, g_perf.h2d_start, g_perf.h2d_end);
    (void)hipEventElapsedTime(&compute_ms, g_perf.h2d_end, g_perf.d2h_start);
    (void)hipEventElapsedTime(&d2h_ms, g_perf.d2h_start, g_perf.d2h_end);
    float total_ms = h2d_ms + compute_ms + d2h_ms;

    g_perf.inference_num++;
    fprintf(stderr,
            "[PERF] inference #%u:\n"
            "  H2D:     %zu tensors, %zu bytes (%.1f MB), %.2f ms\n"
            "  Compute: %.2f ms\n"
            "  D2H:     %zu tensors, %zu bytes (%.1f MB), %.2f ms\n"
            "  Total:   %.2f ms  (H2D %.1f%% | Compute %.1f%% | D2H %.1f%%)\n",
            g_perf.inference_num, g_perf.h2d_count, g_perf.h2d_bytes,
            (double)g_perf.h2d_bytes / (1024.0 * 1024.0), h2d_ms, compute_ms,
            g_perf.d2h_count, g_perf.d2h_bytes,
            (double)g_perf.d2h_bytes / (1024.0 * 1024.0), d2h_ms, total_ms,
            total_ms > 0 ? (h2d_ms / total_ms * 100.0) : 0.0,
            total_ms > 0 ? (compute_ms / total_ms * 100.0) : 0.0,
            total_ms > 0 ? (d2h_ms / total_ms * 100.0) : 0.0);
    auto *op_ps = static_cast<OpProfileState *>(state->op_profile);
    op_profile_resolve_and_print(op_ps);
  }

  return HIPDNN_EP_SUCCESS;
}

// Release input tensor buffer (no D2H transfer needed)
void hipdnn_ep_tensor_free_input(RuntimeState *state, TensorBuffer *buffer) {
  (void)state;
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_free_input: null buffer\n");
    return;
  }

  // Empty input (size_bytes == 0, e.g. KV cache with past_sequence_length==0
  // on prefill): no pool allocation was made; nothing to release.
  if (buffer->size_bytes == 0) {
    buffer->gpu_ptr = nullptr;
    return;
  }

  // Return buffer to pool only if we own it. Aliased buffers are owned by
  // the caller (e.g. OGA's KV cache OrtValue under path A); freeing them
  // would corrupt the caller's allocation.
  if (!buffer->is_aliased) {
    pool_release(buffer->gpu_ptr, buffer->size_bytes);
  }
  buffer->gpu_ptr = nullptr;
  buffer->is_aliased = false;
}

//===----------------------------------------------------------------------===//
// TensorBuffer Field Accessors (Opaque Pattern)
//===----------------------------------------------------------------------===//

void *hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer *buffer) {
  assert(buffer && "get_gpu_ptr: null buffer");
  return buffer ? buffer->gpu_ptr : nullptr;
}

void *hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer *buffer) {
  assert(buffer && "get_host_ptr: null buffer");
  return buffer ? buffer->host_ptr : nullptr;
}

int64_t *hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer *buffer) {
  assert(buffer && "get_shape_ptr: null buffer");
  return buffer ? buffer->shape_ptr : nullptr;
}

size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer *buffer) {
  assert(buffer && "get_rank: null buffer");
  return buffer ? buffer->rank : 0;
}

size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer *buffer) {
  assert(buffer && "get_size_bytes: null buffer");
  return buffer ? buffer->size_bytes : 0;
}
