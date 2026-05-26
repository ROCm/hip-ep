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
  // Per-inference flags for which events were actually recorded.  Required
  // because a model whose outputs are all Category-C deferred (e.g. NonZero)
  // never executes the `finalize_output` body that records `d2h_start` --
  // calling `hipEventElapsedTime` on an unrecorded event returns
  // `hipErrorInvalidHandle` (400) and sets the sticky HIP error state, which
  // then leaks into the next inference's kernel launches (`hipGetLastError`
  // returns the stale 400 and the wrapper interprets it as a launch failure).
  // Reset to false at the start of each inference (in `prepare_input` with
  // index==0); flipped to true wherever the matching `hipEventRecord` runs.
  bool h2d_start_recorded = false;
  bool h2d_end_recorded = false;
  bool d2h_start_recorded = false;
  bool d2h_end_recorded = false;
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

// Helper: Calculate total size in bytes for a tensor
// Returns 0 on overflow, invalid dimensions, OR a legitimately empty tensor
// (any dim == 0). Callers MUST distinguish "empty" from "error" themselves --
// the easiest way is to check shape validity first (rank, non-null, no
// negative dims) and then treat size_bytes==0 as the empty-tensor sentinel.
// We deliberately do NOT log a warning when shape[i] == 0: empty tensors are
// legal ONNX (e.g. text-only inference passes image_features=[0, H]).
static size_t calculateTensorSize(const int64_t *shape, size_t rank,
                                  size_t element_size) {
  if (rank == 0) {
    return element_size; // Rank-0 scalar: 1 element
  }
  if (!shape) {
    return 0;
  }

  // Validate dimensions: shape[i] < 0 is an error (corrupted descriptor);
  // shape[i] == 0 produces total_elements == 0 below and the function
  // returns 0 -- treated by callers as a legitimately empty tensor, not
  // an error.
  for (size_t i = 0; i < rank; i++) {
    if (shape[i] < 0) {
      fprintf(stderr, "Invalid dimension at index %zu: %lld\n", i,
              (long long)shape[i]);
      return 0;
    }
  }

  // Calculate total number of elements with overflow check
  size_t total_elements = 1;
  for (size_t i = 0; i < rank; i++) {
    if (total_elements > SIZE_MAX / static_cast<size_t>(shape[i])) {
      fprintf(stderr, "Tensor size overflow at dimension %zu\n", i);
      return 0;
    }
    total_elements *= static_cast<size_t>(shape[i]);
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

  // Detect an empty input: any dim == 0 makes calculateTensorSize() return 0
  // (it stopped treating shape[i]==0 as an error -- see the helper). Empty
  // tensors are legal ONNX and arrive with `tensor->data == nullptr` (no
  // backing storage for zero elements). They take a separate path here:
  // we allocate a 1-byte sentinel from the pool so that
  //   * downstream wrappers that null-check the buffer pointer don't
  //     fail (mirrors the Cat-C "publish a non-null buffer even at N=0"
  //     convention -- see CLAUDE.md);
  //   * pool_release() (which asserts size_bytes > 0) still works in
  //     free_input;
  //   * the H2D copy is skipped (no source bytes).
  // Negative dims still hit the `< 0` check inside calculateTensorSize and
  // are surfaced as an INVALID_DIMENSION error.
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);
  const bool is_empty_input = (size_bytes == 0);
  if (is_empty_input) {
    // Distinguish "empty" from "error": calculateTensorSize prints to
    // stderr on overflow / negative dim; getting here with shape valid
    // means all dims are non-negative and at least one is zero.
    if (tensor->rank > 0 && tensor->shape) {
      bool has_negative = false;
      for (size_t i = 0; i < tensor->rank; i++) {
        if (tensor->shape[i] < 0) {
          has_negative = true;
          break;
        }
      }
      if (has_negative) {
        return HIPDNN_EP_ERR_INVALID_DIMENSION;
      }
    }
    // Sentinel size for the pool allocator; the real "this tensor has 0
    // elements" signal lives in tensor->shape[i]==0 and is read by
    // downstream wrappers from the shape descriptor.
    size_bytes = 1;
  } else {
    // Non-empty input: tensor->data MUST be a real backing buffer.
    if (!tensor->data) {
      fprintf(stderr,
              "hipdnn_ep_tensor_prepare_input: tensor[%zu].data is null\n",
              index);
      return HIPDNN_EP_ERR_NULL_POINTER;
    }
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_input[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu memory_type=%d%s\n",
      index, tensor->rank, element_size, size_bytes, tensor->memory_type,
      is_empty_input ? " (empty input -> sentinel buffer, H2D skipped)" : "");

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
  //
  // Empty inputs (is_empty_input) ALWAYS take the sentinel pool path
  // regardless of memory_type -- caller's data pointer is null, so the
  // alias fast-path would propagate that null to downstream consumers
  // and trip the same null-pointer checks we're trying to avoid.
  const bool alias_caller_buffer =
      (tensor->memory_type == TENSOR_MEMORY_GPU) && !is_empty_input;
  void *gpu_ptr = nullptr;
  if (alias_caller_buffer) {
    gpu_ptr = tensor->data;
  } else {
    // Allocate GPU buffer (pool reuses across inferences). For
    // is_empty_input this is a 1-byte sentinel; bucket re-use in
    // pool_alloc makes that essentially free across calls.
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
      // Only query events that were actually recorded last inference.
      // Models with all Category-C deferred outputs (e.g. NonZero) never run
      // the body of `finalize_output`, so `d2h_start` is unrecorded -- calling
      // `hipEventElapsedTime` on it returns hipErrorInvalidHandle and pollutes
      // the sticky HIP error state for the next inference's kernel launches.
      float h2d_ms = 0, compute_ms = 0, d2h_ms = 0;
      if (g_perf.h2d_start_recorded && g_perf.h2d_end_recorded)
        (void)hipEventElapsedTime(&h2d_ms, g_perf.h2d_start, g_perf.h2d_end);
      if (g_perf.h2d_end_recorded && g_perf.d2h_start_recorded)
        (void)hipEventElapsedTime(&compute_ms, g_perf.h2d_end,
                                  g_perf.d2h_start);
      if (g_perf.d2h_start_recorded && g_perf.d2h_end_recorded)
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
    g_perf.h2d_start_recorded = false;
    g_perf.h2d_end_recorded = false;
    g_perf.d2h_start_recorded = false;
    g_perf.d2h_end_recorded = false;
    (void)hipEventRecord(g_perf.h2d_start,
                         static_cast<hipStream_t>(state->stream));
    g_perf.h2d_start_recorded = true;
    op_profile_reset(static_cast<OpProfileState *>(state->op_profile));
    g_perf.inference_num++; // marks "window opened"; flush above guards on this
  }

  // H2D transfer (skipped on the alias fast path — caller's buffer is
  // already GPU-accessible, no copy needed -- and on the empty-input
  // sentinel path, where tensor->data is null and there are no source
  // bytes to copy).
  if (!alias_caller_buffer && !is_empty_input) {
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

  // Category-C deferred-output sentinel: the EP knows this output's shape
  // is only computable after inference_compute (e.g. an op publishes its
  // dim to a RuntimeSlot during execution), so it could not yet allocate
  // the ORT OrtValue. The sentinel is `data == nullptr` with a non-null
  // `shape` and the matching `rank`. The EP keeps shape as a stack-
  // resident upper-bound (or sentinel-filled) array so the
  // buildMemrefDescriptor codegen has a valid array to GEP, but no GPU
  // buffer is allocated and no D2H is queued -- main_graph reaches the
  // wrap_* that owns this slot, which publishes its own GPU buffer
  // through the dyn-pool ABI. The EP reads the slot post-compute,
  // ctx.GetOutput's a fresh OrtValue with the resolved shape, and copies
  // the slot buffer into it.
  if (tensor->data == nullptr) {
    if (tensor->rank != expected_rank) {
      fprintf(stderr,
              "hipdnn_ep_tensor_prepare_output: Category-C deferred rank "
              "mismatch (expected %zu, got %zu) at index %zu\n",
              expected_rank, tensor->rank, index);
      return HIPDNN_EP_ERR_RANK_MISMATCH;
    }
    RUNTIME_DEBUG_LOG(
        "[Runtime DEBUG] prepare_output[%zu]: Category-C deferred sentinel "
        "(rank=%zu)\n",
        index, tensor->rank);
    out_buffer->gpu_ptr = nullptr;
    out_buffer->host_ptr = nullptr;
    out_buffer->shape_ptr = tensor->shape;
    out_buffer->rank = tensor->rank;
    out_buffer->size_bytes = 0;
    out_buffer->is_pooled = false;
    out_buffer->is_aliased = false;
    return HIPDNN_EP_SUCCESS;
  }

  // Validate tensor pointers
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].data is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
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
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_output[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu memory_type=%d\n",
      index, tensor->rank, element_size, size_bytes, tensor->memory_type);

  // PERF: record H2D end on first output alloc (after all H2D copies queued)
  if (hipdnn_ep_perf_enabled() && index == 0 && g_perf.initialized) {
    (void)hipEventRecord(g_perf.h2d_end,
                         static_cast<hipStream_t>(state->stream));
    g_perf.h2d_end_recorded = true;
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

  // Category-C deferred sentinel: prepare_output set gpu_ptr = host_ptr =
  // null with size_bytes == 0. Nothing to copy or release here -- the EP
  // handles D2H + ORT allocation post-compute via the dyn-slot ABI.
  if (buffer->gpu_ptr == nullptr && buffer->host_ptr == nullptr &&
      buffer->size_bytes == 0) {
    RUNTIME_DEBUG_LOG(
        "[Runtime DEBUG] finalize_output: Category-C deferred sentinel "
        "skipped\n");
    return HIPDNN_EP_SUCCESS;
  }

  int result = HIPDNN_EP_SUCCESS;

  // PERF: record D2H start on first output finalize (after all compute).
  // We always record, even on the alias fast path, so the [PERF] D2H window
  // is well-defined; aliased outputs just don't add bytes to the accumulator.
  if (hipdnn_ep_perf_enabled() && g_perf.d2h_count == 0 && g_perf.initialized) {
    (void)hipEventRecord(g_perf.d2h_start,
                         static_cast<hipStream_t>(state->stream));
    g_perf.d2h_start_recorded = true;
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
    g_perf.d2h_end_recorded = true;
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
  if (!state) {
    fprintf(stderr, "hipdnn_ep_stream_sync: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // PERF: record D2H end event (after all D2H copies queued)
  if (hipdnn_ep_perf_enabled() && g_perf.initialized) {
    (void)hipEventRecord(g_perf.d2h_end,
                         static_cast<hipStream_t>(state->stream));
    g_perf.d2h_end_recorded = true;
  }

  if (hipStreamSynchronize(static_cast<hipStream_t>(state->stream)) !=
      hipSuccess) {
    fprintf(stderr, "hipdnn_ep_stream_sync: stream sync failed\n");
    return HIPDNN_EP_ERR_STREAM_SYNC_FAILED;
  }

  // PERF: compute and log timing breakdown. Only query events that were
  // actually recorded this inference -- models with all Category-C deferred
  // outputs (e.g. NonZero) skip the body of `finalize_output` that records
  // `d2h_start`, and calling `hipEventElapsedTime` on an unrecorded event
  // returns hipErrorInvalidHandle (400). That sticky error then leaks into
  // the next inference's kernel launches (where `hipGetLastError` returns
  // 400 and the wrapper mis-reports the kernel as failed -- symptom: NonZero
  // returns rc=400, slot publish never happens, MlirCustomOp aborts with
  // "RuntimeSlot ... unpublished" invariant violation).
  if (hipdnn_ep_perf_enabled() && g_perf.initialized) {
    float h2d_ms = 0, compute_ms = 0, d2h_ms = 0;
    if (g_perf.h2d_start_recorded && g_perf.h2d_end_recorded)
      (void)hipEventElapsedTime(&h2d_ms, g_perf.h2d_start, g_perf.h2d_end);
    if (g_perf.h2d_end_recorded && g_perf.d2h_start_recorded)
      (void)hipEventElapsedTime(&compute_ms, g_perf.h2d_end, g_perf.d2h_start);
    if (g_perf.d2h_start_recorded && g_perf.d2h_end_recorded)
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
