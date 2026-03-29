/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "debug_log.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdio>
#include <cstring>

// Macro for best-effort cleanup: logs errors but continues cleanup
#define HIP_CLEANUP(expr)                                                      \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "Warning: " #expr " failed with error %d\n", (int)_err); \
    }                                                                          \
  } while (0)
//==============================================================================
// Per-Inference Performance Measurement (gated on HIPDNN_EP_PERF)
//==============================================================================
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

static struct {
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
} g_perf;

static void perf_ensure_events() {
  if (g_perf.initialized)
    return;
  (void)hipEventCreate(&g_perf.h2d_start);
  (void)hipEventCreate(&g_perf.h2d_end);
  (void)hipEventCreate(&g_perf.d2h_start);
  (void)hipEventCreate(&g_perf.d2h_end);
  g_perf.initialized = true;
}

//==============================================================================
// HIP Graph Capture (gated on HIPDNN_EP_GRAPH)
//==============================================================================
// On inference #2 (after warm-up), captures all main_graph GPU operations into
// a hipGraph_t via stream capture. On inference #3+ the captured graph is
// replayed, skipping main_graph entirely. GPU buffer pointers from the capture
// inference are pinned and reused across replays.

static struct {
  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec = nullptr;
  unsigned inference_num = 0;
  unsigned d2h_count = 0;
  bool capturing = false;
  bool capture_ok = false;
  std::vector<void *> captured_input_ptrs;
  std::vector<void *> captured_output_ptrs;
} g_graph;

//==============================================================================
// GPU Tensor Buffer Pool
//==============================================================================
// Reuses GPU allocations across inferences to eliminate per-call
// hipMalloc/hipFree overhead. Safe because tensor shapes (and therefore sizes)
// are fixed across inferences for a given model.

static std::unordered_map<size_t, std::vector<void *>> g_gpu_buffer_pool;

static void *pool_alloc(size_t size_bytes) {
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
// Returns 0 on error (overflow or invalid dimensions)
static size_t calculateTensorSize(const int64_t *shape, size_t rank,
                                  size_t element_size) {
  if (rank == 0) {
    return element_size; // Rank-0 scalar: 1 element
  }
  if (!shape) {
    return 0;
  }

  // Validate all dimensions are positive
  for (size_t i = 0; i < rank; i++) {
    if (shape[i] <= 0) {
      fprintf(stderr, "Invalid dimension at index %zu: %lld\n", i,
              (long long)shape[i]);
      return 0;
    }
  }

  // Calculate total number of elements with overflow check
  size_t total_elements = 1;
  for (size_t i = 0; i < rank; i++) {
    if (total_elements > SIZE_MAX / (size_t)shape[i]) {
      fprintf(stderr, "Tensor size overflow at dimension %zu\n", i);
      return 0;
    }
    total_elements *= (size_t)shape[i];
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
  unsigned char *bytes = (unsigned char *)tensor;
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

  // Validate tensor pointers
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].data is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
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
            "hipdnn_ep_tensor_prepare_input: rank mismatch (expected %zu, got "
            "%zu)\n",
            expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Read element size from tensor struct (set by EP caller)
  size_t element_size = tensor->element_size;
  if (element_size == 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].element_size is 0, "
            "defaulting to 4\n",
            index);
    element_size = 4;
  }

  // Calculate buffer size
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_input[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu\n",
      index, tensor->rank, element_size, size_bytes);

  // GRAPH: track inference number and reset per-inference counters
  // (must be before replay check which depends on inference_num)
  if (hipdnn_ep_graph_enabled() && index == 0) {
    g_graph.inference_num++;
    g_graph.d2h_count = 0;
  }

  // GRAPH replay: reuse captured GPU pointer instead of pool_alloc.
  // MUST be before pool_alloc to avoid leaking a buffer every call.
  if (hipdnn_ep_graph_enabled() && g_graph.capture_ok &&
      g_graph.inference_num >= 3) {
    if (index >= g_graph.captured_input_ptrs.size()) {
      fprintf(stderr, "[GRAPH] ERROR: input index %zu out of captured range "
                      "(%zu)\n",
              index, g_graph.captured_input_ptrs.size());
      return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
    }
    void *captured_ptr = g_graph.captured_input_ptrs[index];
    if (hipMemcpyAsync(captured_ptr, tensor->data, size_bytes,
                       hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(state->stream)) != hipSuccess) {
      fprintf(stderr, "hipdnn_ep_tensor_prepare_input: H2D transfer failed "
                      "(replay)\n");
      return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
    }
    out_buffer->gpu_ptr = captured_ptr;
    out_buffer->host_ptr = tensor->data;
    out_buffer->shape_ptr = tensor->shape;
    out_buffer->rank = tensor->rank;
    out_buffer->size_bytes = size_bytes;
    out_buffer->is_pooled = true;
    return HIPDNN_EP_SUCCESS;
  }

  // Allocate GPU buffer (pool reuses across inferences)
  void *gpu_ptr = pool_alloc(size_bytes);
  if (!gpu_ptr) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: failed to allocate %zu bytes\n",
            size_bytes);
    return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
  }

  // PERF: record H2D start on first input
  if (hipdnn_ep_perf_enabled() && index == 0) {
    perf_ensure_events();
    g_perf.h2d_bytes = 0;
    g_perf.h2d_count = 0;
    g_perf.d2h_bytes = 0;
    g_perf.d2h_count = 0;
    (void)hipEventRecord(g_perf.h2d_start,
                         static_cast<hipStream_t>(state->stream));
  }

  // H2D transfer
  if (hipMemcpyAsync(gpu_ptr, tensor->data, size_bytes, hipMemcpyHostToDevice,
                     static_cast<hipStream_t>(state->stream)) != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: H2D transfer failed\n");
    HIP_CLEANUP(hipFree(gpu_ptr));
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  }

  // GRAPH: record pointer mapping during capture inference
  if (hipdnn_ep_graph_enabled() && g_graph.inference_num == 2) {
    g_graph.captured_input_ptrs.push_back(gpu_ptr);
  }

  // PERF: accumulate H2D bytes
  if (hipdnn_ep_perf_enabled()) {
    g_perf.h2d_bytes += size_bytes;
    g_perf.h2d_count++;
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false;

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
            "defaulting to 4\n",
            index);
    element_size = 4;
  }

  // Calculate buffer size
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_output[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu\n",
      index, tensor->rank, element_size, size_bytes);

  // PERF: record H2D end on first output alloc (after all H2D copies queued)
  if (hipdnn_ep_perf_enabled() && index == 0 && g_perf.initialized) {
    (void)hipEventRecord(g_perf.h2d_end,
                         static_cast<hipStream_t>(state->stream));
  }

  // GRAPH: begin stream capture on inference #2 (after warm-up)
  // Placed AFTER the PERF event so h2d_end is NOT inside the captured graph.
  if (hipdnn_ep_graph_enabled() && index == 0 &&
      g_graph.inference_num == 2 && !g_graph.capturing) {
    hipError_t err = hipStreamBeginCapture(
        static_cast<hipStream_t>(state->stream),
        hipStreamCaptureModeGlobal);
    if (err == hipSuccess) {
      g_graph.capturing = true;
      fprintf(stderr, "[GRAPH] inference #%u: stream capture STARTED "
                      "(mode=Global)\n", g_graph.inference_num);
    } else {
      fprintf(stderr, "[GRAPH] hipStreamBeginCapture FAILED: %d (%s)\n",
              (int)err, hipGetErrorString(err));
    }
  }

  // GRAPH replay: launch graph at index 0, reuse captured GPU pointers
  if (hipdnn_ep_graph_enabled() && g_graph.capture_ok &&
      g_graph.inference_num >= 3) {
    if (index == 0) {
      hipError_t err = hipGraphLaunch(
          g_graph.graphExec, static_cast<hipStream_t>(state->stream));
      if (err != hipSuccess) {
        fprintf(stderr, "[GRAPH] hipGraphLaunch FAILED: %d (%s)\n", (int)err,
                hipGetErrorString(err));
        return HIPDNN_EP_ERR_STREAM_SYNC_FAILED;
      }
    }
    if (index >= g_graph.captured_output_ptrs.size()) {
      fprintf(stderr, "[GRAPH] ERROR: output index %zu out of captured range "
                      "(%zu)\n",
              index, g_graph.captured_output_ptrs.size());
      return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
    }
    out_buffer->gpu_ptr = g_graph.captured_output_ptrs[index];
    out_buffer->host_ptr = tensor->data;
    out_buffer->shape_ptr = tensor->shape;
    out_buffer->rank = tensor->rank;
    out_buffer->size_bytes = size_bytes;
    out_buffer->is_pooled = true;
    return HIPDNN_EP_SUCCESS;
  }

  // Allocate GPU buffer (pool reuses across inferences)
  void *gpu_ptr = pool_alloc(size_bytes);
  if (!gpu_ptr) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: failed to allocate %zu bytes\n",
            size_bytes);
    return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
  }

  // GRAPH: record pointer mapping during capture inference
  if (hipdnn_ep_graph_enabled() && g_graph.inference_num == 2) {
    g_graph.captured_output_ptrs.push_back(gpu_ptr);
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false;

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

  int result = HIPDNN_EP_SUCCESS;

  // GRAPH: end stream capture on first finalize_output of the capture inference.
  // Placed BEFORE PERF d2h_start and BEFORE the D2H copy so neither is in the
  // captured graph. After EndCapture the stream returns to normal mode.
  if (hipdnn_ep_graph_enabled() && g_graph.capturing &&
      g_graph.d2h_count == 0) {
    g_graph.capturing = false;
    hipError_t err = hipStreamEndCapture(
        static_cast<hipStream_t>(state->stream), &g_graph.graph);
    if (err == hipSuccess) {
      fprintf(stderr, "[GRAPH] hipStreamEndCapture SUCCEEDED\n");
      size_t numNodes = 0;
      (void)hipGraphGetNodes(g_graph.graph, nullptr, &numNodes);
      fprintf(stderr, "[GRAPH] captured graph has %zu nodes\n", numNodes);
      err = hipGraphInstantiate(&g_graph.graphExec, g_graph.graph,
                                nullptr, nullptr, 0);
      if (err == hipSuccess) {
        g_graph.capture_ok = true;
        fprintf(stderr,
                "[GRAPH] hipGraphInstantiate SUCCEEDED -- capture PASSED\n");
      } else {
        fprintf(stderr, "[GRAPH] hipGraphInstantiate FAILED: %d (%s)\n",
                (int)err, hipGetErrorString(err));
      }
    } else {
      fprintf(stderr, "[GRAPH] hipStreamEndCapture FAILED: %d (%s)\n",
              (int)err, hipGetErrorString(err));
      fprintf(stderr, "[GRAPH] Capture FAILED -- stream capture is not "
                      "compatible with current GPU dispatch pipeline.\n");
    }
  }

  // GRAPH: track finalize_output calls per inference
  if (hipdnn_ep_graph_enabled()) {
    g_graph.d2h_count++;
  }

  // PERF: record D2H start on first output finalize (after all compute)
  if (hipdnn_ep_perf_enabled() && g_perf.d2h_count == 0 &&
      g_perf.initialized) {
    (void)hipEventRecord(g_perf.d2h_start,
                         static_cast<hipStream_t>(state->stream));
  }

  // D2H transfer (async -- sync happens once after all outputs)
  if (hipMemcpyAsync(buffer->host_ptr, buffer->gpu_ptr, buffer->size_bytes,
                     hipMemcpyDeviceToHost,
                     static_cast<hipStream_t>(state->stream)) != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: D2H transfer failed\n");
    result = HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;
    // Continue to cleanup even on error (best-effort)
  }

  // PERF: accumulate D2H bytes
  if (hipdnn_ep_perf_enabled()) {
    g_perf.d2h_bytes += buffer->size_bytes;
    g_perf.d2h_count++;
  }

  // Return buffer to pool -- skip when graph is active (buffers are pinned)
  if (hipdnn_ep_graph_enabled() && g_graph.capture_ok) {
    buffer->gpu_ptr = nullptr;
  } else {
    pool_release(buffer->gpu_ptr, buffer->size_bytes);
    buffer->gpu_ptr = nullptr;
  }

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
  }

  if (hipStreamSynchronize(static_cast<hipStream_t>(state->stream)) !=
      hipSuccess) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: stream sync failed\n");
    if (result == HIPDNN_EP_SUCCESS) {
      result = HIPDNN_EP_ERR_STREAM_SYNC_FAILED;
    }
    // Continue to cleanup even on error (best-effort)
  }

  // GRAPH: log replay status (only first few to avoid spam)
  if (hipdnn_ep_graph_enabled() && g_graph.capture_ok &&
      g_graph.inference_num >= 3 && g_graph.inference_num <= 5) {
    fprintf(stderr, "[GRAPH] inference #%u: REPLAY completed\n",
            g_graph.inference_num);
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
  }

  return HIPDNN_EP_SUCCESS;
}

// Release input tensor buffer (no D2H transfer needed)
void hipdnn_ep_tensor_free_input(RuntimeState *state, TensorBuffer *buffer) {
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_free_input: null buffer\n");
    return;
  }

  // Return buffer to pool -- skip when graph is active (buffers are pinned)
  if (hipdnn_ep_graph_enabled() && g_graph.capture_ok) {
    buffer->gpu_ptr = nullptr;
  } else {
    pool_release(buffer->gpu_ptr, buffer->size_bytes);
    buffer->gpu_ptr = nullptr;
  }
}

// Check if graph replay is active (called from generated inference_compute)
extern "C" int hipdnn_ep_graph_should_skip_main(RuntimeState *state) {
  if (hipdnn_ep_graph_enabled() && g_graph.capture_ok &&
      g_graph.inference_num >= 3)
    return 1;
  return 0;
}

//==============================================================================
// TensorBuffer Field Accessors (Opaque Pattern)
//==============================================================================

void *hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer *buffer) {
  return buffer ? buffer->gpu_ptr : nullptr;
}

void *hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer *buffer) {
  return buffer ? buffer->host_ptr : nullptr;
}

int64_t *hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer *buffer) {
  return buffer ? buffer->shape_ptr : nullptr;
}

size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer *buffer) {
  return buffer ? buffer->rank : 0;
}

size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer *buffer) {
  return buffer ? buffer->size_bytes : 0;
}
