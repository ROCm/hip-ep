/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace mlir_compilation {

// ============================================================================
// Data structures — C-compatible, stable ABI
// ============================================================================

// Memory placement of a tensor's `data` pointer. Carried as an int in
// tensor_t so the enum is forward-compatible without breaking the on-the-wire
// ABI between marshalling (onnxruntime_morphizen_ep.dll) and the MLIR-compiled
// model.dll.
//
// IMPORTANT: values are 1:1 with ORT's OrtMemoryInfoDeviceType
// (onnxruntime_c_api.h). MlirCustomOp can therefore write the ORT value
// straight into tensor_t.memory_type with no remapping. Today the runtime
// only special-cases TENSOR_MEMORY_GPU (alias path); CPU / FPGA / NPU all
// fall through to the legacy host H2D / D2H path, preserving existing
// behaviour for hip-test-dll, hip-onnx-runner, and any other host-input
// caller.
//
// Must match the matching enum in `lib/Runtime/hipdnn_ep_runtime.h`.
enum {
  TENSOR_MEMORY_CPU = 0, // == OrtMemoryInfoDeviceType_CPU
  TENSOR_MEMORY_GPU = 1, // == OrtMemoryInfoDeviceType_GPU  (alias path; only
                         // mode optimized today)
  TENSOR_MEMORY_FPGA =
      2, // == OrtMemoryInfoDeviceType_FPGA (treated as host today)
  TENSOR_MEMORY_NPU =
      3, // == OrtMemoryInfoDeviceType_NPU  (treated as host today)
};

// Describes a single tensor passed across the DLL boundary.
//
// Memory ownership: caller-owned. `memory_type` selects the data pointer's
// placement (see the enum above) and tells the DLL whether to copy or alias.
//
// tensor_t is the wire-protocol ABI between three components that are
// intentionally kept decoupled (compiler-emitted model.dll, EP runtime
// DLL, hip-test-dll harness), so we re-declare it here instead of
// sharing a header. The static_assert block below catches any layout
// drift at compile time. Sibling copies live at:
//   * `lib/Runtime/hipdnn_ep_runtime.h`            (extern-C, EP runtime)
//   * `tools/hip-test-dll/hip-test-dll.cpp`        (test driver, local)
struct tensor_t {
  void *data;     // Pointer to contiguous tensor data (caller-owned)
  int64_t *shape; // Pointer to shape array of length `rank` (caller-owned)
  size_t rank;    // Number of dimensions (must match the compiled model's rank)
  size_t element_size; // Bytes per element (e.g. 4=float32, 2=float16, 8=int64)
  int memory_type;     // One of TENSOR_MEMORY_CPU / _GPU / _FPGA / _NPU
};

// Compile-time guard for the wire-protocol ABI described above. The same
// three asserts live in each of the three sibling headers; if you reorder
// / add / remove a field in one copy and forget to mirror it in the others,
// at least one of them fails to build. Per-field offsets (not raw sizeof)
// because trailing padding after `memory_type` is compiler-defined and not
// part of what model.dll actually reads.
static_assert(offsetof(tensor_t, data) == 0,
              "tensor_t.data must remain the first field");
static_assert(offsetof(tensor_t, shape) == sizeof(void *),
              "tensor_t.shape moved -- update all three tensor_t copies");
static_assert(offsetof(tensor_t, memory_type) ==
                  offsetof(tensor_t, element_size) + sizeof(size_t),
              "tensor_t.memory_type moved -- update all three tensor_t "
              "copies");

// A contiguous array of tensor_t descriptors (inputs or outputs).
struct span_t {
  tensor_t *data; // Pointer to array of tensor_t
  size_t count; // Number of tensors (must match the compiled model's I/O count)
};

// EP-side mirror of `hipdnn_output_allocator_t` (lib/Runtime/hipdnn_ep_runtime.h).
// In output-allocator mode the EP installs one of these on the model.dll's
// RuntimeState (via hipdnn_ep_set_output_allocator) before the 2-arg
// inference_compute; the DLL's in-graph hip.alloc_output ops call back through
// `allocate` to obtain each graph-output buffer at the point its shape is known.
// Re-declared here (not shared) for the same decoupling reason as tensor_t.
//
// ABI / forward-compat: `struct_size` MUST stay first; the runtime setter copies
// only min(caller_size, local_size) bytes, so the EP MUST set
// `struct_size = sizeof(output_allocator_t)`. New callbacks are APPENDED after
// `allocate`; existing fields never move. The runtime header explicitly
// requires this EP-side copy to carry the same asserts.
struct output_allocator_t {
  size_t struct_size; // ABI size guard; MUST be first (offset 0)
  void *self;         // opaque EP context (borrowed; runtime never owns/frees)
  void *(*allocate)(void *self, int64_t out_idx, const int64_t *shape,
                    int64_t rank, int64_t elem_size);
};

static_assert(offsetof(output_allocator_t, struct_size) == 0,
              "output_allocator_t.struct_size must remain first (ABI guard)");
static_assert(offsetof(output_allocator_t, self) == sizeof(size_t),
              "output_allocator_t.self moved -- update all copies "
              "(lib/Runtime/hipdnn_ep_runtime.h)");

// ============================================================================
// DLL function pointer typedefs — stable C ABI exported by the compiled model.
//
// Generated by GenerateInterfacePass in the hip-compiler plugin.
// See doc/INTEGRATION-DESIGN.md for the full interface design.
//
// Lifecycle contract:
//   void* state = nullptr;
//   init_fn(&state);           // once: allocates GPU handles, uploads weights
//   compute_fn(state, in, out); // N times: H2D → compute → D2H → stream sync
//   cleanup_fn(state);         // once: releases all GPU resources
//
// All functions return 0 on success, non-zero on failure.
// ============================================================================

// One-time initialisation: creates GPU stream, MIOpen/hipBLAS handles, and
// uploads compiled model weights to device memory.
//
// Parameters:
//   out_state  [out] Receives the allocated opaque context pointer on success.
//                    Pass &state where state is a void* initialised to nullptr.
//
// Return values:
//   0  Success
//   1  Context allocation failed (malloc)
//   2  GPU handle creation failed (stream, MIOpen, or hipBLAS)
//   3  Constant upload to GPU failed
typedef int (*init_fn)(void **out_state);

// Executes one inference pass: copies inputs to GPU, runs the compiled graph,
// copies outputs back to CPU, then synchronises the stream.
//
// Parameters:
//   state    [in]     Opaque context pointer returned by init_fn.
//   inputs   [in]     span_t of host-memory input tensors.
//                     tensor_t.count and each tensor_t.rank must match the
//                     compiled model; tensor_t.shape supplies runtime
//                     dimensions.
//   outputs  [in/out] span_t of host-memory output tensors (caller-allocated).
//                     The DLL writes results into tensor_t.data on success.
//
// Return values:
//   0  Success
//   5  Invalid input (wrong tensor count, rank mismatch, or null pointer)
//   8  Computation failed (GPU kernel error)
//   9  Memory transfer failed (H2D or D2H copy)
typedef int (*compute_fn)(void *state, span_t *inputs, span_t *outputs);

// Releases all GPU resources acquired by init_fn.
// Best-effort: continues freeing remaining resources even if one step fails.
// state is invalid after this call regardless of return value.
//
// Parameters:
//   state  [in] Opaque context pointer returned by init_fn.
//
// Return values:
//   0   Success
//   10  Stream destruction failed
//   11  MIOpen handle destruction failed
//   12  hipBLAS handle destruction failed
//   13  GPU constant memory release failed (potential GPU memory leak)
//   14  Stream synchronisation failed
typedef int (*cleanup_fn)(void *state);

} // namespace mlir_compilation
