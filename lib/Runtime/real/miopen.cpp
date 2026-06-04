/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

// Convenience wrappers for goto cleanup pattern (all functions use 'cleanup'
// label)
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)
#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// MIOpen Convolution Forward Wrapper
//===----------------------------------------------------------------------===//
//
// DESIGN DECISIONS:
//
// 1. GENERIC WRAPPER (Not Specialized)
//    Why use one wrapper for all convolution configurations instead of
//    specialized wrappers per kernel size (1x1, 3x3, 7x7)?
//
//    - MIOpen handles specialization internally via algorithm selection
//    - miopenFindConvolutionForwardAlgorithm() automatically selects optimal
//      kernel (Winograd for 3×3, GEMM for 1×1, FFT for certain configs)
//    - Industry standard: PyTorch, TensorFlow use generic wrappers
//    - No performance benefit from wrapper-level specialization
//    - Avoids complexity explosion (would need 50+ specialized wrappers)
//
//    Sources:
//    -
//    https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
//    -
//    https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/
//
// 2. NO DESCRIPTOR CACHING
//    Why don't we cache tensor/convolution descriptors?
//
//    - No evidence descriptor creation is expensive (likely just CPU struct
//    alloc)
//    - PyTorch/TensorFlow don't cache descriptors - they cache algorithm
//    selection
//    - PyTorch achieves 30-40% speedup from algorithm caching alone (not
//    descriptors)
//    - For dynamic shapes, descriptors change frequently anyway
//    - Descriptor overhead is negligible vs algorithm finding
//
//    Source:
//    - https://docs.pytorch.org/docs/stable/notes/cuda.html
//
// 3. PERFORMANCE OPTIMIZATION OPPORTUNITIES (TODOs below):
//    ✅ Cache algorithm finding results (HIGH PRIORITY - documented as
//    expensive) ✅ Pool workspace memory (eliminate malloc/free from hot path)
//    ❌ Cache descriptors (NOT recommended - negligible benefit, added
//    complexity)
//
//===----------------------------------------------------------------------===//

// TODO: Cache miopenFindConvolutionForwardAlgorithm() results
//
// RATIONALE: Algorithm finding is expensive (benchmarks multiple algorithms on
// GPU). MIOpen documentation explicitly states: "miopenFindConvolution*() is
// expensive in terms of run time and required workspace, so it's highly
// recommended to reserve the required algorithm and workspace to reuse them
// later."
//
// PyTorch achieves 30-40% speedup by caching algorithm selection via
// torch.backends.cudnn.benchmark = True.
//
// IMPLEMENTATION: Store in RuntimeState as AlgorithmCache keyed by:
//   (input_shape, weights_shape, output_shape, pad_h, pad_w, stride_h,
//   stride_w,
//    dilation_h, dilation_w)
//
// For dynamic shapes: Cache hit rate depends on shape variation. If shapes
// change frequently, cache effectiveness is reduced (PyTorch docs warn about
// this).
//
// Sources:
// -
// https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
// - https://docs.pytorch.org/docs/stable/notes/cuda.html

// TODO: Pool workspace memory instead of malloc/free every call
//
// RATIONALE: GPU memory allocation (hipMalloc) is expensive - involves kernel
// launch, synchronization, and memory manager overhead. Current code allocates
// and frees workspace on every inference call (lines 95, 107).
//
// IMPLEMENTATION: Add WorkspacePool to RuntimeState that:
//   - Pre-allocates workspace of maximum required size
//   - Reuses across multiple calls
//   - Grows dynamically if larger workspace needed
//
// BENEFIT: Eliminates malloc/free from hot path.

// MIOpen convolution forward implementation
// Follows opaque RuntimeState pattern - extracts handle/stream from state
int wrap_miopenConvolutionForward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group) {
  OP_PROFILE(
      "conv",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld,k=%lldx%lldx%lld",
                 (long long)input_n, (long long)input_c, (long long)input_h,
                 (long long)input_w, (long long)weights_k, (long long)kernel_h,
                 (long long)kernel_w);
        return std::string(b);
      },
      state);
  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionForward\n");
    return -1;
  }

  // Extract handle and stream from opaque RuntimeState via accessor functions
  // (Maintains abstraction barrier - no direct field access)
  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  hipStream_t hip_stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // Initialize all resource pointers to nullptr for safe cleanup
  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t weights_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  void *find_workspace = nullptr;
  void *workspace = nullptr;
  int result = 0;
  miopenConvAlgoPerf_t perf_results[1];
  int returned_algo_count = 0;
  miopenConvFwdAlgorithm_t algo;
  size_t workspace_size = 0;
  const size_t find_workspace_size = 10 * 1024 * 1024; // 10MB
  float alpha = 1.0f;
  float beta = 0.0f;

  // Create tensor descriptors
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));

  // Set tensor descriptors with explicit NCHW layout (float32).
  // miopenSet4dTensorDescriptor leaves layout as UNKNOWN which triggers
  // warnings in MIOpen 7.12+.
  {
    int in_dims[] = {(int)input_n, (int)input_c, (int)input_h, (int)input_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        input_desc, miopenFloat, miopenTensorNCHW, in_dims, 4));

    int w_dims[] = {(int)weights_k, (int)input_c, (int)kernel_h, (int)kernel_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        weights_desc, miopenFloat, miopenTensorNCHW, w_dims, 4));

    int out_dims[] = {(int)input_n, (int)weights_k, (int)output_h,
                      (int)output_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        output_desc, miopenFloat, miopenTensorNCHW, out_dims, 4));
  }

  // Create convolution descriptor
  // Note: MIOpen padding is per-side, but if pad_top==pad_bottom and
  // pad_left==pad_right, we use the symmetric version
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      conv_desc, miopenConvolution, pad_top, pad_left, stride_h, stride_w,
      dilation_h, dilation_w));

  // Set group count for grouped convolutions (e.g., depthwise convolution)
  // group=1 for standard convolution, group=C for depthwise convolution
  if (group > 1) {
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc, group));
  }

  // Allocate workspace for algorithm search
  // MIOpen's Find API needs workspace to test algorithms
  HIP_CHECK(hipMalloc(&find_workspace, find_workspace_size));

  // Find best algorithm
  // MIOpen 3.x API: returns array of performance results instead of single
  // algorithm
  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
      output_desc, output,
      1,                    // requestAlgoCount - ask for 1 algorithm
      &returned_algo_count, // returnedAlgoCount - how many actually returned
      perf_results,         // perfResults - array to receive results
      find_workspace,       // workspace for algorithm testing
      find_workspace_size,  // workspaceSize
      false));

  // Extract algorithm from performance results
  algo = perf_results[0].fwd_algo;

  // Get actual workspace size needed for this algorithm
  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
      miopen_handle, weights_desc, input_desc, conv_desc, output_desc,
      &workspace_size));

  // Reuse find_workspace if it's large enough, otherwise reallocate
  workspace = find_workspace;
  if (workspace_size > find_workspace_size) {
    hipError_t err = hipFree(find_workspace);
    if (err != hipSuccess) {
      fprintf(stderr, "Warning: hipFree failed for find_workspace: %d\n", err);
    }
    find_workspace = nullptr; // Mark as freed to avoid double-free
    HIP_CHECK(hipMalloc(&workspace, workspace_size));
  }

  // Perform convolution
  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size));

cleanup:
  // Best-effort cleanup: free all allocated resources
  // Continue cleanup even if individual operations fail
  if (workspace) {
    hipError_t err = hipFree(workspace);
    if (err != hipSuccess) {
      fprintf(stderr, "Warning: hipFree failed for workspace: %d\n", err);
    }
  }
  // Free find_workspace only if it wasn't already freed during reallocation
  if (find_workspace && find_workspace != workspace) {
    hipError_t err = hipFree(find_workspace);
    if (err != hipSuccess) {
      fprintf(stderr, "Warning: hipFree failed for find_workspace: %d\n", err);
    }
  }
  if (input_desc) {
    miopenDestroyTensorDescriptor(input_desc);
  }
  if (weights_desc) {
    miopenDestroyTensorDescriptor(weights_desc);
  }
  if (output_desc) {
    miopenDestroyTensorDescriptor(output_desc);
  }
  if (conv_desc) {
    miopenDestroyConvolutionDescriptor(conv_desc);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// MIOpen Softmax (row-wise over last dim) — used by the lowering of
// hip.miopen.softmax (which is the canonical lowering for ONNX Softmax).
//
// Treats the input as a 2-D logical matrix of shape [rows, cols] and runs a
// per-row softmax via miopenSoftmaxForward_V2 with MIOPEN_SOFTMAX_ACCURATE
// (numerically stable: subtracts row max before exp) and MIOPEN_SOFTMAX_MODE_
// INSTANCE (reduces over the per-batch instance, here a single row).
//
// dtype: HIPDNN_EP_DATATYPE_* (only f32/f16/bf16 are valid here; enforced at
// the lowering site as well).
//===----------------------------------------------------------------------===//
extern "C" int hip_miopen_softmax(RuntimeState *state, const void *input,
                                  void *output, int64_t rows, int64_t cols,
                                  int64_t data_type) {
  OP_PROFILE(
      "softmax",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld,dt=%lld", (long long)rows,
                 (long long)cols, (long long)data_type);
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    fprintf(stderr, "Invalid arguments to hip_miopen_softmax\n");
    return -1;
  }

  miopenDataType_t miopen_dt;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    miopen_dt = miopenFloat;
    break;
  case HIPDNN_EP_DATATYPE_HALF:
    miopen_dt = miopenHalf;
    break;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    miopen_dt = miopenBFloat16;
    break;
  default:
    fprintf(stderr, "hip_miopen_softmax: unsupported data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));

  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  int result = 0;

  // Represent [rows, cols] as a 4-D NCHW tensor [rows, cols, 1, 1] so MIOpen's
  // INSTANCE softmax reduces over the cols dimension (the C axis).
  int dims4[4] = {(int)rows, (int)cols, 1, 1};

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));
  MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
      input_desc, miopen_dt, miopenTensorNCHW, dims4, 4));
  MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
      output_desc, miopen_dt, miopenTensorNCHW, dims4, 4));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  MIOPEN_CHECK(miopenSoftmaxForward_V2(
      miopen_handle, &alpha, input_desc, input, &beta, output_desc, output,
      MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_INSTANCE));

cleanup:
  if (input_desc)
    miopenDestroyTensorDescriptor(input_desc);
  if (output_desc)
    miopenDestroyTensorDescriptor(output_desc);
  return result;
}
