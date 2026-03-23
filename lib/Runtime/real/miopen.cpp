/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// Error checking macros
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    miopenStatus_t status = (cmd);                                             \
    if (status != miopenStatusSuccess) {                                       \
      fprintf(stderr, "MIOpen error at %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// =============================================================================
// MIOpen Convolution Forward Wrapper
// =============================================================================
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
// =============================================================================

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
  if (miopenCreateTensorDescriptor(&input_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create input_desc\n");
    result = -1;
    goto cleanup;
  }
  if (miopenCreateTensorDescriptor(&weights_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create weights_desc\n");
    result = -1;
    goto cleanup;
  }
  if (miopenCreateTensorDescriptor(&output_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create output_desc\n");
    result = -1;
    goto cleanup;
  }

  // Set tensor descriptors (assuming float32 data type)
  // Input: [N, C, H, W]
  if (miopenSet4dTensorDescriptor(input_desc, miopenFloat, input_n,
                                  input_c, input_h, input_w) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to set input_desc\n");
    result = -1;
    goto cleanup;
  }

  // Weights: [K, C, R, S] where K=output channels, C=input channels,
  // R=kernel_h, S=kernel_w
  if (miopenSet4dTensorDescriptor(weights_desc, miopenFloat, weights_k,
                                  input_c, kernel_h, kernel_w) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to set weights_desc\n");
    result = -1;
    goto cleanup;
  }

  // Output: [N, K, H', W']
  if (miopenSet4dTensorDescriptor(output_desc, miopenFloat, input_n,
                                  weights_k, output_h, output_w) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to set output_desc\n");
    result = -1;
    goto cleanup;
  }

  // Create convolution descriptor
  // Note: MIOpen padding is per-side, but if pad_top==pad_bottom and
  // pad_left==pad_right, we use the symmetric version
  if (miopenCreateConvolutionDescriptor(&conv_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create conv_desc\n");
    result = -1;
    goto cleanup;
  }
  if (miopenInitConvolutionDescriptor(
      conv_desc, miopenConvolution, pad_top, pad_left, stride_h, stride_w,
      dilation_h, dilation_w) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to init conv_desc\n");
    result = -1;
    goto cleanup;
  }

  // Set group count for grouped convolutions (e.g., depthwise convolution)
  // group=1 for standard convolution, group=C for depthwise convolution
  if (group > 1) {
    if (miopenSetConvolutionGroupCount(conv_desc, group) != miopenStatusSuccess) {
      fprintf(stderr, "MIOpen error: failed to set group count\n");
      result = -1;
      goto cleanup;
    }
  }

  // Allocate workspace for algorithm search
  // MIOpen's Find API needs workspace to test algorithms
  if (hipMalloc(&find_workspace, find_workspace_size) != hipSuccess) {
    fprintf(stderr, "HIP error: failed to allocate find_workspace\n");
    result = -1;
    goto cleanup;
  }

  // Find best algorithm
  // MIOpen 3.x API: returns array of performance results instead of single
  // algorithm
  if (miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
      output_desc, output,
      1,                    // requestAlgoCount - ask for 1 algorithm
      &returned_algo_count, // returnedAlgoCount - how many actually returned
      perf_results,         // perfResults - array to receive results
      find_workspace,       // workspace for algorithm testing
      find_workspace_size,  // workspaceSize
      false) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: algorithm search failed\n");
    result = -1;
    goto cleanup;
  }

  // Extract algorithm from performance results
  algo = perf_results[0].fwd_algo;

  // Get actual workspace size needed for this algorithm
  if (miopenConvolutionForwardGetWorkSpaceSize(
      miopen_handle, weights_desc, input_desc, conv_desc, output_desc,
      &workspace_size) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to get workspace size\n");
    result = -1;
    goto cleanup;
  }

  // Reuse find_workspace if it's large enough, otherwise reallocate
  workspace = find_workspace;
  if (workspace_size > find_workspace_size) {
    hipError_t err = hipFree(find_workspace);
    if (err != hipSuccess) {
      fprintf(stderr, "Warning: hipFree failed for find_workspace: %d\n", err);
    }
    find_workspace = nullptr;  // Mark as freed to avoid double-free
    if (hipMalloc(&workspace, workspace_size) != hipSuccess) {
      fprintf(stderr, "HIP error: failed to allocate workspace\n");
      workspace = nullptr;
      result = -1;
      goto cleanup;
    }
  }

  // Perform convolution
  if (miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: convolution forward failed\n");
    result = -1;
    goto cleanup;
  }

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

// =============================================================================
// MIOpen Activation Forward Wrapper (ReLU)
// =============================================================================
//
// Applies ReLU activation: output = max(0, input)
// Uses miopenActivationForward with MIOPEN_ACTIVATION_RELU mode.
//
// DESIGN: Generic activation wrapper that can be extended for other activation
// functions (sigmoid, tanh, etc.) by passing activation mode parameter.
// =============================================================================

extern "C" int wrap_miopenActivationForward_relu(
    RuntimeState *state, void *input_gpu_ptr, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, void *output_gpu_ptr, int64_t output_n,
    int64_t output_c, int64_t output_h, int64_t output_w) {
  if (!state || !input_gpu_ptr || !output_gpu_ptr) {
    fprintf(stderr, "Invalid arguments to wrap_miopenActivationForward_relu\n");
    return -1;
  }

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));

  // Use parameters directly - no MemRef knowledge!
  void *input_ptr = input_gpu_ptr;
  void *output_ptr = output_gpu_ptr;

  // Dimensions from parameters (not extracted from struct)
  int64_t n = input_n;
  int64_t c = input_c;
  int64_t h = input_h;
  int64_t w = input_w;

  // Initialize all resource pointers to nullptr for safe cleanup
  miopenTensorDescriptor_t input_tensor_desc = nullptr;
  miopenTensorDescriptor_t output_tensor_desc = nullptr;
  miopenActivationDescriptor_t activ_desc = nullptr;
  int result = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  // Create tensor descriptors
  if (miopenCreateTensorDescriptor(&input_tensor_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create input_tensor_desc\n");
    result = -1;
    goto cleanup;
  }
  if (miopenCreateTensorDescriptor(&output_tensor_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create output_tensor_desc\n");
    result = -1;
    goto cleanup;
  }

  if (miopenSet4dTensorDescriptor(
      input_tensor_desc, miopenFloat, static_cast<int>(n), static_cast<int>(c),
      static_cast<int>(h), static_cast<int>(w)) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to set input_tensor_desc\n");
    result = -1;
    goto cleanup;
  }
  if (miopenSet4dTensorDescriptor(
      output_tensor_desc, miopenFloat, static_cast<int>(n), static_cast<int>(c),
      static_cast<int>(h), static_cast<int>(w)) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to set output_tensor_desc\n");
    result = -1;
    goto cleanup;
  }

  // Create activation descriptor for ReLU
  if (miopenCreateActivationDescriptor(&activ_desc) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to create activ_desc\n");
    result = -1;
    goto cleanup;
  }

  // miopenActivationRELU mode with no parameters (alpha, beta, gamma unused for
  // ReLU)
  if (miopenSetActivationDescriptor(activ_desc, miopenActivationRELU,
                                    0.0, 0.0, 0.0) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: failed to set activ_desc\n");
    result = -1;
    goto cleanup;
  }

  // Forward pass
  if (miopenActivationForward(miopen_handle, activ_desc, &alpha,
                              input_tensor_desc, input_ptr, &beta,
                              output_tensor_desc, output_ptr) != miopenStatusSuccess) {
    fprintf(stderr, "MIOpen error: miopenActivationForward failed\n");
    result = -1;
    goto cleanup;
  }

cleanup:
  // Best-effort cleanup: free all allocated resources
  // Continue cleanup even if individual operations fail
  if (activ_desc) {
    miopenDestroyActivationDescriptor(activ_desc);
  }
  if (input_tensor_desc) {
    miopenDestroyTensorDescriptor(input_tensor_desc);
  }
  if (output_tensor_desc) {
    miopenDestroyTensorDescriptor(output_tensor_desc);
  }

  return result;
}
