/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <vector>

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
  const size_t find_workspace_size = 256 * 1024 * 1024; // 256MB
  float alpha = 1.0f;
  float beta = 0.0f;

  // Check inputs near the NaN-producing operation (op ~492)
  {
    int next_op = g_nan_trace_counter + 1;
    if (next_op >= 485 && next_op <= 500 && !g_nan_first_found) {
      nan_trace_check_input("conv_fwd", next_op, "data", input,
                            input_n * input_c * input_h * input_w);
      nan_trace_check_input("conv_fwd", next_op, "weights", weights,
                            weights_k * input_c * kernel_h * kernel_w);
      if (bias)
        nan_trace_check_input("conv_fwd", next_op, "bias", bias, weights_k);
      fprintf(stderr,
              "[NAN_TRACE] op#%d conv_fwd shapes: in=[%lld,%lld,%lld,%lld] "
              "w_k=%lld kern=[%lld,%lld] stride=[%lld,%lld] pad=[%lld,%lld,%lld,%lld] "
              "dil=[%lld,%lld] group=%lld out=[_,%lld,%lld,%lld]\n",
              next_op, (long long)input_n, (long long)input_c,
              (long long)input_h, (long long)input_w, (long long)weights_k,
              (long long)kernel_h, (long long)kernel_w,
              (long long)stride_h, (long long)stride_w,
              (long long)pad_top, (long long)pad_left,
              (long long)pad_bottom, (long long)pad_right,
              (long long)dilation_h, (long long)dilation_w,
              (long long)group, (long long)weights_k,
              (long long)output_h, (long long)output_w);
      fflush(stderr);
    }
  }

  // Create tensor descriptors
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));

  // Use miopenSet4dTensorDescriptor for convolution I/O descriptors.
  // miopenSetNdTensorDescriptorWithLayout can set zero strides for unit
  // dimensions (e.g. H=1), which breaks MIOpen convolution kernels.
  {
    int in_c_per_group = (group > 1) ? (int)(input_c / group) : (int)input_c;
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc, miopenFloat, (int)input_n, (int)input_c,
        (int)input_h, (int)input_w));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        weights_desc, miopenFloat, (int)weights_k, in_c_per_group,
        (int)kernel_h, (int)kernel_w));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc, miopenFloat, (int)input_n, (int)weights_k,
        (int)output_h, (int)output_w));
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

  // Find best algorithm. Request multiple candidates and use exhaustive
  // search to avoid problematic algorithms (Winograd for non-3x3 kernels).
  {
    const int kMaxAlgos = 8;
    miopenConvAlgoPerf_t all_results[8];
    int total_returned = 0;
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
        output_desc, output,
        kMaxAlgos,       // requestAlgoCount
        &total_returned, // returnedAlgoCount
        all_results,     // perfResults
        find_workspace,  // workspace for algorithm testing
        find_workspace_size,
        true)); // exhaustiveSearch = true

    bool winograd_safe =
        (kernel_h == kernel_w) && (kernel_h == 3 || kernel_h == 5) &&
        (input_h > 1) && (input_w > 1);

    int best_idx = -1;
    for (int i = 0; i < total_returned; ++i) {
      if (!winograd_safe &&
          all_results[i].fwd_algo == miopenConvolutionFwdAlgoWinograd)
        continue;
      best_idx = i;
      break;
    }

    if (best_idx < 0 && !winograd_safe) {
      // All returned algorithms are Winograd for a shape we previously
      // considered unsafe.  Accept the best Winograd result rather than
      // falling back to host-side computation (which violates the
      // zero-CPU-fallback requirement).
      fprintf(stderr,
              "[conv_fwd] NOTE: using Winograd for "
              "in=[%lld,%lld,%lld,%lld] kern=[%lld,%lld]\n",
              (long long)input_n, (long long)input_c,
              (long long)input_h, (long long)input_w,
              (long long)kernel_h, (long long)kernel_w);
      best_idx = 0; // Accept the best Winograd result
    }

    if (best_idx < 0)
      best_idx = 0;

    algo = all_results[best_idx].fwd_algo;
    returned_algo_count = total_returned;
    perf_results[0] = all_results[best_idx];
  }

  // Use workspace size reported by the selected algorithm.
  // miopenConvolutionForwardGetWorkSpaceSize returns the maximum over all
  // algorithms and can return 0 when the selected algorithm actually needs
  // workspace, causing "0 provided, N required" errors.
  workspace_size = perf_results[0].memory;

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

  // Zero output buffer before convolution (belt-and-suspenders; MIOpen with
  // beta=0 overwrites output fully, but zeroing prevents stale data from
  // masking errors during NAN_TRACE debugging).
  {
    int64_t out_elems = input_n * weights_k * output_h * output_w;
    hipMemsetAsync(output, 0, out_elems * sizeof(float), hip_stream);
  }

  // Log algorithm choice near the NaN-producing op
  {
    int next_op = g_nan_trace_counter + 1;
    if (next_op >= 491 && next_op <= 493) {
      fprintf(stderr,
              "[NAN_TRACE] op#%d conv_fwd: algo=%d, workspace_size=%zu, "
              "returned_algo_count=%d\n",
              next_op, (int)algo, workspace_size, returned_algo_count);
      fflush(stderr);
    }
  }

  // Perform convolution
  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size));

  // Check output BEFORE bias addition
  {
    int next_op = g_nan_trace_counter + 1;
    if (next_op >= 491 && next_op <= 493 && !g_nan_first_found) {
      nan_trace_check_input("conv_fwd", next_op, "PRE_BIAS_output", output,
                            input_n * weights_k * output_h * output_w);
    }
  }

  if (bias) {
    miopenTensorDescriptor_t bias_desc = nullptr;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    int bias_dims[] = {1, (int)weights_k, 1, 1};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, miopenFloat, miopenTensorNCHW, bias_dims, 4));
    float alpha_a = 1.0f, alpha_b = 1.0f, beta_c = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha_a,
                                output_desc, output, &alpha_b, bias_desc, bias,
                                &beta_c, output_desc, output));
    miopenDestroyTensorDescriptor(bias_desc);
  }

  nan_trace_check("conv_fwd", output,
                  input_n * weights_k * output_h * output_w);

cleanup:
  if (workspace) {
    hipError_t err = hipFree(workspace);
    if (err != hipSuccess) {
      fprintf(stderr, "Warning: hipFree failed for workspace: %d\n", err);
    }
  }
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
