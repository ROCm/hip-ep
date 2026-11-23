/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime wrappers of hip.quantize_q8_1, hip.skip_rms_norm_q8_1,
// hip.kquant_matmul, hip.kquant_matmul_q8_1 and hip.kquant_swiglu_q8_1: the
// operands' extents checked, scratch taken from the session, and the MMQ tile
// product chosen where it takes the shape.

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "kquant_abi.h"
#include "kquant_contract.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

int kernelFormat(int64_t formatId) {
  switch (formatId) {
  case HIP_KQUANT_FORMAT_Q4_K:
  case HIP_KQUANT_FORMAT_Q5_K:
  case HIP_KQUANT_FORMAT_Q6_K:
    return static_cast<int>(formatId);
  default:
    return -1;
  }
}

bool isValidQ8SumMode(int64_t sumMode) {
  return sumMode == HIP_Q8_1_SUM_PREQUANTIZATION ||
         sumMode == HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED ||
         sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM;
}

bool isValidQ8SumModeForFormat(int64_t sumMode, int64_t formatId) {
  if (!isValidQ8SumMode(sumMode))
    return false;
  (void)formatId;
  return true;
}

bool hasValidQ8RowGrouping(int64_t M, int64_t batchCount,
                           int64_t rowsPerToken) {
  int64_t totalRows = 0;
  return rowsPerToken > 0 &&
         hipdnn_ep::kquant::checkedMultiply(M, batchCount, totalRows) &&
         totalRows % rowsPerToken == 0;
}

bool getMatmulBufferBytes(int64_t M, int64_t N, int64_t K, int64_t batchCount,
                          int64_t formatId, size_t &q8Bytes,
                          size_t &packedWeightBytes, size_t &outputBytes) {
  int64_t blockElements = 0;
  int64_t blockBytes = 0;
  int64_t rows = 0;
  int64_t q8Blocks = 0;
  int64_t packedBlocks = 0;
  int64_t outputElements = 0;
  int64_t q8ByteCount = 0;
  int64_t packedByteCount = 0;
  int64_t outputByteCount = 0;
  if (!hipdnn_ep::kquant::getKQuantBlockElements(formatId, blockElements) ||
      !hipdnn_ep::kquant::getKQuantBlockBytes(formatId, blockBytes) ||
      !hipdnn_ep::kquant::checkedMultiply(batchCount, M, rows) ||
      !hipdnn_ep::kquant::checkedMultiply(rows, K / HIP_Q8_1_BLOCK_ELEMENTS,
                                          q8Blocks) ||
      !hipdnn_ep::kquant::checkedMultiply(q8Blocks, HIP_Q8_1_BLOCK_SIZE,
                                          q8ByteCount) ||
      !hipdnn_ep::kquant::checkedMultiply(N, K / blockElements, packedBlocks) ||
      !hipdnn_ep::kquant::checkedMultiply(packedBlocks, blockBytes,
                                          packedByteCount) ||
      !hipdnn_ep::kquant::checkedMultiply(rows, N, outputElements) ||
      !hipdnn_ep::kquant::checkedMultiply(
          outputElements, int64_t{sizeof(float)}, outputByteCount)) {
    return false;
  }
  if (static_cast<uint64_t>(q8ByteCount) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(packedByteCount) >
          std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(outputByteCount) >
          std::numeric_limits<size_t>::max()) {
    return false;
  }
  q8Bytes = static_cast<size_t>(q8ByteCount);
  packedWeightBytes = static_cast<size_t>(packedByteCount);
  outputBytes = static_cast<size_t>(outputByteCount);
  return true;
}

bool getMmqWorkspaceBytes(size_t q8Bytes, size_t &workspaceBytes) {
  int64_t paddingBytes = 0;
  if (!hipdnn_ep::kquant::checkedMultiply(
          int64_t{HIP_KQUANT_MMQ_INPUT_TILE *
                  HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE},
          HIP_Q8_1_BLOCK_SIZE, paddingBytes) ||
      static_cast<uint64_t>(paddingBytes) >
          std::numeric_limits<size_t>::max() - q8Bytes) {
    return false;
  }
  workspaceBytes = q8Bytes + static_cast<size_t>(paddingBytes);
  return true;
}

void *getMmqWorkspace(RuntimeState *state, size_t bytes, const void *q8,
                      size_t q8Bytes, const void *packedWeights,
                      size_t packedWeightBytes, const void *output,
                      size_t outputBytes) {
  void *oldWorkspace = hipdnn_ep_state_get_workspace(state);
  const size_t oldWorkspaceBytes = hipdnn_ep_state_get_workspace_size(state);
  if (oldWorkspace && oldWorkspaceBytes > 0 &&
      (hipdnn_ep::kquant::rangesOverlap(oldWorkspace, oldWorkspaceBytes, q8,
                                        q8Bytes) ||
       hipdnn_ep::kquant::rangesOverlap(oldWorkspace, oldWorkspaceBytes,
                                        packedWeights, packedWeightBytes) ||
       hipdnn_ep::kquant::rangesOverlap(oldWorkspace, oldWorkspaceBytes, output,
                                        outputBytes))) {
    return nullptr;
  }
  if (hipdnn_ep_state_ensure_workspace(state, bytes) != 0)
    return nullptr;
  void *workspace = hipdnn_ep_state_get_workspace(state);
  if (!workspace || !hipdnn_ep::kquant::hasQ8Alignment(workspace) ||
      hipdnn_ep::kquant::rangesOverlap(workspace, bytes, q8, q8Bytes) ||
      hipdnn_ep::kquant::rangesOverlap(workspace, bytes, packedWeights,
                                       packedWeightBytes) ||
      hipdnn_ep::kquant::rangesOverlap(workspace, bytes, output, outputBytes)) {
    return nullptr;
  }
  return workspace;
}

} // namespace

int wrap_quantize_q8_1(RuntimeState *state, const float *activation,
                       void *quantizedActivation, int64_t M, int64_t K,
                       int64_t batchCount, int64_t sumMode,
                       int64_t rowsPerToken) {

  if (!state || !activation || !quantizedActivation) {
    std::fprintf(stderr, "wrap_quantize_q8_1: null required argument\n");
    return -1;
  }
  if (!hasValidQ8RowGrouping(M, batchCount, rowsPerToken) || K <= 0 ||
      K % HIP_Q8_1_BLOCK_ELEMENTS != 0 || !isValidQ8SumMode(sumMode)) {
    std::fprintf(stderr,
                 "wrap_quantize_q8_1: dimensions must be positive and "
                 "K must be divisible by 32, and sum_mode must be valid\n");
    return -1;
  }
  if (!hipdnn_ep::kquant::hasQ8Alignment(quantizedActivation)) {
    std::fprintf(stderr, "wrap_quantize_q8_1: output has invalid alignment\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    std::fprintf(stderr, "wrap_quantize_q8_1: null HIP stream\n");
    return -1;
  }
  return hip_quantize_q8_1(stream, activation, quantizedActivation, M, K,
                           batchCount, static_cast<int>(sumMode), rowsPerToken);
}

int wrap_skip_rms_norm_q8_1(RuntimeState *state, const float *lhs,
                            const float *rhs, const float *scale,
                            float *residual, void *quantizedActivation,
                            int64_t lhsDim0, int64_t lhsDim1, int64_t lhsDim2,
                            int64_t rhsDim0, int64_t rhsDim1, int64_t rhsDim2,
                            int64_t scaleElements, int64_t residualDim0,
                            int64_t residualDim1, int64_t residualDim2,
                            int64_t quantizedDim0, int64_t quantizedDim1,
                            int64_t quantizedDim2, int64_t K, float epsilon,
                            int64_t sumMode, int64_t rowsPerToken) {

  if (!state || !lhs || !rhs || !scale || !residual || !quantizedActivation) {
    std::fprintf(stderr, "wrap_skip_rms_norm_q8_1: null required argument\n");
    return -1;
  }
  if (scaleElements != K || !std::isfinite(epsilon) || epsilon < 0.0f ||
      !isValidQ8SumMode(sumMode) || rowsPerToken <= 0) {
    std::fprintf(stderr, "wrap_skip_rms_norm_q8_1: inconsistent dimensions or "
                         "invalid epsilon or sum_mode\n");
    return -1;
  }
  hipdnn_ep::kquant::AddRmsNormQ8Layout layout{};
  if (!hipdnn_ep::kquant::validateAddRmsNormQ8Layout(
          {lhsDim0, lhsDim1, lhsDim2}, {rhsDim0, rhsDim1, rhsDim2},
          {residualDim0, residualDim1, residualDim2},
          {quantizedDim0, quantizedDim1, quantizedDim2}, K, layout)) {
    std::fprintf(stderr, "wrap_skip_rms_norm_q8_1: mismatched or overflowing "
                         "tensor dimensions\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(quantizedActivation) %
          HIP_Q8_1_BLOCK_ALIGNMENT !=
      0) {
    std::fprintf(stderr,
                 "wrap_skip_rms_norm_q8_1: output has invalid alignment\n");
    return -1;
  }
  if (!hipdnn_ep::kquant::hasDisjointAddRmsNormQ8Outputs(
          lhs, rhs, scale, residual, quantizedActivation, layout)) {
    std::fprintf(stderr, "wrap_skip_rms_norm_q8_1: output buffers must not "
                         "overlap inputs or each other\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    std::fprintf(stderr, "wrap_skip_rms_norm_q8_1: null HIP stream\n");
    return -1;
  }
  return hip_skip_rms_norm_q8_1(
      stream, lhs, rhs, scale, residual, quantizedActivation, layout.rowCount,
      K, epsilon, static_cast<int>(sumMode), rowsPerToken);
}

int wrap_kquant_matmul_q8_1(RuntimeState *state,
                            const void *quantizedActivation,
                            const void *packedWeights, float *output, int64_t M,
                            int64_t N, int64_t K, int64_t batchCount,
                            int64_t formatId, int64_t sumMode,
                            int64_t rowsPerToken) {

  if (!state || !quantizedActivation || !packedWeights || !output) {
    std::fprintf(stderr, "wrap_kquant_matmul_q8_1: null required argument\n");
    return -1;
  }
  int64_t blockElements = 0;
  if (rowsPerToken != 1 ||
      !hasValidQ8RowGrouping(M, batchCount, rowsPerToken) || N <= 0 || K <= 0 ||
      !hipdnn_ep::kquant::getKQuantBlockElements(formatId, blockElements) ||
      K % blockElements != 0 || !isValidQ8SumModeForFormat(sumMode, formatId) ||
      (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM &&
       batchCount != 1)) {
    std::fprintf(
        stderr,
        "wrap_kquant_matmul_q8_1: dimensions must be positive "
        "and K must be divisible by the selected format's block size, and "
        "sum_mode must match the format\n");
    return -1;
  }
  const int mappedFormat = kernelFormat(formatId);
  if (mappedFormat < 0) {
    std::fprintf(stderr,
                 "wrap_kquant_matmul_q8_1: unsupported format_id=%lld\n",
                 static_cast<long long>(formatId));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    std::fprintf(stderr, "wrap_kquant_matmul_q8_1: null HIP stream\n");
    return -1;
  }
  if (hip_kquant_mmq_is_supported(M, N, K, batchCount, mappedFormat)) {
    size_t q8Bytes = 0;
    size_t packedWeightBytes = 0;
    size_t outputBytes = 0;
    size_t mmqWorkspaceBytes = 0;
    if (!getMatmulBufferBytes(M, N, K, batchCount, formatId, q8Bytes,
                              packedWeightBytes, outputBytes) ||
        !getMmqWorkspaceBytes(q8Bytes, mmqWorkspaceBytes)) {
      std::fprintf(stderr, "wrap_kquant_matmul_q8_1: buffer size "
                           "overflow\n");
      return -1;
    }
    void *mmqWorkspace =
        getMmqWorkspace(state, mmqWorkspaceBytes, quantizedActivation, q8Bytes,
                        packedWeights, packedWeightBytes, output, outputBytes);
    if (mmqWorkspace)
      return hip_kquant_mmq_from_q8_1(stream, quantizedActivation, mmqWorkspace,
                                      packedWeights, output, M, N, K,
                                      batchCount, mappedFormat);
  }
  return hip_kquant_matmul_q8_1(stream, quantizedActivation, packedWeights,
                                output, M, N, K, batchCount, mappedFormat,
                                static_cast<int>(sumMode), rowsPerToken);
}

int wrap_kquant_swiglu_q8_1(RuntimeState *state,
                            const void *quantizedActivation,
                            const void *gateWeights, const void *upWeights,
                            float *output, int64_t q8Dim0, int64_t q8Dim1,
                            int64_t q8Dim2, int64_t gateWeightBytes,
                            int64_t upWeightBytes, int64_t outputDim0,
                            int64_t outputDim1, int64_t outputDim2, int64_t K,
                            int64_t N, int64_t gateFormatId, int64_t upFormatId,
                            int64_t sumMode, int64_t rowsPerToken) {

  if (!state || !quantizedActivation || !gateWeights || !upWeights || !output) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: null required argument\n");
    return -1;
  }
  if (!hipdnn_ep::kquant::isKQuantSwiGluFormat(gateFormatId) ||
      !hipdnn_ep::kquant::isKQuantSwiGluFormat(upFormatId) ||
      !isValidQ8SumModeForFormat(sumMode, gateFormatId) ||
      !isValidQ8SumModeForFormat(sumMode, upFormatId)) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: only Q4_K and Q6_K "
                         "weights are supported\n");
    return -1;
  }
  hipdnn_ep::kquant::KQuantSwiGluQ8Layout layout{};
  if (!hipdnn_ep::kquant::validateKQuantSwiGluQ8Layout(
          {q8Dim0, q8Dim1, q8Dim2}, gateWeightBytes, upWeightBytes,
          {outputDim0, outputDim1, outputDim2}, K, N, gateFormatId, upFormatId,
          layout)) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: invalid or overflowing "
                         "shape, format, or packed byte count\n");
    return -1;
  }
  if (rowsPerToken != 1 ||
      !hasValidQ8RowGrouping(layout.M, layout.batchCount, rowsPerToken)) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: invalid "
                         "rows_per_token\n");
    return -1;
  }
  if (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM &&
      layout.batchCount != 1) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: sum_mode 2 requires "
                         "one ordinary batch\n");
    return -1;
  }
  if (!hipdnn_ep::kquant::hasQ8Alignment(quantizedActivation) ||
      !hipdnn_ep::kquant::hasKQuantWeightAlignment(gateWeights, gateFormatId) ||
      !hipdnn_ep::kquant::hasKQuantWeightAlignment(upWeights, upFormatId) ||
      !hipdnn_ep::kquant::hasF32Alignment(output)) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: invalid input alignment\n");
    return -1;
  }
  if (!hipdnn_ep::kquant::hasDisjointKQuantSwiGluQ8Output(
          quantizedActivation, gateWeights, upWeights, output,
          /*scratch=*/nullptr, layout)) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: output overlaps an input\n");
    return -1;
  }

  void *scratch = nullptr;
  size_t scratchBytes = 0;
  if (layout.M > 1) {
    scratchBytes = layout.outputBytes;
    if (hipdnn_ep_state_ensure_matmul_dp4a_scratch(state, scratchBytes) != 0) {
      std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: cannot allocate fallback "
                           "scratch\n");
      return -1;
    }
    scratch = hipdnn_ep_state_get_matmul_dp4a_scratch(state);
    if (!scratch || !hipdnn_ep::kquant::hasF32Alignment(scratch) ||
        !hipdnn_ep::kquant::hasDisjointKQuantSwiGluQ8Output(
            quantizedActivation, gateWeights, upWeights, output, scratch,
            layout)) {
      std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: invalid or overlapping "
                           "fallback scratch\n");
      return -1;
    }
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: null HIP stream\n");
    return -1;
  }
  const int mappedGateFormat = kernelFormat(gateFormatId);
  const int mappedUpFormat = kernelFormat(upFormatId);
  if (mappedGateFormat == mappedUpFormat &&
      hip_kquant_mmq_is_supported(layout.M, N, K, layout.batchCount,
                                  mappedGateFormat)) {
    size_t mmqWorkspaceBytes = 0;
    if (!getMmqWorkspaceBytes(layout.q8Bytes, mmqWorkspaceBytes)) {
      std::fprintf(stderr, "wrap_kquant_swiglu_q8_1: MMQ workspace size "
                           "overflow\n");
      return -1;
    }
    void *mmqWorkspace = getMmqWorkspace(
        state, mmqWorkspaceBytes, quantizedActivation, layout.q8Bytes,
        gateWeights, layout.gateWeightBytes, output, layout.outputBytes);
    if (mmqWorkspace &&
        !hipdnn_ep::kquant::rangesOverlap(mmqWorkspace, mmqWorkspaceBytes,
                                          upWeights, layout.upWeightBytes) &&
        !hipdnn_ep::kquant::rangesOverlap(mmqWorkspace, mmqWorkspaceBytes,
                                          scratch, scratchBytes)) {
      int status = hip_kquant_mmq_from_q8_1(
          stream, quantizedActivation, mmqWorkspace, gateWeights,
          static_cast<float *>(scratch), layout.M, N, K, layout.batchCount,
          mappedGateFormat);
      if (status != 0)
        return status;
      // The up product reuses the packed activations.
      status = hip_kquant_mmq_packed_q8_1(stream, mmqWorkspace, upWeights,
                                          output, layout.M, N, K,
                                          layout.batchCount, mappedUpFormat);
      if (status != 0)
        return status;
      return hip_swiglu(stream, scratch, output, output, layout.outputElements,
                        HIP_DTYPE_FLOAT32);
    }
  }
  return hip_kquant_swiglu_q8_1(
      stream, quantizedActivation, gateWeights, upWeights, output, layout.M, N,
      K, layout.batchCount, kernelFormat(gateFormatId),
      kernelFormat(upFormatId), static_cast<int>(sumMode), rowsPerToken,
      scratch, scratchBytes);
}

int wrap_kquant_matmul(RuntimeState *state, const float *activation,
                       const void *packedWeights, float *output, int64_t M,
                       int64_t N, int64_t K, int64_t batchCount,
                       int64_t formatId) {
  if (!state || !activation || !packedWeights || !output) {
    std::fprintf(stderr, "wrap_kquant_matmul: null required argument\n");
    return -1;
  }
  int64_t blockElements = 0;
  if (M <= 0 || N <= 0 || K <= 0 || batchCount <= 0 ||
      !hipdnn_ep::kquant::getKQuantBlockElements(formatId, blockElements) ||
      K % blockElements != 0) {
    std::fprintf(stderr,
                 "wrap_kquant_matmul: dimensions must be positive and "
                 "K must be divisible by the selected format's block size\n");
    return -1;
  }
  const int mappedFormat = kernelFormat(formatId);
  if (mappedFormat < 0) {
    std::fprintf(stderr, "wrap_kquant_matmul: unsupported format_id=%lld\n",
                 static_cast<long long>(formatId));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    std::fprintf(stderr, "wrap_kquant_matmul: null HIP stream\n");
    return -1;
  }
  void *q8Scratch = nullptr;
  size_t q8DataBytes = 0;
  size_t packedWeightBytes = 0;
  size_t outputBytes = 0;
  if (!getMatmulBufferBytes(M, N, K, batchCount, formatId, q8DataBytes,
                            packedWeightBytes, outputBytes)) {
    std::fprintf(stderr, "wrap_kquant_matmul: Q8_1 scratch size overflow\n");
    return -1;
  }
  bool useMmq = hip_kquant_mmq_is_supported(M, N, K, batchCount, mappedFormat);
  // Products that read partial sums go through standard Q8_1 blocks; the
  // others quantize straight into the MMQ layout.
  const bool mmqFromQ8 = useMmq && hip_kquant_mmq_reads_q8_1(mappedFormat);
  if (useMmq && !mmqFromQ8 &&
      reinterpret_cast<uintptr_t>(activation) % HIP_KQUANT_MMQ_F32_ALIGNMENT !=
          0) {
    // The direct MMQ quantizer loads float4 values. Preserve the general
    // wrapper contract by falling back when an otherwise valid F32 input does
    // not satisfy that fast-path alignment.
    useMmq = false;
  }
  size_t q8ScratchBytes = q8DataBytes;
  if (useMmq && !mmqFromQ8) {
    if (!getMmqWorkspaceBytes(q8DataBytes, q8ScratchBytes)) {
      std::fprintf(stderr, "wrap_kquant_matmul: MMQ scratch size "
                           "overflow\n");
      return -1;
    }
  }
  // A successful ensure guarantees at least the requested size. When the MMQ
  // layout's larger scratch cannot be had, fall back to plain Q8_1 blocks.
  if (hipdnn_ep_state_ensure_matmul_dp4a_scratch(state, q8ScratchBytes) != 0) {
    if (!useMmq || q8ScratchBytes == q8DataBytes ||
        hipdnn_ep_state_ensure_matmul_dp4a_scratch(state, q8DataBytes) != 0) {
      std::fprintf(stderr,
                   "wrap_kquant_matmul: cannot allocate Q8_1 scratch\n");
      return -1;
    }
    useMmq = false;
    q8ScratchBytes = q8DataBytes;
  }
  q8Scratch = hipdnn_ep_state_get_matmul_dp4a_scratch(state);
  if (!q8Scratch) {
    std::fprintf(stderr,
                 "wrap_kquant_matmul: allocated Q8_1 scratch is null\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(q8Scratch) % HIP_Q8_1_BLOCK_ALIGNMENT != 0) {
    std::fprintf(stderr, "wrap_kquant_matmul: Q8_1 scratch has invalid "
                         "alignment\n");
    return -1;
  }
  if (useMmq && mmqFromQ8) {
    size_t mmqWorkspaceBytes = 0;
    if (!getMmqWorkspaceBytes(q8DataBytes, mmqWorkspaceBytes)) {
      std::fprintf(stderr, "wrap_kquant_matmul: MMQ workspace size "
                           "overflow\n");
      return -1;
    }
    void *mmqWorkspace =
        getMmqWorkspace(state, mmqWorkspaceBytes, q8Scratch, q8DataBytes,
                        packedWeights, packedWeightBytes, output, outputBytes);
    if (mmqWorkspace) {
      int status =
          hip_quantize_q8_1(stream, activation, q8Scratch, M, K, batchCount,
                            HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED,
                            /*rowsPerToken=*/1);
      if (status != 0)
        return status;
      return hip_kquant_mmq_from_q8_1(stream, q8Scratch, mmqWorkspace,
                                      packedWeights, output, M, N, K,
                                      batchCount, mappedFormat);
    }
  }
  if (useMmq && !mmqFromQ8) {
    return hip_kquant_mmq_f32(stream, activation, q8Scratch, packedWeights,
                              output, M, N, K, batchCount, mappedFormat);
  }
  return hip_kquant_matmul(stream, activation, packedWeights, output, M, N, K,
                           batchCount, mappedFormat, q8Scratch, q8ScratchBytes);
}
