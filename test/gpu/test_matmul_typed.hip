/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"
#include "op_profile.h"
#include "op_state.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

extern "C" int8_t hipdnn_ep_op_state_construct_matmul(RuntimeState *state,
                                                      int32_t slot);

namespace {

hipStream_t testStream = nullptr;
hipblasLtHandle_t testHandle = nullptr;
OpState *testOpState = nullptr;
void *testWorkspace = nullptr;
size_t testWorkspaceSize = 0;

template <typename T> T fromFloat(float value);
template <> float fromFloat(float value) { return value; }
template <> __half fromFloat(float value) { return __float2half(value); }
template <> __hip_bfloat16 fromFloat(float value) {
  return __float2bfloat16(value);
}

template <typename T> float toFloat(T value);
template <> float toFloat(float value) { return value; }
template <> float toFloat(__half value) { return __half2float(value); }
template <> float toFloat(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

bool checkHip(hipError_t status, const char *operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

bool checkHipblas(hipblasStatus_t status, const char *operation) {
  if (status == HIPBLAS_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s failed with status %d\n", operation,
               static_cast<int>(status));
  return false;
}

float inputValue(size_t index) {
  const int value = static_cast<int>((index * 17 + 5) % 61) - 30;
  return static_cast<float>(value) / 32.0f;
}

float weightValue(size_t batch, size_t row, size_t column) {
  const int value =
      static_cast<int>((batch * 31 + row * 13 + column * 7 + 3) % 47) - 23;
  return static_cast<float>(value) / 32.0f;
}

template <typename InputT, typename OutputT>
bool runCase(const char *label, int64_t batchCount, int64_t M, int64_t N,
             int64_t K, bool transB, bool broadcastB, int64_t inputType,
             int64_t outputType, bool legacyEntryPoint = false) {
  const size_t inputCount = static_cast<size_t>(batchCount * M * K);
  const size_t weightBatches = broadcastB ? 1 : static_cast<size_t>(batchCount);
  const size_t weightCount = weightBatches * static_cast<size_t>(K * N);
  const size_t outputCount = static_cast<size_t>(batchCount * M * N);
  std::vector<InputT> input(inputCount);
  std::vector<InputT> weights(weightCount);
  std::vector<OutputT> output(outputCount);
  std::vector<float> expected(outputCount);

  for (size_t index = 0; index < input.size(); ++index)
    input[index] = fromFloat<InputT>(inputValue(index));
  for (size_t batch = 0; batch < weightBatches; ++batch) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t k = 0; k < K; ++k) {
        const size_t index = transB
                                 ? (batch * N + static_cast<size_t>(n)) * K + k
                                 : (batch * K + static_cast<size_t>(k)) * N + n;
        weights[index] = fromFloat<InputT>(weightValue(batch, n, k));
      }
    }
  }

  for (int64_t batch = 0; batch < batchCount; ++batch) {
    const size_t weightBatch = broadcastB ? 0 : static_cast<size_t>(batch);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          const size_t inputIndex =
              (static_cast<size_t>(batch) * M + m) * K + k;
          const size_t weightIndex = transB ? (weightBatch * N + n) * K + k
                                            : (weightBatch * K + k) * N + n;
          sum = std::fma(toFloat(input[inputIndex]),
                         toFloat(weights[weightIndex]), sum);
        }
        expected[(static_cast<size_t>(batch) * M + m) * N + n] = sum;
      }
    }
  }

  InputT *deviceInput = nullptr;
  InputT *deviceWeights = nullptr;
  OutputT *deviceOutput = nullptr;
  bool passed = checkHip(hipMalloc(reinterpret_cast<void **>(&deviceInput),
                                   input.size() * sizeof(InputT)),
                         "hipMalloc input") &&
                checkHip(hipMalloc(reinterpret_cast<void **>(&deviceWeights),
                                   weights.size() * sizeof(InputT)),
                         "hipMalloc weights") &&
                checkHip(hipMalloc(reinterpret_cast<void **>(&deviceOutput),
                                   output.size() * sizeof(OutputT)),
                         "hipMalloc output");
  if (passed) {
    passed = checkHip(hipMemcpyAsync(deviceInput, input.data(),
                                     input.size() * sizeof(InputT),
                                     hipMemcpyHostToDevice, testStream),
                      "hipMemcpyAsync input") &&
             checkHip(hipMemcpyAsync(deviceWeights, weights.data(),
                                     weights.size() * sizeof(InputT),
                                     hipMemcpyHostToDevice, testStream),
                      "hipMemcpyAsync weights");
  }

  const int64_t bBatchStride = broadcastB ? 0 : K * N;
  int status = -1;
  if (passed && legacyEntryPoint) {
    status = wrap_hipblasLtMatmul(
        reinterpret_cast<RuntimeState *>(1), 0, deviceInput, deviceWeights,
        deviceOutput, M, N, K, batchCount, sizeof(InputT), bBatchStride,
        /*transA=*/0, transB);
  } else if (passed) {
    status = wrap_hipblasLtMatmulTyped(
        reinterpret_cast<RuntimeState *>(1), 0, deviceInput, deviceWeights,
        deviceOutput, M, N, K, batchCount, inputType, outputType, bBatchStride,
        /*transA=*/0, transB);
  }
  if (passed && status != 0) {
    std::fprintf(stderr, "%s: matmul returned %d\n", label, status);
    passed = false;
  }
  if (passed)
    passed = checkHip(hipMemcpyAsync(output.data(), deviceOutput,
                                     output.size() * sizeof(OutputT),
                                     hipMemcpyDeviceToHost, testStream),
                      "hipMemcpyAsync output") &&
             checkHip(hipStreamSynchronize(testStream), "hipStreamSynchronize");

  const float absoluteTolerance =
      outputType == HIPDNN_EP_DATATYPE_HALF ? 2.0e-2f : 2.0e-3f;
  const float relativeTolerance =
      outputType == HIPDNN_EP_DATATYPE_HALF ? 2.0e-3f : 2.0e-4f;
  for (size_t index = 0; passed && index < output.size(); ++index) {
    const float actual = toFloat(output[index]);
    const float difference = std::abs(actual - expected[index]);
    const float tolerance =
        absoluteTolerance + relativeTolerance * std::abs(expected[index]);
    if (!std::isfinite(actual) || difference > tolerance) {
      std::fprintf(stderr,
                   "%s output %zu: expected %.9g, got %.9g "
                   "(difference %.9g, tolerance %.9g)\n",
                   label, index, expected[index], actual, difference,
                   tolerance);
      passed = false;
    }
  }

  if (deviceOutput)
    passed &= checkHip(hipFree(deviceOutput), "hipFree output");
  if (deviceWeights)
    passed &= checkHip(hipFree(deviceWeights), "hipFree weights");
  if (deviceInput)
    passed &= checkHip(hipFree(deviceInput), "hipFree input");
  return passed;
}

} // namespace

extern "C" void *hipdnn_ep_state_get_hipblas_handle(RuntimeState *) {
  return testHandle;
}

extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *) {
  return testStream;
}

extern "C" void *hipdnn_ep_state_get_workspace(RuntimeState *) {
  return testWorkspace;
}

extern "C" size_t hipdnn_ep_state_get_workspace_size(RuntimeState *) {
  return testWorkspaceSize;
}

extern "C" int hipdnn_ep_state_ensure_workspace(RuntimeState *,
                                                size_t requiredSize) {
  if (requiredSize <= testWorkspaceSize)
    return 0;
  if (testWorkspace && hipFree(testWorkspace) != hipSuccess)
    return -1;
  testWorkspace = nullptr;
  testWorkspaceSize = 0;
  if (hipMalloc(&testWorkspace, requiredSize) != hipSuccess)
    return -1;
  testWorkspaceSize = requiredSize;
  return 0;
}

extern "C" void *hipdnn_ep_state_get_op_profile(RuntimeState *) {
  return nullptr;
}

extern "C" void *hipdnn_ep_op_state_get(RuntimeState *, int slot) {
  return slot == 0 ? testOpState : nullptr;
}

void hipdnn_ep_op_state_set(RuntimeState *, int32_t slot, OpState *value) {
  if (slot == 0)
    testOpState = value;
}

bool op_profile_is_active(OpProfileState *) { return false; }
void op_profile_ensure_epoch(OpProfileState *, hipStream_t) {}
int op_profile_acquire_marker(OpProfileState *) { return -1; }
hipEvent_t op_profile_get_marker_event(OpProfileState *, int) {
  return nullptr;
}
void op_profile_add_pending(OpProfileState *, const std::string &,
                            const std::string &, int, double, int64_t, double) {
}

bool rejectsInvalidArguments() {
  auto *state = reinterpret_cast<RuntimeState *>(uintptr_t{1});
  auto *buffer = reinterpret_cast<void *>(uintptr_t{4});
  const auto rejects = [&](int64_t M, int64_t N, int64_t K, int64_t batchCount,
                           int64_t bBatchStride, int64_t transA,
                           int64_t transB) {
    return wrap_hipblasLtMatmulTyped(state, 0, buffer, buffer, buffer, M, N, K,
                                     batchCount, HIPDNN_EP_DATATYPE_BFLOAT16,
                                     HIPDNN_EP_DATATYPE_FLOAT, bBatchStride,
                                     transA, transB) == -1;
  };
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  return rejects(0, 3, 5, 4, 15, 0, 1) && rejects(2, -3, 5, 4, 15, 0, 1) &&
         rejects(2, 3, 5, 0, 15, 0, 1) && rejects(2, 3, 5, 4, -1, 0, 1) &&
         rejects(2, 3, 5, 4, 14, 0, 1) && rejects(2, 3, 5, 4, 15, 0, 2) &&
         rejects(kMax, 1, 2, 1, 0, 0, 1) && rejects(kMax, 2, 1, 1, 0, 0, 1) &&
         rejects(2, 2, 1, kMax, 0, 0, 1) &&
         rejects(1, 5, int64_t{1} << 62, 2, int64_t{1} << 62, 0, 1) &&
         rejects(1, kMax / 4 + 1, 1, 1, 0, 0, 1) &&
         wrap_hipblasLtMatmul(state, 0, buffer, buffer, buffer,
                              /*M=*/0, /*N=*/3, /*K=*/5,
                              /*batch_count=*/4, /*elem_size=*/4,
                              /*b_batch_stride=*/15, /*transA=*/0,
                              /*transB=*/1) == -1;
}

int main() {
  if (!rejectsInvalidArguments())
    return 1;

  int deviceCount = 0;
  const hipError_t countStatus = hipGetDeviceCount(&deviceCount);
  if (countStatus == hipErrorNoDevice || deviceCount == 0)
    return 77;
  if (!checkHip(countStatus, "hipGetDeviceCount"))
    return 1;

  int selectedDevice = -1;
  for (int device = 0; device < deviceCount; ++device) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) == hipSuccess &&
        (std::strcmp(properties.gcnArchName, "gfx1151") == 0 ||
         std::strncmp(properties.gcnArchName, "gfx1151:", 8) == 0)) {
      selectedDevice = device;
      break;
    }
  }
  if (selectedDevice < 0)
    return 77;
  if (!checkHip(hipSetDevice(selectedDevice), "hipSetDevice") ||
      !checkHip(hipStreamCreate(&testStream), "hipStreamCreate") ||
      !checkHipblas(hipblasLtCreate(&testHandle), "hipblasLtCreate") ||
      hipdnn_ep_op_state_construct_matmul(reinterpret_cast<RuntimeState *>(1),
                                          0) != 0)
    return 1;

  bool passed =
      runCase<__hip_bfloat16, float>(
          "bf16 decode transposed B", 1, 1, 1025, 1026, true, true,
          HIPDNN_EP_DATATYPE_BFLOAT16, HIPDNN_EP_DATATYPE_FLOAT) &&
      runCase<__hip_bfloat16, float>("bf16 M>1", 1, 3, 11, 17, false, true,
                                     HIPDNN_EP_DATATYPE_BFLOAT16,
                                     HIPDNN_EP_DATATYPE_FLOAT) &&
      runCase<__hip_bfloat16, float>("bf16 broadcast B", 3, 2, 16, 32, false,
                                     true, HIPDNN_EP_DATATYPE_BFLOAT16,
                                     HIPDNN_EP_DATATYPE_FLOAT) &&
      runCase<__hip_bfloat16, float>("bf16 batched B", 3, 2, 16, 32, false,
                                     false, HIPDNN_EP_DATATYPE_BFLOAT16,
                                     HIPDNN_EP_DATATYPE_FLOAT) &&
      runCase<__half, __half>("f16 legacy", 2, 3, 9, 13, false, true,
                              HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_HALF,
                              true) &&
      runCase<float, float>("f32 legacy", 2, 2, 7, 11, false, false,
                            HIPDNN_EP_DATATYPE_FLOAT, HIPDNN_EP_DATATYPE_FLOAT,
                            true);

  if (testOpState) {
    testOpState->deletor(testOpState);
    testOpState = nullptr;
  }
  if (testWorkspace)
    passed &= checkHip(hipFree(testWorkspace), "hipFree workspace");
  if (testHandle)
    passed &= checkHipblas(hipblasLtDestroy(testHandle), "hipblasLtDestroy");
  if (testStream)
    passed &= checkHip(hipStreamDestroy(testStream), "hipStreamDestroy");
  return passed ? 0 : 1;
}
