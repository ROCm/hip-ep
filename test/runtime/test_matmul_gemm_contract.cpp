/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"
#include "matmul_gemm_contract.h"
#include "runtime_state_internal.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" void hipdnn_ep_mock_reset_blas_dispatch_counts();
extern "C" int hipdnn_ep_mock_matmul_dispatch_count();
extern "C" int hipdnn_ep_mock_gemm_dispatch_count();

namespace {

int failures = 0;
int errorFlagCalls = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

template <size_t N> bool allZero(const std::array<unsigned char, N> &buffer) {
  for (unsigned char byte : buffer)
    if (byte != 0)
      return false;
  return true;
}

void testSharedSizing() {
  using namespace hipdnn_ep::blas_contract;
  OutputSize size;
  CHECK(computeOutputSize(2, 3, 4, 2, size));
  CHECK(size.elements == 24);
  CHECK(size.bytes == 48);
  CHECK(!computeOutputSize(1, -1, 4, 2, size));
  CHECK(!computeOutputSize(1, std::numeric_limits<int64_t>::max(), 2, 8, size));
  CHECK(validateMatrixProducts(2, 3, 4, 4));
  CHECK(!validateMatrixProducts(std::numeric_limits<int64_t>::max(), 3, 4, 4));
  CHECK(validateMatmul(true, 2, 3, 4, 5, 1, 2, 1, 1, 0, 0, true, true, true)
            .status == ValidationStatus::Failure);
  CHECK(
      validateGemm(2, 3, 4, 4, 0, 0, 1, false, 0, 0, true, true, true).status ==
      ValidationStatus::Success);
}

void testMatmulWrapper() {
  RuntimeState state{};
  std::array<unsigned char, 48> a{};
  std::array<unsigned char, 64> b{};
  std::array<unsigned char, 24> output{};

  hipdnn_ep_mock_reset_blas_dispatch_counts();
  errorFlagCalls = 0;
  output.fill(0x5a);
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/true,
                             /*M=*/2, /*N=*/3, /*K_a=*/4, /*K_b=*/4,
                             /*batch=*/2, /*elem=*/2, /*a_batches=*/2,
                             /*b_batches=*/1, /*a_stride=*/8,
                             /*b_stride=*/0) == 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 1);
  CHECK(errorFlagCalls == 0);

  output.fill(0x5a);
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(), true,
                             2, 3, 4, 5, 2, 2, 2, 1, 8, 0) != 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 1);
  CHECK(errorFlagCalls == 1);
  CHECK(allZero(output));

  CHECK(wrap_hipblasLtMatmul(&state, 0, nullptr, nullptr, nullptr,
                             /*batch_axes_valid=*/true,
                             /*M=*/0, /*N=*/3, /*K_a=*/4, /*K_b=*/4,
                             /*batch=*/2, /*elem=*/2, /*a_batches=*/2,
                             /*b_batches=*/1, /*a_stride=*/0,
                             /*b_stride=*/0) == 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 1);
  CHECK(errorFlagCalls == 1);

  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/true,
                             std::numeric_limits<int64_t>::max(), 3, 4, 4, 2, 2,
                             2, 2, 8, 12) != 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 1);
  CHECK(errorFlagCalls == 2);
}

void testDynamicPartialBatchRejected() {
  RuntimeState state{};
  std::array<unsigned char, 32> a{};
  std::array<unsigned char, 144> b{};
  std::array<unsigned char, 72> output{};

  hipdnn_ep_mock_reset_blas_dispatch_counts();
  errorFlagCalls = 0;
  output.fill(0x5a);
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/true,
                             /*M=*/2, /*N=*/3, /*K_a=*/4, /*K_b=*/4,
                             /*output batches=*/6, /*elem=*/2,
                             /*partial A batches=*/2, /*full B batches=*/6,
                             /*lowering-selected A stride=*/0,
                             /*B stride=*/12) != 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 0);
  CHECK(errorFlagCalls == 1);
  CHECK(allZero(output));
}

void testPerAxisBatchValidation() {
  RuntimeState state{};
  std::array<unsigned char, 96> a{};
  std::array<unsigned char, 144> b{};
  std::array<unsigned char, 72> output{};

  hipdnn_ep_mock_reset_blas_dispatch_counts();
  errorFlagCalls = 0;

  // Equal dynamic axes: [2,3] with [2,3].
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/true, /*M=*/2, /*N=*/3,
                             /*K_a=*/4, /*K_b=*/4, /*output batches=*/6,
                             /*elem=*/2,
                             /*A batches=*/6, /*B batches=*/6, /*A stride=*/8,
                             /*B stride=*/12) == 0);

  // Valid unit broadcast: [1,1] with [2,3].
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/true, /*M=*/2, /*N=*/3,
                             /*K_a=*/4, /*K_b=*/4, /*output batches=*/6,
                             /*elem=*/2,
                             /*A batches=*/1, /*B batches=*/6, /*A stride=*/0,
                             /*B stride=*/12) == 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 2);
  CHECK(errorFlagCalls == 0);

  // Equal flattened products cannot hide incompatible axes: [2,3] vs [3,2].
  output.fill(0x5a);
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/false, /*M=*/2, /*N=*/3,
                             /*K_a=*/4, /*K_b=*/4, /*output batches=*/6,
                             /*elem=*/2,
                             /*A batches=*/6, /*B batches=*/6, /*A stride=*/8,
                             /*B stride=*/12) != 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 2);
  CHECK(errorFlagCalls == 1);
  CHECK(allZero(output));

  // A caller-provided output extent that disagrees with the exact broadcast
  // choice is rejected through the same lowering validity bit.
  output.fill(0x5a);
  CHECK(wrap_hipblasLtMatmul(&state, 0, a.data(), b.data(), output.data(),
                             /*batch_axes_valid=*/false, /*M=*/2, /*N=*/3,
                             /*K_a=*/4, /*K_b=*/4, /*output batches=*/6,
                             /*elem=*/2,
                             /*A batches=*/6, /*B batches=*/6, /*A stride=*/8,
                             /*B stride=*/12) != 0);
  CHECK(hipdnn_ep_mock_matmul_dispatch_count() == 2);
  CHECK(errorFlagCalls == 2);
  CHECK(allZero(output));
}

void testGemmWrapper() {
  RuntimeState state{};
  std::array<unsigned char, 32> a{};
  std::array<unsigned char, 32> b{};
  std::array<unsigned char, 24> output{};

  hipdnn_ep_mock_reset_blas_dispatch_counts();
  errorFlagCalls = 0;
  output.fill(0x5a);
  CHECK(wrap_gemm(&state, 0, a.data(), b.data(), nullptr, output.data(),
                  /*M=*/2, /*N=*/3, /*K_a=*/4, /*K_b=*/4, 1.0f, 0.0f,
                  /*transA=*/1, /*transB=*/1, /*f32=*/1,
                  /*cDim0=*/0, /*cDim1=*/0) == 0);
  CHECK(hipdnn_ep_mock_gemm_dispatch_count() == 1);
  CHECK(errorFlagCalls == 0);

  output.fill(0x5a);
  CHECK(wrap_gemm(&state, 0, a.data(), b.data(), nullptr, output.data(), 2, 3,
                  4, 5, 1.0f, 0.0f, 0, 0, 1, 0, 0) != 0);
  CHECK(hipdnn_ep_mock_gemm_dispatch_count() == 1);
  CHECK(errorFlagCalls == 1);
  CHECK(allZero(output));

  CHECK(wrap_gemm(&state, 0, nullptr, nullptr, nullptr, nullptr,
                  /*M=*/2, /*N=*/0, /*K_a=*/4, /*K_b=*/4, 1.0f, 0.0f, 0, 0, 1,
                  0, 0) == 0);
  CHECK(hipdnn_ep_mock_gemm_dispatch_count() == 1);
  CHECK(errorFlagCalls == 1);
}

} // namespace

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *) {
  ++errorFlagCalls;
  return 0;
}

int main() {
  testSharedSizing();
  testMatmulWrapper();
  testDynamicPartialBatchRejected();
  testPerAxisBatchValidation();
  testGemmWrapper();
  if (failures == 0) {
    std::printf("MatMul/Gemm contract unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "MatMul/Gemm contract unit test: %d FAILURE(S)\n",
               failures);
  return 1;
}
