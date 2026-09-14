/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* No-LUT MatMulNBits autotune resolver for the standalone single-kernel test.
 *
 * matmul_nbits_kernel.hip calls hipdnn_ep::matmul_nbits_autotune::resolve() (the
 * offline-LUT lookup shared by the bits=4/8 WMMA and GEMV configs). The real
 * implementation (hip/autotune/matmul_nbits/matmul_nbits_autotune.cpp) reads a
 * FlatBuffers table whose generated header and flatbuffers runtime are
 * produced by the CMake build. This example builds with only the HIP SDK (no
 * CMake, no flatbuffers), so it links this stub instead.
 *
 * resolve() returns Source::None, i.e. "no table" — exactly the path taken on
 * any GPU arch without a compiled LUT. The kernel then runs its normal runtime
 * autotune sweep to pick the best config, so correctness and the benchmarked
 * timings are unaffected; only the one-time first-encounter LUT shortcut is
 * skipped, which is irrelevant to a benchmark that already warms up per shape.
 */
#include "matmul_nbits_autotune.h"

namespace hipdnn_ep {
namespace matmul_nbits_autotune {

Result resolve(const Request&, WmmaValidator, GemvValidator, void*) {
  return Result{};  // Source::None -> caller runs the runtime sweep
}

Stats stats() { return Stats{}; }

}  // namespace matmul_nbits_autotune
}  // namespace hipdnn_ep
