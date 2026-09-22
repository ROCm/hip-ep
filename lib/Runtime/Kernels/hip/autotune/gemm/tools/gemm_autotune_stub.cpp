/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Flatbuffers-free definition of gemm_autotune::resolve/stats, returning a
 * miss. For standalone/dev builds that link gemm_kernel.hip WITHOUT the real
 * flatbuffer resolver (which needs flatc-generated headers) -- e.g. the
 * test/example/gemm Makefile. The production custom_kernels_<arch> DLL links
 * the real gemm_autotune.cpp instead; exactly one definition per build.
 *
 * A miss means every shape falls through to the heuristic / runtime autotune,
 * which is the behaviour the standalone harness wants anyway.
 */
#include "../gemm_autotune.h"

namespace hipdnn_ep {
namespace gemm_autotune {

Result resolve(const Request &, WmmaValidator, GemvValidator,
               TiledFmaValidator, void *) {
  return Result{};
}

Stats stats() { return Stats{}; }

} // namespace gemm_autotune
} // namespace hipdnn_ep
