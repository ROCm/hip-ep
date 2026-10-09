/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Runs ckSelectGemmInstance in online mode over a shapes CSV written by
// `update_lut.py extract` and records each winner by instance name, the input
// `update_lut.py build` turns into lut/<arch>.json.
//
//   ck_gemm_autotune_sweep <shapes.csv> <winners.csv>

#include "ck_gemm_select.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace {

std::vector<std::string> split(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    out.push_back(cell);
  }
  return out;
}

int64_t spanElems(int64_t perBatch, int64_t stride, int64_t batch) {
  return stride * (batch - 1) + perBatch;
}

// fill() writes whole 32-bit words, so an odd f16 count needs the tail padded.
bool alloc(void **p, int64_t bytes) {
  return hipMalloc(p, (bytes + 3) & ~int64_t(3)) == hipSuccess;
}

// 0.1 in the operand's type keeps long-K sums finite.
bool fill(void *p, int64_t elems, int dtype) {
  if (dtype == HIP_DTYPE_FLOAT16) {
    const size_t words = static_cast<size_t>((elems + 1) / 2);
    return hipMemsetD32(p, 0x2E662E66, words) == hipSuccess;
  }
  return hipMemsetD32(p, 0x3DCCCCCD, static_cast<size_t>(elems)) ==
         hipSuccess;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <shapes.csv> <winners.csv>\n", argv[0]);
    return 2;
  }
#ifdef _WIN32
  SetEnvironmentVariableA("HIPDNN_CK_GEMM_AUTOTUNE_MODE", "online");
#else
  setenv("HIPDNN_CK_GEMM_AUTOTUNE_MODE", "online", 1);
#endif

  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  if (!in || !out) {
    fprintf(stderr, "cannot open %s or %s\n", argv[1], argv[2]);
    return 2;
  }
  std::string header;
  std::getline(in, header);
  const std::vector<std::string> cols = split(header);
  std::map<std::string, size_t> col;
  for (size_t i = 0; i < cols.size(); ++i) {
    col[cols[i]] = i;
  }
  out << header << ",instance\n";

  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess) {
    fprintf(stderr, "hipStreamCreate failed\n");
    return 1;
  }
  hipDeviceProp_t props;
  (void)hipGetDeviceProperties(&props, 0);
  fprintf(stderr, "#SWEEP arch=%s\n", props.gcnArchName);

  std::string line;
  int row = 0;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    ++row;
    const std::vector<std::string> v = split(line);
    auto I = [&](const char *name) { return std::stoll(v[col.at(name)]); };
    const int64_t m = I("m"), n = I("n"), k = I("k"), batch = I("batch");
    const int transA = static_cast<int>(I("trans_a"));
    const int ab = static_cast<int>(I("ab")), d = static_cast<int>(I("d"));
    const bool bias = I("bias") != 0;
    const int64_t lda = I("lda"), ldb = I("ldb"), ldd = I("ldd");
    const int64_t sA = I("stride_a"), sB = I("stride_b"), sD = I("stride_d");
    const float alpha = std::stof(v[col.at("alpha")]);

    const int64_t abBytes = ab == HIP_DTYPE_FLOAT16 ? 2 : 4;
    const int64_t dBytes = d == HIP_DTYPE_FLOAT16 ? 2 : 4;
    const int64_t aElems = spanElems(lda * (transA ? m : k), sA, batch);
    const int64_t bElems = spanElems(ldb * n, sB, batch);
    const int64_t dElems = spanElems(ldd * n, sD, batch);

    void *A = nullptr, *B = nullptr, *D = nullptr, *bias_p = nullptr;
    const bool ok = alloc(&A, aElems * abBytes) &&
                    alloc(&B, bElems * abBytes) &&
                    alloc(&D, dElems * dBytes) &&
                    (!bias || alloc(&bias_p, m * dBytes)) &&
                    fill(A, aElems, ab) && fill(B, bElems, ab) &&
                    (!bias || fill(bias_p, m, d));
    int inst = -1;
    if (ok) {
      inst = ckSelectGemmInstance(stream, A, B, bias_p, D, m, n, k, batch,
                                  transA, /*transB=*/0, ab, d, alpha, lda, ldb,
                                  ldd, sA, sB, sD);
    } else {
      (void)hipGetLastError();
    }
    (void)hipFree(A);
    (void)hipFree(B);
    (void)hipFree(D);
    (void)hipFree(bias_p);

    const char *name = inst >= 0 ? hip_ck_gemm_instance_name(inst) : nullptr;
    out << line << "," << (name ? name : "-") << "\n";
    out.flush();
    fprintf(stderr, "[%d] m=%lld n=%lld k=%lld batch=%lld transA=%d ab=%d d=%d "
                    "bias=%d -> %s%s\n",
            row, (long long)m, (long long)n, (long long)k, (long long)batch,
            transA, ab, d, (int)bias, name ? name : "-",
            ok ? "" : " (buffer setup failed)");
  }
  (void)hipStreamDestroy(stream);
  return 0;
}
