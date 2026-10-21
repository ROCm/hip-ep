// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// Test only. Keep the model's K-by-N weight layout and FP16 storage.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "head_matvec.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

constexpr int N = 131072, K = 2688;
void check(hipError_t s) {
  if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
void check(hipblasStatus_t s) {
  if (s != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BLAS status " + std::to_string(s));
}
__host__ __device__ half sample(unsigned i, unsigned salt) {
  i ^= salt; i ^= i >> 16; i *= 0x7feb352du;
  i ^= i >> 15; i *= 0x846ca68bu; i ^= i >> 16;
  return __float2half((int(i % 20001) - 10000) * 0.00003f);
}
__global__ void init(half *x, half *w) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < K) x[i] = sample(i, 17);
  if (i < unsigned(K) * N) w[i] = sample(i, 41);
}

template<int Columns, int Vec, int Splits>
__global__ void partial(const half *x, const half *w, half *y, float *scratch) {
  constexpr int Rows = 256 / Columns;
  int lane = threadIdx.x % Columns, row = threadIdx.x / Columns;
  int col = (blockIdx.x * Columns + lane) * Vec;
  float accum[Vec] = {};
  for (int k = blockIdx.y * Rows + row; k < K; k += Splits * Rows) {
    float xv = __half2float(x[k]);
    const half *weight = w + size_t(k) * N + col;
    if constexpr (Vec == 1) {
      accum[0] = __fmaf_rn(xv, __half2float(*weight), accum[0]);
    } else {
#pragma unroll
      for (int v = 0; v < Vec / 2; ++v) {
        float2 pair = __half22float2(reinterpret_cast<const half2 *>(weight)[v]);
        accum[2*v] = __fmaf_rn(xv, pair.x, accum[2*v]);
        accum[2*v+1] = __fmaf_rn(xv, pair.y, accum[2*v+1]);
      }
    }
  }
  __shared__ float values[256 * Vec];
#pragma unroll
  for (int v = 0; v < Vec; ++v) values[threadIdx.x * Vec + v] = accum[v];
  __syncthreads();
  if (row == 0) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      float sum = 0;
#pragma unroll
      for (int r = 0; r < Rows; ++r) sum += values[(r * Columns + lane) * Vec + v];
      if constexpr (Splits == 1) y[col+v] = __float2half(sum);
      else scratch[blockIdx.y * N + col+v] = sum;
    }
  }
}
template<int Splits>
__global__ void finish(const float *scratch, half *y) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
#pragma unroll
  for (int s = 0; s < Splits; ++s) sum += scratch[s * N + col];
  y[col] = __float2half(sum);
}
template<int C, int V, int S>
void launch(const half *x, const half *w, half *y, float *scratch, hipStream_t stream) {
  static_assert(N % (C * V) == 0);
  hipLaunchKernelGGL((partial<C,V,S>), dim3(N/(C*V), S), dim3(256), 0, stream,
                    x, w, y, scratch);
  check(hipGetLastError());
  if constexpr (S > 1) {
    hipLaunchKernelGGL((finish<S>), dim3(N/256), dim3(256), 0, stream, scratch, y);
    check(hipGetLastError());
  }
}

int main() try {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  half *x, *w, *y, *ref; float *scratch;
  check(hipMalloc(&x, K*2)); check(hipMalloc(&w, size_t(K)*N*2));
  check(hipMalloc(&y, N*2)); check(hipMalloc(&ref, N*2));
  check(hipMalloc(&scratch, N*16*4));
  hipStream_t stream; check(hipStreamCreate(&stream));
  hipLaunchKernelGGL(init, dim3((size_t(K)*N+255)/256), dim3(256), 0, stream, x, w);
  check(hipGetLastError()); check(hipStreamSynchronize(stream));

  hipblasLtHandle_t handle; check(hipblasLtCreate(&handle));
  hipblasLtMatmulDesc_t desc;
  check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasLtMatrixLayout_t a, b, c;
  check(hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, N, K, N));
  check(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, K, 1, K));
  check(hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, N, 1, N));
  std::vector<int> indices{914};
  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  check(hipblaslt_ext::getAlgosFromIndex(handle, indices, algorithms));
  if (algorithms.size() != 1) throw std::runtime_error("Expected solution 914");
  auto algo = algorithms[0].algo;
  float alpha = 1, beta = 0; size_t workspaceBytes = 0;
  check(hipblaslt_ext::matmulIsAlgoSupported(handle, desc, &alpha, a, b, &beta,
                                           c, c, algo, workspaceBytes));
  if (workspaceBytes > 256*1024*1024) throw std::runtime_error("Workspace too large");
  void *workspace = nullptr;
  if (workspaceBytes) check(hipMalloc(&workspace, workspaceBytes));
  auto blas = [&] {
    check(hipblasLtMatmul(handle, desc, &alpha, w, a, x, b, &beta, ref, c, ref,
                          c, &algo, workspace, workspaceBytes, stream));
  };
  blas(); check(hipStreamSynchronize(stream));
  std::vector<half> reference(N), observed(N);
  check(hipMemcpy(reference.data(), ref, N*2, hipMemcpyDeviceToHost));
  // This scalar oracle checks the test fixture, not model execution.
  for (unsigned i = 0; i < 31; ++i) {
    unsigned col = (i * 3571) % N;
    double sum = 0;
    for (unsigned k = 0; k < K; ++k)
      sum += double(__half2float(sample(k, 17))) * __half2float(sample(k*N+col, 41));
    if (std::abs(__half2float(reference[col]) - sum) > 0.003)
      throw std::runtime_error("BLAS disagrees with fixture oracle");
  }
  struct Candidate { const char *name; std::function<void()> run; };
  std::vector<Candidate> candidates;
#define ADD(C,V,S) candidates.push_back({#C "x" #V "/" #S, [&] { launch<C,V,S>(x,w,y,scratch,stream); }})
  ADD(32,1,1); ADD(32,1,4); ADD(32,1,16);
  ADD(32,2,1); ADD(32,2,4); ADD(32,2,16);
  ADD(32,4,1); ADD(32,4,4); ADD(32,4,16);
  ADD(64,2,1); ADD(64,2,4); ADD(64,2,16);
  ADD(64,4,1); ADD(64,4,4); ADD(64,4,16);
  ADD(128,2,1); ADD(128,2,4); ADD(128,2,16);
#undef ADD
  candidates.push_back({"integrated64x4/4", [&] {
    check(static_cast<hipError_t>(hip_head_matvec_f16(stream, x, w, y, scratch, N, K)));
  }});
  for (auto shape : {std::pair{N-1,K}, std::pair{N,K-1}})
    if (hip_head_matvec_f16(stream, x, w, y, scratch, shape.first, shape.second) !=
        hipErrorInvalidValue) throw std::runtime_error("Invalid shape accepted");
  if (hip_head_matvec_f16(stream, x, w, x, scratch, N, K) != hipErrorInvalidValue ||
      hip_head_matvec_f16(stream, nullptr, w, y, scratch, N, K) != hipErrorInvalidValue)
    throw std::runtime_error("Invalid pointer accepted");
  for (auto &candidate : candidates) {
    candidate.run(); check(hipStreamSynchronize(stream));
    check(hipMemcpy(observed.data(), y, N*2, hipMemcpyDeviceToHost));
    double e2 = 0, r2 = 0, maxError = 0; size_t unequal = 0;
    for (int col = 0; col < N; ++col) {
      float value = __half2float(observed[col]), base = __half2float(reference[col]);
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite candidate");
      double error = value - base;
      e2 += error*error; r2 += double(base)*base;
      maxError = std::max(maxError, std::abs(error)); unequal += value != base;
    }
    double relative = std::sqrt(e2/r2);
    std::printf("CHECK %s unequal=%zu max_abs=%.9g rel_l2=%.9g\n",
                candidate.name, unequal, maxError, relative);
    if (relative > 0.001 || maxError > 0.008)
      throw std::runtime_error("Candidate exceeds probe tolerance");
  }
  hipEvent_t start, end;
  check(hipEventCreate(&start)); check(hipEventCreate(&end));
  auto time = [&](const std::function<void()> &run) {
    for (int i = 0; i < 3; ++i) run();
    check(hipEventRecord(start, stream));
    for (int i = 0; i < 12; ++i) run();
    check(hipEventRecord(end, stream)); check(hipEventSynchronize(end));
    float elapsed; check(hipEventElapsedTime(&elapsed, start, end));
    return elapsed / 12;
  };
  // One weight matrix exceeds L2. Reverse both candidate and pair order.
  for (int round = 0; round < 4; ++round) {
    for (auto &candidate : candidates) {
      float base, tested;
      if (round % 2) { tested = time(candidate.run); base = time(blas); }
      else { base = time(blas); tested = time(candidate.run); }
      std::printf("TIME round=%d tile=%s candidate_ms=%.6f blas914_ms=%.6f GBps=%.2f\n",
                  round, candidate.name, tested, base, double(K)*N*2/(tested*1e6));
    }
    std::reverse(candidates.begin(), candidates.end());
  }
  check(hipEventDestroy(start)); check(hipEventDestroy(end));
  check(hipblasLtMatrixLayoutDestroy(a)); check(hipblasLtMatrixLayoutDestroy(b));
  check(hipblasLtMatrixLayoutDestroy(c)); check(hipblasLtMatmulDescDestroy(desc));
  check(hipblasLtDestroy(handle)); check(hipStreamDestroy(stream));
  if (workspace) check(hipFree(workspace));
  check(hipFree(x)); check(hipFree(w)); check(hipFree(y));
  check(hipFree(ref)); check(hipFree(scratch));
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
}
