// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// Test only. Compare sequential-K FMA with pinned BLAS 1278.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "small_matvec.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

constexpr int N = 128, K = 2688, Rotations = 64;
void check(hipError_t s) {
  if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
void check(hipblasStatus_t s) {
  if (s != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BLAS status " + std::to_string(s));
}
__device__ float sample(unsigned i, unsigned salt) {
  i ^= salt; i ^= i >> 16; i *= 0x7feb352du;
  i ^= i >> 15; i *= 0x846ca68bu; i ^= i >> 16;
  return __half2float(__float2half((int(i%20001)-10000)*0.00003f));
}
__global__ void init(float *x, float *w) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < unsigned(K)*Rotations) x[i] = sample(i, 17);
  if (i < unsigned(K)*N*Rotations) w[i] = sample(i, 41);
}
template<int Columns>
__global__ void sequential(const float *x, const float *w, float *y) {
  if (threadIdx.x >= Columns) return;
  int col = blockIdx.x * Columns + threadIdx.x;
  float sum = 0;
  for (int j = 0; j < K; j += 8) {
#pragma unroll
    for (int p = 0; p < 8; ++p)
      sum = __fmaf_rn(x[j+p], w[(j+p)*N+col], sum);
  }
  y[col] = sum;
}
template<int Columns>
void launch(const float *x, const float *w, float *y, hipStream_t stream) {
  hipLaunchKernelGGL((sequential<Columns>), dim3(N/Columns), dim3(32),
                    0, stream, x, w, y);
  check(hipGetLastError());
}
template<int Columns, int Chunk>
__global__ void prefetched(const float *x, const float *w, float *y) {
  if (threadIdx.x >= Columns) return;
  int col=blockIdx.x*Columns+threadIdx.x;
  float sum=0;
  for (int j=0; j<K; j+=Chunk) {
    float xv[Chunk], wv[Chunk];
#pragma unroll
    for (int p=0; p<Chunk; ++p) {
      xv[p]=x[j+p];
      wv[p]=w[(j+p)*N+col];
    }
#pragma unroll
    for (int p=0; p<Chunk; ++p) sum=__fmaf_rn(xv[p],wv[p],sum);
  }
  y[col]=sum;
}
template<int Columns, int Chunk>
void launchPrefetch(const float *x, const float *w, float *y, hipStream_t stream) {
  static_assert(K%Chunk==0);
  hipLaunchKernelGGL((prefetched<Columns,Chunk>), dim3(N/Columns), dim3(32),
                    0, stream, x,w,y);
  check(hipGetLastError());
}
int main() try {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  float *x, *w, *y, *reference;
  check(hipMalloc(&x, size_t(K)*Rotations*4));
  check(hipMalloc(&w, size_t(K)*N*Rotations*4));
  check(hipMalloc(&y, N*4)); check(hipMalloc(&reference, N*Rotations*4));
  hipStream_t stream; check(hipStreamCreate(&stream));
  hipLaunchKernelGGL(init, dim3((size_t(K)*N*Rotations+255)/256), dim3(256),
                    0, stream, x, w);
  check(hipGetLastError()); check(hipStreamSynchronize(stream));
  hipblasLtHandle_t handle; check(hipblasLtCreate(&handle));
  hipblasLtMatmulDesc_t desc;
  check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasLtMatrixLayout_t a,b,c;
  check(hipblasLtMatrixLayoutCreate(&a, HIP_R_32F, N, K, N));
  check(hipblasLtMatrixLayoutCreate(&b, HIP_R_32F, K, 1, K));
  check(hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, N, 1, N));
  std::vector<int> indices{1278};
  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  check(hipblaslt_ext::getAlgosFromIndex(handle, indices, algorithms));
  if (algorithms.size() != 1) throw std::runtime_error("Expected solution 1278");
  auto algo = algorithms[0].algo;
  float alpha=1, beta=0; size_t workspaceBytes=0;
  check(hipblaslt_ext::matmulIsAlgoSupported(handle, desc, &alpha, a, b, &beta,
                                           c, c, algo, workspaceBytes));
  if (workspaceBytes > 256*1024*1024) throw std::runtime_error("Workspace too large");
  void *workspace=nullptr;
  if (workspaceBytes) check(hipMalloc(&workspace, workspaceBytes));
  auto blas = [&](int r) {
    check(hipblasLtMatmul(handle, desc, &alpha, w+size_t(r)*K*N, a, x+r*K, b,
                          &beta, reference+r*N, c, reference+r*N, c, &algo,
                          workspace, workspaceBytes, stream));
  };
  for (int r=0; r<Rotations; ++r) blas(r);
  check(hipStreamSynchronize(stream));
  std::vector<float> expected(Rotations*N), actual(N);
  check(hipMemcpy(expected.data(), reference, expected.size()*4, hipMemcpyDeviceToHost));
  struct Candidate { const char *name; std::function<void(int)> run; bool exact=true; };
  std::vector<Candidate> candidates;
#define ADD(C) candidates.push_back({#C, [&](int r) { launch<C>(x+r*K,w+size_t(r)*K*N,y,stream); }})
  ADD(1); ADD(2); ADD(4); ADD(8); ADD(16); ADD(32);
#undef ADD
#define ADD(C,P) candidates.push_back({#C "/pre" #P, [&](int r) { launchPrefetch<C,P>(x+r*K,w+size_t(r)*K*N,y,stream); }})
  ADD(4,32); ADD(8,32); ADD(32,32);
  ADD(4,64); ADD(8,64); ADD(32,64);
  ADD(4,128); ADD(8,128); ADD(32,128);
#undef ADD
  candidates.push_back({"integrated", [&](int r) {
    check(static_cast<hipError_t>(hip_sequential_router_f32(
        stream,x+r*K,w+size_t(r)*K*N,y,N,K)));
  }});
  if (hip_sequential_router_f32(stream,x,w,y,N-1,K) != hipErrorInvalidValue ||
      hip_sequential_router_f32(stream,x,w,y,N,K-1) != hipErrorInvalidValue ||
      hip_sequential_router_f32(stream,nullptr,w,y,N,K) != hipErrorInvalidValue ||
      hip_sequential_router_f32(stream,x,w,x,N,K) != hipErrorInvalidValue)
    throw std::runtime_error("Invalid router arguments accepted");
  for (auto &candidate : candidates) {
    size_t unequal=0; double error2=0, reference2=0, maxError=0;
    for (int r=0; r<Rotations; ++r) {
      candidate.run(r); check(hipStreamSynchronize(stream));
      check(hipMemcpy(actual.data(), y, N*4, hipMemcpyDeviceToHost));
      for (int n=0; n<N; ++n) {
        float base=expected[r*N+n];
        if (!std::isfinite(actual[n]) || !std::isfinite(base))
          throw std::runtime_error("Non-finite probe output");
        unequal += std::memcmp(&actual[n], &base, sizeof(float)) != 0;
        double error=double(actual[n])-base;
        error2 += error*error; reference2 += double(base)*base;
        maxError=std::max(maxError, std::abs(error));
      }
    }
    candidate.exact=unequal==0;
    if (std::strcmp(candidate.name,"integrated")==0 && unequal)
      throw std::runtime_error("Integrated router differs from BLAS");
    std::printf("CHECK columns=%s values=%d unequal=%zu max_abs=%.9g rel_l2=%.9g\n",
                candidate.name, N*Rotations, unequal, maxError, std::sqrt(error2/reference2));
  }
  hipEvent_t start,end;
  check(hipEventCreate(&start)); check(hipEventCreate(&end));
  auto time = [&](const std::function<void(int)> &run) {
    for (int r=0; r<Rotations; ++r) run(r);
    check(hipEventRecord(start, stream));
    for (int r=0; r<2*Rotations; ++r) run(r%Rotations);
    check(hipEventRecord(end, stream)); check(hipEventSynchronize(end));
    float elapsed; check(hipEventElapsedTime(&elapsed,start,end));
    return elapsed/(2*Rotations);
  };
  for (int round=0; round<4; ++round) {
    for (auto &candidate : candidates) {
      float base,tested;
      if (round%2) { tested=time(candidate.run); base=time(blas); }
      else { base=time(blas); tested=time(candidate.run); }
      std::printf("TIME round=%d columns=%s exact=%d candidate_ms=%.6f blas1278_ms=%.6f\n",
                  round,candidate.name,candidate.exact,tested,base);
    }
    std::reverse(candidates.begin(),candidates.end());
  }
  check(hipEventDestroy(start)); check(hipEventDestroy(end));
  check(hipblasLtMatrixLayoutDestroy(a)); check(hipblasLtMatrixLayoutDestroy(b));
  check(hipblasLtMatrixLayoutDestroy(c)); check(hipblasLtMatmulDescDestroy(desc));
  check(hipblasLtDestroy(handle)); check(hipStreamDestroy(stream));
  if (workspace) check(hipFree(workspace));
  check(hipFree(x)); check(hipFree(w)); check(hipFree(y)); check(hipFree(reference));
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr,"%s\n",e.what()); return 1;
}
