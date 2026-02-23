/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_attention.cpp - Main driver for 3D attention test
//--------------===//
//
// Single-head attention: Q/K/V projections, transpose, score, scale,
// softmax, output. All tensors are 3D [B,S,D].
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <vector>

struct MemRef3D { float *a, *al; int64_t o, s[3], st[3]; };

extern "C" __declspec(dllimport) MemRef3D attention(
    float* X_a, float* X_al, int64_t X_o, int64_t X_s0, int64_t X_s1,
    int64_t X_s2, int64_t X_st0, int64_t X_st1, int64_t X_st2, float* Wq_a,
    float* Wq_al, int64_t Wq_o, int64_t Wq_s0, int64_t Wq_s1, int64_t Wq_st0,
    int64_t Wq_st1, float* Wk_a, float* Wk_al, int64_t Wk_o, int64_t Wk_s0,
    int64_t Wk_s1, int64_t Wk_st0, int64_t Wk_st1, float* Wv_a, float* Wv_al,
    int64_t Wv_o, int64_t Wv_s0, int64_t Wv_s1, int64_t Wv_st0, int64_t Wv_st1,
    float* sc_a, float* sc_al, int64_t sc_o, int64_t sc_s0, int64_t sc_s1,
    int64_t sc_s2, int64_t sc_st0, int64_t sc_st1, int64_t sc_st2, float* out_a,
    float* out_al, int64_t out_o, int64_t out_s0, int64_t out_s1,
    int64_t out_s2, int64_t out_st0, int64_t out_st1, int64_t out_st2);

static void cpu_matmul(const float* A, const float* B, float* C, int64_t M,
                       int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float s = 0;
      for (int64_t k = 0; k < K; ++k)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

static void cpu_softmax(const float* in, float* out, int64_t rows,
                        int64_t cols) {
  for (int64_t n = 0; n < rows; ++n) {
    float mx = in[n * cols];
    for (int64_t d = 1; d < cols; ++d)
      mx = std::fmax(mx, in[n * cols + d]);
    float sum = 0;
    for (int64_t d = 0; d < cols; ++d) {
      out[n * cols + d] = std::exp(in[n * cols + d] - mx);
      sum += out[n * cols + d];
    }
    for (int64_t d = 0; d < cols; ++d)
      out[n * cols + d] /= sum;
  }
}

#define HIP_CHECK(call)                                            \
  do {                                                             \
    hipError_t e = (call);                                         \
    if (e != hipSuccess) {                                         \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
              hipGetErrorString(e));                               \
      exit(1);                                                     \
    }                                                              \
  } while (0)

int main() {
  const int64_t B = 2, S = 4, D = 8;
  const float scale_val = 1.0f / std::sqrt((float)D);
  printf("3D Attention: B=%lld S=%lld D=%lld scale=%.4f\n", B, S, D, scale_val);

  std::vector<float> h_X(B * S * D), h_Wq(D * D), h_Wk(D * D), h_Wv(D * D);
  std::vector<float> h_scale(B * S * S, scale_val);
  std::vector<float> h_out(B * S * D, 0), h_ref(B * S * D);

  for (int64_t i = 0; i < B * S * D; ++i)
    h_X[i] = float(i % 7) * 0.1f;
  for (int64_t i = 0; i < D * D; ++i)
    h_Wq[i] = (i / D == i % D) ? 1.0f : 0.0f;
  for (int64_t i = 0; i < D * D; ++i)
    h_Wk[i] = h_Wq[i];
  for (int64_t i = 0; i < D * D; ++i)
    h_Wv[i] = h_Wq[i];

  // CPU reference per batch
  for (int64_t b = 0; b < B; b++) {
    std::vector<float> Q(S * D), K(S * D), V(S * D), KT(D * S), sc(S * S),
        scaled(S * S), attn(S * S);
    cpu_matmul(&h_X[b * S * D], h_Wq.data(), Q.data(), S, D, D);
    cpu_matmul(&h_X[b * S * D], h_Wk.data(), K.data(), S, D, D);
    cpu_matmul(&h_X[b * S * D], h_Wv.data(), V.data(), S, D, D);
    for (int64_t i = 0; i < S; i++)
      for (int64_t j = 0; j < D; j++)
        KT[j * S + i] = K[i * D + j];
    cpu_matmul(Q.data(), KT.data(), sc.data(), S, D, S);
    for (int64_t i = 0; i < S * S; i++)
      scaled[i] = sc[i] * scale_val;
    cpu_softmax(scaled.data(), attn.data(), S, S);
    cpu_matmul(attn.data(), V.data(), &h_ref[b * S * D], S, S, D);
  }

  float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_scale, *d_out;
  HIP_CHECK(hipMalloc(&d_X, B * S * D * 4));
  HIP_CHECK(hipMalloc(&d_Wq, D * D * 4));
  HIP_CHECK(hipMalloc(&d_Wk, D * D * 4));
  HIP_CHECK(hipMalloc(&d_Wv, D * D * 4));
  HIP_CHECK(hipMalloc(&d_scale, B * S * S * 4));
  HIP_CHECK(hipMalloc(&d_out, B * S * D * 4));
  HIP_CHECK(hipMemcpy(d_X, h_X.data(), B * S * D * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_Wq, h_Wq.data(), D * D * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_Wk, h_Wk.data(), D * D * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_Wv, h_Wv.data(), D * D * 4, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_scale, h_scale.data(), B * S * S * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, B * S * D * 4));

  printf("Running attention on GPU...\n");
  attention(d_X, d_X, 0, B, S, D, S * D, D, 1, d_Wq, d_Wq, 0, D, D, D, 1, d_Wk,
            d_Wk, 0, D, D, D, 1, d_Wv, d_Wv, 0, D, D, D, 1, d_scale, d_scale, 0,
            B, S, S, S * S, S, 1, d_out, d_out, 0, B, S, D, S * D, D, 1);

  HIP_CHECK(
      hipMemcpy(h_out.data(), d_out, B * S * D * 4, hipMemcpyDeviceToHost));

  printf("\nGPU output (batch 0, first 2 rows):\n");
  for (int64_t i = 0; i < 2; ++i) {
    printf("  [");
    for (int64_t j = 0; j < D; ++j)
      printf(" %6.3f", h_out[i * D + j]);
    printf(" ]\n");
  }
  printf("CPU reference (batch 0, first 2 rows):\n");
  for (int64_t i = 0; i < 2; ++i) {
    printf("  [");
    for (int64_t j = 0; j < D; ++j)
      printf(" %6.3f", h_ref[i * D + j]);
    printf(" ]\n");
  }

  float mx = 0;
  for (int64_t i = 0; i < B * S * D; ++i) {
    float d = std::fabs(h_out[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("\nMax diff: %e\n%s\n", mx, mx < 1e-2f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_X));
  HIP_CHECK(hipFree(d_Wq));
  HIP_CHECK(hipFree(d_Wk));
  HIP_CHECK(hipFree(d_Wv));
  HIP_CHECK(hipFree(d_scale));
  HIP_CHECK(hipFree(d_out));
  return (mx < 1e-2f) ? 0 : 1;
}
