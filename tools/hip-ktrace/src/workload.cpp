/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TILE 16
#define ELEM_BLOCK 256

// Register-blocked GEMM variants. All use 16x16 threads and a GEMM_BK deep
// k-step; the TM x TN template parameters set how many outputs each thread
// accumulates, so the block tile is (16*TM) x (16*TN). Several instantiations
// are dispatched every round, which puts them on one timeline under identical
// clocks -- a fairer comparison than separate runs.
#define GEMM_BK 16
#define GEMM_THREADS_X 16
#define GEMM_THREADS_Y 16
// Shared-tile row padding, in floats. Padding by one float offsets the banks
// but leaves an odd row stride, which costs 16-byte alignment and makes the
// compiler emit ds_load_2addr_b32 where ds_load_b128 would do; measured at
// 25.6% of the 4x8 variant's cost against 16.2% for 8x4. Four floats offsets
// the banks just as well and keeps the stride 16-byte aligned.
#define GEMM_PAD 4
// Largest 16*TM over the instantiations below; --size must be a multiple of it.
#define GEMM_MAX_TILE 128

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t s__ = (expr);                                                   \
    if (s__ != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)s__,                \
              hipGetErrorString(s__), __FILE__, __LINE__);                     \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__global__ void GemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int N) {
  __shared__ float ta[TILE][TILE];
  __shared__ float tb[TILE][TILE];
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < N / TILE; ++t) {
    ta[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
    tb[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k)
      acc += ta[threadIdx.y][k] * tb[k][threadIdx.x];
    __syncthreads();
  }
  if (row < N && col < N)
    C[row * N + col] = acc;
}

// Register-blocked counterpart of GemmKernel. Each thread accumulates a TM x TN
// sub-tile instead of a single element, so one value read from LDS feeds TN
// fused multiply-adds rather than one, and the global-load latency is amortised
// over TM*TN times the arithmetic.
//
// The shared tiles are padded by GEMM_PAD floats: an unpadded row stride that
// is a multiple of the 32 LDS banks would serialise the per-thread column
// reads.
template <int TM, int TN>
__device__ void GemmTiledBody(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, int N) {
  constexpr int BM = GEMM_THREADS_Y * TM;
  constexpr int BN = GEMM_THREADS_X * TN;
  constexpr int THREADS = GEMM_THREADS_X * GEMM_THREADS_Y;

  __shared__ float as[GEMM_BK][BM + GEMM_PAD]; // A tile, stored transposed
  __shared__ float bs[GEMM_BK][BN + GEMM_PAD];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  // Output sub-tile this thread owns inside the block tile.
  const int threadRow = threadIdx.y * TM;
  const int threadCol = threadIdx.x * TN;

  // Global-load lane assignments, chosen so neighbouring threads read
  // neighbouring addresses.
  const int aRow = tid / GEMM_BK;
  const int aCol = tid % GEMM_BK;
  const int bRow = tid / BN;
  const int bCol = tid % BN;

  float acc[TM][TN] = {};

  for (int kTile = 0; kTile < N; kTile += GEMM_BK) {
    for (int off = 0; off + aRow < BM; off += THREADS / GEMM_BK)
      as[aCol][aRow + off] = A[(blockRow + aRow + off) * N + kTile + aCol];
    for (int off = 0; off + bRow < GEMM_BK; off += THREADS / BN)
      bs[bRow + off][bCol] = B[(kTile + bRow + off) * N + blockCol + bCol];
    __syncthreads();

    for (int k = 0; k < GEMM_BK; ++k) {
      float a[TM];
      float b[TN];
      for (int i = 0; i < TM; ++i)
        a[i] = as[k][threadRow + i];
      for (int j = 0; j < TN; ++j)
        b[j] = bs[k][threadCol + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    const int row = blockRow + threadRow + i;
    if (row >= N)
      continue;
    for (int j = 0; j < TN; ++j) {
      const int col = blockCol + threadCol + j;
      if (col < N)
        C[row * N + col] = acc[i][j];
    }
  }
}

// One named kernel per configuration, so each appears under its own name in a
// trace instead of as a mangled template instantiation.
__global__ void GemmKernelTiled2x2(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int N) {
  GemmTiledBody<2, 2>(A, B, C, N);
}

__global__ void GemmKernelTiled4x4(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int N) {
  GemmTiledBody<4, 4>(A, B, C, N);
}

__global__ void GemmKernelTiled8x8(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int N) {
  GemmTiledBody<8, 8>(A, B, C, N);
}

// Rectangular blocks sit between 4x4 and 8x8 on both axes that matter: they
// need (TM+TN)/(TM*TN) = 0.375 LDS reads per FMA against 0.5 and 0.25, while
// their LDS footprint stays below the point where 8x8 loses occupancy.
__global__ void GemmKernelTiled4x8(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int N) {
  GemmTiledBody<4, 8>(A, B, C, N);
}

__global__ void GemmKernelTiled8x4(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int N) {
  GemmTiledBody<8, 4>(A, B, C, N);
}

__global__ void VecAddKernel(const float *A, const float *B, float *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

__global__ void MulKernel(const float *A, const float *B, float *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] * B[i];
}

__global__ void TransposeKernel(const float *in, float *out, int N) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
  if (x < N && y < N)
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  if (x < N && y < N)
    out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void SaxpyKernel(float a, const float *X, float *Y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    Y[i] = a * X[i] + Y[i];
}

__global__ void ReluKernel(float *A, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] = A[i] > 0.0f ? A[i] : 0.0f;
}

__global__ void SqrtKernel(float *A, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] = sqrtf(fabsf(A[i]) + 1e-6f);
}

__global__ void ScaleKernel(float *A, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] *= s;
}

int main(int argc, char **argv) {
  unsigned rounds = 2000;
  unsigned N = 256;
  unsigned stream_count = 0; // 0 means HIP's null/default stream
  bool graph = false;
  bool copies = false;
  bool abrupt_exit = false;
  bool verify = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rounds") && i + 1 < argc)
      rounds = (unsigned)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--size") && i + 1 < argc)
      N = (unsigned)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--streams") && i + 1 < argc)
      stream_count = (unsigned)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--graph"))
      graph = true;
    else if (!strcmp(argv[i], "--copies"))
      copies = true;
    else if (!strcmp(argv[i], "--abrupt-exit"))
      abrupt_exit = true;
    else if (!strcmp(argv[i], "--verify"))
      verify = true;
  }
  if (rounds == 0)
    rounds = 1;
  if (N % GEMM_MAX_TILE) {
    fprintf(stderr, "--size must be a multiple of %d\n", GEMM_MAX_TILE);
    return 2;
  }
  if (graph && stream_count > 1) {
    fprintf(stderr,
            "--graph and --streams > 1 are separate validation modes\n");
    return 2;
  }

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("device     : %s (%s)\n", props.name, props.gcnArchName);
  printf("mode       : %s", graph ? "hipGraph replay" : "direct launch");
  if (!graph)
    printf(" (%u stream%s)", stream_count ? stream_count : 1,
           (stream_count == 1 || stream_count == 0) ? "" : "s");
  printf("\n");
  printf("dispatches : %u  (%u rounds x 13 kernels)\n", rounds * 13, rounds);

  const int n = (int)(N * N);
  const size_t bytes = (size_t)n * sizeof(float);
  float *A, *B, *C, *X, *Y;
  HIP_CHECK(hipMalloc(&A, bytes));
  HIP_CHECK(hipMalloc(&B, bytes));
  HIP_CHECK(hipMalloc(&C, bytes));
  HIP_CHECK(hipMalloc(&X, bytes));
  HIP_CHECK(hipMalloc(&Y, bytes));
  HIP_CHECK(hipMemset(A, 0, bytes));
  HIP_CHECK(hipMemset(B, 0, bytes));
  HIP_CHECK(hipMemset(C, 0, bytes));
  HIP_CHECK(hipMemset(X, 0, bytes));
  HIP_CHECK(hipMemset(Y, 0, bytes));
  if (copies) {
    float host = 1.0f;
    HIP_CHECK(hipMemcpy(A, &host, sizeof(host), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(&host, A, sizeof(host), hipMemcpyDeviceToHost));
  }

  const dim3 eb(ELEM_BLOCK), eg((n + ELEM_BLOCK - 1) / ELEM_BLOCK);
  const dim3 gb(TILE, TILE), gg(N / TILE, N / TILE);
  const dim3 tb(GEMM_THREADS_X, GEMM_THREADS_Y);
  const dim3 tg2(N / (GEMM_THREADS_X * 2), N / (GEMM_THREADS_Y * 2));
  const dim3 tg4(N / (GEMM_THREADS_X * 4), N / (GEMM_THREADS_Y * 4));
  const dim3 tg8(N / (GEMM_THREADS_X * 8), N / (GEMM_THREADS_Y * 8));
  const dim3 tg48(N / (GEMM_THREADS_X * 8), N / (GEMM_THREADS_Y * 4));
  const dim3 tg84(N / (GEMM_THREADS_X * 4), N / (GEMM_THREADS_Y * 8));

  // --verify checks GemmKernelTiled against GemmKernel. The rotation runs on
  // zero-filled buffers, where any GEMM returns zero and a wrong kernel would
  // still compare equal, so this fills A and B with distinct values first.
  if (verify) {
    std::vector<float> ha((size_t)n), hb((size_t)n), h1((size_t)n),
        h2((size_t)n);
    for (int i = 0; i < n; ++i) {
      ha[i] = (float)((i % 13) - 6) * 0.25f;
      hb[i] = (float)((i % 7) - 3) * 0.5f;
    }
    HIP_CHECK(hipMemcpy(A, ha.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B, hb.data(), bytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(C, 0, bytes));
    hipLaunchKernelGGL(GemmKernel, gg, gb, 0, nullptr, A, B, C, (int)N);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h1.data(), C, bytes, hipMemcpyDeviceToHost));

    double magnitude = 0.0;
    for (int i = 0; i < n; ++i)
      if (fabs((double)h1[i]) > magnitude)
        magnitude = fabs((double)h1[i]);
    if (magnitude == 0.0) {
      fprintf(stderr, "verify FAILED: reference output is all zero\n");
      return EXIT_FAILURE;
    }
    const double tolerance = magnitude * 1e-5;

    struct Variant {
      const char *name;
      void (*kernel)(const float *, const float *, float *, int);
      dim3 grid;
    };
    const Variant variants[] = {
        {"tiled2x2", GemmKernelTiled2x2, tg2},
        {"tiled4x4", GemmKernelTiled4x4, tg4},
        {"tiled8x8", GemmKernelTiled8x8, tg8},
        {"tiled4x8", GemmKernelTiled4x8, tg48},
        {"tiled8x4", GemmKernelTiled8x4, tg84},
    };

    bool ok = true;
    for (const Variant &v : variants) {
      HIP_CHECK(hipMemset(C, 0, bytes));
      hipLaunchKernelGGL(v.kernel, v.grid, tb, 0, nullptr, A, B, C, (int)N);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(h2.data(), C, bytes, hipMemcpyDeviceToHost));

      double worst = 0.0;
      int at = -1;
      for (int i = 0; i < n; ++i) {
        const double diff = fabs((double)h1[i] - (double)h2[i]);
        if (diff > worst) {
          worst = diff;
          at = i;
        }
      }
      printf("verify     : %-8s max |naive - tiled| = %g at %d  (tolerance %g) "
             " %s\n",
             v.name, worst, at, tolerance, worst > tolerance ? "FAILED" : "ok");
      if (worst > tolerance)
        ok = false;
    }
    printf("verify     : reference magnitude up to %g\n", magnitude);
    if (!ok)
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
  }

#define ROTATION(stream)                                                       \
  do {                                                                         \
    hipLaunchKernelGGL(GemmKernel, gg, gb, 0, stream, A, B, C, (int)N);        \
    hipLaunchKernelGGL(GemmKernelTiled2x2, tg2, tb, 0, stream, A, B, C,        \
                       (int)N);                                                \
    hipLaunchKernelGGL(GemmKernelTiled4x4, tg4, tb, 0, stream, A, B, C,        \
                       (int)N);                                                \
    hipLaunchKernelGGL(GemmKernelTiled8x8, tg8, tb, 0, stream, A, B, C,        \
                       (int)N);                                                \
    hipLaunchKernelGGL(GemmKernelTiled4x8, tg48, tb, 0, stream, A, B, C,       \
                       (int)N);                                                \
    hipLaunchKernelGGL(GemmKernelTiled8x4, tg84, tb, 0, stream, A, B, C,       \
                       (int)N);                                                \
    hipLaunchKernelGGL(VecAddKernel, eg, eb, 0, stream, A, B, C, n);           \
    hipLaunchKernelGGL(MulKernel, eg, eb, 0, stream, A, B, C, n);              \
    hipLaunchKernelGGL(TransposeKernel, gg, gb, 0, stream, A, B, (int)N);      \
    hipLaunchKernelGGL(SaxpyKernel, eg, eb, 0, stream, 2.0f, X, Y, n);         \
    hipLaunchKernelGGL(ReluKernel, eg, eb, 0, stream, C, n);                   \
    hipLaunchKernelGGL(SqrtKernel, eg, eb, 0, stream, C, n);                   \
    hipLaunchKernelGGL(ScaleKernel, eg, eb, 0, stream, C, 1.0001f, n);         \
  } while (0)

  HIP_CHECK(hipDeviceSynchronize());
  const auto t0 = std::chrono::steady_clock::now();

  if (graph) {
    hipStream_t stream = nullptr;
    hipGraph_t g = nullptr;
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    ROTATION(stream);
    HIP_CHECK(hipStreamEndCapture(stream, &g));
    HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (unsigned r = 0; r < rounds; ++r)
      HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(g));
    HIP_CHECK(hipStreamDestroy(stream));
  } else {
    std::vector<hipStream_t> streams(stream_count, nullptr);
    for (hipStream_t &stream : streams)
      HIP_CHECK(hipStreamCreate(&stream));
    for (unsigned r = 0; r < rounds; ++r)
      ROTATION(streams.empty() ? nullptr : streams[r % streams.size()]);
    for (hipStream_t stream : streams)
      HIP_CHECK(hipStreamSynchronize(stream));
    for (hipStream_t stream : streams)
      HIP_CHECK(hipStreamDestroy(stream));
  }
#undef ROTATION

  HIP_CHECK(hipDeviceSynchronize());
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  printf("wall       : %.3f s  ->  %.3f us / dispatch\n", wall,
         wall * 1e6 / (rounds * 13.0));
  std::fflush(stdout);
  if (abrupt_exit)
    std::_Exit(7);

  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(C));
  HIP_CHECK(hipFree(X));
  HIP_CHECK(hipFree(Y));
  return 0;
}
