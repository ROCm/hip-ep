/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- softmax_kernel.hip - Row-wise 2D softmax (fp16, fp32) --------------===//
//
// Standalone Softmax over the last dim of a flattened [rows, cols] fp16 or
// fp32 row-major buffer, with warp-shuffle reductions for max + sum. Matches
// ONNX Softmax semantics (`axis = -1` along the last dim).
//
// Every kernel here reads a source and writes a distinct destination, and
// tolerates the two being the same buffer. That matters more than it looks:
// `hip.miopen.softmax` is destination-passing with separate input and output
// buffers, so a source-and-destination kernel lets the runtime softmax
// straight from one to the other. An in-place-only kernel forces the caller
// to copy input over output first, and that copy moves exactly as many bytes
// as the softmax does -- doubling the op's traffic however good the kernel is.
//
// Used by the `hip_miopen_softmax` runtime entry point (lib/Runtime/real/
// activation.cpp) for `onnx.Softmax` outside fused attention paths — most
// notably the SigLIP / ViT self-attention scores in vision encoders. The
// pre-existing `softmax_inplace_kernel` in `gqa_kernel.hip` is column-wise
// over an internal layout convention specific to the GQA pipeline; using
// it for the row-wise standalone case requires the caller to launch one
// block per (head, col) pair which the existing `hip_gqa_softmax_inplace`
// launcher does NOT do — hence this distinct kernel.
//
// Numerics: exp2f(x * LOG2E) instead of expf(x) for the native HW
// instruction, fp32 accumulators, single max + single sum sweep per row.
//
// Launch shapes, picked by width in the entry points at the bottom:
//   - rows of exactly 1, 2 or 4 vectors per lane: one wavefront per row,
//     16-byte loads, no LDS and no barrier;
//   - rows up to WAVE_SIZE * 16: one masked wavefront per row, several rows
//     per block, still shuffle-only;
//   - rows up to 32768: one block per row, 256 threads up to 8192 columns and
//     1024 past that, the row held in registers so it is read once and
//     written once (16-byte loads when the buffers and the row stride allow);
//   - wider rows: the three-pass general kernel.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

// Pulls in the HIP_KERNEL_API (dllexport) declaration of
// hip_softmax_row_2d_inplace so this TU exports it from the per-arch SHARED
// kernel DLL (the native artifact link resolves it via the import lib).
#include "hip_custom_kernels.h"

#ifndef WAVE_SIZE
#if defined(__GFX9__) || defined(__GFX9_NEXT__)
#define WAVE_SIZE 64
#else
#define WAVE_SIZE 32
#endif
#endif

#ifndef LOG2E
#define LOG2E 1.44269504088896340736f
#endif

// Wavefronts per block for the wave-per-row kernels: one row each, sized so a
// block stays at the 256-thread shape the other launches use.
static constexpr int kWavesPerBlock = 8;

// Largest elements-per-lane of the masked wave kernel; wider rows go to a
// block per row.
static constexpr int kMaxWaveElems = 16;

// Block per row: 256 threads up to kRowCols, 1024 up to kWideRowCols. Each
// thread keeps at most kMaxRowElems values in registers at either size. The
// block reductions fold one wave of per-wave partials, so a block may hold at
// most WAVE_SIZE waves; 1024 threads is 32 waves on a 32-lane wave.
static constexpr int kRowThreads = 256;
static constexpr int kWideRowThreads = 1024;
static constexpr int kMaxRowElems = 32;
static constexpr int kRowCols = kRowThreads * kMaxRowElems;
static constexpr int kWideRowCols = kWideRowThreads * kMaxRowElems;

namespace {

__device__ inline float warp_reduce_max_f(float v) {
  for (int off = WAVE_SIZE / 2; off > 0; off /= 2)
    v = fmaxf(v, __shfl_xor(v, off));
  return v;
}

__device__ inline float warp_reduce_sum_f(float v) {
  for (int off = WAVE_SIZE / 2; off > 0; off /= 2)
    v += __shfl_xor(v, off);
  return v;
}

// Block-wide reductions through LDS, for the kernels whose rows span more than
// one wavefront.
//
// `slot` is the cell the result is broadcast through, and callers must give
// each reduction in a kernel its own. Broadcasting through smem[0] instead --
// which is the obvious thing to do, since the per-wave partials are dead by
// then -- races: after the barrier that publishes the broadcast, a fast wave
// can store its next partial into smem[0] before a slow wave has read the
// previous result out of it. The window is short enough that a kernel doing
// global memory work between the two reductions usually survives it, and short
// enough that one holding its row in registers does not.
__device__ inline float block_reduce_max_f(float v, float* smem, int tid,
                                             int num_waves, int slot) {
  v = warp_reduce_max_f(v);
  if (tid % WAVE_SIZE == 0)
    smem[tid / WAVE_SIZE] = v;
  __syncthreads();
  if (tid < WAVE_SIZE) {
    float t = (tid < num_waves) ? smem[tid] : -INFINITY;
    t = warp_reduce_max_f(t);
    if (tid == 0)
      smem[slot] = t;
  }
  __syncthreads();
  return smem[slot];
}

__device__ inline float block_reduce_sum_f(float v, float* smem, int tid,
                                             int num_waves, int slot) {
  v = warp_reduce_sum_f(v);
  if (tid % WAVE_SIZE == 0)
    smem[tid / WAVE_SIZE] = v;
  __syncthreads();
  if (tid < WAVE_SIZE) {
    float t = (tid < num_waves) ? smem[tid] : 0.0f;
    t = warp_reduce_sum_f(t);
    if (tid == 0)
      smem[slot] = t;
  }
  __syncthreads();
  return smem[slot];
}

// LDS cells a block-per-row kernel needs: one partial per wave, plus a
// dedicated broadcast cell for each of the two reductions.
__device__ __host__ constexpr int softmax_smem_floats(int num_waves) {
  return num_waves + 2;
}

__device__ inline float softmax_as_float(__half v) { return __half2float(v); }
__device__ inline float softmax_as_float(float v) { return v; }

__device__ inline void softmax_store(__half* dst, float v) {
  *dst = __float2half(v);
}
__device__ inline void softmax_store(float* dst, float v) { *dst = v; }

// Smallest power of two >= n, for picking a register-array bucket.
__host__ inline int softmax_pow2_ceil(int n) {
  int p = 1;
  while (p < n)
    p *= 2;
  return p;
}

// --- Wave-per-row kernels ----------------------------------------------

// One wavefront per row, `ELEMS` elements per lane held in registers. Valid
// when `cols == WAVE_SIZE * ELEMS` and ELEMS is a multiple of the 16-byte
// vector width, which is the shape every windowed-attention softmax has.
//
// Two things change against the block-per-row form. The row is read and
// written in 16-byte transactions rather than 2-byte ones, so a wave moves a
// full set of cache lines per memory instruction instead of a fraction of one.
// And because a row never spans more than one wavefront, both reductions are
// pure `__shfl_xor` -- the kernel needs no LDS and no `__syncthreads` at all.
//
// Lane vectors are strided by WAVE_SIZE rather than packed per lane so that
// every individual load remains coalesced across the wave when ELEMS spans
// more than one vector.
template <int ELEMS>
__global__ void softmax_row_2d_wave_kernel(const __half* in, __half* out,
                                             int rows, int cols) {
  constexpr int kVecElems = 8; // 8 halves == 16 bytes
  constexpr int kVecs = ELEMS / kVecElems;

  const int lane = threadIdx.x % WAVE_SIZE;
  const int wave = threadIdx.x / WAVE_SIZE;
  const int row = blockIdx.x * (int)(blockDim.x / WAVE_SIZE) + wave;
  if (row >= rows)
    return;
  const uint4* in_vec =
      reinterpret_cast<const uint4*>(in + (size_t)row * cols);
  uint4* out_vec = reinterpret_cast<uint4*>(out + (size_t)row * cols);

  uint4 raw[kVecs];
  float x[ELEMS];
  float local_max = -INFINITY;
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    raw[v] = in_vec[lane + v * WAVE_SIZE];
    const __half* h = reinterpret_cast<const __half*>(&raw[v]);
#pragma unroll
    for (int e = 0; e < kVecElems; ++e) {
      const float val = __half2float(h[e]);
      x[v * kVecElems + e] = val;
      local_max = fmaxf(local_max, val);
    }
  }
  const float row_max = warp_reduce_max_f(local_max);

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    x[i] = exp2f((x[i] - row_max) * LOG2E);
    local_sum += x[i];
  }
  const float inv = 1.0f / warp_reduce_sum_f(local_sum);

#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    __half* h = reinterpret_cast<__half*>(&raw[v]);
#pragma unroll
    for (int e = 0; e < kVecElems; ++e)
      h[e] = __float2half(x[v * kVecElems + e] * inv);
    out_vec[lane + v * WAVE_SIZE] = raw[v];
  }
}

// fp32 counterpart of the kernel above. Only the element type and the 16-byte
// vector width differ.
template <int ELEMS>
__global__ void softmax_row_2d_wave_fp32_kernel(const float* in, float* out,
                                                  int rows, int cols) {
  constexpr int kVecElems = 4; // 4 floats == 16 bytes
  constexpr int kVecs = ELEMS / kVecElems;

  const int lane = threadIdx.x % WAVE_SIZE;
  const int wave = threadIdx.x / WAVE_SIZE;
  const int row = blockIdx.x * (int)(blockDim.x / WAVE_SIZE) + wave;
  if (row >= rows)
    return;
  const float4* in_vec =
      reinterpret_cast<const float4*>(in + (size_t)row * cols);
  float4* out_vec = reinterpret_cast<float4*>(out + (size_t)row * cols);

  float4 raw[kVecs];
  float x[ELEMS];
  float local_max = -INFINITY;
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    raw[v] = in_vec[lane + v * WAVE_SIZE];
    const float* f = reinterpret_cast<const float*>(&raw[v]);
#pragma unroll
    for (int e = 0; e < kVecElems; ++e) {
      x[v * kVecElems + e] = f[e];
      local_max = fmaxf(local_max, f[e]);
    }
  }
  const float row_max = warp_reduce_max_f(local_max);

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    x[i] = exp2f((x[i] - row_max) * LOG2E);
    local_sum += x[i];
  }
  const float inv = 1.0f / warp_reduce_sum_f(local_sum);

#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    float* f = reinterpret_cast<float*>(&raw[v]);
#pragma unroll
    for (int e = 0; e < kVecElems; ++e)
      f[e] = x[v * kVecElems + e] * inv;
    out_vec[lane + v * WAVE_SIZE] = raw[v];
  }
}

// One wavefront per row of any width up to WAVE_SIZE * ELEMS, several rows
// per block. Element `lane + i * WAVE_SIZE` lives in lane `lane`, so every
// load stays coalesced; lanes past `cols` are masked. Both reductions are
// shuffles, so a short row never pays for LDS or a barrier.
template <typename T, int ELEMS>
__global__ __launch_bounds__(kWavesPerBlock * WAVE_SIZE) void
softmax_row_2d_masked_wave_kernel(const T* in, T* out, int rows, int cols) {
  const int lane = threadIdx.x % WAVE_SIZE;
  const int row = blockIdx.x * kWavesPerBlock + threadIdx.x / WAVE_SIZE;
  if (row >= rows)
    return;
  const T* in_row = in + (size_t)row * cols;
  T* out_row = out + (size_t)row * cols;

  float x[ELEMS];
  float local_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int j = lane + i * WAVE_SIZE;
    x[i] = (j < cols) ? softmax_as_float(in_row[j]) : -INFINITY;
    local_max = fmaxf(local_max, x[i]);
  }
  const float row_max = warp_reduce_max_f(local_max);

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int j = lane + i * WAVE_SIZE;
    x[i] = (j < cols) ? exp2f((x[i] - row_max) * LOG2E) : 0.0f;
    local_sum += x[i];
  }
  const float inv = 1.0f / warp_reduce_sum_f(local_sum);

#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int j = lane + i * WAVE_SIZE;
    if (j < cols)
      softmax_store(out_row + j, x[i] * inv);
  }
}

// --- Block-per-row register kernels --------------------------------------
//
// The general kernel below re-reads the row from memory in each of its three
// passes because the value it needs is not kept anywhere. When a thread owns a
// fixed slice of the row, that slice fits in registers and the row only has
// to be read once and written once. Every thread reads its whole slice before
// the first barrier and writes only that slice after the last one, which is
// what keeps the source-aliases-destination case correct.

// THREADS threads per row, CHUNKS 16-byte vectors per thread. Valid when both
// buffers are 16-byte aligned and `cols` is a multiple of the vector width, so
// every row starts on a 16-byte boundary.
template <typename T, int THREADS, int CHUNKS>
__global__ __launch_bounds__(THREADS) void softmax_row_2d_regvec_kernel(
    const T* in, T* out, int rows, int cols) {
  constexpr int kVec = 16 / sizeof(T);
  constexpr int kElems = CHUNKS * kVec;
  constexpr int kWaves = THREADS / WAVE_SIZE;
  __shared__ float smem[softmax_smem_floats(kWaves)];

  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const int tid = threadIdx.x;
  const int nvec = cols / kVec;
  const uint4* in_vec = reinterpret_cast<const uint4*>(in + (size_t)row * cols);
  uint4* out_vec = reinterpret_cast<uint4*>(out + (size_t)row * cols);

  float x[kElems];
  float local_max = -INFINITY;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int v = tid + c * THREADS;
    const bool active = v < nvec;
    uint4 raw = active ? in_vec[v] : uint4{0, 0, 0, 0};
    const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const float val = active ? softmax_as_float(e[k]) : -INFINITY;
      x[c * kVec + k] = val;
      local_max = fmaxf(local_max, val);
    }
  }
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, kWaves, kWaves);

  float local_sum = 0.0f;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const bool active = tid + c * THREADS < nvec;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const int i = c * kVec + k;
      x[i] = active ? exp2f((x[i] - row_max) * LOG2E) : 0.0f;
      local_sum += x[i];
    }
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, kWaves, kWaves + 1);

#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int v = tid + c * THREADS;
    if (v < nvec) {
      uint4 raw;
      T* e = reinterpret_cast<T*>(&raw);
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        softmax_store(e + k, x[c * kVec + k] * inv);
      out_vec[v] = raw;
    }
  }
}

// Scalar form of the kernel above for rows that are misaligned or not a
// multiple of the vector width: THREADS threads per row, ELEMS elements per
// thread strided by THREADS.
template <typename T, int THREADS, int ELEMS>
__global__ __launch_bounds__(THREADS) void softmax_row_2d_regstride_kernel(
    const T* in, T* out, int rows, int cols) {
  constexpr int kWaves = THREADS / WAVE_SIZE;
  __shared__ float smem[softmax_smem_floats(kWaves)];

  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const int tid = threadIdx.x;
  const T* in_row = in + (size_t)row * cols;
  T* out_row = out + (size_t)row * cols;

  float x[ELEMS];
  float local_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int j = tid + i * THREADS;
    x[i] = (j < cols) ? softmax_as_float(in_row[j]) : -INFINITY;
    local_max = fmaxf(local_max, x[i]);
  }
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, kWaves, kWaves);

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int j = tid + i * THREADS;
    x[i] = (j < cols) ? exp2f((x[i] - row_max) * LOG2E) : 0.0f;
    local_sum += x[i];
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, kWaves, kWaves + 1);

#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int j = tid + i * THREADS;
    if (j < cols)
      softmax_store(out_row + j, x[i] * inv);
  }
}

// --- Three-pass fallback -------------------------------------------------
//
// The fallback for rows too long to cache in registers. Three passes over the
// row, so the only pass that may read what an earlier pass wrote is the third,
// and a thread there re-reads exactly the elements it wrote itself under the
// same `tid`-strided loop. No thread ever reads another thread's slice, which
// is what makes the source-aliases-destination case correct without a barrier
// beyond the ones the reductions already impose.
template <typename T>
__global__ __launch_bounds__(kRowThreads) void softmax_row_2d_general_kernel(
    const T* in, T* out, int rows, int cols) {
  constexpr int kWaves = kRowThreads / WAVE_SIZE;
  __shared__ float smem[softmax_smem_floats(kWaves)];

  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const T* in_row = in + (size_t)row * cols;
  T* out_row = out + (size_t)row * cols;
  const int tid = threadIdx.x;

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int j = tid; j < cols; j += kRowThreads)
    local_max = fmaxf(local_max, softmax_as_float(in_row[j]));
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, kWaves, kWaves);

  // Pass 2: exp2f + sum.
  float local_sum = 0.0f;
  for (int j = tid; j < cols; j += kRowThreads) {
    const float e = exp2f((softmax_as_float(in_row[j]) - row_max) * LOG2E);
    softmax_store(out_row + j, e);
    local_sum += e;
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, kWaves, kWaves + 1);

  // Pass 3: normalize.
  for (int j = tid; j < cols; j += kRowThreads)
    softmax_store(out_row + j, softmax_as_float(out_row[j]) * inv);
}

// --- Launchers -------------------------------------------------------------

template <typename T>
__host__ void launch_masked_wave(const T* in, T* out, int rows, int cols,
                                 hipStream_t stream) {
  const int grid = (rows + kWavesPerBlock - 1) / kWavesPerBlock;
  const int block = kWavesPerBlock * WAVE_SIZE;
  switch (softmax_pow2_ceil((cols + WAVE_SIZE - 1) / WAVE_SIZE)) {
  case 1:
    softmax_row_2d_masked_wave_kernel<T, 1>
        <<<grid, block, 0, stream>>>(in, out, rows, cols);
    break;
  case 2:
    softmax_row_2d_masked_wave_kernel<T, 2>
        <<<grid, block, 0, stream>>>(in, out, rows, cols);
    break;
  case 4:
    softmax_row_2d_masked_wave_kernel<T, 4>
        <<<grid, block, 0, stream>>>(in, out, rows, cols);
    break;
  case 8:
    softmax_row_2d_masked_wave_kernel<T, 8>
        <<<grid, block, 0, stream>>>(in, out, rows, cols);
    break;
  default:
    softmax_row_2d_masked_wave_kernel<T, kMaxWaveElems>
        <<<grid, block, 0, stream>>>(in, out, rows, cols);
    break;
  }
}

// Register bucket for a block of THREADS threads: the smallest power-of-two
// number of 16-byte vectors per thread that covers the row.
template <typename T, int THREADS>
__host__ void launch_row_vec_at(const T* in, T* out, int rows, int cols,
                                hipStream_t stream) {
  constexpr int kVec = 16 / sizeof(T);
  constexpr int kMaxChunks = kMaxRowElems / kVec;
  const int nvec = cols / kVec;
  switch (softmax_pow2_ceil((nvec + THREADS - 1) / THREADS)) {
  case 1:
    softmax_row_2d_regvec_kernel<T, THREADS, 1>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  case 2:
    softmax_row_2d_regvec_kernel<T, THREADS, 2>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  case 4:
    softmax_row_2d_regvec_kernel<T, THREADS, 4>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  default:
    softmax_row_2d_regvec_kernel<T, THREADS, kMaxChunks>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  }
}

template <typename T, int THREADS>
__host__ void launch_row_scalar_at(const T* in, T* out, int rows, int cols,
                                   hipStream_t stream) {
  switch (softmax_pow2_ceil((cols + THREADS - 1) / THREADS)) {
  case 1:
  case 2:
  case 4:
    softmax_row_2d_regstride_kernel<T, THREADS, 4>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  case 8:
    softmax_row_2d_regstride_kernel<T, THREADS, 8>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  case 16:
    softmax_row_2d_regstride_kernel<T, THREADS, 16>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  default:
    softmax_row_2d_regstride_kernel<T, THREADS, kMaxRowElems>
        <<<rows, THREADS, 0, stream>>>(in, out, rows, cols);
    break;
  }
}

// Rows up to kRowCols take kRowThreads threads, wider rows kWideRowThreads.
template <typename T>
__host__ void launch_row_vec(const T* in, T* out, int rows, int cols,
                             hipStream_t stream) {
  if (cols > kRowCols)
    launch_row_vec_at<T, kWideRowThreads>(in, out, rows, cols, stream);
  else
    launch_row_vec_at<T, kRowThreads>(in, out, rows, cols, stream);
}

template <typename T>
__host__ void launch_row_scalar(const T* in, T* out, int rows, int cols,
                                hipStream_t stream) {
  if (cols > kRowCols)
    launch_row_scalar_at<T, kWideRowThreads>(in, out, rows, cols, stream);
  else
    launch_row_scalar_at<T, kRowThreads>(in, out, rows, cols, stream);
}

// Launch order shared by both element types, after the exact-width wave
// kernels: masked wave, block-per-row in registers, three passes.
template <typename T>
__host__ int launch_softmax_rows(const T* in, T* out, int rows, int cols,
                                 hipStream_t stream) {
  constexpr int kVec = 16 / sizeof(T);
  if (cols <= WAVE_SIZE * kMaxWaveElems) {
    launch_masked_wave(in, out, rows, cols, stream);
  } else if (cols <= kWideRowCols) {
    const bool aligned = (reinterpret_cast<uintptr_t>(in) % 16) == 0 &&
                         (reinterpret_cast<uintptr_t>(out) % 16) == 0 &&
                         cols % kVec == 0;
    if (aligned)
      launch_row_vec(in, out, rows, cols, stream);
    else
      launch_row_scalar(in, out, rows, cols, stream);
  } else {
    softmax_row_2d_general_kernel<T>
        <<<rows, kRowThreads, 0, stream>>>(in, out, rows, cols);
  }
  return hipGetLastError();
}

} // namespace

extern "C" int hip_softmax_row_2d(void* stream_ptr, const void* input,
                                    void* output, int rows, int cols) {
  if (!input || !output || rows <= 0 || cols <= 0)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();

  const __half* in = static_cast<const __half*>(input);
  __half* out = static_cast<__half*>(output);

  // Wave-per-row when the row is exactly one wavefront wide at the vector
  // width and both buffers are 16-byte aligned; the row stride is then a
  // multiple of 16 bytes on its own. This covers the windowed-attention
  // widths (256 and 512 on a 32-lane wave).
  if ((reinterpret_cast<uintptr_t>(in) % 16) == 0 &&
      (reinterpret_cast<uintptr_t>(out) % 16) == 0) {
    const int grid = (rows + kWavesPerBlock - 1) / kWavesPerBlock;
    const int block = kWavesPerBlock * WAVE_SIZE;
    if (cols == WAVE_SIZE * 8) {
      softmax_row_2d_wave_kernel<8>
          <<<grid, block, 0, stream>>>(in, out, rows, cols);
      return hipGetLastError();
    }
    if (cols == WAVE_SIZE * 16) {
      softmax_row_2d_wave_kernel<16>
          <<<grid, block, 0, stream>>>(in, out, rows, cols);
      return hipGetLastError();
    }
    if (cols == WAVE_SIZE * 32) {
      softmax_row_2d_wave_kernel<32>
          <<<grid, block, 0, stream>>>(in, out, rows, cols);
      return hipGetLastError();
    }
  }

  return launch_softmax_rows(in, out, rows, cols, stream);
}

// fp32 variant — for models where Softmax input is fp32 (e.g. Qwen VLM).
extern "C" int hip_softmax_row_2d_fp32(void* stream_ptr, const void* input,
                                         void* output, int rows, int cols) {
  if (!input || !output || rows <= 0 || cols <= 0)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();

  const float* in = static_cast<const float*>(input);
  float* out = static_cast<float*>(output);

  if ((reinterpret_cast<uintptr_t>(in) % 16) == 0 &&
      (reinterpret_cast<uintptr_t>(out) % 16) == 0) {
    const int grid = (rows + kWavesPerBlock - 1) / kWavesPerBlock;
    const int block = kWavesPerBlock * WAVE_SIZE;
    if (cols == WAVE_SIZE * 4) {
      softmax_row_2d_wave_fp32_kernel<4>
          <<<grid, block, 0, stream>>>(in, out, rows, cols);
      return hipGetLastError();
    }
    if (cols == WAVE_SIZE * 8) {
      softmax_row_2d_wave_fp32_kernel<8>
          <<<grid, block, 0, stream>>>(in, out, rows, cols);
      return hipGetLastError();
    }
    if (cols == WAVE_SIZE * 16) {
      softmax_row_2d_wave_fp32_kernel<16>
          <<<grid, block, 0, stream>>>(in, out, rows, cols);
      return hipGetLastError();
    }
  }

  return launch_softmax_rows(in, out, rows, cols, stream);
}

// Kept for callers that hold a single buffer. Every kernel above tolerates
// its source and destination being the same pointer.
extern "C" int hip_softmax_row_2d_inplace(void* stream_ptr, void* data,
                                            int rows, int cols) {
  return hip_softmax_row_2d(stream_ptr, data, data, rows, cols);
}

extern "C" int hip_softmax_row_2d_inplace_fp32(void* stream_ptr, void* data,
                                                 int rows, int cols) {
  return hip_softmax_row_2d_fp32(stream_ptr, data, data, rows, cols);
}
