/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- softmax_kernel.hip - Row-wise 2D softmax (fp16) --------------------===//
//
// Standalone Softmax over the last dim of a flattened [rows, cols] fp16
// row-major buffer. One block per row, cooperative warp-shuffle reductions
// for max + sum, then an inverse-multiply pass. Matches ONNX Softmax
// semantics (`axis = -1` along the last dim).
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
// Three launch shapes, picked by width in the entry points at the bottom:
// one wavefront per row holding the row in registers (no LDS, no barrier),
// one thread per element with the element in a register, and the general
// strided form for rows too wide to cache. LDS use for the latter two is
// `softmax_smem_floats(num_waves) * sizeof(float)` per block.
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
__device__ __host__ inline int softmax_smem_floats(int num_waves) {
  return num_waves + 2;
}

// --- Register-cached variants ------------------------------------------
//
// The general kernel below re-reads the row from memory in each of its three
// passes because the value it needs is not kept anywhere. When a thread owns a
// fixed slice of the row, that value fits in a register and the row only has
// to be read once and written once. Both variants below exploit that; they
// differ in how a row is divided among threads.

// One element per thread, block per row. Valid when `cols <= blockDim.x`, so
// the general kernel's strided loop would run exactly one iteration anyway.
// Reductions still cross the whole block, so the barriers remain.
__global__ void softmax_row_2d_regcache_kernel(const __half* in, __half* out,
                                                 int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const __half* in_row = in + (size_t)row * cols;
  __half* out_row = out + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / WAVE_SIZE;
  const bool active = tid < cols;

  extern __shared__ float smem[];

  const float x = active ? __half2float(in_row[tid]) : -INFINITY;
  const float row_max =
      block_reduce_max_f(x, smem, tid, num_waves, num_waves);
  const float e = active ? exp2f((x - row_max) * LOG2E) : 0.0f;
  const float inv =
      1.0f / block_reduce_sum_f(e, smem, tid, num_waves, num_waves + 1);

  if (active)
    out_row[tid] = __float2half(e * inv);
}

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

// The fallback for rows too long to cache in registers. Three passes over the
// row, so the only pass that may read what an earlier pass wrote is the third,
// and a thread there re-reads exactly the elements it wrote itself under the
// same `tid`-strided loop. No thread ever reads another thread's slice, which
// is what makes the source-aliases-destination case correct without a barrier
// beyond the ones the reductions already impose.
__global__ void softmax_row_2d_general_kernel(const __half* in, __half* out,
                                                int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const __half* in_row = in + (size_t)row * cols;
  __half* out_row = out + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int j = tid; j < cols; j += blockDim.x)
    local_max = fmaxf(local_max, __half2float(in_row[j]));
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, num_waves, num_waves);

  // Pass 2: exp2f + sum.
  float local_sum = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float e = exp2f((__half2float(in_row[j]) - row_max) * LOG2E);
    out_row[j] = __float2half(e);
    local_sum += e;
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, num_waves, num_waves + 1);

  // Pass 3: normalize.
  for (int j = tid; j < cols; j += blockDim.x)
    out_row[j] = __float2half(__half2float(out_row[j]) * inv);
}

// fp32 counterparts of the two register-cached variants above. Only the
// element type and the 16-byte vector width differ.
__global__ void softmax_row_2d_regcache_fp32_kernel(const float* in, float* out,
                                                      int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const float* in_row = in + (size_t)row * cols;
  float* out_row = out + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / WAVE_SIZE;
  const bool active = tid < cols;

  extern __shared__ float smem[];

  const float x = active ? in_row[tid] : -INFINITY;
  const float row_max =
      block_reduce_max_f(x, smem, tid, num_waves, num_waves);
  const float e = active ? exp2f((x - row_max) * LOG2E) : 0.0f;
  const float inv =
      1.0f / block_reduce_sum_f(e, smem, tid, num_waves, num_waves + 1);

  if (active)
    out_row[tid] = e * inv;
}

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

// fp32 variant — same algorithm, reads float instead of __half.
// Used by Qwen VLM and other models where Softmax input is fp32.
__global__ void softmax_row_2d_general_fp32_kernel(const float* in, float* out,
                                                     int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const float* in_row = in + (size_t)row * cols;
  float* out_row = out + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int j = tid; j < cols; j += blockDim.x)
    local_max = fmaxf(local_max, in_row[j]);
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, num_waves, num_waves);

  // Pass 2: exp2f + sum.
  float local_sum = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float e = exp2f((in_row[j] - row_max) * LOG2E);
    out_row[j] = e;
    local_sum += e;
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, num_waves, num_waves + 1);

  // Pass 3: normalize.
  for (int j = tid; j < cols; j += blockDim.x)
    out_row[j] = out_row[j] * inv;
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

  // Block per row. Pick block size: power of two up to min(cols, 256). 32
  // minimum for one wave on RDNA / 64 on CDNA. SoftMax for vision attention
  // has cols up to ~4096, well-served by 256 threads × 16 iters.
  int threads = 32;
  while (threads < cols && threads < 256)
    threads *= 2;
  if (threads > 256)
    threads = 256;
  int num_waves = (threads + WAVE_SIZE - 1) / WAVE_SIZE;
  size_t smem_bytes =
      static_cast<size_t>(softmax_smem_floats(num_waves)) * sizeof(float);
  if (cols <= threads) {
    softmax_row_2d_regcache_kernel<<<rows, threads, smem_bytes, stream>>>(
        in, out, rows, cols);
    return hipGetLastError();
  }
  softmax_row_2d_general_kernel<<<rows, threads, smem_bytes, stream>>>(
      in, out, rows, cols);
  return hipGetLastError();
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

  // Block per row. Pick block size: power of two up to min(cols, 256). 32
  // minimum for one wave on RDNA / 64 on CDNA. SoftMax for vision attention
  // has cols up to ~4096, well-served by 256 threads × 16 iters.
  int threads = 32;
  while (threads < cols && threads < 256)
    threads *= 2;
  if (threads > 256)
    threads = 256;
  int num_waves = (threads + WAVE_SIZE - 1) / WAVE_SIZE;
  size_t smem_bytes =
      static_cast<size_t>(softmax_smem_floats(num_waves)) * sizeof(float);
  if (cols <= threads) {
    softmax_row_2d_regcache_fp32_kernel<<<rows, threads, smem_bytes, stream>>>(
        in, out, rows, cols);
    return hipGetLastError();
  }
  softmax_row_2d_general_fp32_kernel<<<rows, threads, smem_bytes, stream>>>(
      in, out, rows, cols);
  return hipGetLastError();
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
