/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- softmax_kernel.hip - Row-wise 2D softmax ----------------------------===//
//
// Standalone Softmax over the last dim of a flattened [rows, cols] row-major
// buffer of fp16, bf16 or fp32. Cooperative warp-shuffle reductions for max +
// sum, then an inverse-multiply. Matches ONNX Softmax semantics (`axis = -1`
// along the last dim). All arithmetic is fp32 whatever the storage type.
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
// activation.cpp) for `onnx.Softmax` outside fused attention paths. The
// pre-existing `softmax_inplace_kernel` in `gqa_kernel.hip` is column-wise
// over an internal layout convention specific to the GQA pipeline; using
// it for the row-wise standalone case requires the caller to launch one
// block per (head, col) pair which the existing `hip_gqa_softmax_inplace`
// launcher does NOT do — hence this distinct kernel.
//
// Numerics: exp2f(x * LOG2E) instead of expf(x) for the native HW
// instruction, fp32 accumulators, single max + single sum sweep per row.
//
// Four launch shapes, picked by width in `launch_softmax` at the bottom:
//   wave     one wavefront per row, the row in registers, 16-byte accesses,
//            no LDS and no barrier. cols == wave * {1,2,4} vectors per lane.
//   regcache one block per row, one element per thread. cols <= threads.
//   block    one block per row, up to kBlockMaxElems elements per thread in
//            registers, vector accesses. cols <= 256 * kBlockMaxElems. The
//            row is read once and written once.
//   general  one block per row, three strided passes over memory. Any cols.
// This is the usual three-tier row-softmax design (warp per row, block per
// row with the row cached, uncached fallback), as in OneFlow's softmax.cuh
// and PyTorch's SoftMax.cu.
// LDS use for the block-per-row shapes is
// `softmax_smem_floats(num_waves) * sizeof(float)` per block.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <initializer_list>

// Pulls in the HIP_KERNEL_API (dllexport) declarations of the entry points so
// this TU exports them from the per-arch SHARED kernel DLL (the native
// artifact link resolves them via the import lib).
#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"

#ifndef LOG2E
#define LOG2E 1.44269504088896340736f
#endif

// Wavefronts per block for the wave-per-row kernels: one row each.
static constexpr int kWavesPerBlock = 8;

// Largest per-thread element count of the block kernel, and its block size.
// 32 fp32 values per thread stay well clear of a register spill.
static constexpr int kBlockMaxElems = 32;
static constexpr int kMaxThreads = 256;

namespace {

template <typename T> struct SoftmaxIO;
template <> struct SoftmaxIO<__half> {
  __device__ static float load(__half v) { return __half2float(v); }
  __device__ static __half store(float v) { return __float2half(v); }
};
template <> struct SoftmaxIO<__hip_bfloat16> {
  __device__ static float load(__hip_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __hip_bfloat16 store(float v) {
    return __float2bfloat16(v);
  }
};
template <> struct SoftmaxIO<float> {
  __device__ static float load(float v) { return v; }
  __device__ static float store(float v) { return v; }
};

// Plain integer storage for one vector access of a given byte width, so a
// load or store compiles to a single global_load/store of that width.
template <int BYTES> struct RawVec;
template <> struct RawVec<16> { using type = uint4; };
template <> struct RawVec<8> { using type = uint2; };
template <> struct RawVec<4> { using type = uint32_t; };
template <> struct RawVec<2> { using type = uint16_t; };

__device__ inline float warp_reduce_max_f(float v) {
  for (int off = HIPDNN_WAVE_SIZE / 2; off > 0; off /= 2)
    v = fmaxf(v, __shfl_xor(v, off));
  return v;
}

__device__ inline float warp_reduce_sum_f(float v) {
  for (int off = HIPDNN_WAVE_SIZE / 2; off > 0; off /= 2)
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
  if (tid % HIPDNN_WAVE_SIZE == 0)
    smem[tid / HIPDNN_WAVE_SIZE] = v;
  __syncthreads();
  if (tid < HIPDNN_WAVE_SIZE) {
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
  if (tid % HIPDNN_WAVE_SIZE == 0)
    smem[tid / HIPDNN_WAVE_SIZE] = v;
  __syncthreads();
  if (tid < HIPDNN_WAVE_SIZE) {
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
// to be read once and written once. The three variants below exploit that;
// they differ in how a row is divided among threads.

// One element per thread, block per row. Valid when `cols <= blockDim.x`, so
// the general kernel's strided loop would run exactly one iteration anyway.
// Reductions still cross the whole block, so the barriers remain.
template <typename T>
__global__ void softmax_row_2d_regcache_kernel(const T* in, T* out, int rows,
                                               int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const T* in_row = in + (size_t)row * cols;
  T* out_row = out + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / HIPDNN_WAVE_SIZE;
  const bool active = tid < cols;

  extern __shared__ float smem[];

  const float x = active ? SoftmaxIO<T>::load(in_row[tid]) : -INFINITY;
  const float row_max =
      block_reduce_max_f(x, smem, tid, num_waves, num_waves);
  const float e = active ? exp2f((x - row_max) * LOG2E) : 0.0f;
  const float inv =
      1.0f / block_reduce_sum_f(e, smem, tid, num_waves, num_waves + 1);

  if (active)
    out_row[tid] = SoftmaxIO<T>::store(e * inv);
}

// One wavefront per row, `ELEMS` elements per lane held in registers. Valid
// when `cols == wave * ELEMS` and ELEMS is a multiple of the 16-byte vector
// width, which is the shape every windowed-attention softmax has.
//
// Two things change against the block-per-row form. The row is read and
// written in 16-byte transactions rather than element-sized ones, so a wave
// moves a full set of cache lines per memory instruction instead of a
// fraction of one. And because a row never spans more than one wavefront,
// both reductions are pure `__shfl_xor` -- the kernel needs no LDS and no
// `__syncthreads` at all.
//
// Lane vectors are strided by the wave size rather than packed per lane so
// that every individual load remains coalesced across the wave when ELEMS
// spans more than one vector.
template <typename T, int ELEMS>
__global__ void softmax_row_2d_wave_kernel(const T* in, T* out, int rows,
                                           int cols) {
  constexpr int kVecElems = 16 / sizeof(T);
  constexpr int kVecs = ELEMS / kVecElems;
  static_assert(ELEMS % kVecElems == 0, "ELEMS must fill whole vectors");

  const int lane = threadIdx.x % HIPDNN_WAVE_SIZE;
  const int wave = threadIdx.x / HIPDNN_WAVE_SIZE;
  const int row = blockIdx.x * (int)(blockDim.x / HIPDNN_WAVE_SIZE) + wave;
  if (row >= rows)
    return;
  const uint4* in_vec =
      reinterpret_cast<const uint4*>(in + (size_t)row * cols);
  uint4* out_vec = reinterpret_cast<uint4*>(out + (size_t)row * cols);

  float x[ELEMS];
  float local_max = -INFINITY;
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    const uint4 raw = in_vec[lane + v * HIPDNN_WAVE_SIZE];
    const T* h = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int e = 0; e < kVecElems; ++e) {
      const float val = SoftmaxIO<T>::load(h[e]);
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
    uint4 raw;
    T* h = reinterpret_cast<T*>(&raw);
#pragma unroll
    for (int e = 0; e < kVecElems; ++e)
      h[e] = SoftmaxIO<T>::store(x[v * kVecElems + e] * inv);
    out_vec[lane + v * HIPDNN_WAVE_SIZE] = raw;
  }
}

// One block per row, up to `ELEMS` elements per thread held in fp32
// registers, moved in `VEC`-element vectors. Valid when
// `blockDim.x * ELEMS >= cols`, `cols % VEC == 0` and both buffers are
// aligned to the vector width; the launcher picks the widest such VEC.
//
// This is the long-row counterpart of the wave kernel: a row too wide for one
// wavefront is still read once and written once, where the general kernel
// reads it three times and writes it twice. Thread `tid` owns vectors
// `tid + k * blockDim.x`, so each vector access is coalesced across the
// block. All loads are issued before the first reduction so the memory
// system sees every request of the row at once.
//
// A thread writes back exactly the vectors it read, after both reductions,
// so the source-aliases-destination case needs no extra barrier.
template <typename T, int ELEMS, int VEC>
__global__ void __launch_bounds__(kMaxThreads)
    softmax_row_2d_block_kernel(const T* in, T* out, int rows, int cols) {
  static_assert(ELEMS % VEC == 0, "ELEMS must fill whole vectors");
  constexpr int kVecs = ELEMS / VEC;
  using Raw = typename RawVec<VEC * sizeof(T)>::type;

  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const Raw* in_vec = reinterpret_cast<const Raw*>(in + (size_t)row * cols);
  Raw* out_vec = reinterpret_cast<Raw*>(out + (size_t)row * cols);

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / HIPDNN_WAVE_SIZE;
  const int row_vecs = cols / VEC;

  extern __shared__ float smem[];

  float x[ELEMS];
  float local_max = -INFINITY;
#pragma unroll
  for (int k = 0; k < kVecs; ++k) {
    const int v = tid + k * (int)blockDim.x;
    if (v < row_vecs) {
      const Raw raw = in_vec[v];
      const T* h = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float val = SoftmaxIO<T>::load(h[e]);
        x[k * VEC + e] = val;
        local_max = fmaxf(local_max, val);
      }
    } else {
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        x[k * VEC + e] = -INFINITY;
    }
  }
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, num_waves, num_waves);

  // Padding slots hold -inf, and exp2f(-inf) == 0, so they add nothing.
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    x[i] = exp2f((x[i] - row_max) * LOG2E);
    local_sum += x[i];
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, num_waves, num_waves + 1);

#pragma unroll
  for (int k = 0; k < kVecs; ++k) {
    const int v = tid + k * (int)blockDim.x;
    if (v < row_vecs) {
      Raw raw;
      T* h = reinterpret_cast<T*>(&raw);
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        h[e] = SoftmaxIO<T>::store(x[k * VEC + e] * inv);
      out_vec[v] = raw;
    }
  }
}

// The fallback for rows too long to cache in registers. Three passes over the
// row, so the only pass that may read what an earlier pass wrote is the third,
// and a thread there re-reads exactly the elements it wrote itself under the
// same `tid`-strided loop. No thread ever reads another thread's slice, which
// is what makes the source-aliases-destination case correct without a barrier
// beyond the ones the reductions already impose.
template <typename T>
__global__ void softmax_row_2d_general_kernel(const T* in, T* out, int rows,
                                              int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  const T* in_row = in + (size_t)row * cols;
  T* out_row = out + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int num_waves = blockDim.x / HIPDNN_WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int j = tid; j < cols; j += blockDim.x)
    local_max = fmaxf(local_max, SoftmaxIO<T>::load(in_row[j]));
  const float row_max =
      block_reduce_max_f(local_max, smem, tid, num_waves, num_waves);

  // Pass 2: exp2f + sum.
  float local_sum = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float e = exp2f((SoftmaxIO<T>::load(in_row[j]) - row_max) * LOG2E);
    out_row[j] = SoftmaxIO<T>::store(e);
    local_sum += e;
  }
  const float inv =
      1.0f / block_reduce_sum_f(local_sum, smem, tid, num_waves, num_waves + 1);

  // Pass 3: normalize.
  for (int j = tid; j < cols; j += blockDim.x)
    out_row[j] = SoftmaxIO<T>::store(SoftmaxIO<T>::load(out_row[j]) * inv);
}

// --- Host dispatch ------------------------------------------------------

size_t smem_bytes_for(int threads, int wave) {
  return static_cast<size_t>(softmax_smem_floats(threads / wave)) *
         sizeof(float);
}

template <typename T, int ELEMS>
int launch_wave(hipStream_t stream, const T* in, T* out, int rows, int cols,
                int wave) {
  const int grid = (rows + kWavesPerBlock - 1) / kWavesPerBlock;
  softmax_row_2d_wave_kernel<T, ELEMS>
      <<<grid, kWavesPerBlock * wave, 0, stream>>>(in, out, rows, cols);
  return hipGetLastError();
}

template <typename T, int ELEMS, int VEC>
int launch_block(hipStream_t stream, const T* in, T* out, int rows, int cols,
                 int threads, int wave) {
  softmax_row_2d_block_kernel<T, ELEMS, VEC>
      <<<rows, threads, smem_bytes_for(threads, wave), stream>>>(in, out, rows,
                                                                 cols);
  return hipGetLastError();
}

template <typename T, int ELEMS>
int launch_block_vec(hipStream_t stream, const T* in, T* out, int rows,
                     int cols, int threads, int wave, int vec_bytes) {
  switch (vec_bytes) {
  case 16:
    return launch_block<T, ELEMS, 16 / sizeof(T)>(stream, in, out, rows, cols,
                                                  threads, wave);
  case 8:
    return launch_block<T, ELEMS, 8 / sizeof(T)>(stream, in, out, rows, cols,
                                                 threads, wave);
  default:
    return launch_block<T, ELEMS, 1>(stream, in, out, rows, cols, threads,
                                     wave);
  }
}

// Widest access, in bytes, that divides the row and is aligned in both
// buffers. The row stride is cols * sizeof(T), so every row start is aligned
// as soon as the first one is. 8 bytes is kept next to 16 because a common
// attention width (1500) is a multiple of 4 halves but not of 8.
int pick_vec_bytes(const void* in, const void* out, int cols, int elem_bytes) {
  const uintptr_t addr_bits = reinterpret_cast<uintptr_t>(in) |
                              reinterpret_cast<uintptr_t>(out);
  const int64_t row_bytes = static_cast<int64_t>(cols) * elem_bytes;
  for (int bytes : {16, 8}) {
    if (bytes > elem_bytes && row_bytes % bytes == 0 &&
        addr_bits % bytes == 0)
      return bytes;
  }
  return elem_bytes;
}

template <typename T>
int launch_softmax(void* stream_ptr, const void* input, void* output, int rows,
                   int cols) {
  if (!input || !output || rows <= 0 || cols <= 0)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();

  const T* in = static_cast<const T*>(input);
  T* out = static_cast<T*>(output);
  const int wave = hipdnn_device_wave_size();

  // Wave-per-row when the row is exactly one wavefront wide at one, two or
  // four 16-byte vectors per lane and both buffers are 16-byte aligned; the
  // row stride is then a multiple of 16 bytes on its own.
  constexpr int kVecElems = 16 / sizeof(T);
  if ((reinterpret_cast<uintptr_t>(in) % 16) == 0 &&
      (reinterpret_cast<uintptr_t>(out) % 16) == 0) {
    if (cols == wave * kVecElems)
      return launch_wave<T, kVecElems>(stream, in, out, rows, cols, wave);
    if (cols == wave * kVecElems * 2)
      return launch_wave<T, kVecElems * 2>(stream, in, out, rows, cols, wave);
    if (cols == wave * kVecElems * 4)
      return launch_wave<T, kVecElems * 4>(stream, in, out, rows, cols, wave);
  }

  // One element per thread: the smallest power-of-two block of at least one
  // wave that covers the row.
  int threads = wave;
  while (threads < cols && threads < kMaxThreads)
    threads *= 2;
  if (cols <= threads) {
    softmax_row_2d_regcache_kernel<T>
        <<<rows, threads, smem_bytes_for(threads, wave), stream>>>(in, out,
                                                                   rows, cols);
    return hipGetLastError();
  }

  // Row in registers: the fewest elements per thread, then the smallest block,
  // that covers the row.
  if (cols <= kMaxThreads * kBlockMaxElems) {
    const int vec_bytes =
        pick_vec_bytes(in, out, cols, static_cast<int>(sizeof(T)));
    for (int elems : {8, 16, kBlockMaxElems}) {
      for (int t = wave; t <= kMaxThreads; t *= 2) {
        if (t * elems < cols)
          continue;
        switch (elems) {
        case 8:
          return launch_block_vec<T, 8>(stream, in, out, rows, cols, t, wave,
                                        vec_bytes);
        case 16:
          return launch_block_vec<T, 16>(stream, in, out, rows, cols, t, wave,
                                         vec_bytes);
        default:
          return launch_block_vec<T, kBlockMaxElems>(stream, in, out, rows,
                                                     cols, t, wave, vec_bytes);
        }
      }
    }
  }

  softmax_row_2d_general_kernel<T>
      <<<rows, kMaxThreads, smem_bytes_for(kMaxThreads, wave), stream>>>(
          in, out, rows, cols);
  return hipGetLastError();
}

} // namespace

extern "C" int hip_softmax_row_2d(void* stream_ptr, const void* input,
                                    void* output, int rows, int cols) {
  return launch_softmax<__half>(stream_ptr, input, output, rows, cols);
}

extern "C" int hip_softmax_row_2d_bf16(void* stream_ptr, const void* input,
                                         void* output, int rows, int cols) {
  return launch_softmax<__hip_bfloat16>(stream_ptr, input, output, rows, cols);
}

extern "C" int hip_softmax_row_2d_fp32(void* stream_ptr, const void* input,
                                         void* output, int rows, int cols) {
  return launch_softmax<float>(stream_ptr, input, output, rows, cols);
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
