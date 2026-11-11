/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft GroupNorm. Mean and variance are computed in FP32 (FP64 for
 * f64) over each (n, group), then the per-channel affine is applied.
 * activation 1 is SiLU.
 *
 * channels_last 0: (N, C, spatial). channels_last 1: (N, spatial, C).
 *
 * Large groups use a split reduction so that the grid fills the device even
 * when N * groups is small:
 *
 *   stats:    grid of (channel block, pixel block) tiles. Each block reduces
 *             its tile to one (mean, M2, count) partial per group it covers.
 *   finalize: one block per (n, group) merges its partials in a fixed order
 *             (Chan's parallel variance) into (mean, inv_std).
 *   apply:    a second, vectorized read that normalizes, applies gamma/beta
 *             and the optional SiLU.
 *
 * Every stage carries a mean and a sum of squared deviations (M2), never a
 * raw sum of squares, so the variance does not cancel when |mean| >> stddev.
 *
 * No step uses atomics, so the output is bitwise deterministic for a given
 * device and workspace size. Small groups, and calls without a big enough
 * workspace, use a fused kernel with one block per (n, group).
 *
 * Launch sizes come from the device's reported unit count and the shape; there
 * are no per-architecture tables.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBlock = 256;
// Below this many elements per group the fused kernel's launch saving beats
// the parallelism of the split path. HIPDNN_EP_GROUPNORM_SPLIT_SIZE overrides
// it.
constexpr int64_t kMinSplitCount = 16384;
// The split targets this many rounds of the blocks a unit (a CU, or a WGP on
// RDNA) holds at once, so a unit that finishes its first round early still
// has work.
constexpr int64_t kResidentRounds = 2;
// Each thread should stream at least this many vectors per split block.
constexpr int64_t kMinVectorsPerThread = 4;
// Partials per group. Bounds the workspace and the finalize merge length.
constexpr int64_t kMaxPartials = 1024;
constexpr size_t kWorkspaceAlign = 256;

template <typename T> struct GroupNormTraits;

template <> struct GroupNormTraits<float> {
  using Compute = float;
  __device__ static inline Compute load(float v) { return v; }
  __device__ static inline float store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
  __device__ static inline Compute exp_fn(Compute v) { return expf(v); }
};

template <> struct GroupNormTraits<__half> {
  using Compute = float;
  __device__ static inline Compute load(__half v) { return __half2float(v); }
  __device__ static inline __half store(Compute v) { return __float2half(v); }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
  __device__ static inline Compute exp_fn(Compute v) { return expf(v); }
};

template <> struct GroupNormTraits<__hip_bfloat16> {
  using Compute = float;
  __device__ static inline Compute load(__hip_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ static inline __hip_bfloat16 store(Compute v) {
    return __float2bfloat16(v);
  }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
  __device__ static inline Compute exp_fn(Compute v) { return expf(v); }
};

template <> struct GroupNormTraits<double> {
  using Compute = double;
  __device__ static inline Compute load(double v) { return v; }
  __device__ static inline double store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrt(v); }
  __device__ static inline Compute exp_fn(Compute v) { return exp(v); }
};

template <typename T, int V> struct alignas(sizeof(T) * V) GnVec {
  T v[V];
};

// `d` is x - mean. Folding the mean into the bias instead (x * a + b) cancels
// two terms of size |mean| * a, which costs accuracy when |mean| >> stddev.
template <typename Tr>
__device__ inline typename Tr::Compute
applyAffine(typename Tr::Compute d, typename Tr::Compute a,
            typename Tr::Compute b, int activation) {
  using A = typename Tr::Compute;
  A y = d * a + b;
  if (activation)
    y = y / (A(1) + Tr::exp_fn(-y));
  return y;
}

// Chan et al. merge of (count, mean, M2) running statistics.
template <typename A>
__device__ inline void chanMerge(int64_t &n, A &mean, A &m2, int64_t nb,
                                 A meanb, A m2b) {
  if (nb == 0)
    return;
  if (n == 0) {
    n = nb;
    mean = meanb;
    m2 = m2b;
    return;
  }
  int64_t total = n + nb;
  A delta = meanb - mean;
  A fb = A(nb) / A(total);
  mean += delta * fb;
  m2 += m2b + delta * delta * A(n) * fb;
  n = total;
}

// Per-thread Welford statistics, one update per loaded vector: the vector's
// own mean and M2 are exact (two passes over registers), then merged. M2 is
// a sum of squared deviations throughout, so no step subtracts two large
// sums and the variance stays accurate for any |mean| / stddev, even when
// the shift is an outlier. Values are taken relative to the thread's first
// value so the running mean stays small and its per-update rounding does not
// accumulate at the scale of |mean|.
template <typename A> struct WelfordAcc {
  int64_t count = 0;
  A shift = A(0);
  A mean = A(0); // relative to shift
  A m2 = A(0);

  template <int V> __device__ inline void add(const A (&v)[V]) {
    if (count == 0)
      shift = v[0];
    A d[V];
    A s = A(0);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      d[k] = v[k] - shift;
      s += d[k];
    }
    const A vm = s * (A(1) / A(V));
    A q = A(0);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const A e = d[k] - vm;
      q += e * e;
    }
    chanMerge(count, mean, m2, int64_t(V), vm, q);
  }
  __device__ inline A absoluteMean() const { return shift + mean; }
};

// Deterministic tree merge of per-thread (count, mean, M2) in shared memory.
// The result is in slot 0. blockDim.x must be a power of two.
template <typename A>
__device__ inline void blockChanReduce(int64_t *sCount, A *sMean, A *sM2) {
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int64_t n = sCount[threadIdx.x];
      A mean = sMean[threadIdx.x];
      A m2 = sM2[threadIdx.x];
      int o = threadIdx.x + stride;
      chanMerge(n, mean, m2, sCount[o], sMean[o], sM2[o]);
      sCount[threadIdx.x] = n;
      sMean[threadIdx.x] = mean;
      sM2[threadIdx.x] = m2;
    }
    __syncthreads();
  }
}

template <typename A> struct Partials {
  A *mean;
  A *m2;
  int64_t *count;
  A *stats; // [N * groups][2] = {mean, inv_std}
};

// Channels-last tiling. A block is tx x ty threads: tx threads cover V
// channels each, ty threads stride over the block's pixel range. A channel
// block holds groupsPerBlock whole groups, or, when one group is wider than a
// block (subBlocks > 1), one slice of a group. Grid x = n * channelBlocks + cb,
// y = pixel block. Partials per group = subBlocks * gridDim.y.
struct NhwcTile {
  int channelsPerGroup;
  int tx;
  int groupsPerBlock;
  int subBlocks;
  int channelBlocks;
  int64_t pixelsPerBlock;
};

struct NhwcLane {
  int64_t n;
  int64_t firstGroup; // first group of the channel block
  int groupsHere;     // whole groups in the block (1 for a slice)
  int sub;            // slice of the group (0 unless subBlocks > 1)
  int lx, ly, ty;
  int64_t c0; // first channel of this thread
  bool active;
};

template <int V>
__device__ inline NhwcLane nhwcLane(const NhwcTile &t, int64_t groups) {
  NhwcLane l;
  l.lx = threadIdx.x % t.tx;
  l.ly = threadIdx.x / t.tx;
  l.ty = blockDim.x / t.tx;
  l.n = blockIdx.x / t.channelBlocks;
  const int cb = blockIdx.x - static_cast<int>(l.n) * t.channelBlocks;
  const int threadsPerGroup = t.channelsPerGroup / V;
  int lane; // vector index within the first group
  if (t.subBlocks > 1) {
    l.firstGroup = cb / t.subBlocks;
    l.sub = cb - static_cast<int>(l.firstGroup) * t.subBlocks;
    l.groupsHere = 1;
    lane = l.sub * t.tx + l.lx;
    l.active = lane < threadsPerGroup;
  } else {
    l.firstGroup = static_cast<int64_t>(cb) * t.groupsPerBlock;
    l.sub = 0;
    l.groupsHere = static_cast<int>(
        std::min<int64_t>(t.groupsPerBlock, groups - l.firstGroup));
    lane = l.lx;
    l.active = lane < l.groupsHere * threadsPerGroup;
  }
  l.c0 = l.firstGroup * t.channelsPerGroup + int64_t(lane) * V;
  return l;
}

template <typename T, int V>
__global__ void __launch_bounds__(kBlock)
    gn_stats_nhwc(const T *__restrict__ x, Partials<typename GroupNormTraits<T>::Compute> p,
                  int64_t channels, int64_t spatial, int64_t groups,
                  NhwcTile t) {
  using Tr = GroupNormTraits<T>;
  using A = typename Tr::Compute;
  __shared__ int64_t sCount[kBlock];
  __shared__ A sMean[kBlock];
  __shared__ A sM2[kBlock];

  const NhwcLane l = nhwcLane<V>(t, groups);
  const int pb = blockIdx.y;
  const int64_t p0 = int64_t(pb) * t.pixelsPerBlock;
  const int64_t p1 = std::min(spatial, p0 + t.pixelsPerBlock);

  WelfordAcc<A> acc;
  if (l.active) {
    const T *ptr = x + (l.n * spatial + p0 + l.ly) * channels + l.c0;
    const int64_t step = int64_t(l.ty) * channels;
    for (int64_t px = p0 + l.ly; px < p1; px += l.ty, ptr += step) {
      GnVec<T, V> v = *reinterpret_cast<const GnVec<T, V> *>(ptr);
      A f[V];
#pragma unroll
      for (int k = 0; k < V; ++k)
        f[k] = Tr::load(v.v[k]);
      acc.add(f);
    }
  }
  const int tid = threadIdx.x;
  sCount[tid] = acc.count;
  sMean[tid] = acc.absoluteMean();
  sM2[tid] = acc.m2;
  __syncthreads();

  if (tid < l.groupsHere) {
    const int span = t.subBlocks > 1 ? t.tx : t.channelsPerGroup / V;
    int64_t cnt = 0;
    A gm = A(0), gm2 = A(0);
    for (int yy = 0; yy < l.ty; ++yy)
      for (int xx = tid * span; xx < (tid + 1) * span; ++xx) {
        int e = yy * t.tx + xx;
        chanMerge(cnt, gm, gm2, sCount[e], sMean[e], sM2[e]);
      }
    int64_t slot =
        ((l.n * groups + l.firstGroup + tid) * t.subBlocks + l.sub) *
            gridDim.y +
        pb;
    p.mean[slot] = gm;
    p.m2[slot] = gm2;
    p.count[slot] = cnt;
  }
}

// Channels-first stats. A group is one contiguous range of
// channelsPerGroup * spatial elements. Grid x = n * groups + g, y = chunk.
template <typename T, int V>
__global__ void __launch_bounds__(kBlock)
    gn_stats_nchw(const T *__restrict__ x, Partials<typename GroupNormTraits<T>::Compute> p,
                  int64_t groupSize, int64_t chunk) {
  using Tr = GroupNormTraits<T>;
  using A = typename Tr::Compute;
  __shared__ int64_t sCount[kBlock];
  __shared__ A sMean[kBlock];
  __shared__ A sM2[kBlock];

  const int64_t ng = blockIdx.x;
  const int pb = blockIdx.y;
  const int64_t start = int64_t(pb) * chunk;
  const int64_t end = std::min(groupSize, start + chunk);
  const T *base = x + ng * groupSize;

  WelfordAcc<A> acc;
  for (int64_t i = start + int64_t(threadIdx.x) * V; i < end;
       i += int64_t(blockDim.x) * V) {
    GnVec<T, V> v = *reinterpret_cast<const GnVec<T, V> *>(base + i);
    A f[V];
#pragma unroll
    for (int k = 0; k < V; ++k)
      f[k] = Tr::load(v.v[k]);
    acc.add(f);
  }
  sCount[threadIdx.x] = acc.count;
  sMean[threadIdx.x] = acc.absoluteMean();
  sM2[threadIdx.x] = acc.m2;
  __syncthreads();
  blockChanReduce(sCount, sMean, sM2);

  if (threadIdx.x == 0) {
    int64_t slot = ng * gridDim.y + pb;
    p.mean[slot] = sMean[0];
    p.m2[slot] = sM2[0];
    p.count[slot] = sCount[0];
  }
}

// One block per (n, group): each thread merges a fixed strided subset of the
// partials, then a tree merge combines the threads.
template <typename A>
__global__ void __launch_bounds__(kBlock)
    gn_finalize(Partials<A> p, int partialsPerGroup, float epsilon) {
  __shared__ int64_t sCount[kBlock];
  __shared__ A sMean[kBlock];
  __shared__ A sM2[kBlock];

  const int64_t ng = blockIdx.x;
  int64_t cnt = 0;
  A mean = A(0), m2 = A(0);
  for (int i = threadIdx.x; i < partialsPerGroup; i += blockDim.x) {
    int64_t slot = ng * partialsPerGroup + i;
    chanMerge(cnt, mean, m2, p.count[slot], p.mean[slot], p.m2[slot]);
  }
  sCount[threadIdx.x] = cnt;
  sMean[threadIdx.x] = mean;
  sM2[threadIdx.x] = m2;
  __syncthreads();
  blockChanReduce(sCount, sMean, sM2);

  if (threadIdx.x == 0) {
    A var = sCount[0] > 0 ? sM2[0] / A(sCount[0]) : A(0);
    A e = A(epsilon) + var;
    p.stats[ng * 2] = sMean[0];
    p.stats[ng * 2 + 1] = A(1) / sqrt(e);
  }
}

// Same tiling as gn_stats_nhwc; each thread owns V fixed channels, so the
// affine terms are loaded once.
template <typename T, int V>
__global__ void __launch_bounds__(kBlock)
    gn_apply_nhwc(const T *__restrict__ x, const T *__restrict__ scale,
                  const T *__restrict__ bias, T *__restrict__ y,
                  const typename GroupNormTraits<T>::Compute *__restrict__ stats,
                  int64_t channels, int64_t spatial, int64_t groups,
                  NhwcTile t, int activation) {
  using Tr = GroupNormTraits<T>;
  using A = typename Tr::Compute;
  const NhwcLane l = nhwcLane<V>(t, groups);
  if (!l.active)
    return;

  const int64_t c0 = l.c0;
  const int64_t ng = l.n * groups + c0 / t.channelsPerGroup;
  const A mean = stats[ng * 2];
  const A invStd = stats[ng * 2 + 1];
  A a[V], b[V];
#pragma unroll
  for (int k = 0; k < V; ++k) {
    a[k] = Tr::load(scale[c0 + k]) * invStd;
    b[k] = Tr::load(bias[c0 + k]);
  }

  const int64_t p0 = int64_t(blockIdx.y) * t.pixelsPerBlock;
  const int64_t p1 = std::min(spatial, p0 + t.pixelsPerBlock);
  const int64_t offset = (l.n * spatial + p0 + l.ly) * channels + c0;
  const T *src = x + offset;
  T *dst = y + offset;
  const int64_t step = int64_t(l.ty) * channels;
  for (int64_t px = p0 + l.ly; px < p1; px += l.ty, src += step, dst += step) {
    GnVec<T, V> v = *reinterpret_cast<const GnVec<T, V> *>(src);
    GnVec<T, V> o;
#pragma unroll
    for (int k = 0; k < V; ++k)
      o.v[k] = Tr::store(
          applyAffine<Tr>(Tr::load(v.v[k]) - mean, a[k], b[k], activation));
    *reinterpret_cast<GnVec<T, V> *>(dst) = o;
  }
}

// Channels-first apply. Grid x = n * C + c (one contiguous row of spatial
// elements), y = chunk of that row.
template <typename T, int V>
__global__ void __launch_bounds__(kBlock)
    gn_apply_nchw(const T *__restrict__ x, const T *__restrict__ scale,
                  const T *__restrict__ bias, T *__restrict__ y,
                  const typename GroupNormTraits<T>::Compute *__restrict__ stats,
                  int64_t channels, int64_t spatial, int64_t groups,
                  int channelsPerGroup, int64_t chunk, int activation) {
  using Tr = GroupNormTraits<T>;
  using A = typename Tr::Compute;
  const int64_t row = blockIdx.x;
  const int64_t n = row / channels;
  const int64_t c = row - n * channels;
  const int64_t ng = n * groups + c / channelsPerGroup;
  const A mean = stats[ng * 2];
  const A a = Tr::load(scale[c]) * stats[ng * 2 + 1];
  const A b = Tr::load(bias[c]);

  const int64_t start = int64_t(blockIdx.y) * chunk;
  const int64_t end = std::min(spatial, start + chunk);
  const T *src = x + row * spatial;
  T *dst = y + row * spatial;
  for (int64_t i = start + int64_t(threadIdx.x) * V; i < end;
       i += int64_t(blockDim.x) * V) {
    GnVec<T, V> v = *reinterpret_cast<const GnVec<T, V> *>(src + i);
    GnVec<T, V> o;
#pragma unroll
    for (int k = 0; k < V; ++k)
      o.v[k] = Tr::store(
          applyAffine<Tr>(Tr::load(v.v[k]) - mean, a, b, activation));
    *reinterpret_cast<GnVec<T, V> *>(dst + i) = o;
  }
}

template <typename A> __device__ inline A blockSum(A *sBuf, A v) {
  sBuf[threadIdx.x] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      sBuf[threadIdx.x] += sBuf[threadIdx.x + stride];
    __syncthreads();
  }
  A r = sBuf[0];
  __syncthreads();
  return r;
}

// Fused path: one block per (n, group), exact two-pass variance, then apply.
// Channels-last elements are visited pixel-major so that a group's channels
// at one pixel are read together.
template <typename T>
__global__ void __launch_bounds__(kBlock)
    gn_fused(const T *__restrict__ x, const T *__restrict__ scale,
             const T *__restrict__ bias, T *__restrict__ y, int64_t channels,
             int64_t spatial, int64_t groups, int channelsLast, int activation,
             float epsilon) {
  using Tr = GroupNormTraits<T>;
  using A = typename Tr::Compute;
  __shared__ A sBuf[kBlock];

  const int64_t ng = blockIdx.x;
  const int64_t n = ng / groups;
  const int64_t g = ng - n * groups;
  const int64_t cpg = channels / groups;
  const int64_t count = cpg * spatial;
  const int64_t c0 = g * cpg;

  auto offsetOf = [&](int64_t i, int64_t &c) {
    if (channelsLast) {
      int64_t px = i / cpg;
      int64_t lc = i - px * cpg;
      c = c0 + lc;
      return (n * spatial + px) * channels + c;
    }
    int64_t lc = i / spatial;
    c = c0 + lc;
    return ng * count + i;
  };

  // Summing relative to the group's first value keeps the rounding of the
  // mean at the scale of the spread rather than of |mean|.
  int64_t firstC;
  const A shift = Tr::load(x[offsetOf(0, firstC)]);
  A local = A(0);
  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    int64_t c;
    local += Tr::load(x[offsetOf(i, c)]) - shift;
  }
  const A mean = shift + blockSum(sBuf, local) / A(count);

  local = A(0);
  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    int64_t c;
    A d = Tr::load(x[offsetOf(i, c)]) - mean;
    local += d * d;
  }
  const A var = blockSum(sBuf, local) / A(count);
  const A invStd = A(1) / sqrt(var + A(epsilon));

  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    int64_t c;
    int64_t off = offsetOf(i, c);
    A a = Tr::load(scale[c]) * invStd;
    y[off] = Tr::store(applyAffine<Tr>(Tr::load(x[off]) - mean, a,
                                       Tr::load(bias[c]), activation));
  }
}

// The number of split blocks that keeps the current device busy.
hipError_t splitTargetBlocks(int64_t &blocks) {
  int dev = 0;
  hipError_t err = hipGetDevice(&dev);
  if (err != hipSuccess)
    return err;
  int units = 0, threadsPerUnit = 0;
  err = hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount,
                              dev);
  if (err != hipSuccess)
    return err;
  err = hipDeviceGetAttribute(&threadsPerUnit,
                              hipDeviceAttributeMaxThreadsPerMultiProcessor,
                              dev);
  if (err != hipSuccess)
    return err;
  if (units <= 0 || threadsPerUnit < kBlock)
    return hipErrorInvalidDevice;
  blocks = int64_t(units) * (threadsPerUnit / kBlock) * kResidentRounds;
  return hipSuccess;
}

// Read once: the env lookup would otherwise run on every launch.
int64_t minSplitCount() {
  static const int64_t value = [] {
    char buf[32] = {};
#ifdef _WIN32
    unsigned long len = GetEnvironmentVariableA(
        "HIPDNN_EP_GROUPNORM_SPLIT_SIZE", buf, sizeof(buf));
    if (len == 0 || len >= sizeof(buf))
      return kMinSplitCount;
#else
    const char *v = getenv("HIPDNN_EP_GROUPNORM_SPLIT_SIZE");
    if (!v)
      return kMinSplitCount;
    snprintf(buf, sizeof(buf), "%s", v);
#endif
    char *end = nullptr;
    long long parsed = strtoll(buf, &end, 10);
    return end != buf && *end == '\0' && parsed > 0 ? int64_t(parsed)
                                                    : kMinSplitCount;
  }();
  return value;
}

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

size_t alignUp(size_t v) {
  return (v + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

int elemSizeOf(int hipDtype) {
  switch (hipDtype) {
  case HIP_DTYPE_FLOAT16:
  case HIP_DTYPE_BFLOAT16:
    return 2;
  case HIP_DTYPE_FLOAT32:
    return 4;
  case HIP_DTYPE_FLOAT64:
    return 8;
  default:
    return 0;
  }
}

int accSizeOf(int hipDtype) { return hipDtype == HIP_DTYPE_FLOAT64 ? 8 : 4; }

struct Plan {
  bool split = false;
  int vec = 1;
  int partials = 1; // pixel blocks (channels_last) or chunks per group
  // channels_last tiling
  NhwcTile tile{};
  // channels_first tiling
  int64_t statsChunk = 0;
  int applyChunks = 0;
  int64_t applyChunk = 0;

  int slotsPerGroup() const {
    return partials * (tile.subBlocks > 0 ? tile.subBlocks : 1);
  }
};

size_t workspaceBytesFor(const Plan &plan, int64_t ng, int accSize) {
  if (!plan.split)
    return 0;
  size_t slots = static_cast<size_t>(ng) * plan.slotsPerGroup();
  return 2 * alignUp(slots * accSize) + alignUp(slots * sizeof(int64_t)) +
         alignUp(static_cast<size_t>(ng) * 2 * accSize);
}

// Lays out the split partitions for `partials` blocks per group (an upper
// bound; the returned plan may use fewer).
void layoutPartials(Plan &plan, int64_t partials, int64_t spatial,
                    int64_t groupSize, int channelsLast) {
  if (channelsLast) {
    plan.tile.pixelsPerBlock = ceilDiv(spatial, partials);
    plan.partials =
        static_cast<int>(ceilDiv(spatial, plan.tile.pixelsPerBlock));
  } else {
    int64_t chunk = ceilDiv(groupSize, partials);
    chunk = ceilDiv(chunk, plan.vec) * plan.vec;
    plan.statsChunk = chunk;
    plan.partials = static_cast<int>(ceilDiv(groupSize, chunk));
  }
}

// `aligned(v)` tells whether all tensors allow v-wide vector access. Fails
// only when the device cannot be queried.
template <typename AlignFn>
hipError_t makePlan(int64_t n, int64_t c, int64_t spatial, int64_t groups,
                    int channelsLast, int elemSize, AlignFn aligned,
                    Plan &plan) {
  plan = Plan{};
  const int64_t cpg = c / groups;
  const int64_t groupSize = cpg * spatial;
  int vec = 16 / elemSize;
  const int64_t divisor = channelsLast ? cpg : spatial;
  while (vec > 1 && (divisor % vec != 0 || !aligned(vec)))
    vec /= 2;
  plan.vec = vec;

  if (groupSize < minSplitCount() || cpg > INT32_MAX)
    return hipSuccess;

  int64_t targetBlocks = 0;
  if (hipError_t err = splitTargetBlocks(targetBlocks); err != hipSuccess)
    return err;
  int64_t partials;
  int64_t maxPartials = kMaxPartials;
  if (channelsLast) {
    const int64_t threadsPerGroup = cpg / vec;
    NhwcTile &t = plan.tile;
    t.channelsPerGroup = static_cast<int>(cpg);
    if (threadsPerGroup > kBlock) {
      t.subBlocks = static_cast<int>(ceilDiv(threadsPerGroup, kBlock));
      t.tx = static_cast<int>(ceilDiv(threadsPerGroup, t.subBlocks));
      t.groupsPerBlock = 1;
      t.channelBlocks = static_cast<int>(groups * t.subBlocks);
      maxPartials = std::max<int64_t>(1, kMaxPartials / t.subBlocks);
    } else {
      t.subBlocks = 1;
      t.groupsPerBlock = static_cast<int>(std::min<int64_t>(
          groups, std::max<int64_t>(1, kBlock / threadsPerGroup)));
      t.tx = static_cast<int>(t.groupsPerBlock * threadsPerGroup);
      t.channelBlocks = static_cast<int>(ceilDiv(groups, t.groupsPerBlock));
    }
    const int64_t ty = std::max(1, kBlock / t.tx);
    const int64_t base = n * t.channelBlocks;
    partials = ceilDiv(targetBlocks, base);
    partials = std::min(partials, ceilDiv(spatial, ty * kMinVectorsPerThread));
  } else {
    const int64_t base = n * groups;
    partials = ceilDiv(targetBlocks, base);
    partials = std::min(
        partials, ceilDiv(groupSize, int64_t(kBlock) * vec * kMinVectorsPerThread));
    int64_t rowChunks = ceilDiv(targetBlocks, n * c);
    rowChunks = std::min(
        rowChunks, ceilDiv(spatial, int64_t(kBlock) * vec * kMinVectorsPerThread));
    rowChunks = std::max<int64_t>(1, rowChunks);
    plan.applyChunk = ceilDiv(ceilDiv(spatial, rowChunks), vec) * vec;
    plan.applyChunks = static_cast<int>(ceilDiv(spatial, plan.applyChunk));
  }
  partials = std::max<int64_t>(1, std::min(partials, maxPartials));
  layoutPartials(plan, partials, spatial, groupSize, channelsLast);
  plan.split = true;
  return hipSuccess;
}

// Shrinks the split so that its partials fit in `bytes`, or falls back to the
// fused kernel.
void fitWorkspace(Plan &plan, int64_t ng, int accSize, size_t bytes,
                  int64_t spatial, int64_t groupSize, int channelsLast) {
  if (!plan.split || workspaceBytesFor(plan, ng, accSize) <= bytes)
    return;
  size_t fixed = alignUp(static_cast<size_t>(ng) * 2 * accSize) +
                 3 * kWorkspaceAlign;
  size_t perPartial = static_cast<size_t>(ng) *
                      (2 * accSize + sizeof(int64_t)) *
                      (plan.slotsPerGroup() / plan.partials);
  int64_t cap =
      bytes > fixed ? static_cast<int64_t>((bytes - fixed) / perPartial) : 0;
  if (cap < 1) {
    plan.split = false;
    return;
  }
  layoutPartials(plan, std::min<int64_t>(cap, plan.partials), spatial,
                 groupSize, channelsLast);
  if (workspaceBytesFor(plan, ng, accSize) > bytes)
    plan.split = false;
}

template <typename A>
Partials<A> carvePartials(void *workspace, int64_t ng, int partials) {
  size_t slots = static_cast<size_t>(ng) * partials;
  char *p = static_cast<char *>(workspace);
  Partials<A> out;
  out.mean = reinterpret_cast<A *>(p);
  p += alignUp(slots * sizeof(A));
  out.m2 = reinterpret_cast<A *>(p);
  p += alignUp(slots * sizeof(A));
  out.count = reinterpret_cast<int64_t *>(p);
  p += alignUp(slots * sizeof(int64_t));
  out.stats = reinterpret_cast<A *>(p);
  return out;
}

template <typename T, int V>
void launchSplit(hipStream_t stream, const T *x, const T *scale, const T *bias,
                 T *y, int64_t n, int64_t c, int64_t spatial, int64_t groups,
                 int channelsLast, int activation, float epsilon,
                 const Plan &plan, void *workspace) {
  using A = typename GroupNormTraits<T>::Compute;
  const int64_t ng = n * groups;
  const int cpg = static_cast<int>(c / groups);
  const int slots = plan.slotsPerGroup();
  Partials<A> parts = carvePartials<A>(workspace, ng, slots);
  const dim3 block(kBlock);
  if (channelsLast) {
    const NhwcTile &t = plan.tile;
    const int ty = std::max(1, kBlock / t.tx);
    const dim3 tile(t.tx * ty);
    const dim3 grid(static_cast<unsigned>(n * t.channelBlocks),
                    static_cast<unsigned>(plan.partials));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_stats_nhwc<T, V>), grid, tile, 0,
                       stream, x, parts, c, spatial, groups, t);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_finalize<A>),
                       dim3(static_cast<unsigned>(ng)), block, 0, stream,
                       parts, slots, epsilon);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_apply_nhwc<T, V>), grid, tile, 0,
                       stream, x, scale, bias, y, parts.stats, c, spatial,
                       groups, t, activation);
  } else {
    const dim3 statsGrid(static_cast<unsigned>(ng),
                         static_cast<unsigned>(plan.partials));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_stats_nchw<T, V>), statsGrid, block,
                       0, stream, x, parts, int64_t(cpg) * spatial,
                       plan.statsChunk);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_finalize<A>),
                       dim3(static_cast<unsigned>(ng)), block, 0, stream,
                       parts, slots, epsilon);
    const dim3 applyGrid(static_cast<unsigned>(n * c),
                         static_cast<unsigned>(plan.applyChunks));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_apply_nchw<T, V>), applyGrid, block,
                       0, stream, x, scale, bias, y, parts.stats, c, spatial,
                       groups, cpg, plan.applyChunk, activation);
  }
}

template <typename T>
void launchGroupNorm(hipStream_t stream, const void *input, const void *scale,
                     const void *bias, void *output, int64_t n, int64_t c,
                     int64_t spatial, int64_t groups, int channelsLast,
                     int activation, float epsilon, const Plan &plan,
                     void *workspace) {
  const T *x = static_cast<const T *>(input);
  const T *s = static_cast<const T *>(scale);
  const T *b = static_cast<const T *>(bias);
  T *y = static_cast<T *>(output);
  if (!plan.split) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gn_fused<T>),
                       dim3(static_cast<unsigned>(n * groups)), dim3(kBlock),
                       0, stream, x, s, b, y, c, spatial, groups, channelsLast,
                       activation, epsilon);
    return;
  }
  constexpr int kMaxVec = 16 / sizeof(T);
  switch (plan.vec) {
  case 8:
    if constexpr (kMaxVec >= 8) {
      launchSplit<T, 8>(stream, x, s, b, y, n, c, spatial, groups,
                        channelsLast, activation, epsilon, plan, workspace);
      return;
    }
    break;
  case 4:
    if constexpr (kMaxVec >= 4) {
      launchSplit<T, 4>(stream, x, s, b, y, n, c, spatial, groups,
                        channelsLast, activation, epsilon, plan, workspace);
      return;
    }
    break;
  case 2:
    launchSplit<T, 2>(stream, x, s, b, y, n, c, spatial, groups, channelsLast,
                      activation, epsilon, plan, workspace);
    return;
  default:
    break;
  }
  launchSplit<T, 1>(stream, x, s, b, y, n, c, spatial, groups, channelsLast,
                    activation, epsilon, plan, workspace);
}

bool validArgs(int64_t n, int64_t c, int64_t spatial, int64_t groups,
               int channelsLast, int activation) {
  if (groups <= 0 || c % groups != 0 ||
      (channelsLast != 0 && channelsLast != 1) ||
      (activation != 0 && activation != 1))
    return false;
  return n <= static_cast<int64_t>(UINT32_MAX) / groups &&
         n <= static_cast<int64_t>(UINT32_MAX) / c;
}

} // namespace

extern "C" size_t hip_group_norm_workspace_bytes(int64_t n, int64_t c,
                                                 int64_t spatial,
                                                 int64_t groups,
                                                 int channels_last,
                                                 int hip_dtype) {
  if (n <= 0 || c <= 0 || spatial <= 0 ||
      !validArgs(n, c, spatial, groups, channels_last, 0))
    return 0;
  int elemSize = elemSizeOf(hip_dtype);
  if (elemSize == 0)
    return 0;
  // A failed device query leaves the plan unsplit; hip_group_norm then
  // reports the same failure.
  Plan plan;
  if (makePlan(n, c, spatial, groups, channels_last, elemSize,
               [](int) { return true; }, plan) != hipSuccess)
    return 0;
  return workspaceBytesFor(plan, n * groups, accSizeOf(hip_dtype));
}

extern "C" int hip_group_norm(void *stream, const void *input,
                              const void *scale, const void *bias, void *output,
                              int64_t n, int64_t c, int64_t spatial,
                              int64_t groups, int channels_last, int activation,
                              float epsilon, void *workspace,
                              size_t workspace_bytes, int hip_dtype) {
  if (n <= 0 || c <= 0 || spatial <= 0)
    return 0;
  if (!validArgs(n, c, spatial, groups, channels_last, activation))
    return -1;
  const int elemSize = elemSizeOf(hip_dtype);
  if (elemSize == 0) {
    fprintf(stderr, "[custom_kernels] hip_group_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  auto aligned = [&](int vec) {
    uintptr_t mask = static_cast<uintptr_t>(vec) * elemSize - 1;
    return ((reinterpret_cast<uintptr_t>(input) |
             reinterpret_cast<uintptr_t>(output)) &
            mask) == 0;
  };
  Plan plan;
  if (hipError_t err = makePlan(n, c, spatial, groups, channels_last,
                                elemSize, aligned, plan);
      err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_group_norm: device query failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  if (!workspace)
    workspace_bytes = 0;
  fitWorkspace(plan, n * groups, accSizeOf(hip_dtype), workspace_bytes,
               spatial, (c / groups) * spatial, channels_last);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_group_norm: dtype=%d n=%lld c=%lld spatial=%lld "
      "groups=%lld channels_last=%d activation=%d eps=%f split=%d vec=%d "
      "partials=%d\n",
      hip_dtype, (long long)n, (long long)c, (long long)spatial,
      (long long)groups, channels_last, activation, epsilon, plan.split ? 1 : 0,
      plan.vec, plan.partials);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    launchGroupNorm<__half>(hip_stream, input, scale, bias, output, n, c,
                            spatial, groups, channels_last, activation, epsilon,
                            plan, workspace);
    break;
  case HIP_DTYPE_BFLOAT16:
    launchGroupNorm<__hip_bfloat16>(hip_stream, input, scale, bias, output, n,
                                    c, spatial, groups, channels_last,
                                    activation, epsilon, plan, workspace);
    break;
  case HIP_DTYPE_FLOAT32:
    launchGroupNorm<float>(hip_stream, input, scale, bias, output, n, c,
                           spatial, groups, channels_last, activation, epsilon,
                           plan, workspace);
    break;
  default:
    launchGroupNorm<double>(hip_stream, input, scale, bias, output, n, c,
                            spatial, groups, channels_last, activation, epsilon,
                            plan, workspace);
    break;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_group_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
