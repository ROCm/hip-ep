/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * RoPE (Rotary Positional Embedding) kernel for FP16.
 * Optimized for RDNA 3.5/4 (gfx1150/gfx1151, wave32).
 *
 * Two execution paths, auto-selected per shape via runtime autotune:
 *
 *  Float4 fast path  (128-bit vectorized, flat grid)
 *    - No LDS / no __syncthreads: cos/sin served via L0/L1 cache
 *    - __hfma2 fused-multiply-add reduces ALU count
 *    - Flat grid: uniform thread utilization for any num_heads
 *    - Bit-shift index decomposition: zero integer division
 *
 *  Half2 fallback  (block-per-position, LDS-cached cos/sin)
 *    - Handles arbitrary dimension combinations
 *
 * Autotune: on the first call for each (npos, H, D, RD, interleaved)
 * shape, all applicable configs are benchmarked and the fastest is
 * cached for subsequent calls.
 *
 * API: hip_rope_forward()
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#include "hip_custom_kernels.h"

/* ================================================================
 * HALF-ROTATED  —  float4 fast path (templated on block size)
 * ================================================================ */
template <int BLOCK>
__launch_bounds__(BLOCK)
__global__ void rope_hr_f4(
    const float4* __restrict__ in,
    const int64_t* __restrict__ pid,
    const float4* __restrict__ cc,
    const float4* __restrict__ sc,
    float4* __restrict__       out,
    const int f4_stride,
    const int f4_head,
    const int f4_half,
    const int fh_mask,
    const int fh_shift,
    const int nh_mask,
    const int nh_shift,
    const int max_pos,
    const int total)
{
    const int gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= total) return;

    const int chunk = gid & fh_mask;
    const int tmp   = gid >> fh_shift;
    const int h     = tmp  & nh_mask;
    const int bs    = tmp  >> nh_shift;

    const int64_t pr = pid[bs];
    const int pos = static_cast<int>(pr < 0 ? 0 :
                        (pr >= max_pos ? max_pos - 1 : pr));

    const int ho = bs * f4_stride + h * f4_head;
    const int co = pos * f4_half + chunk;

    float4 x0 = in[ho + chunk];
    float4 x1 = in[ho + f4_half + chunk];
    float4 cv = cc[co];
    float4 sv = sc[co];

    __half2 *va = reinterpret_cast<__half2*>(&x0);
    __half2 *vb = reinterpret_cast<__half2*>(&x1);
    __half2 *vc = reinterpret_cast<__half2*>(&cv);
    __half2 *vs = reinterpret_cast<__half2*>(&sv);

    float4 r0, r1;
    __half2 *d0 = reinterpret_cast<__half2*>(&r0);
    __half2 *d1 = reinterpret_cast<__half2*>(&r1);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        d0[i] = __hfma2(va[i], vc[i], __hneg2(__hmul2(vb[i], vs[i])));
        d1[i] = __hfma2(vb[i], vc[i],          __hmul2(va[i], vs[i]));
    }

    out[ho + chunk]            = r0;
    out[ho + f4_half + chunk]  = r1;
}

/* ================================================================
 * INTERLEAVED  —  float4 fast path (templated on block size)
 * ================================================================ */
template <int BLOCK>
__launch_bounds__(BLOCK)
__global__ void rope_il_f4(
    const float4* __restrict__ in,
    const int64_t* __restrict__ pid,
    const float4* __restrict__ cc,
    const float4* __restrict__ sc,
    float4* __restrict__       out,
    const int f4_stride,
    const int f4_head,
    const int cph,
    const int cm,
    const int cs,
    const int nm,
    const int ns,
    const int max_pos,
    const int total)
{
    const int gid = blockIdx.x * BLOCK + threadIdx.x;
    if (gid >= total) return;

    const int ch = gid & cm;
    const int tmp = gid >> cs;
    const int h   = tmp  & nm;
    const int bs  = tmp  >> ns;

    const int64_t pr = pid[bs];
    const int pos = static_cast<int>(pr < 0 ? 0 :
                        (pr >= max_pos ? max_pos - 1 : pr));

    const int ho = bs * f4_stride + h * f4_head;
    const int co = pos * cph + ch;

    float4 da = in[ho + ch * 2];
    float4 db = in[ho + ch * 2 + 1];
    float4 cv = cc[co];
    float4 sv = sc[co];

    __half *ia = reinterpret_cast<__half*>(&da);
    __half *ib = reinterpret_cast<__half*>(&db);
    __half *cH = reinterpret_cast<__half*>(&cv);
    __half *sH = reinterpret_cast<__half*>(&sv);

    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        __half x0 = ia[2*p], x1 = ia[2*p+1];
        __half c_ = cH[p],   s_ = sH[p];
        ia[2*p]   = __hsub(__hmul(x0, c_), __hmul(x1, s_));
        ia[2*p+1] = __hadd(__hmul(x0, s_), __hmul(x1, c_));
    }
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        __half x0 = ib[2*p], x1 = ib[2*p+1];
        __half c_ = cH[4+p], s_ = sH[4+p];
        ib[2*p]   = __hsub(__hmul(x0, c_), __hmul(x1, s_));
        ib[2*p+1] = __hadd(__hmul(x0, s_), __hmul(x1, c_));
    }

    out[ho + ch * 2]     = da;
    out[ho + ch * 2 + 1] = db;
}

/* ================================================================
 * PASSTHROUGH COPY  —  float4
 * ================================================================ */
__launch_bounds__(256)
__global__ void rope_pass_f4(
    const float4* __restrict__ in,
    float4* __restrict__       out,
    const int f4_stride,
    const int f4_head,
    const int f4_rot,
    const int f4_pass,
    const int num_heads,
    const int total)
{
    const int gid = blockIdx.x * 256 + threadIdx.x;
    if (gid >= total) return;

    int t   = gid % f4_pass;
    int tmp = gid / f4_pass;
    int h   = tmp % num_heads;
    int bs  = tmp / num_heads;

    int off = bs * f4_stride + h * f4_head + f4_rot + t;
    out[off] = in[off];
}

/* ================================================================
 * HALF-ROTATED  —  fallback (templated on block size)
 * ================================================================ */
template <int BLOCK>
__launch_bounds__(BLOCK)
__global__ void rope_hr_fb(
    const __half* __restrict__ in,
    const int64_t* __restrict__ pid,
    const __half* __restrict__ cc,
    const __half* __restrict__ sc,
    __half* __restrict__       out,
    const int nh,
    const int hd,
    const int hr,
    const int max_pos)
{
    const int bs = blockIdx.x;
    const int64_t pr = pid[bs];
    const int pos = static_cast<int>(pr < 0 ? 0 :
                        (pr >= max_pos ? max_pos - 1 : pr));

    extern __shared__ __half sm[];
    __half *lc = sm, *ls = sm + hr;
    const __half *gc = cc + pos * hr, *gs = sc + pos * hr;
    for (int i = threadIdx.x; i < hr; i += BLOCK)
    { lc[i] = gc[i]; ls[i] = gs[i]; }
    __syncthreads();

    const int base = bs * nh * hd;
    const int h2n  = hr >> 1;
    const int tot  = nh * h2n;

    for (int i = threadIdx.x; i < tot; i += BLOCK) {
        int h = i / h2n;
        int d = (i - h * h2n) << 1;
        int off = base + h * hd;

        __half2 x0 = *reinterpret_cast<const __half2*>(&in[off + d]);
        __half2 x1 = *reinterpret_cast<const __half2*>(&in[off + d + hr]);
        __half2 c2 = *reinterpret_cast<const __half2*>(&lc[d]);
        __half2 s2 = *reinterpret_cast<const __half2*>(&ls[d]);

        *reinterpret_cast<__half2*>(&out[off + d]) =
            __hfma2(x0, c2, __hneg2(__hmul2(x1, s2)));
        *reinterpret_cast<__half2*>(&out[off + d + hr]) =
            __hfma2(x1, c2, __hmul2(x0, s2));
    }

    if ((hr & 1) && threadIdx.x < nh) {
        int d = hr - 1, off = base + threadIdx.x * hd;
        __half x0 = in[off + d], x1 = in[off + d + hr];
        __half cv = lc[d], sv = ls[d];
        out[off + d]      = __hsub(__hmul(x0, cv), __hmul(x1, sv));
        out[off + d + hr] = __hadd(__hmul(x0, sv), __hmul(x1, cv));
    }

    const int rfull = hr << 1;
    const int pass  = hd - rfull;
    if (pass > 0) {
        const int pt = nh * pass;
        for (int i = threadIdx.x; i < pt; i += BLOCK) {
            int h = i / pass;
            int d = rfull + i - h * pass;
            int off = base + h * hd + d;
            out[off] = in[off];
        }
    }
}

/* ================================================================
 * INTERLEAVED  —  fallback (templated on block size)
 * ================================================================ */
template <int BLOCK>
__launch_bounds__(BLOCK)
__global__ void rope_il_fb(
    const __half* __restrict__ in,
    const int64_t* __restrict__ pid,
    const __half* __restrict__ cc,
    const __half* __restrict__ sc,
    __half* __restrict__       out,
    const int nh,
    const int hd,
    const int hr,
    const int max_pos)
{
    const int bs = blockIdx.x;
    const int64_t pr = pid[bs];
    const int pos = static_cast<int>(pr < 0 ? 0 :
                        (pr >= max_pos ? max_pos - 1 : pr));

    extern __shared__ __half sm[];
    __half *lc = sm, *ls = sm + hr;
    const __half *gc = cc + pos * hr, *gs = sc + pos * hr;
    for (int i = threadIdx.x; i < hr; i += BLOCK)
    { lc[i] = gc[i]; ls[i] = gs[i]; }
    __syncthreads();

    const int base = bs * nh * hd;
    const int tot  = nh * hr;

    for (int i = threadIdx.x; i < tot; i += BLOCK) {
        int h = i / hr;
        int d = i - h * hr;
        int off = base + h * hd;

        __half2 xy = *reinterpret_cast<const __half2*>(&in[off + d * 2]);
        __half x0 = __low2half(xy), x1 = __high2half(xy);
        __half cv = lc[d], sv = ls[d];

        __half o0 = __hsub(__hmul(x0, cv), __hmul(x1, sv));
        __half o1 = __hadd(__hmul(x0, sv), __hmul(x1, cv));
        *reinterpret_cast<__half2*>(&out[off + d * 2]) =
            __halves2half2(o0, o1);
    }

    const int rfull = hr << 1;
    const int pass  = hd - rfull;
    if (pass > 0) {
        const int pt = nh * pass;
        for (int i = threadIdx.x; i < pt; i += BLOCK) {
            int h = i / pass;
            int d = rfull + i - h * pass;
            int off = base + h * hd + d;
            out[off] = in[off];
        }
    }
}

/* ================================================================
 * Autotune infrastructure
 * ================================================================ */

static int ilog2_(int x)
{
    int n = 0;
    while (x > 1) { x >>= 1; ++n; }
    return n;
}

static bool ipow2_(int x) { return x > 0 && (x & (x - 1)) == 0; }

struct RopeConfig {
    int  block_size;
    bool use_f4;
};

static constexpr RopeConfig kRopeConfigs[] = {
    { 64, true},   // 0
    {128, true},   // 1
    {256, true},   // 2
    {128, false},  // 3
    {256, false},  // 4
};
static constexpr int kNumRopeConfigs =
    static_cast<int>(sizeof(kRopeConfigs) / sizeof(kRopeConfigs[0]));

struct RopeLaunchParams {
    const __half*   ih;
    const int64_t*  ps;
    const __half*   ch;
    const __half*   sh;
    __half*         oh;
    int H, D, hr, M, N;
    bool interleaved;
    bool f4ok;
};

static void launchRopeConfig(
    int config_id, hipStream_t st,
    const RopeLaunchParams& p)
{
    auto& cfg = kRopeConfigs[config_id];
    const int BS = cfg.block_size;

    if (cfg.use_f4 && p.f4ok) {
        const auto* i4 = reinterpret_cast<const float4*>(p.ih);
        const auto* c4 = reinterpret_cast<const float4*>(p.ch);
        const auto* s4 = reinterpret_cast<const float4*>(p.sh);
        auto*       o4 = reinterpret_cast<float4*>(p.oh);
        const int fh = p.D >> 3;
        const int fs = p.H * fh;

        if (p.interleaved) {
            const int cp  = p.hr >> 3;
            const int tot = p.N * p.H * cp;
            dim3 g((tot + BS - 1) / BS);

#define ROPE_IL_F4_CASE(B) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rope_il_f4<B>), \
        g, dim3(B), 0, st, \
        i4, p.ps, c4, s4, o4, \
        fs, fh, cp, cp-1, ilog2_(cp), p.H-1, ilog2_(p.H), p.M, tot)

            switch (BS) {
                case  64: ROPE_IL_F4_CASE(64);  break;
                case 128: ROPE_IL_F4_CASE(128); break;
                default:  ROPE_IL_F4_CASE(256); break;
            }
#undef ROPE_IL_F4_CASE
        } else {
            const int fp  = p.hr >> 3;
            const int tot = p.N * p.H * fp;
            dim3 g((tot + BS - 1) / BS);

#define ROPE_HR_F4_CASE(B) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rope_hr_f4<B>), \
        g, dim3(B), 0, st, \
        i4, p.ps, c4, s4, o4, \
        fs, fh, fp, fp-1, ilog2_(fp), p.H-1, ilog2_(p.H), p.M, tot)

            switch (BS) {
                case  64: ROPE_HR_F4_CASE(64);  break;
                case 128: ROPE_HR_F4_CASE(128); break;
                default:  ROPE_HR_F4_CASE(256); break;
            }
#undef ROPE_HR_F4_CASE
        }

        if (p.D > p.hr * 2 && (p.D - p.hr * 2) % 8 == 0) {
            const int fr = (p.hr * 2) >> 3;
            const int fp = (p.D - p.hr * 2) >> 3;
            const int fh = p.D >> 3;
            const int fs = p.H * fh;
            const int pt = p.N * p.H * fp;
            hipLaunchKernelGGL(rope_pass_f4,
                dim3((pt + 255) >> 8), dim3(256), 0, st,
                i4, o4, fs, fh, fr, fp, p.H, pt);
        }
    } else {
        dim3 g(p.N);
        dim3 b(BS);
        int smem = 2 * p.hr * static_cast<int>(sizeof(__half));

        if (p.interleaved) {
#define ROPE_IL_FB_CASE(B) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rope_il_fb<B>), \
        g, b, smem, st, \
        p.ih, p.ps, p.ch, p.sh, p.oh, p.H, p.D, p.hr, p.M)

            switch (BS) {
                case 128: ROPE_IL_FB_CASE(128); break;
                default:  ROPE_IL_FB_CASE(256); break;
            }
#undef ROPE_IL_FB_CASE
        } else {
#define ROPE_HR_FB_CASE(B) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rope_hr_fb<B>), \
        g, b, smem, st, \
        p.ih, p.ps, p.ch, p.sh, p.oh, p.H, p.D, p.hr, p.M)

            switch (BS) {
                case 128: ROPE_HR_FB_CASE(128); break;
                default:  ROPE_HR_FB_CASE(256); break;
            }
#undef ROPE_HR_FB_CASE
        }
    }
}

static int tuneRopeConfig(
    hipStream_t st,
    const RopeLaunchParams& p)
{
    hipEvent_t ev0, ev1;
    (void)hipEventCreate(&ev0);
    (void)hipEventCreate(&ev1);

    float best_ms = 1e30f;
    int   best_id = 2;
    constexpr int WARMUP = 2;
    constexpr int ITERS  = 10;

    for (int cid = 0; cid < kNumRopeConfigs; cid++) {
        auto& cfg = kRopeConfigs[cid];
        if (cfg.use_f4 && !p.f4ok) continue;

        for (int w = 0; w < WARMUP; w++)
            launchRopeConfig(cid, st, p);

        (void)hipEventRecord(ev0, st);
        for (int i = 0; i < ITERS; i++)
            launchRopeConfig(cid, st, p);
        (void)hipEventRecord(ev1, st);
        (void)hipEventSynchronize(ev1);

        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, ev0, ev1);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);
    return best_id;
}

static std::unordered_map<std::string, int> rope_cache;
static std::mutex rope_tune_mutex;

static std::string ropeCacheKey(int N, int H, int D, int hr, int interleaved) {
    return std::to_string(N) + "_" + std::to_string(H) + "_" +
           std::to_string(D) + "_" + std::to_string(hr) + "_" +
           std::to_string(interleaved);
}

static int getOrTuneRopeConfig(
    hipStream_t st,
    const RopeLaunchParams& p)
{
    std::string key = ropeCacheKey(p.N, p.H, p.D, p.hr,
                                   p.interleaved ? 1 : 0);

    std::lock_guard<std::mutex> lock(rope_tune_mutex);

    auto it = rope_cache.find(key);
    if (it != rope_cache.end())
        return it->second;

    int best = tuneRopeConfig(st, p);
    rope_cache[key] = best;
    return best;
}

/* ================================================================
 * Host dispatcher
 * ================================================================ */

extern "C" int hip_rope_forward(
    void* stream_ptr,
    const void* input,
    const void* position_ids,
    const void* cos_cache,
    const void* sin_cache,
    void* output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_seq_len_i64,
    int64_t interleaved,
    int64_t element_size_bytes)
{
    if (!input || !position_ids || !cos_cache || !sin_cache || !output) {
        fprintf(stderr, "hip_rope_forward: null pointer\n");
        return -1;
    }
    if (element_size_bytes != 2) {
        fprintf(stderr, "hip_rope_forward: fp16 only (got %lld)\n",
                (long long)element_size_bytes);
        return -1;
    }

    hipStream_t st = static_cast<hipStream_t>(stream_ptr);
    const int H  = static_cast<int>(num_heads);
    const int D  = static_cast<int>(head_dim);
    const int R  = static_cast<int>(rotary_dim);
    const int hr = R / 2;
    const int M  = static_cast<int>(max_seq_len_i64);
    const int N  = static_cast<int>(batch_size) * static_cast<int>(seq_len);

    bool f4ok = (hr >= 8)
             && (hr % 8 == 0)
             && (D % 8 == 0)
             && ipow2_(H)
             && ipow2_(hr >> 3);

    if (f4ok && R < D && (D - R) % 8 != 0)
        f4ok = false;

    RopeLaunchParams p;
    p.ih = static_cast<const __half*>(input);
    p.ps = static_cast<const int64_t*>(position_ids);
    p.ch = static_cast<const __half*>(cos_cache);
    p.sh = static_cast<const __half*>(sin_cache);
    p.oh = static_cast<__half*>(output);
    p.H  = H;
    p.D  = D;
    p.hr = hr;
    p.M  = M;
    p.N  = N;
    p.interleaved = (interleaved != 0);
    p.f4ok = f4ok;

    int config_id = getOrTuneRopeConfig(st, p);
    launchRopeConfig(config_id, st, p);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "hip_rope_forward: %s\n", hipGetErrorString(err));
        return -1;
    }
    return 0;
}
