/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <type_traits>

#include "hip_custom_kernels.h"

//===----------------------------------------------------------------------===//
// hip_causal_conv_step_decode -- single-step fused causal depthwise 1D conv.
//
// Replaces the MIOpen "build virtual buffer with hipMemcpy2DAsync, run
// depthwise conv, run bias + SiLU as separate ops" chain for the seq_len == 1
// decode path. Single launch, one block per (batch, channel), each thread is
// responsible for one channel.
//
// At decode the math is laughably small:
//   per channel: read k-1 past-state elements + 1 input element + k weights,
//   compute k FMAs, optional bias, optional SiLU, write 1 output + k-1 state.
//
// On a Mamba/Gated DeltaNet block with C=8192, k=4 this is ~96 KB of VRAM
// reads and ~16 KB of VRAM writes. Runs in ~1-20 µs end-to-end.
//
// We unroll on kernel_size at the call site (template <int K>) so the inner
// loops collapse to a fixed small number of FMAs and the per-thread state
// register array is statically sized.
//===----------------------------------------------------------------------===//

namespace {

template <typename T> __device__ __forceinline__ float to_f32(T x);
template <> __device__ __forceinline__ float to_f32(float x) { return x; }
template <> __device__ __forceinline__ float to_f32(__half x) {
    return __half2float(x);
}

template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ float from_f32(float x) { return x; }
template <> __device__ __forceinline__ __half from_f32(float x) {
    return __float2half(x);
}

// Per-(b,c) thread:
//   1) Loads past_state[b,c,0..K-2] into registers (or zeros if past is null).
//   2) Loads input[b,c,0], weight[c,0,0..K-1], optional bias[c].
//   3) Computes the conv dot product in fp32.
//   4) Applies optional SiLU.
//   5) Writes output[b,c,0] and present_state[b,c,0..K-2].
//
// Memory layout (matches wrap_causal_conv_with_state):
//   input,  output         flat [b*C + c]                 (one element/plane)
//   past_state, present_state  flat [(b*C + c)*(K-1) + j]
//   weight                     flat [c*K + j]
//   bias                       flat [c]
template <typename T, int K, int HAS_BIAS, int ACT_SILU>
__global__ void causal_conv_step_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ past_state,
    T* __restrict__ output,
    T* __restrict__ present_state,
    int batch_size,
    int channels) {

    // One thread per (b, c) plane. Grid = ceil(B*C / blockDim.x).
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels;
    if (idx >= total)
        return;

    // (b, c) for this thread
    const int c = idx % channels;
    // const int b = idx / channels;  // not needed beyond `idx`

    // Load past_state[b,c,0..K-2] into a small register array. K is a template
    // constant so this unrolls fully and `state[]` lives in registers.
    constexpr int STATE_LEN = K - 1;
    float state[STATE_LEN > 0 ? STATE_LEN : 1];
    if (past_state != nullptr) {
        const T* ps = past_state + static_cast<int64_t>(idx) * STATE_LEN;
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            state[j] = to_f32<T>(ps[j]);
    } else {
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            state[j] = 0.0f;
    }

    // Load input[b,c,0] and weight[c,0,0..K-1] (weight is per-channel only).
    const float x = to_f32<T>(input[idx]);
    float w[K];
    {
        const T* wp = weight + static_cast<int64_t>(c) * K;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            w[j] = to_f32<T>(wp[j]);
    }

    // Conv dot product: virtual = [past_state[0..K-2], x]
    //   y = sum_{j=0..K-2} w[j] * state[j] + w[K-1] * x
    float y = w[K - 1] * x;
    #pragma unroll
    for (int j = 0; j < STATE_LEN; ++j)
        y += w[j] * state[j];

    if (HAS_BIAS)
        y += to_f32<T>(bias[c]);

    if (ACT_SILU) {
        // SiLU(y) = y * sigmoid(y) = y / (1 + exp(-y))
        const float s = 1.0f / (1.0f + expf(-y));
        y *= s;
    }

    output[idx] = from_f32<T>(y);

    // Shift state forward by one timestep:
    //   present_state[b,c,0..K-3] = past_state[b,c,1..K-2]
    //   present_state[b,c,K-2]    = x
    if (STATE_LEN > 0) {
        T* nps = present_state + static_cast<int64_t>(idx) * STATE_LEN;
        #pragma unroll
        for (int j = 0; j < STATE_LEN - 1; ++j)
            nps[j] = from_f32<T>(state[j + 1]);
        nps[STATE_LEN - 1] = from_f32<T>(x);
    }
}

// Dispatcher: stamps out the small cross-product of (T, K, HAS_BIAS,
// ACT_SILU) template instantiations we actually use at runtime. Only the K
// values currently observed in supported models (Mamba uses k=4) need fast
// paths; we provide k in {2,3,4,5,6,7,8} for future hybrids.
template <typename T>
hipError_t launch_for_K(
    int K, hipStream_t stream,
    const T* input, const T* weight, const T* bias, const T* past_state,
    T* output, T* present_state,
    int batch_size, int channels, int has_bias, int act_silu) {

    const int total = batch_size * channels;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    auto launch = [&](auto k_const, auto bias_const, auto act_const) {
        constexpr int Kc = decltype(k_const)::value;
        constexpr int Bc = decltype(bias_const)::value;
        constexpr int Ac = decltype(act_const)::value;
        hipLaunchKernelGGL(
            (causal_conv_step_kernel<T, Kc, Bc, Ac>),
            dim3(grid), dim3(block), 0, stream,
            input, weight, bias, past_state, output, present_state,
            batch_size, channels);
        return hipGetLastError();
    };

    // Compile-time switch on K so we get fully-unrolled inner loops.
#define DISPATCH_K_BIAS_ACT(KVAL)                                             \
    case KVAL: {                                                              \
        using K_t = std::integral_constant<int, KVAL>;                        \
        if (has_bias && act_silu)                                             \
            return launch(K_t{},                                              \
                          std::integral_constant<int, 1>{},                   \
                          std::integral_constant<int, 1>{});                  \
        if (has_bias)                                                         \
            return launch(K_t{},                                              \
                          std::integral_constant<int, 1>{},                   \
                          std::integral_constant<int, 0>{});                  \
        if (act_silu)                                                         \
            return launch(K_t{},                                              \
                          std::integral_constant<int, 0>{},                   \
                          std::integral_constant<int, 1>{});                  \
        return launch(K_t{},                                                  \
                      std::integral_constant<int, 0>{},                       \
                      std::integral_constant<int, 0>{});                      \
    }

    switch (K) {
        DISPATCH_K_BIAS_ACT(1)
        DISPATCH_K_BIAS_ACT(2)
        DISPATCH_K_BIAS_ACT(3)
        DISPATCH_K_BIAS_ACT(4)
        DISPATCH_K_BIAS_ACT(5)
        DISPATCH_K_BIAS_ACT(6)
        DISPATCH_K_BIAS_ACT(7)
        DISPATCH_K_BIAS_ACT(8)
        default:
            return hipErrorInvalidValue;
    }
#undef DISPATCH_K_BIAS_ACT
}

// Prefill (seq_len > 1) fused causal depthwise 1D conv + bias + SiLU.
// A thread slides a length-K window over its slice of the virtual sequence
// [past_state[0..K-2], input[0..L-1]], writing that slice's outputs; the thread
// holding the tail also writes the trailing K-1 present_state. fp32 accumulate
// -- numerically matches causal_conv_step_kernel at L=1. Replaces the MIOpen
// prefill path (Find + 3 pitched memcpys + conv + bias + activation + mul)
// with a single launch. Layout matches wrap_causal_conv_with_state:
//   input, output            flat [(b*C + c)*L + t]
//   past_state, present_state flat [(b*C + c)*(K-1) + j]
//   weight                    flat [c*K + j];   bias flat [c]
//
// One block owns one (b, c) plane and a TILE-long run of its outputs: lane i
// produces output t0 + i, and the block stages virtual[t0 .. t0+TILE+K-2] in LDS
// first so every global access is contiguous across the wave.
//
// Why it is shaped this way rather than one-thread-per-plane walking t. That
// form fixes the grid at B*C, so a 4k-token prefill launches exactly as many
// threads as a 40-token one -- 8192 threads, ~6 waves per CU, far too few to
// cover DRAM latency, which is why it sat at 105 GB/s. But simply splitting t
// across more blocks while keeping one thread per plane is *worse*, measured
// 2.6x worse: each lane's addresses are seq_len apart, so a wave touches 32
// different cache lines per load and only recovers the cost by consuming the
// rest of each line on later iterations of its own serial walk. Multiply the
// resident lane count and those lines no longer survive in cache between uses,
// and the same traffic is refetched per element. Coalescing across lanes is what
// actually removes the problem: consecutive lanes read consecutive t, so one
// line serves the whole wave immediately and no cache retention is required.
//
// Layout matches wrap_causal_conv_with_state:
//   input, output            flat [(b*C + c)*L + t]
//   past_state, present_state flat [(b*C + c)*(K-1) + j]
//   weight                    flat [c*K + j];   bias flat [c]
template <typename T, int K, int HAS_BIAS, int ACT_SILU, int TILE>
__global__ void causal_conv_prefill_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ past_state,
    T* __restrict__ output,
    T* __restrict__ present_state,
    int batch_size,
    int channels,
    int seq_len) {

    const int plane = static_cast<int>(blockIdx.x);
    const int total = batch_size * channels;
    if (plane >= total)
        return;

    const int t0 = static_cast<int>(blockIdx.y) * TILE;
    if (t0 >= seq_len)
        return;

    const int c = plane % channels;
    const int lane = static_cast<int>(threadIdx.x);
    constexpr int STATE_LEN = K - 1;
    constexpr int SMEM_LEN = TILE + K - 1;

    const T* in_row = input + static_cast<int64_t>(plane) * seq_len;
    T* out_row = output + static_cast<int64_t>(plane) * seq_len;

    // smem[j] = virtual[t0 + j], where the virtual sequence is
    //   virtual[i] = past_state[i]      for i < K-1
    //                input[i - (K-1)]   otherwise,
    // so output t reads virtual[t .. t+K-1] = smem[t-t0 .. t-t0+K-1]. Only the
    // first tile of the first block touches past_state; every later tile reads
    // its whole window from input.
    __shared__ float smem[SMEM_LEN];
    {
        const T* ps = (STATE_LEN > 0 && past_state != nullptr)
                          ? past_state + static_cast<int64_t>(plane) * STATE_LEN
                          : nullptr;
        for (int j = lane; j < SMEM_LEN; j += TILE) {
            const int vi = t0 + j;
            float v;
            if (vi < STATE_LEN) {
                v = ps ? to_f32<T>(ps[vi]) : 0.0f;
            } else {
                const int ii = vi - STATE_LEN;
                v = (ii < seq_len) ? to_f32<T>(in_row[ii]) : 0.0f;
            }
            smem[j] = v;
        }
    }

    // Per-channel weights (K taps) and optional bias: one channel per block, so
    // every lane reads the same values and the load broadcasts.
    float w[K];
    {
        const T* wp = weight + static_cast<int64_t>(c) * K;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            w[j] = to_f32<T>(wp[j]);
    }
    const float bias_v = HAS_BIAS ? to_f32<T>(bias[c]) : 0.0f;

    __syncthreads();

    const int t = t0 + lane;
    if (t < seq_len) {
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            y += w[j] * smem[lane + j];
        if (HAS_BIAS)
            y += bias_v;
        if (ACT_SILU) {
            const float s = 1.0f / (1.0f + expf(-y));
            y *= s;
        }
        out_row[t] = from_f32<T>(y);
    }

    // present_state[j] = virtual[L + j]: the trailing K-1 of the virtual
    // sequence, owned by whichever block covers the final output.
    if (STATE_LEN > 0 && t0 + TILE >= seq_len && lane < STATE_LEN) {
        T* nps = present_state + static_cast<int64_t>(plane) * STATE_LEN;
        nps[lane] = from_f32<T>(smem[seq_len - t0 + lane]);
    }
}

// Channels-last prefill: input and output are [B, L, C] instead of [B, C, L].
// The recurrent state keeps its native [B, C, K-1] layout either way.
//
// This exists because the ONNX export brackets the conv with a Transpose pair --
// [B,L,C] -> [B,C,L] in, and back out -- purely because ONNX Conv wants its
// channels first. Those two transposes move 65 MB each on Qwen3.6-35B-A3B and
// together cost more than the convolution they surround. Nothing about the
// arithmetic needs channels first, so the layout is a parameter here and the
// transposes can be folded away by the caller.
//
// The shape falls out differently from the channels-first kernel. There, time is
// the contiguous axis, so a wave must cover consecutive t and the K-deep window
// overlaps between neighbouring lanes -- hence the LDS staging. Here channels are
// contiguous, so a lane owns one channel and walks time by itself: the window
// lives entirely in that lane's registers with no sharing, no LDS and no halo
// staging, while each load still coalesces because neighbouring lanes read
// neighbouring channels at the same t. A block covers CTILE channels x TTILE
// timesteps and re-reads only the K-1 timesteps preceding its tile.
template <typename T, int K, int HAS_BIAS, int ACT_SILU, int CTILE, int TTILE>
__global__ void causal_conv_prefill_nlc_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ past_state,
    T* __restrict__ output,
    T* __restrict__ present_state,
    int batch_size,
    int channels,
    int seq_len) {

    const int c = static_cast<int>(blockIdx.x) * CTILE + static_cast<int>(threadIdx.x);
    const int b = static_cast<int>(blockIdx.z);
    const int t0 = static_cast<int>(blockIdx.y) * TTILE;
    if (c >= channels || t0 >= seq_len || b >= batch_size)
        return;

    constexpr int STATE_LEN = K - 1;
    const int64_t plane = static_cast<int64_t>(b) * channels + c;
    const T* ps = (STATE_LEN > 0 && past_state != nullptr)
                      ? past_state + plane * STATE_LEN
                      : nullptr;
    const T* in_b = input + static_cast<int64_t>(b) * seq_len * channels;
    T* out_b = output + static_cast<int64_t>(b) * seq_len * channels;

    // The virtual sequence the convolution slides over, same definition as the
    // channels-first kernel: virtual[i] = past_state[i] for i < K-1, else
    // input[i - (K-1)]. Output t reads virtual[t .. t+K-1].
    auto virt = [&](int i) -> float {
        if (i < STATE_LEN)
            return ps ? to_f32<T>(ps[i]) : 0.0f;
        const int ti = i - STATE_LEN;
        return (ti < seq_len)
                   ? to_f32<T>(in_b[static_cast<int64_t>(ti) * channels + c])
                   : 0.0f;
    };

    // One channel per lane, so each lane reads its own K taps. Strided by K
    // across the wave, but the whole filter bank is C*K elements and is reused
    // by every timestep this lane walks.
    float w[K];
    {
        const T* wp = weight + static_cast<int64_t>(c) * K;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            w[j] = to_f32<T>(wp[j]);
    }
    const float bias_v = HAS_BIAS ? to_f32<T>(bias[c]) : 0.0f;

    // Seed with the K-1 virtual elements preceding this tile's first output.
    float win[K];
    #pragma unroll
    for (int j = 0; j < STATE_LEN; ++j)
        win[j] = virt(t0 + j);

    const int tend = (t0 + TTILE < seq_len) ? t0 + TTILE : seq_len;
    for (int t = t0; t < tend; ++t) {
        win[K - 1] = to_f32<T>(in_b[static_cast<int64_t>(t) * channels + c]);
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            y += w[j] * win[j];
        if (HAS_BIAS)
            y += bias_v;
        if (ACT_SILU) {
            const float s = 1.0f / (1.0f + expf(-y));
            y *= s;
        }
        out_b[static_cast<int64_t>(t) * channels + c] = from_f32<T>(y);
        #pragma unroll
        for (int j = 0; j < K - 1; ++j)
            win[j] = win[j + 1];
    }

    // present_state[j] = virtual[L + j], written by the tile holding the tail.
    if (STATE_LEN > 0 && tend == seq_len) {
        T* nps = present_state + plane * STATE_LEN;
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            nps[j] = from_f32<T>(virt(seq_len + j));
    }
}

template <typename T>
hipError_t launch_prefill_nlc_for_K(
    int K, hipStream_t stream,
    const T* input, const T* weight, const T* bias, const T* past_state,
    T* output, T* present_state,
    int batch_size, int channels, int seq_len, int has_bias, int act_silu) {

    // CTILE is the block size, so it sets the coalesced width; 256 covers four
    // 64-lane waves of consecutive channels. TTILE trades halo re-reads against
    // block count: at K=4 a 64-step tile re-reads 3 of every 64 timesteps, ~5%,
    // while still giving 8192/256 x 3985/64 = 2016 blocks on this model.
    constexpr int kCTile = 256;
    constexpr int kTTile = 64;
    const dim3 grid((channels + kCTile - 1) / kCTile,
                    (seq_len + kTTile - 1) / kTTile,
                    batch_size);

    auto launch = [&](auto k_const, auto bias_const, auto act_const) {
        constexpr int Kc = decltype(k_const)::value;
        constexpr int Bc = decltype(bias_const)::value;
        constexpr int Ac = decltype(act_const)::value;
        hipLaunchKernelGGL(
            (causal_conv_prefill_nlc_kernel<T, Kc, Bc, Ac, kCTile, kTTile>),
            grid, dim3(kCTile), 0, stream,
            input, weight, bias, past_state, output, present_state,
            batch_size, channels, seq_len);
        return hipGetLastError();
    };

#define DISPATCH_PREFILL_NLC_K(KVAL)                                          \
    case KVAL: {                                                              \
        using K_t = std::integral_constant<int, KVAL>;                        \
        if (has_bias && act_silu)                                             \
            return launch(K_t{}, std::integral_constant<int, 1>{},            \
                          std::integral_constant<int, 1>{});                  \
        if (has_bias)                                                         \
            return launch(K_t{}, std::integral_constant<int, 1>{},            \
                          std::integral_constant<int, 0>{});                  \
        if (act_silu)                                                         \
            return launch(K_t{}, std::integral_constant<int, 0>{},            \
                          std::integral_constant<int, 1>{});                  \
        return launch(K_t{}, std::integral_constant<int, 0>{},                \
                      std::integral_constant<int, 0>{});                      \
    }

    switch (K) {
        DISPATCH_PREFILL_NLC_K(1)
        DISPATCH_PREFILL_NLC_K(2)
        DISPATCH_PREFILL_NLC_K(3)
        DISPATCH_PREFILL_NLC_K(4)
        DISPATCH_PREFILL_NLC_K(5)
        DISPATCH_PREFILL_NLC_K(6)
        DISPATCH_PREFILL_NLC_K(7)
        DISPATCH_PREFILL_NLC_K(8)
        default:
            return hipErrorInvalidValue;
    }
#undef DISPATCH_PREFILL_NLC_K
}

template <typename T>
hipError_t launch_prefill_for_K(
    int K, hipStream_t stream,
    const T* input, const T* weight, const T* bias, const T* past_state,
    T* output, T* present_state,
    int batch_size, int channels, int seq_len, int has_bias, int act_silu) {

    // One block per (b, c) plane per kTile outputs. kTile is also the block
    // size, so it sets the coalesced load width and the LDS staging buffer at
    // once; 256 keeps that buffer near 1 KB and the halo re-read to K-1 elements
    // per 256 outputs.
    constexpr int kTile = 256;
    const int planes = batch_size * channels;
    const int tiles = (seq_len + kTile - 1) / kTile;

    auto launch = [&](auto k_const, auto bias_const, auto act_const) {
        constexpr int Kc = decltype(k_const)::value;
        constexpr int Bc = decltype(bias_const)::value;
        constexpr int Ac = decltype(act_const)::value;
        hipLaunchKernelGGL(
            (causal_conv_prefill_kernel<T, Kc, Bc, Ac, kTile>),
            dim3(planes, tiles), dim3(kTile), 0, stream,
            input, weight, bias, past_state, output, present_state,
            batch_size, channels, seq_len);
        return hipGetLastError();
    };

#define DISPATCH_PREFILL_K(KVAL)                                              \
    case KVAL: {                                                              \
        using K_t = std::integral_constant<int, KVAL>;                        \
        if (has_bias && act_silu)                                             \
            return launch(K_t{}, std::integral_constant<int, 1>{},            \
                          std::integral_constant<int, 1>{});                  \
        if (has_bias)                                                         \
            return launch(K_t{}, std::integral_constant<int, 1>{},            \
                          std::integral_constant<int, 0>{});                  \
        if (act_silu)                                                         \
            return launch(K_t{}, std::integral_constant<int, 0>{},            \
                          std::integral_constant<int, 1>{});                  \
        return launch(K_t{}, std::integral_constant<int, 0>{},                \
                      std::integral_constant<int, 0>{});                      \
    }

    switch (K) {
        DISPATCH_PREFILL_K(1)
        DISPATCH_PREFILL_K(2)
        DISPATCH_PREFILL_K(3)
        DISPATCH_PREFILL_K(4)
        DISPATCH_PREFILL_K(5)
        DISPATCH_PREFILL_K(6)
        DISPATCH_PREFILL_K(7)
        DISPATCH_PREFILL_K(8)
        default:
            return hipErrorInvalidValue;
    }
#undef DISPATCH_PREFILL_K
}

} // namespace

extern "C" int hip_causal_conv_step_decode(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes) {

    if (!input || !weight || !output || !present_state) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: null required "
                "pointer (input=%p weight=%p output=%p present_state=%p)\n",
                input, weight, output, present_state);
        return -1;
    }
    if (batch_size <= 0 || channels <= 0) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: invalid "
                "batch=%lld channels=%lld\n",
                (long long)batch_size, (long long)channels);
        return -1;
    }
    if (kernel_size < 1 || kernel_size > 8) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: kernel_size=%lld "
                "outside supported range [1,8]\n",
                (long long)kernel_size);
        return -1;
    }
    if (activation != 0 && activation != 1) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: unsupported "
                "activation=%lld (0=none, 1=SiLU)\n",
                (long long)activation);
        return -1;
    }

    hipStream_t s = static_cast<hipStream_t>(stream);
    const int K = static_cast<int>(kernel_size);
    const int B = static_cast<int>(batch_size);
    const int C = static_cast<int>(channels);
    const int has_bias = bias ? 1 : 0;
    const int act_silu = (activation == 1) ? 1 : 0;

    hipError_t err;
    if (element_size_bytes == 4) {
        err = launch_for_K<float>(
            K, s,
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<const float*>(bias),
            static_cast<const float*>(past_state),
            static_cast<float*>(output),
            static_cast<float*>(present_state),
            B, C, has_bias, act_silu);
    } else if (element_size_bytes == 2) {
        err = launch_for_K<__half>(
            K, s,
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<const __half*>(past_state),
            static_cast<__half*>(output),
            static_cast<__half*>(present_state),
            B, C, has_bias, act_silu);
    } else {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: unsupported "
                "element_size_bytes=%lld (expect 2 or 4)\n",
                (long long)element_size_bytes);
        return -1;
    }

    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: launch failed "
                "(%s)\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}

namespace {

// Shared body of the two prefill entry points: identical contract and
// validation, differing only in which launcher (and so which activation layout)
// they use. `who` names the caller in diagnostics.
template <typename Launch>
int prefill_common(
    const char* who,
    const void* input, const void* weight, const void* output,
    const void* present_state,
    int64_t batch_size, int64_t channels, int64_t seq_len,
    int64_t kernel_size, int64_t activation, int64_t element_size_bytes,
    Launch&& launch) {

    if (!input || !weight || !output || !present_state) {
        fprintf(stderr,
                "[custom_kernels] %s: null required "
                "pointer (input=%p weight=%p output=%p present_state=%p)\n",
                who, input, weight, output, present_state);
        return -1;
    }
    if (batch_size <= 0 || channels <= 0 || seq_len <= 0) {
        fprintf(stderr,
                "[custom_kernels] %s: invalid "
                "batch=%lld channels=%lld seq_len=%lld\n",
                who, (long long)batch_size, (long long)channels,
                (long long)seq_len);
        return -1;
    }
    if (kernel_size < 1 || kernel_size > 8) {
        fprintf(stderr,
                "[custom_kernels] %s: kernel_size=%lld "
                "outside supported range [1,8]\n",
                who, (long long)kernel_size);
        return -1;
    }
    if (activation != 0 && activation != 1) {
        fprintf(stderr,
                "[custom_kernels] %s: unsupported "
                "activation=%lld (0=none, 1=SiLU)\n",
                who, (long long)activation);
        return -1;
    }
    if (element_size_bytes != 2 && element_size_bytes != 4) {
        fprintf(stderr,
                "[custom_kernels] %s: unsupported "
                "element_size_bytes=%lld (expect 2 or 4)\n",
                who, (long long)element_size_bytes);
        return -1;
    }

    const hipError_t err = launch(element_size_bytes == 4);
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] %s: launch failed (%s)\n", who,
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}

} // namespace

extern "C" int hip_causal_conv_prefill(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t seq_len,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes) {

    return prefill_common(
        "hip_causal_conv_prefill", input, weight, output, present_state,
        batch_size, channels, seq_len, kernel_size, activation,
        element_size_bytes,
        [&](bool is_f32) {
            hipStream_t s = static_cast<hipStream_t>(stream);
            const int K = static_cast<int>(kernel_size);
            const int B = static_cast<int>(batch_size);
            const int C = static_cast<int>(channels);
            const int L = static_cast<int>(seq_len);
            const int has_bias = bias ? 1 : 0;
            const int act_silu = (activation == 1) ? 1 : 0;
            if (is_f32)
                return launch_prefill_for_K<float>(
                    K, s,
                    static_cast<const float*>(input),
                    static_cast<const float*>(weight),
                    static_cast<const float*>(bias),
                    static_cast<const float*>(past_state),
                    static_cast<float*>(output),
                    static_cast<float*>(present_state),
                    B, C, L, has_bias, act_silu);
            return launch_prefill_for_K<__half>(
                K, s,
                static_cast<const __half*>(input),
                static_cast<const __half*>(weight),
                static_cast<const __half*>(bias),
                static_cast<const __half*>(past_state),
                static_cast<__half*>(output),
                static_cast<__half*>(present_state),
                B, C, L, has_bias, act_silu);
        });
}

extern "C" int hip_causal_conv_prefill_nlc(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t seq_len,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes) {

    return prefill_common(
        "hip_causal_conv_prefill_nlc", input, weight, output, present_state,
        batch_size, channels, seq_len, kernel_size, activation,
        element_size_bytes,
        [&](bool is_f32) {
            hipStream_t s = static_cast<hipStream_t>(stream);
            const int K = static_cast<int>(kernel_size);
            const int B = static_cast<int>(batch_size);
            const int C = static_cast<int>(channels);
            const int L = static_cast<int>(seq_len);
            const int has_bias = bias ? 1 : 0;
            const int act_silu = (activation == 1) ? 1 : 0;
            if (is_f32)
                return launch_prefill_nlc_for_K<float>(
                    K, s,
                    static_cast<const float*>(input),
                    static_cast<const float*>(weight),
                    static_cast<const float*>(bias),
                    static_cast<const float*>(past_state),
                    static_cast<float*>(output),
                    static_cast<float*>(present_state),
                    B, C, L, has_bias, act_silu);
            return launch_prefill_nlc_for_K<__half>(
                K, s,
                static_cast<const __half*>(input),
                static_cast<const __half*>(weight),
                static_cast<const __half*>(bias),
                static_cast<const __half*>(past_state),
                static_cast<__half*>(output),
                static_cast<__half*>(present_state),
                B, C, L, has_bias, act_silu);
        });
}
