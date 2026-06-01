/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ck_dsl single-pass GQA attention decode shim.
 *
 * Drop-in alternative to hip_gqa_fused_decode for the sq == 1 short-context
 * (skv < kFlashDecodeMinSkv) regime where the EP currently dispatches to the
 * cooperative-dot-product fused_decode. On gfx1151 (Strix Halo) the ck_dsl
 * WMMA-based single-pass kernel runs 2.3-2.8x faster than the baseline at
 * skv in {64, 128, 192}; see docs/design/ck-dsl-fmha-decode.md for the
 * empirical sweep + the est. +9.5% Mistral 7B AWQ b128 decode TPS at L=128.
 *
 * The underlying kernel is `wmma_fmha_fwd` from the ck_dsl prototype
 * (ROCm/rocm-libraries `users/vanantha/ck-dsl-prototype` branch,
 * `projects/composablekernel/python/ck_dsl/instances/gfx1151/wmma_fmha_fwd.py`).
 * It expects seqlen_q = 16 (the WMMA M-tile), so we drive the kernel with
 * Q-broadcast strides (stride_q_token = stride_o_token = 0) — the single
 * decode Q row drives all 16 WMMA rows in-register and all 16 output rows
 * write the *same* value to the *same* output slot. No scratch buffer
 * required; no Q replication kernel required.
 *
 * Gating: this shim is opt-in via HIPDNN_EP_CK_DSL_FMHA_DECODE=1. Default off
 * so production routing remains hip_gqa_fused_decode until reviewers
 * validate the swap. The dispatch call site lives in real/gqa.cpp.
 */

#include "../hipdnn_ep_runtime.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ck_dsl_fmha_decode_gfx1151_hsaco.h"

namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kBlockM = 16;         // WMMA M-tile (== ck_dsl _BLOCK_M)
constexpr int kWaveSize = 32;       // wave32 RDNA kernel

// Kernel symbol name as exported by the ck_dsl-emitted HSACO.
// This name comes from `WmmaFmhaFwdSpec.kernel_name()`:
//   ck_dsl_wmma_fmha_fwd_wmma16x16x16_H128_HQ32_HK8_fp16_none_vgather
// If the spec ever changes (e.g. add causal mask, change atom shape, change
// HQ/HK), regenerate the HSACO and update this symbol.
constexpr const char *kKernelSymbol =
    "ck_dsl_wmma_fmha_fwd_wmma16x16x16_H128_HQ32_HK8_fp16_none_vgather";

// Shape preconditions the kernel was compiled for. The wrapper rejects calls
// that don't match — falling back to the caller's existing code path.
constexpr int kHeadSize = 128;
constexpr int kNumQueryHeads = 32;
constexpr int kNumKvHeads = 8;

// HIP module + function handles, lazily loaded once per process.
hipModule_t   g_module   = nullptr;
hipFunction_t g_function = nullptr;
std::once_flag g_load_once;
std::atomic<bool> g_load_ok{false};

void load_module_once() {
    std::call_once(g_load_once, []() {
        hipError_t err = hipModuleLoadData(&g_module, kCkDslFmhaDecodeGfx1151Hsaco);
        if (err != hipSuccess) {
            fprintf(stderr,
                    "ck_dsl_fmha_decode: hipModuleLoadData failed: %s\n",
                    hipGetErrorString(err));
            return;
        }
        err = hipModuleGetFunction(&g_function, g_module, kKernelSymbol);
        if (err != hipSuccess) {
            fprintf(stderr,
                    "ck_dsl_fmha_decode: hipModuleGetFunction(%s) failed: %s\n",
                    kKernelSymbol, hipGetErrorString(err));
            hipModuleUnload(g_module);
            g_module = nullptr;
            return;
        }
        g_load_ok.store(true, std::memory_order_release);
    });
}

// Packed args for hipModuleLaunchKernel "extra" buffer.
//
// ABI order matches `WmmaFmhaFwdSpec.kernel_name()`'s param list (see
// `instances/gfx1151/wmma_fmha_fwd.py::_declare_params`):
//   Q, K, V, O           : 4 × void* (8 bytes each)
//   scale_log2           : float
//   seqlen_q, seqlen_k   : 2 × int32
//   stride_q_token/head  : 2 × int32
//   stride_k_token/head  : 2 × int32
//   stride_v_token/head  : 2 × int32
//   stride_o_token/head  : 2 × int32
// The struct is `#pragma pack`'d to natural-with-no-padding to match how the
// kernel's MLIR-style ABI reads scalar args inline after the pointers.
#pragma pack(push, 1)
struct KernelArgs {
    const void *Q;
    const void *K;
    const void *V;
    void       *O;
    float       scale_log2;
    int32_t     seqlen_q;
    int32_t     seqlen_k;
    int32_t     stride_q_token;
    int32_t     stride_q_head;
    int32_t     stride_k_token;
    int32_t     stride_k_head;
    int32_t     stride_v_token;
    int32_t     stride_v_head;
    int32_t     stride_o_token;
    int32_t     stride_o_head;
};
#pragma pack(pop)

} // namespace

// Same C ABI as hip_gqa_fused_decode (hip_custom_kernels.h:486). Drop-in:
// callers can substitute one for the other.
//
// Returns:
//   0           on successful launch.
//   -2 (RejectFallback) when the runtime spec doesn't match what this HSACO
//                       was compiled for (caller must fall back to baseline).
//   hipError_t  on a HIP failure (matches hip_gqa_fused_decode's contract).
extern "C" int ck_dsl_gqa_fmha_decode(
    void *stream,
    const void *Q, const void *Kcache, const void *Vcache,
    void *O,
    int B, int H, int G, int d, int skv, int max_seq,
    float scale,
    const void * /*seqlens_k*/) {
    constexpr int kRejectFallback = -2;

    // Hard preconditions for this HSACO. Anything outside is a "not our shape"
    // — return RejectFallback so the dispatcher routes back to the baseline.
    if (B != 1 || d != kHeadSize || H != kNumQueryHeads || G != kNumKvHeads) {
        return kRejectFallback;
    }
    if (skv <= 0 || max_seq < skv) {
        return kRejectFallback;
    }

    load_module_once();
    if (!g_load_ok.load(std::memory_order_acquire)) {
        return kRejectFallback;
    }

    KernelArgs args;
    args.Q              = Q;
    args.K              = Kcache;
    args.V              = Vcache;
    args.O              = O;
    args.scale_log2     = scale * kLog2e;
    args.seqlen_q       = kBlockM;             // 16 — the WMMA tile
    args.seqlen_k       = skv;
    // Broadcast Q across 16 WMMA rows: all rows read the same Q[0,*,h,:] data.
    // The single decode Q has shape [1, 1, H, d] = stride_q_head=d, no token
    // stride within batch — feed the same memory to all 16 logical rows.
    args.stride_q_token = 0;
    args.stride_q_head  = d;
    // Baseline KV cache is BNSD [B, G, max_seq, d] row-major.
    args.stride_k_token = d;
    args.stride_k_head  = max_seq * d;
    args.stride_v_token = d;
    args.stride_v_head  = max_seq * d;
    // Output O is [B, 1, H, d]: only 1 real row, but the kernel writes 16
    // *identical* rows (broadcast Q -> broadcast O). Send all 16 rows to the
    // same slot — every write produces the same value. No scratch needed.
    args.stride_o_token = 0;
    args.stride_o_head  = d;

    size_t args_size = sizeof(args);
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &args_size,
        HIP_LAUNCH_PARAM_END,
    };

    // Grid: (seqlen_q / BLOCK_M = 1, num_q_heads = 32, batch = 1).
    // Block: one wave32 per CTA.
    hipError_t err = hipModuleLaunchKernel(
        g_function,
        /*gx*/ 1, /*gy*/ H, /*gz*/ 1,
        /*bx*/ kWaveSize, /*by*/ 1, /*bz*/ 1,
        /*sharedMemBytes*/ 0,
        /*stream*/ static_cast<hipStream_t>(stream),
        /*kernelParams*/ nullptr,
        /*extra*/ extra);
    if (err != hipSuccess) {
        fprintf(stderr,
                "ck_dsl_fmha_decode: hipModuleLaunchKernel failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}
