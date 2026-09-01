/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Fused FA-2 WMMA prefill kernel for com.microsoft.MultiHeadAttention.
//
// Replaces the decomposed hipBLASLt pipeline (Q/K/V transpose -> score GEMM ->
// softmax -> value GEMM -> O transpose) that materializes the full fp32 score
// matrix S[B,N,Sq,Skv] in global memory. For the Qwen VLM vision encoder
// (B=1, N=16, Sq=Skv=7296, H=72) that matrix is ~3.4 GB and the op is entirely
// HBM-bandwidth bound (~2.9 s / prefill). Flash-attention keeps the running
// (m, l, O) softmax state in registers and never writes S to DRAM.
//
// This is the non-causal (bidirectional), self-attention (N_q == N_kv, HpG==1)
// specialization modeled on the GQA flash-prefill v7 kernel (gqa_kernel.hip):
//   * Q resident in registers as 16x16 WMMA fragments (D-tiled).
//   * K streamed from global straight into WMMA fragments.
//   * V staged into LDS; P (probabilities) transposed through LDS.
//   * Score / value GEMMs run on the RDNA3.5 WMMA matrix unit (wave32).
//   * Intra-wave online softmax in log2e space.
//
// Head dim H need not be a multiple of 16: the kernel is templated on the
// padded D (next multiple of 16) and the tail D-tile is masked-loaded (valid
// columns real, padded columns zero) so the padded dims contribute 0 to the
// QK dot and produce discarded (unwritten) output columns.
//
// Layout (matches the runtime call site after the K/V transpose):
//   Q      : [B, Sq,  N, H]   fp16   (BSND == BSHD, no transpose needed)
//   Kcache : [B, N, max_seq, H]  fp16   (BNSD, transposed from BSND)
//   Vcache : [B, N, max_seq, H]  fp16   (BNSD)
//   O      : [B, Sq,  N, H]   fp16   (BSND, no transpose needed)
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hip_custom_kernels.h"

#include "rtc/hiprtc_module.h"

// The __global__ definitions live in a header so the AOT build below and the
// hipRTC path compile byte-identical source. Keeping it outside the anonymous
// namespace gives the kernel the same linkage on both paths.
#include "hip/rtc/multi_head_attention_device.h"

// The same header, embedded as bytes at build time.
extern "C" const unsigned char kMhaDeviceSource[];
extern "C" const size_t kMhaDeviceSource_size;

namespace {

static inline size_t mha_prefill_smem_bytes(int nw, int bkv, int d_pad, int mt) {
  const size_t v_lds = (size_t)bkv * (d_pad + 2) * sizeof(_Float16);
  const size_t p_lds = (size_t)(nw * mt * 16) * (bkv + 2) * sizeof(_Float16);
  return v_lds + p_lds;
}

// Default launch config (correctness-first; autotune can be layered later).
// rows/block = NW*MT*16 = 64; V+P LDS at d_pad=80,BKV=64 ~= 19 KB.
constexpr int kMhaNW = 4;
constexpr int kMhaBKV = 64;
constexpr int kMhaMT = 1;

// All of these get registered up front: hipRTC name expressions must be added
// before the program is compiled.
constexpr int kMhaDPads[] = {16, 32, 48, 64, 80, 96, 112, 128, 256};

std::string mhaKernelName(int d_pad) {
  return "mha_flash_prefill_kernel<" + std::to_string(kMhaNW) + ", " +
         std::to_string(kMhaBKV) + ", " + std::to_string(d_pad) + ", " +
         std::to_string(kMhaMT) + ">";
}

std::vector<std::string> mhaKernelNames() {
  std::vector<std::string> names;
  for (int d_pad : kMhaDPads) {
    names.push_back(mhaKernelName(d_pad));
  }
  return names;
}

hipdnn_ep::rtc::Module& mhaRtcModule() {
  static hipdnn_ep::rtc::Module module(
      "multi_head_attention", reinterpret_cast<const char*>(kMhaDeviceSource),
      kMhaDeviceSource_size, mhaKernelNames());
  return module;
}

// Returns false when the RTC path is unavailable, leaving the caller to fall
// through to AOT. Once the kernel has been launched the result is reported
// through *status instead.
//
// Argument packing is unchecked by the compiler here: the order and types must
// track mha_flash_prefill_kernel's signature in the device header by hand.
bool tryLaunchMhaRtc(int d_pad, hipStream_t stream, dim3 grid, size_t smem,
                     const _Float16* Q, const _Float16* K, const _Float16* V,
                     _Float16* O, int N, int sq, int skv, int d, int max_seq,
                     float scale, int* status) {
  const std::string name = mhaKernelName(d_pad);
  hipFunction_t fn = mhaRtcModule().getFunction(name);
  if (!fn) {
    fprintf(stderr,
            "[custom_kernels] hip_mha_flash_prefill: RTC kernel %s unavailable, "
            "using AOT\n",
            name.c_str());
    return false;
  }

  void* args[] = {&Q, &K, &V, &O, &N, &sq, &skv, &d, &max_seq, &scale};
  hipError_t err =
      hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, kMhaNW * 32, 1, 1,
                            static_cast<unsigned>(smem), stream, args, nullptr);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_mha_flash_prefill RTC launch failed: %s\n",
            hipGetErrorString(err));
  }
  *status = static_cast<int>(err);
  return true;
}

template <int D_PAD>
static inline int launch_mha_prefill(hipStream_t stream, const _Float16 *Q,
                                     const _Float16 *K, const _Float16 *V,
                                     _Float16 *O, int B, int N, int sq, int skv,
                                     int d, int max_seq, float scale) {
  const int rows = kMhaNW * kMhaMT * 16;
  const int num_q_tiles = (sq + rows - 1) / rows;
  dim3 grid(num_q_tiles * B, N);
  const size_t smem = mha_prefill_smem_bytes(kMhaNW, kMhaBKV, D_PAD, kMhaMT);
  (void)hipGetLastError();

  if (hipdnn_ep::rtc::enabled()) {
    int status = 0;
    if (tryLaunchMhaRtc(D_PAD, stream, grid, smem, Q, K, V, O, N, sq, skv, d,
                        max_seq, scale, &status)) {
      return status;
    }
  }

  mha_flash_prefill_kernel<kMhaNW, kMhaBKV, D_PAD, kMhaMT>
      <<<grid, kMhaNW * 32, smem, stream>>>(Q, K, V, O, N, sq, skv, d, max_seq,
                                            scale);
  return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

}  // namespace

// hip_mha_flash_prefill: fused non-causal FA-2 prefill for MultiHeadAttention.
// Q [B,sq,N,d], K/V [B,N,max_seq,d] (BNSD), O [B,sq,N,d]; fp16. Returns 0 on
// success, -1 if the head dim is unsupported (d_pad > 256).
extern "C" int hip_mha_flash_prefill(void *stream, const void *Q,
                                     const void *Kcache, const void *Vcache,
                                     void *O, int B, int N, int sq, int skv,
                                     int d, int max_seq, float scale) {
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  const _Float16 *q = reinterpret_cast<const _Float16 *>(Q);
  const _Float16 *k = reinterpret_cast<const _Float16 *>(Kcache);
  const _Float16 *v = reinterpret_cast<const _Float16 *>(Vcache);
  _Float16 *o = reinterpret_cast<_Float16 *>(O);
  const int d_pad = ((d + 15) / 16) * 16;

  switch (d_pad) {
  case 16:  return launch_mha_prefill<16>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 32:  return launch_mha_prefill<32>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 48:  return launch_mha_prefill<48>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 64:  return launch_mha_prefill<64>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 80:  return launch_mha_prefill<80>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 96:  return launch_mha_prefill<96>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 112: return launch_mha_prefill<112>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 128: return launch_mha_prefill<128>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 256: return launch_mha_prefill<256>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  default:
    fprintf(stderr, "[hip_mha_flash_prefill] unsupported head dim d=%d (d_pad=%d)\n",
            d, d_pad);
    return -1;
  }
}
