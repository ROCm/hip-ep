/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// T0.1 spike -- Candidate A ("HIP owns"), the NPU-touching half.
//
// THIS FILE DOES NOT COMPILE ON A MACHINE WITHOUT THE REAL DynamicDispatch /
// XRT / ryzen_mm SDKs INSTALLED. It is not meant to. CMakeLists.txt only adds
// its target when those packages resolve to real CMake targets -- see the
// README in this directory for why, and for exactly what to do on the
// gfx1151 NPU host to build and run it.
//
// What this program proves, if it links and runs:
//   1. hipHostMalloc(Mapped|Coherent) memory -- the exact allocation shape
//      morphizen's EP allocator uses -- can be bound directly into a
//      DynamicDispatch kernel via bind_bo(), without copying through a
//      RyzenMM-allocated intermediate. This is the specific combination
//      docs/design/hybrid-npu-gpu-tasks.md's T0.1 entry calls unverified:
//      "whether XDNA accepts a pointer the HIP driver has already pinned."
//   2. The xrt::bo returned by bind_bo() maps back to the SAME address HIP
//      handed out (aliasing, not copying).
//   3. One physical allocation is written by the CPU, computed on by the
//      NPU, read by the CPU with no explicit copy, then mutated in place by
//      a HIP kernel (gpu_touch.hip) and read by the CPU again -- closing the
//      three-way loop the T0.1 gate asks for.
//   4. bind_bo()'s cost is measured, not assumed.
//   5. xrt::bo::sync()'s per-run cost is measured. Registration aliasing
//      (check 2) only rules out a copy at bind time; sync() runs on every
//      inference. On a UMA part an imported buffer should need cache
//      maintenance at most, so a cost proportional to the buffer size would
//      mean a staging copy inside XRT -- a per-inference boundary copy that
//      the EP's own counters can never observe, because it happens below
//      them. Zero copy would be silently false while every value still
//      compared equal.
//
// Provenance of the API calls below: cribbed from the read-only hybrid-llm
// reference tree named in docs/design/hybrid-npu-gpu-tasks.md's
// "In another repository" table --
//   onnx_custom_ops/hybrid_llm/npu/binary_elemwise_npu_kernel.hpp (the
//     bind_bo/set_tensor_shape/get_buffer_reqs/run call sequence),
//   onnx_custom_ops/hybrid_llm/npu/mul.hpp (the concrete ElwMul<bf16,bf16,bf16>
//     instantiation and its 'MUL_' alloc tag),
//   onnx_custom_ops/hybrid_llm/npu/npu_op.cpp (getCommonAttrs()'s exact
//     attribute map shape).
// That tree does not vendor DynamicDispatch/include/ops/op_interface.hpp
// itself (it is an external package there too), so the following are
// EXPLICITLY UNVERIFIED against the real header and must be confirmed on
// the NPU host before trusting a compile error -- or its silent absence --
// here:
//   - The exact declaration of `::Tensor`, `OpArgMap`, and `NPUBufferSpan`
//     (their include paths below are a best guess based on sibling
//     `#include` lines in mul.hpp / matmulnbits.cpp).
//   - Whether bind_bo() has a `read_only` overload -- the task text mentions
//     `bind_bo(void*, size, read_only)`, but every call site found in the
//     reference tree uses the two-argument form `bind_bo(ptr, len)`. Try the
//     two-argument form first; it is what is actually exercised in
//     production there.
//   - The DDKernel constructor's first argument's exact meaning (passed as
//     `true` at every reference call site; read as "attach to real
//     hardware" but never confirmed against the header).

#include <hip/hip_runtime.h>

#include "bf16_util.hpp"
#include "gpu_touch.hpp"

// --- DynamicDispatch headers -------------------------------------------
// Paths as included by mul.hpp / matmulnbits.cpp in the hybrid-llm
// reference tree. CONFIRM these resolve as-is against the NPU host's
// DynamicDispatch install; adjust if its include layout has moved.
#include <xrt/xrt_bo.h>

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <ops/op_interface.hpp>  // Tensor, OpArgMap, NPUBufferSpan (assumed)
#include <ops/transformer/binary_elementwise_op.hpp>  // ElwMul<...>
#include <string>
#include <types/data_meta_types.hpp>  // meta_types::bfloat16
#include <vector>

namespace {

using DDElwMul = ryzenai::dynamic_dispatch::transformer::ElwMul<
    ryzenai::dynamic_dispatch::meta_types::bfloat16,
    ryzenai::dynamic_dispatch::meta_types::bfloat16,
    ryzenai::dynamic_dispatch::meta_types::bfloat16>;

// GPU and NPU both define a `Tensor` type in different namespaces in the
// reference tree; `::Tensor` is the NPU (DynamicDispatch) one. Mirrors the
// `using NPUTensor = ::Tensor;` alias at the top of matmulnbits.cpp.
using NpuTensor = ::Tensor;

bool CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", what, hipGetErrorString(err));
    return false;
  }
  return true;
}

}  // namespace

int main() {
  constexpr size_t kM = 1, kK = 64;  // trivial shape, only needs to be valid
  constexpr size_t kElems = kM * kK;
  constexpr size_t kBytes = kElems * sizeof(uint16_t);

  // --- Candidate A allocation: exactly HipGpuAllocator::AllocImpl's cold
  // path (see morphizen/ort-bridge/src/morphizen-hip-gpu-allocator.cpp). ---
  void *lhs_ptr = nullptr, *rhs_ptr = nullptr, *out_ptr = nullptr;
  if (!CheckHip(hipHostMalloc(&lhs_ptr, kBytes,
                              hipHostMallocMapped | hipHostMallocCoherent),
                "hipHostMalloc(lhs)") ||
      !CheckHip(hipHostMalloc(&rhs_ptr, kBytes,
                              hipHostMallocMapped | hipHostMallocCoherent),
                "hipHostMalloc(rhs)") ||
      !CheckHip(hipHostMalloc(&out_ptr, kBytes,
                              hipHostMallocMapped | hipHostMallocCoherent),
                "hipHostMalloc(out)")) {
    return 1;
  }

  bool ok = true;
  try {
    // getCommonAttrs()'s shape, per npu_op.cpp -- values are placeholders;
    // this spike does not go through OrtKernelInfo, so there is no real
    // op_version/pdi_name to read. Confirm on the NPU host whether the
    // kernel tolerates empty/default values or needs a real PDI name.
    std::map<std::string, std::any> attrs = {
        {std::string("op_version"), std::string("v1")},
        {std::string("pdi_name"), std::string("")},
        {std::string("preemption"), false},
        {std::string("lora"), false},
        {std::string("qos"), std::map<std::string, uint32_t>{}},
    };

    auto kernel = std::make_unique<DDElwMul>(/*load_xrt=*/true, attrs);

    NpuTensor in0{nullptr, {kM, kK}, "bfloat16"};
    NpuTensor in1{nullptr, {kM, kK}, "bfloat16"};
    NpuTensor out{nullptr, {kM, kK}, "bfloat16"};
    std::vector<NpuTensor> ins{in0, in1}, outs{out};
    std::map<std::string, std::any> shape_attr;
    kernel->set_tensor_shape(ins, outs, shape_attr);

    // --- The untested combination: bind_bo() directly on HIP-pinned
    // memory, not on a RyzenMM-allocated buffer. ---
    const auto t0 = std::chrono::steady_clock::now();
    xrt::bo lhs_bo = kernel->bind_bo(lhs_ptr, kBytes);
    xrt::bo rhs_bo = kernel->bind_bo(rhs_ptr, kBytes);
    xrt::bo out_bo = kernel->bind_bo(out_ptr, kBytes);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("bind_bo() x3 on HIP-pinned memory: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Aliasing check: the BO's own view of the buffer must be the address
    // HIP handed out, not a copy XRT made internally on bind.
    if (lhs_bo.map() != lhs_ptr) {
      std::fprintf(stderr,
                   "FAIL: aliasing -- lhs_bo.map() (%p) != hipHostMalloc "
                   "pointer (%p). bind_bo() copied instead of registering.\n",
                   lhs_bo.map(), lhs_ptr);
      ok = false;
    }

    // Write a recognizable pattern from the CPU.
    auto* lhs_u16 = static_cast<uint16_t*>(lhs_ptr);
    auto* rhs_u16 = static_cast<uint16_t*>(rhs_ptr);
    for (size_t i = 0; i < kElems; ++i) {
      lhs_u16[i] = npu_spike::FloatToBf16(static_cast<float>(i) + 1.0f);
      rhs_u16[i] = npu_spike::FloatToBf16(2.0f);
    }
    lhs_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    rhs_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<NPUBufferSpan> npu_in = {{lhs_bo, 0, lhs_bo.size()},
                                         {rhs_bo, 0, rhs_bo.size()}};
    std::vector<NPUBufferSpan> npu_out = {{out_bo, 0, out_bo.size()}};
    kernel->run(npu_in, npu_out);
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Read the NPU's output through the ORIGINAL hipHostMalloc pointer, not
    // through out_bo.map() -- proving the NPU wrote through the same
    // physical memory HIP owns, not a side buffer DD/XRT copied out to.
    auto* out_u16 = static_cast<uint16_t*>(out_ptr);
    for (size_t i = 0; i < kElems; ++i) {
      const float expect = (static_cast<float>(i) + 1.0f) * 2.0f;
      const float got = npu_spike::Bf16ToFloat(out_u16[i]);
      if (std::fabs(got - expect) > 1e-1f) {
        std::fprintf(stderr,
                     "FAIL: element %zu = %f, expected %f (NPU output not "
                     "observed through the HIP pointer)\n",
                     i, got, expect);
        ok = false;
        break;
      }
    }
    std::printf("CPU write -> NPU compute -> CPU read (no copy): %s\n",
                ok ? "PASS" : "FAIL");

    // Close the loop: a HIP kernel mutates the NPU's output in place, and
    // the CPU reads that mutation with no explicit copy either. This is
    // the same gpu_touch.hip helper alloc_probe.cpp uses for its CPU<->GPU
    // check; here it runs on a buffer the NPU just wrote.
    constexpr uint16_t kDelta = 0x0001;
    if (!GpuBumpBf16InPlace(out_ptr, kElems, kDelta)) {
      std::fprintf(stderr, "FAIL: GpuBumpBf16InPlace on NPU output\n");
      ok = false;
    } else {
      bool gpu_ok = true;
      for (size_t i = 0; i < kElems; ++i) {
        const uint16_t before =
            npu_spike::FloatToBf16((static_cast<float>(i) + 1.0f) * 2.0f);
        const uint16_t expected = static_cast<uint16_t>(before + kDelta);
        if (out_u16[i] != expected) {
          gpu_ok = false;
          break;
        }
      }
      std::printf(
          "NPU write -> GPU kernel mutates in place -> CPU read (no "
          "copy): %s\n",
          gpu_ok ? "PASS" : "FAIL");
      ok = ok && gpu_ok;
    }

    // --- Does sync() move bytes? (header note 5) ---
    // Deliberately measured on its own large buffer rather than on the
    // compute path above: that path's tensors are 128 bytes, where a copy and
    // a no-op are indistinguishable. A copy's cost scales with size, so the
    // question is only answerable at a size where copying would be visible.
    // No kernel runs on this buffer; bind_bo + sync is the whole probe.
    constexpr size_t kSyncBytes = 64u * 1024u * 1024u;
    void* big_ptr = nullptr;
    if (CheckHip(hipHostMalloc(&big_ptr, kSyncBytes,
                               hipHostMallocMapped | hipHostMallocCoherent),
                 "hipHostMalloc(sync probe)")) {
      double best_to = 1.0e30, best_from = 1.0e30;
      {
        // Scoped so the BO's destructor releases the registration before the
        // pages below it are freed, without assuming xrt::bo is
        // default-constructible or reassignable.
        xrt::bo big_bo = kernel->bind_bo(big_ptr, kSyncBytes);
        std::memset(big_ptr, 0xA5, kSyncBytes);  // fault the pages in first

        // Best of several: a single timing on a fresh mapping measures page
        // faults and first-touch cost rather than steady-state sync.
        for (int rep = 0; rep < 5; ++rep) {
          const auto a0 = std::chrono::steady_clock::now();
          big_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          const auto a1 = std::chrono::steady_clock::now();
          big_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          const auto a2 = std::chrono::steady_clock::now();
          best_to = std::min(
              best_to,
              std::chrono::duration<double, std::milli>(a1 - a0).count());
          best_from = std::min(
              best_from,
              std::chrono::duration<double, std::milli>(a2 - a1).count());
        }
      }

      std::printf("\n=== sync() cost on %zu MiB (does it copy?) ===\n",
                  kSyncBytes / (1024u * 1024u));
      std::printf("  sync(TO_DEVICE)  : %8.3f ms  -> %7.1f GB/s implied\n",
                  best_to, kSyncBytes / (best_to * 1.0e6));
      std::printf("  sync(FROM_DEVICE): %8.3f ms  -> %7.1f GB/s implied\n",
                  best_from, kSyncBytes / (best_from * 1.0e6));
      std::printf(
          "  Interpretation: a figure in the range of this part's DRAM\n"
          "  bandwidth means XRT moved the bytes, so zero copy is false even\n"
          "  though every value above compared equal. A cost far below that,\n"
          "  or independent of size, is cache maintenance and is fine.\n"
          "  This is NOT asserted here -- it is a measurement to record\n"
          "  against Decision 3. See the design doc's open question.\n");

      hipHostFree(big_ptr);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    ok = false;
  }

  hipHostFree(lhs_ptr);
  hipHostFree(rhs_ptr);
  hipHostFree(out_ptr);

  std::printf("\n=== npu_probe (Candidate A): %s ===\n",
              ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
