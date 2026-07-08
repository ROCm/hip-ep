/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * AMDMLSS MHA fast-path shim.
 *
 * Bridges the EP's MultiHeadAttention runtime to a pre-tuned AMDMLSS gfx1151
 * CK-WMMA attention code object for small sequences (see the seq-based
 * dispatch in lib/Runtime/real/multi_head_attention.cpp and the crossover in
 * AMDMLSS-main/docs/gfx1151_benchmark.md).
 *
 * It reuses the AMDMLSS C API to emit the code object, then loads and launches
 * it with raw HIP on the caller's stream (the same technique validated by the
 * standalone mlss-bench harness). Per-shape state (module, function, launch
 * geometry, self-describing arg list, double-pointer slots) is cached.
 */

#include <hip/hip_runtime.h>

#include <amdmlss/amdmlss_api.h>

#include "hip_custom_kernels.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ArgDesc {
  int32_t place;
  MLSSenum type;
  bool isPointer;
  uint32_t indirection;
  std::string name;
};

std::vector<ArgDesc> readArgs(const MLSSbinary &bin) {
  MLSSvoid *raw = nullptr;
  MLSSsize n = 0;
  MLSSenum t = 0;
  std::vector<ArgDesc> out;
  if (mlssVectorRetrieveData(bin.m_argList, &raw, &n, &t) != MLSS_SUCCESS || !raw)
    return out;
  const MLSSarg *a = static_cast<const MLSSarg *>(raw);
  for (MLSSsize i = 0; i < n; ++i)
    out.push_back({a[i].m_place, a[i].m_type, static_cast<bool>(a[i].m_isPointer),
                   a[i].m_indirectionLevel, a[i].m_name ? a[i].m_name : ""});
  for (size_t i = 1; i < out.size(); ++i) { // insertion sort by place (tiny n)
    ArgDesc key = out[i];
    size_t j = i;
    while (j > 0 && out[j - 1].place > key.place) { out[j] = out[j - 1]; --j; }
    out[j] = key;
  }
  return out;
}

size_t typeSize(MLSSenum t) {
  switch (t) {
  case MLSS_BOOL: case MLSS_INT8: case MLSS_UINT8: return 1;
  case MLSS_INT16: case MLSS_UINT16: case MLSS_FLOAT16: case MLSS_BFLOAT16: return 2;
  case MLSS_INT32: case MLSS_UINT32: case MLSS_FLOAT32: case MLSS_ENUM: return 4;
  case MLSS_INT64: case MLSS_UINT64: case MLSS_FLOAT64: case MLSS_ENUM64: return 8;
  default: return 4;
  }
}

struct PtrArg {
  std::string name;   // "Q"/"K"/"V"/"output"
  void *slot = nullptr;      // device slot holding the buffer pointer (indir>=2)
  size_t kargOff = 0;        // byte offset in karg where this arg's value sits
  bool indirect = false;     // indirection >= 2 (double pointer)
  void *lastBuf = nullptr;   // last buffer pointer staged into slot
};
struct ScalarArg {
  std::string name;
  size_t kargOff = 0;
  size_t size = 0;
};

struct Entry {
  bool usable = false;
  hipModule_t mod{};
  hipFunction_t fn{};
  MLSSdim3 grid{}, blocks{};
  uint64_t lds = 0;
  std::vector<uint8_t> karg;        // prebuilt kernarg buffer (layout fixed per shape)
  std::vector<PtrArg> ptrArgs;      // pointer args + their slots/offsets
  std::vector<ScalarArg> scalarArgs;
};

std::mutex g_mu;
std::unordered_map<uint64_t, Entry> g_cache;

uint64_t shapeKey(int64_t N, int64_t Sq, int64_t Skv, int64_t H) {
  // pack into 64 bits: N(12) Sq(20) Skv(20) H(12)
  return (uint64_t(N & 0xFFF) << 52) | (uint64_t(Sq & 0xFFFFF) << 32) |
         (uint64_t(Skv & 0xFFFFF) << 12) | uint64_t(H & 0xFFF);
}

// Build/compile/load the AMDMLSS MHA code object for a shape. Caller holds g_mu.
Entry buildEntry(int64_t B, int64_t N, int64_t Sq, int64_t Skv, int64_t H,
                 float scale) {
  Entry e;
  MLSScontext ctx = 0;
  MLSSstring op = const_cast<MLSSstring>(MLSS_MHA);
  if (mlssCreateContext(&ctx, const_cast<MLSSstring>(MLSS_GFXAUTOFIND), op) != MLSS_SUCCESS)
    return e;
  uint32_t b = (uint32_t)B, h = (uint32_t)N, qs = (uint32_t)Sq, kvs = (uint32_t)Skv,
           hd = (uint32_t)H, kv0 = 0, packing = MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
  MLSSenum dt = MLSS_FLOAT16;
  auto S = [&](MLSSenum a, const void *v) { mlssSetParameterByEnum(&ctx, op, a, v); };
  S(MLSS_ATTR_MHA_BATCH, &b); S(MLSS_ATTR_MHA_QSEQ, &qs); S(MLSS_ATTR_MHA_KVSEQ, &kvs);
  S(MLSS_ATTR_MHA_KDIM, &kv0); S(MLSS_ATTR_MHA_VDIM, &kv0); S(MLSS_ATTR_MHA_SIZEHEADS, &hd);
  S(MLSS_ATTR_MHA_PACKING, &packing); S(MLSS_ATTR_MHA_HEADCOUNT, &h);
  S(MLSS_ATTR_MHA_SCALE, &scale); S(MLSS_ATTR_MHA_DATATYPE, &dt);

  MLSSstatus *st = nullptr; MLSSsize nst = 0;
  if (mlssGetCaps(ctx, &st, &nst) != MLSS_SUCCESS) return e;
  MLSSbinary *bins = nullptr; MLSSsize nb = 0;
  if (mlssGetBinaries(ctx, &bins, &nb) != MLSS_SUCCESS || nb == 0) return e;

  // Prefer non-relocatable with the fewest args (contiguous no-strides variant).
  const MLSSbinary *bin = nullptr; size_t bestArgs = SIZE_MAX;
  for (MLSSsize i = 0; i < nb; ++i) {
    if (static_cast<bool>(bins[i].m_isRelocatable)) continue;
    size_t na = readArgs(bins[i]).size();
    if (na < bestArgs) { bestArgs = na; bin = &bins[i]; }
  }
  if (!bin) return e;

  if (hipModuleLoadData(&e.mod, bin->m_binaries) != hipSuccess) return e;
  if (hipModuleGetFunction(&e.fn, e.mod, bin->m_pKernelName) != hipSuccess) {
    hipModuleUnload(e.mod); return e;
  }
  e.grid = bin->m_grid; e.blocks = bin->m_blocks; e.lds = bin->m_sharedMemInBytes;

  // Prebuild the kernarg buffer once. Pointer args store the device *slot*
  // address (stable); the buffer pointer is staged into the slot per-call
  // (async, only when it changes). Scalar values are patched per-call (host).
  auto align = [&](size_t a) { while (e.karg.size() % a) e.karg.push_back(0); };
  for (auto &a : readArgs(*bin)) {
    if (a.isPointer) {
      void *slot = nullptr;
      if (a.indirection >= 2) {
        if (hipMalloc(&slot, sizeof(void *)) != hipSuccess) { hipModuleUnload(e.mod); return e; }
      }
      align(sizeof(void *));
      PtrArg pa; pa.name = a.name; pa.slot = slot; pa.kargOff = e.karg.size();
      pa.indirect = (a.indirection >= 2);
      e.ptrArgs.push_back(pa);
      e.karg.resize(e.karg.size() + sizeof(void *), 0);
      // For indirect args the kernarg holds the (stable) slot address.
      if (slot) std::memcpy(&e.karg[pa.kargOff], &slot, sizeof(void *));
    } else {
      size_t sz = typeSize(a.type);
      align(sz);
      ScalarArg sa; sa.name = a.name; sa.kargOff = e.karg.size(); sa.size = sz;
      e.scalarArgs.push_back(sa);
      e.karg.resize(e.karg.size() + sz, 0);
    }
  }
  while (e.karg.size() % 8) e.karg.push_back(0);
  e.usable = true;
  return e;
}

} // namespace

extern "C" HIP_KERNEL_API long long hipdnn_ep_amdmlss_max_seq(void) {
  const char *env = std::getenv("HIPDNN_MLSS_MAX_SEQ");
  if (env && *env) {
    char *end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) return v;
  }
  // Default OFF (0) for the PREFILL gate: in-EP the AMDMLSS "fallback" kernel is
  // latency-bound for multi-token prefill (only B*H workgroups) and loses to the
  // EP's sequence-tiled path. Opt in with HIPDNN_MLSS_MAX_SEQ=<n> to route
  // prefill with Sq,Skv <= n. Decode (Sq==1) is gated separately below.
  return 0;
}

namespace {
// Decode gate: route single-query (Sq==1) attention when Skv >= this value.
// Decode is where the compact per-head AMDMLSS kernel wins in-EP (the EP's
// decode path scales worse with context). Default 512 (measured crossover on
// gfx1151); set HIPDNN_MLSS_DECODE_MIN_KV=0 to disable decode routing.
long long decodeMinKv() {
  const char *env = std::getenv("HIPDNN_MLSS_DECODE_MIN_KV");
  if (env && *env) {
    char *end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) return v;
  }
  return 512;
}
} // namespace

extern "C" HIP_KERNEL_API int
hipdnn_ep_amdmlss_mha(void *stream, const void *q, const void *k, const void *v,
                      void *out, long long batch, long long num_heads,
                      long long seq_q, long long seq_kv, long long head_dim,
                      float scale) {
  // Two independent gates: prefill (Sq,Skv <= max_seq; default off) and decode
  // (Sq==1 && Skv >= decode_min_kv; default on for Skv>=512, the measured win).
  const long long maxSeq = hipdnn_ep_amdmlss_max_seq();
  const long long dmk = decodeMinKv();
  const bool prefillOk = (maxSeq > 0 && seq_q <= maxSeq && seq_kv <= maxSeq);
  const bool decodeOk = (dmk > 0 && seq_q == 1 && seq_kv >= dmk);
  if (!prefillOk && !decodeOk) return 1; // let the EP handle it

  std::lock_guard<std::mutex> lock(g_mu);
  uint64_t key = shapeKey(num_heads, seq_q, seq_kv, head_dim);
  auto it = g_cache.find(key);
  if (it == g_cache.end()) {
    Entry e = buildEntry(batch, num_heads, seq_q, seq_kv, head_dim, scale);
    it = g_cache.emplace(key, std::move(e)).first;
  }
  Entry &e = it->second;
  if (!e.usable) return 1;
  hipStream_t hstream = reinterpret_cast<hipStream_t>(stream);

  // Patch scalar values into the prebuilt kernarg (host-side, cheap). Values
  // are fixed per shape, but re-patching keeps correctness if scale varies.
  auto patchScalar = [&](const char *n, int32_t val) {
    for (auto &s : e.scalarArgs) if (s.name == n) { std::memcpy(&e.karg[s.kargOff], &val, (s.size < 4 ? s.size : 4)); }
  };
  patchScalar("batch_size", (int32_t)batch);
  patchScalar("q_sequence_length", (int32_t)seq_q);
  patchScalar("kv_sequence_length", (int32_t)seq_kv);
  patchScalar("sequence_length", (int32_t)seq_q);
  patchScalar("head_num", (int32_t)num_heads);
  patchScalar("head_dim", (int32_t)head_dim);
  for (auto &s : e.scalarArgs) if (s.name == "scale") std::memcpy(&e.karg[s.kargOff], &scale, 4);

  // Stage buffer pointers. For double-pointer args write the buffer ptr into
  // the device slot (async, on the EP stream) only when it changed; for direct
  // pointers patch the kernarg bytes. No synchronous copies on the hot path.
  for (auto &pa : e.ptrArgs) {
    void *buf = nullptr;
    if (pa.name == "Q") buf = const_cast<void *>(q);
    else if (pa.name == "K") buf = const_cast<void *>(k);
    else if (pa.name == "V") buf = const_cast<void *>(v);
    else if (pa.name == "output") buf = out;
    else return 1;
    if (pa.indirect) {
      if (buf != pa.lastBuf) {
        if (hipMemcpyAsync(pa.slot, &buf, sizeof(void *), hipMemcpyHostToDevice, hstream) != hipSuccess)
          return 1;
        pa.lastBuf = buf;
      }
      // kernarg already holds the stable slot address.
    } else {
      std::memcpy(&e.karg[pa.kargOff], &buf, sizeof(void *));
    }
  }

  size_t kargSize = e.karg.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, e.karg.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &kargSize,
                    HIP_LAUNCH_PARAM_END};
  hipError_t le = hipModuleLaunchKernel(
      e.fn, e.grid.m_x, e.grid.m_y, e.grid.m_z, e.blocks.m_x, e.blocks.m_y,
      e.blocks.m_z, (unsigned)e.lds, hstream,
      nullptr, reinterpret_cast<void **>(&config));
  if (le != hipSuccess) {
    fprintf(stderr, "[mlss_mha_shim] launch failed: %s\n", hipGetErrorString(le));
    return 1;
  }
  return 0;
}
