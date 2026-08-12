/*
 * Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Implementation of the GQA dispatch cost model. See gqa_cost_model.h for the
// model, its derivation and its measured accuracy.
//
// Pure host C++: no HIP, no device code, no device query. The CU count arrives
// through hip_gqa_shape_t so this stays callable from the runtime (tier 2 of
// gqa_autotune.cpp) and from the kernel TU alike, and so it can be unit-tested
// without a GPU.

#include "gqa_cost_model.h"

#include <cmath>
#include <cstddef>

namespace {

// ---------------------------------------------------------------------------
// Compile-time kernel resources. Regenerate after editing gqa_kernel.hip with:
//   hipcc -c -O3 -std=c++17 -x hip --offload-arch=<arch> -Iinclude \
//         -Rpass-analysis=kernel-resource-usage hip/gqa_kernel.hip -o /dev/null \
//         2> res.txt
//   python RdpCapture/ops_analyze/gqa/tools/parse_kernel_resources.py res.txt
// ---------------------------------------------------------------------------
struct KernelRes {
  int key;             // packed selector, see the lookup sites
  int scratch_bytes;   // per-lane spill; 0 means the config fits in registers
  int blocks_per_cu;   // concurrent blocks one CU holds (register or LDS bound)
  int waves_per_block; // block size / wavefront size
};

// v5 @ d=64. key = m_tiles*1000 + bkv*10 + has_window.
// MT1_BKV32 is the only instantiation that does not spill; MT2 buys half the
// blocks and 2x the KV reuse and pays 120-180 bytes/lane of scratch for it.
const KernelRes kResV5D64[] = {
    {1 * 1000 + 32 * 10 + 0,    0, 12, 1}, {1 * 1000 + 32 * 10 + 1,    0, 12, 1},
    {1 * 1000 + 64 * 10 + 0,   32,  6, 1}, {1 * 1000 + 64 * 10 + 1,   68,  6, 1},
    {2 * 1000 + 32 * 10 + 0,  120, 10, 1}, {2 * 1000 + 32 * 10 + 1,  180, 10, 1},
    {2 * 1000 + 64 * 10 + 0,  532,  5, 1}, {2 * 1000 + 64 * 10 + 1,  660,  5, 1},
};

// v7 @ d=128. key = num_waves*1000 + bkv*10 + m_tiles. Every one of these spills.
const KernelRes kResV7D128[] = {
    {1 * 1000 + 32 * 10 + 1,  108, 6, 1}, {1 * 1000 + 32 * 10 + 2,  760, 6, 1},
    {1 * 1000 + 64 * 10 + 1,  720, 3, 1}, {1 * 1000 + 64 * 10 + 2, 1272, 3, 1},
    {2 * 1000 + 32 * 10 + 1,  112, 6, 2}, {2 * 1000 + 32 * 10 + 2,  764, 5, 2},
    {2 * 1000 + 64 * 10 + 1,  724, 3, 2}, {2 * 1000 + 64 * 10 + 2, 1276, 2, 2},
    {4 * 1000 + 32 * 10 + 1,  112, 5, 4}, {4 * 1000 + 32 * 10 + 2,  764, 3, 4},
    {4 * 1000 + 64 * 10 + 1,  724, 2, 4}, {4 * 1000 + 64 * 10 + 2, 1276, 1, 4},
};

// v8 @ d=256. key = num_waves*1000 + m_tiles*100 + bkv.
// ND4_MT1_BKV32 is the only spill-free one (VGPR 188 vs 256 elsewhere), which is
// why it wins every d=256 shape measured.
const KernelRes kResV8D256[] = {
    {2 * 1000 + 1 * 100 + 32,  124, 3, 2}, {2 * 1000 + 1 * 100 + 64,  728, 1, 2},
    {2 * 1000 + 2 * 100 + 32,  816, 2, 2}, {2 * 1000 + 2 * 100 + 64, 1412, 1, 2},
    {4 * 1000 + 1 * 100 + 32,    0, 2, 4}, {4 * 1000 + 2 * 100 + 32,  324, 1, 4},
};

// Decode WMMA. key = head_dim*100 + heads_per_group. Only these are templated;
// every other geometry runs the HpG-generic scalar kernel.
const KernelRes kResDecodeWmma[] = {
    { 64 * 100 + 4, 136, 10, 1},
    { 64 * 100 + 8, 136, 10, 1},
    {128 * 100 + 4,   0,  3, 1},
};

// Fitted coefficients, indexed by hip_gqa_path_t:
//   {c0 fixed, c1 math, c2 kv, c3 spill/trip, c4 spill/block, c5 reduce}
// c5 applies to the decode reduce only. Produced by
// RdpCapture/ops_analyze/gqa/tools/fit_cost_model.py --emit
const double kCoef[4][6] = {
    /* v5     */ {3.39370e-04, 9.54993e-08, 1.86066e-10, 5.25459e-07, 8.77615e-06, 0.0},
    /* v7     */ {3.10516e-05, 4.28160e-11, 3.43113e-12, 1.19215e-08, 2.54788e-09, 0.0},
    /* v8     */ {3.90956e-07, 2.08697e-13, 1.53545e-14, 1.81272e-08, 5.15129e-07, 0.0},
    /* decode */ {1.04935e-04, 5.23006e-06, 2.92390e-05, 3.60196e-08, 6.47168e-07, 4.99497e-05},
};

// Both decode kernels walk the KV cache 16 keys at a time.
constexpr int kDecodeKeysPerStep = 16;
constexpr int kDefaultCuCount = 20;

const KernelRes *lookup(const KernelRes *tab, size_t n, int key) {
  for (size_t i = 0; i < n; ++i)
    if (tab[i].key == key) return &tab[i];
  return nullptr;
}

template <size_t N>
const KernelRes *lookup(const KernelRes (&tab)[N], int key) {
  return lookup(tab, N, key);
}

}  // namespace

HIP_GQA_COST_MODEL_API double hip_gqa_config_score(
    const hip_gqa_shape_t *shape, const hip_gqa_config_t *cand) {
  if (!shape || !cand) return 0.0;
  if (shape->kv_heads <= 0 || shape->num_heads % shape->kv_heads != 0) return 0.0;

  const int cus = shape->cu_count > 0 ? shape->cu_count : kDefaultCuCount;
  const int hpg = shape->num_heads / shape->kv_heads;
  const int d = shape->head_dim;
  const int sq = shape->q_len;
  const int skv = shape->kv_len;
  const int past = skv - sq;
  const int win = shape->window;
  const int eff_kv = (win > 0 && win < skv) ? win : skv;

  const KernelRes *res = nullptr;
  double blocks = 0.0;    // thread blocks the launch creates
  double kv_tiles = 0.0;  // KV tiles ONE block walks, averaged over blocks
  int tile_w = 0;         // KV keys per tile
  double extra = 0.0;     // path-specific additive cost (decode reduce)
  KernelRes scalar_decode{};

  if (cand->path == HIP_GQA_PATH_DECODE) {
    if (cand->splits < 1) return 0.0;
    if (cand->use_wmma) {
      res = lookup(kResDecodeWmma, d * 100 + hpg);
      if (!res) return 0.0;  // WMMA is not templated for this geometry
    } else {
      // The scalar kernel is HpG-generic: one wave per query head, so the block
      // is HpG waves and occupancy follows from that alone.
      scalar_decode.key = 0;
      scalar_decode.scratch_bytes = 0;
      scalar_decode.waves_per_block = hpg;
      scalar_decode.blocks_per_cu = (hpg == 1) ? 64 : (hpg >= 16 ? 4 : 8);
      res = &scalar_decode;
    }
    tile_w = kDecodeKeysPerStep;
    blocks = static_cast<double>(shape->batch) * shape->kv_heads * cand->splits;
    const double span = std::ceil(static_cast<double>(eff_kv) / cand->splits);
    kv_tiles = std::ceil(span / tile_w);
    if (kv_tiles < 1.0) kv_tiles = 1.0;
    // FA-2 reduce: one block per query head, each scanning `splits` partials.
    const int red_bpc = (d == 64) ? 32 : (d == 128 ? 16 : 8);
    extra = std::ceil(static_cast<double>(shape->batch) * shape->num_heads /
                      (cus * red_bpc)) *
            cand->splits;
  } else {
    int rows = 0;  // query rows one block owns
    if (cand->path == HIP_GQA_PATH_PREFILL_V5) {
      if (d != 64) return 0.0;
      rows = cand->m_tiles * 16;
      res = lookup(kResV5D64,
                   cand->m_tiles * 1000 + cand->bkv * 10 + (win > 0 ? 1 : 0));
    } else if (cand->path == HIP_GQA_PATH_PREFILL_V7) {
      if (d != 128) return 0.0;
      rows = cand->num_waves * cand->m_tiles * 16;
      res = lookup(kResV7D128,
                   cand->num_waves * 1000 + cand->bkv * 10 + cand->m_tiles);
    } else if (cand->path == HIP_GQA_PATH_PREFILL_V8) {
      if (d != 256) return 0.0;
      rows = cand->m_tiles * 16;
      res = lookup(kResV8D256,
                   cand->num_waves * 1000 + cand->m_tiles * 100 + cand->bkv);
    } else {
      return 0.0;
    }
    if (!res || rows <= 0 || cand->bkv <= 0) return 0.0;

    tile_w = cand->bkv;
    const int q_tiles = (sq + rows - 1) / rows;
    blocks = static_cast<double>(q_tiles) * shape->num_heads * shape->batch;

    // Causal masking means later query tiles walk more of the cache, and a
    // window truncates the front of that range. Summing the per-tile spans is
    // what keeps chunked prefill (large past) from being scored like a fresh one.
    double sum_tiles = 0.0;
    for (int i = 0; i < q_tiles; ++i) {
      const int kv_end_i = past + (i + 1) * rows;
      const int kv_end = kv_end_i < skv ? kv_end_i : skv;
      int kv_lo = 0;
      if (win > 0) {
        kv_lo = past + i * rows + 1 - win;
        if (kv_lo < 0) kv_lo = 0;
      }
      const int span = kv_end - kv_lo;
      if (span > 0) sum_tiles += std::ceil(static_cast<double>(span) / tile_w);
    }
    kv_tiles = sum_tiles / (q_tiles > 0 ? q_tiles : 1);
  }

  if (!res || res->blocks_per_cu <= 0) return 0.0;

  const double resident = static_cast<double>(cus) * res->blocks_per_cu;
  const double waves = std::ceil(blocks / resident);
  const double *c = kCoef[static_cast<int>(cand->path)];

  const double per_block =
      c[0] +
      c[1] * kv_tiles * tile_w * d / static_cast<double>(res->waves_per_block) +
      c[2] * kv_tiles * tile_w * d +
      c[3] * kv_tiles * res->scratch_bytes +
      c[4] * res->scratch_bytes;

  const double cost = waves * per_block + c[5] * extra;
  if (!(cost > 0.0)) return 0.0;

  // Useful work is a property of the shape, not of the config, so dividing by
  // the predicted cost turns the ranking into a throughput: same ordering, but
  // the number also says how efficiently this shape runs at all.
  const double q_rows = (sq > 0) ? static_cast<double>(sq) : 1.0;
  const double macs = static_cast<double>(shape->batch) * shape->num_heads *
                      q_rows * static_cast<double>(eff_kv) *
                      static_cast<double>(d) * 2.0;
  return macs / cost;
}
