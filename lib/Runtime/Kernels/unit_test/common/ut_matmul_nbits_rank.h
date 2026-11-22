/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Standalone UT-only controls, never linked into the production kernel DLL.
// The adapter compiles the unchanged production source and adjusts its existing
// in-process selection maps. Calls still enter hip_matmul_nbits, including all
// fp32 cast, zero-point conversion, K-pad, transpose and dequant wrappers.
struct UtNbitsShape { int m, n, k, bits, gs, zero, zp_bytes; };
struct UtNbitsConfig {
  int route = 0;  // 1 WMMA, 2 GEMV, 3 M-blocked heuristic, 4 naive
  int id = -1;
  int bm = 0, bn = 0, swizzle = 0, wt_m = 0, wt_n = 0, bk = 0;
  int fused = 1, ext = 0, variant = 0, threads = 0, tile_n = 0, tile_m = 0;
};
extern "C" int hipdnn_ut_nbits_config(const UtNbitsShape*, int id, UtNbitsConfig*);
extern "C" int hipdnn_ut_nbits_select(const UtNbitsShape*, const UtNbitsConfig*);
extern "C" int hipdnn_ut_nbits_config_count(const UtNbitsShape*);
