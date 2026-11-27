/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cstdint>
#include <string>

// Standalone UT-only controls; never part of the production kernel ABI.
struct UtGemmShape {
  int m, n, k, ta, tb, dtype, crows, ccols;
  float alpha, beta;
  int output = -1, batches = 1, a_group = 1, b_group = 1;
  int64_t stride_a = -1, stride_b = -1, stride_c = -1, stride_y = -1;
};
struct UtGemmConfig {
  int id = -1;
  std::string geometry;
  std::string launch;
};
extern "C" int hipdnn_ut_gemm_config(const UtGemmShape*, int, UtGemmConfig*);
extern "C" int hipdnn_ut_gemm_config_count(const UtGemmShape*);
extern "C" int hipdnn_ut_gemm_select(const UtGemmShape*, const UtGemmConfig*);
