// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "matmul_nbits_autotune.h"
#include <cstdio>
#include <initializer_list>

int main() {
  using namespace hipdnn_ep::matmul_nbits_autotune;
  for (auto phase : {Phase::DecodeDp4a, Phase::Prefill})
    for (int n : {1856, 2688}) {
      Request request;
      request.phase = phase;
      request.m = phase == Phase::Prefill ? 43 : 1;
      request.n = n;
      request.k = n == 1856 ? 2688 : 1856;
      request.group_size = 64;
      request.has_zp = true;
      auto result = resolve(
          request, [](void *, const WmmaAnswer &) { return true; },
          [](void *, const GemvAnswer &) { return true; }, nullptr);
      std::printf("N=%d K=%d source=%d threads=%d tile_n=%d distance=%.3f ",
                  request.n, request.k, int(result.source), result.gemv.threads,
                  result.gemv.tile_n, result.distance);
      std::printf("M=%d WMMA=%dx%d wt=%dx%d bk=%d fused=%d\n", request.m,
                  result.wmma.bm, result.wmma.bn, result.wmma.wt_m,
                  result.wmma.wt_n, result.wmma.bk, result.wmma.fused);
    }
}
