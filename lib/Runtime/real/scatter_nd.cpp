/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

#include <cstdio>

// Stub implementation: ScatterND has no kernel today. The stub logs its
// parameters (data / indices / updates ranks and the resolved reduction
// string) and returns success so that pipelines exercising ScatterND can
// still link and run end-to-end for IR-shape debugging. Models that hit
// this path will produce uninitialised / garbage output — a real kernel
// must (a) copy `data` into `output`, then (b) for each index tuple in
// `indices`, write the corresponding `updates` slice into `output`
// combined via the requested `reduction` op.
static const char *reductionName(int64_t id) {
  switch (id) {
  case 0:
    return "none";
  case 1:
    return "add";
  case 2:
    return "mul";
  case 3:
    return "min";
  case 4:
    return "max";
  default:
    return "<unknown>";
  }
}

int wrap_scatter_nd(RuntimeState *state, void *data, void *indices,
                    void *updates, void *output,
                    const int64_t *data_shape, int64_t data_rank,
                    const int64_t *indices_shape, int64_t indices_rank,
                    const int64_t *updates_shape, int64_t updates_rank,
                    const int64_t *output_shape, int64_t output_rank,
                    int64_t reduction_id, int64_t data_type) {
  (void)state;
  (void)data;
  (void)indices;
  (void)updates;
  (void)output;
  (void)data_shape;
  (void)indices_shape;
  (void)updates_shape;
  (void)output_shape;

  std::fprintf(stderr,
               "[STUB] wrap_scatter_nd("
               "data_rank=%lld, indices_rank=%lld, updates_rank=%lld, "
               "output_rank=%lld, reduction=%s(%lld), data_type=%s(%lld))\n",
               (long long)data_rank, (long long)indices_rank,
               (long long)updates_rank, (long long)output_rank,
               reductionName(reduction_id), (long long)reduction_id,
               hipdnn_ep_datatype_name(data_type), (long long)data_type);
  std::fflush(stderr);
  return 0;
}
