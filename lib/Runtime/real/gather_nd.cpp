/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_gather_nd(RuntimeState *state, void *data, void *indices, void *output,
                   const int64_t *data_shape, int64_t data_rank,
                   const int64_t *indices_shape, int64_t indices_rank,
                   const int64_t *output_shape, int64_t output_rank,
                   int64_t batch_dims, int64_t data_type) {
  (void)state;
  (void)data;
  (void)indices;
  (void)output;
  (void)data_shape;
  (void)data_rank;
  (void)indices_shape;
  (void)indices_rank;
  (void)output_shape;
  (void)output_rank;
  (void)batch_dims;
  (void)data_type;
  return 0;
}
