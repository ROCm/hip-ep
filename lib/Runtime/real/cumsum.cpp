/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_cumsum(RuntimeState *state, void *x, void *axis, void *y,
                const int64_t *data_shape, int64_t data_rank,
                int64_t num_elements, int64_t data_type, int64_t axis_dtype,
                int64_t exclusive, int64_t reverse) {
  (void)state;
  (void)x;
  (void)axis;
  (void)y;
  (void)data_shape;
  (void)data_rank;
  (void)num_elements;
  (void)data_type;
  (void)axis_dtype;
  (void)exclusive;
  (void)reverse;
  return 0;
}
