/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_pad(RuntimeState *state, void *data, void *pads, void *constant_value,
             void *axes, void *output, const int64_t *data_shape,
             int64_t data_rank, const int64_t *output_shape,
             int64_t output_rank, int64_t pads_num_elements,
             int64_t axes_num_elements, int64_t data_type, int64_t mode_id) {
  (void)state;
  (void)data;
  (void)pads;
  (void)constant_value;
  (void)axes;
  (void)output;
  (void)data_shape;
  (void)data_rank;
  (void)output_shape;
  (void)output_rank;
  (void)pads_num_elements;
  (void)axes_num_elements;
  (void)data_type;
  (void)mode_id;
  return 0;
}
