/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_reduce_prod(RuntimeState *state, void *data, void *axes, void *output,
                     int64_t data_num_elements, int64_t output_num_elements,
                     int64_t axes_num_elements, int64_t data_type,
                     int64_t keepdims, int64_t noop_with_empty_axes) {
  (void)state;
  (void)data;
  (void)axes;
  (void)output;
  (void)data_num_elements;
  (void)output_num_elements;
  (void)axes_num_elements;
  (void)data_type;
  (void)keepdims;
  (void)noop_with_empty_axes;
  return 0;
}
