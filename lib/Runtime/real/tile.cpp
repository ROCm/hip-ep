/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_tile(RuntimeState *state, void *input, void *repeats, void *output,
              const int64_t *input_shape, int64_t input_rank,
              const int64_t *output_shape, int64_t output_rank,
              int64_t data_type) {
  (void)state;
  (void)input;
  (void)repeats;
  (void)output;
  (void)input_shape;
  (void)input_rank;
  (void)output_shape;
  (void)output_rank;
  (void)data_type;
  return 0;
}
