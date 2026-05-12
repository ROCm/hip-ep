/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_equal(RuntimeState *state, void *a, void *b, void *output,
               int64_t num_elements, int64_t data_type) {
  (void)state;
  (void)a;
  (void)b;
  (void)output;
  (void)num_elements;
  (void)data_type;
  return 0;
}
