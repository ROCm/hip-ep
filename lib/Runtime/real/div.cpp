/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

int wrap_div(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t num_elements, int64_t data_type) {
  (void)state;
  (void)lhs;
  (void)rhs;
  (void)output;
  (void)num_elements;
  (void)data_type;
  return 0;
}
