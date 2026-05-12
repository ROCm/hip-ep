/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_layer_normalization(RuntimeState *state, void *input, void *scale,
                             void *bias, void *output, void *mean,
                             void *inv_std, int64_t input_num_elements,
                             int64_t scale_num_elements,
                             int64_t element_size_bytes, int64_t axis,
                             float epsilon, int64_t stash_type) {
  // TODO: implement full LayerNormalization via MIOpen
  (void)state;
  (void)input;
  (void)scale;
  (void)bias;
  (void)output;
  (void)mean;
  (void)inv_std;
  (void)input_num_elements;
  (void)scale_num_elements;
  (void)element_size_bytes;
  (void)axis;
  (void)epsilon;
  (void)stash_type;
  fprintf(stderr,
          "[STUB] wrap_layer_normalization: not yet implemented "
          "(input_num_elements=%lld, scale_num_elements=%lld, axis=%lld, "
          "epsilon=%f, stash_type=%lld, bias=%s, mean=%s, inv_std=%s)\n",
          (long long)input_num_elements, (long long)scale_num_elements,
          (long long)axis, epsilon, (long long)stash_type,
          bias ? "yes" : "null", mean ? "yes" : "null",
          inv_std ? "yes" : "null");
  return -1;
}
