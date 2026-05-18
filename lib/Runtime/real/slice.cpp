/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

#include <cstdio>

// Stub implementation: the OnnxToHip decompose pattern rewrites the common
// case (compile-time constant indices + positive unit stride) to
// tensor.extract_slice, so this entry point only fires for unsupported
// configurations (non-constant indices or negative steps). The stub only
// logs its parameters and returns success so models that exercise this
// path can still link and run end-to-end (with incorrect Slice output) for
// development and IR-shape debugging.
int wrap_slice(RuntimeState *state, void *data, void *starts, void *ends,
               void *axes, void *steps, void *output,
               const int64_t *data_shape, int64_t data_rank,
               const int64_t *output_shape, int64_t output_rank,
               int64_t starts_num_elements, int64_t axes_num_elements,
               int64_t steps_num_elements, int64_t data_type) {
  (void)state;
  (void)data;
  (void)starts;
  (void)ends;
  (void)axes;
  (void)steps;
  (void)output;
  (void)data_shape;
  (void)output_shape;

  std::fprintf(stderr,
               "[STUB] wrap_slice("
               "data_rank=%lld, output_rank=%lld, starts_n=%lld, axes_n=%lld, "
               "steps_n=%lld, data_type=%s(%lld))\n",
               (long long)data_rank, (long long)output_rank,
               (long long)starts_num_elements, (long long)axes_num_elements,
               (long long)steps_num_elements,
               hipdnn_ep_datatype_name(data_type), (long long)data_type);
  std::fflush(stderr);
  return 0;
}
