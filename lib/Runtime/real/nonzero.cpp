/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

#include <cstdio>
#include <stdexcept>
#include <string>

// Stub implementation of ONNX NonZero.
//
// NonZero returns the row-major indices of non-zero elements of `input`
// as an int64 tensor of shape [R, N], where R is the input rank and N
// is the (data-dependent) number of non-zero elements. There is no GPU
// kernel for this op yet -- the compiler still emits a call into this
// symbol so that models containing NonZero can compile end-to-end.
//
// Per the project convention for unimplemented runtime ops, this stub
// logs its parameters and throws a std::runtime_error so any inference
// path that actually executes the op fails loudly instead of silently
// producing uninitialised output.
int wrap_nonzero(RuntimeState *state, void *input, void *output,
                 int64_t input_num_elements, int64_t input_rank,
                 int64_t output_capacity, int64_t input_data_type) {
  (void)state;
  (void)input;
  (void)output;

  std::fprintf(stderr,
               "[STUB] wrap_nonzero(input=%p, output=%p, "
               "input_num_elements=%lld, input_rank=%lld, "
               "output_capacity=%lld, input_data_type=%s(%lld))\n",
               input, output, (long long)input_num_elements,
               (long long)input_rank, (long long)output_capacity,
               hipdnn_ep_datatype_name(input_data_type),
               (long long)input_data_type);
  std::fflush(stderr);
  return 0;
}
