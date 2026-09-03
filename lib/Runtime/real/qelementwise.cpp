/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A,B) = Q(DQ(A) [OP] DQ(B))
// OP belongs to {+,-,*,/}

// a = (A - z_a) * S
// b = (B - z_b) * s_b
// OUT = round( (a [OP] b) * s_out ) + z_out
// --->
// let M_a = s_a/s_out, M_b = s_b/s_out
// OUT = round( (M_a * (A - z_a) [OP] M_b * (B - z_b)) + z_out
#include "../hipdnn_ep_runtime.h"

int wrap_qelementwise(RuntimeState *state, void *lhs, void *rhs, void *output,
                     int64_t kind, const int64_t *lhs_shape, int64_t lhs_rank,
                     const int64_t *rhs_shape, int64_t rhs_rank,
                     const int64_t *out_shape, int64_t out_rank,
                     int64_t data_type, float M_a, int64_t lhs_zp, float M_b,
                     int64_t rhs_zp, int64_t output_zp) {
  (void)state;
  (void)lhs;
  (void)rhs;
  (void)output;
  (void)kind;
  (void)lhs_shape;
  (void)lhs_rank;
  (void)rhs_shape;
  (void)rhs_rank;
  (void)out_shape;
  (void)out_rank;
  (void)data_type;
  (void)M_a;
  (void)lhs_zp;
  (void)M_b;
  (void)rhs_zp;
  (void)output_zp;
  return 0;
}
