/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// wrap_matmul_nbits keeps the signature that model artifacts compiled before
// wrap_matmul_nbits_ex call: the model cache key covers the graph, not the
// runtime ABI, so such an artifact can meet a newer runtime. This calls the
// old 18-argument entry point on the mock runtime; CTest checks the output
// (PASS_REGULAR_EXPRESSION) for the forwarded wrap_matmul_nbits_ex call with
// b_row_bytes = 0, the arrival layout those artifacts were compiled for.

#include "hipdnn_ep_runtime.h"

#include <cstdio>

extern "C" {
// The two state accessors mock_gpu.cpp references; the matmul entry points
// under test do not reach them.
void *hipdnn_ep_state_get_stream(RuntimeState *) { return nullptr; }
int hipdnn_ep_state_set_error_flag(RuntimeState *, int) { return 0; }
}

int main() {
  alignas(16) unsigned char state[64] = {};
  alignas(16) unsigned char tensor[64] = {};
  auto *s = reinterpret_cast<RuntimeState *>(state);
  int rc = wrap_matmul_nbits(s, /*op_state_slot=*/0, tensor, tensor, tensor,
                             /*zero_points=*/nullptr, /*g_idx=*/nullptr,
                             /*bias=*/nullptr, tensor, /*M=*/1, /*N=*/64,
                             /*K=*/64, /*batch_count=*/1, /*bits=*/4,
                             /*block_size=*/32, /*elem_size=*/2,
                             /*zp_elem_size=*/1, /*scale_elem_size=*/2);
  std::printf("wrap_matmul_nbits rc=%d\n", rc);
  return rc == 0 ? 0 : 1;
}
