/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- hipdnn_ep_runtime_if.cpp - ONNX If driver ------------------------===//

#include "hipdnn_ep_runtime.h"

namespace {

typedef int (*HipdnnEpIfBranchFn)(RuntimeState *state, void **output_descs,
                                  void **capture_descs);

} // namespace

extern "C" int hipdnn_ep_run_if(RuntimeState *state, bool cond,
                                HipdnnEpIfBranchFn then_fn,
                                HipdnnEpIfBranchFn else_fn, int32_t num_outputs,
                                int32_t num_captures, void **output_descs,
                                void **capture_descs) {
  if (!state || !then_fn || !else_fn)
    return -1;
  HipdnnEpIfBranchFn branch = cond ? then_fn : else_fn;
  (void)num_outputs;
  (void)num_captures;
  return branch(state, output_descs, capture_descs);
}
