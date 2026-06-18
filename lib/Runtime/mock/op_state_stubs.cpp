/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- op_state_stubs.cpp - Mock per-op state constructors --------------===//
//
// Mock-runtime counterparts of the per-op state constructors that the real
// runtime defines in each op's translation unit (e.g. MatmulState +
// hipdnn_ep_op_state_construct_matmul in real/matmul.cpp). The mock has no HIP,
// so each stub allocates a trivial OpState whose only job is to be non-null
// (so the generated op-states init succeeds) and to be freed generically by
// its deletor at cleanup. See docs/design/op-state-slots-design.md.
//
//===----------------------------------------------------------------------===//

#include "../op_state.h"

// MatMulNBits: real runtime owns a zero_points unpack cache (MatmulNbitsState
// in real/matmul_nbits.cpp); the mock owns no device memory.
extern "C" OpState *hipdnn_ep_op_state_construct_matmul_nbits(RuntimeState *) {
  return make_op_state<OpState>();
}

// MatMul: real runtime holds a shared_ptr to a device-wide hipBLASLt algo
// table (MatmulState in real/matmul.cpp); the mock owns no device/hipBLASLt
// resources.
extern "C" OpState *hipdnn_ep_op_state_construct_matmul(RuntimeState *) {
  return make_op_state<OpState>();
}

// CausalConvWithState: real runtime owns a per-shape MIOpen descriptor/algo
// cache (CausalConvState in real/causal_conv_with_state.cpp); the mock owns no
// device/MIOpen resources.
extern "C" OpState *
hipdnn_ep_op_state_construct_causal_conv_with_state(RuntimeState *) {
  return make_op_state<OpState>();
}

// GQA: real runtime owns a per-GEMM-shape hipBLASLt descriptor/algo cache
// (GqaState in real/gqa.cpp); the mock owns no device/hipBLASLt resources.
extern "C" OpState *hipdnn_ep_op_state_construct_gqa(RuntimeState *) {
  return make_op_state<OpState>();
}

// MultiHeadAttention: real runtime owns a per-GEMM-shape hipBLASLt
// descriptor/algo cache (MhaState in real/multi_head_attention.cpp); the mock
// owns no device/hipBLASLt resources.
extern "C" OpState *
hipdnn_ep_op_state_construct_multi_head_attention(RuntimeState *) {
  return make_op_state<OpState>();
}
