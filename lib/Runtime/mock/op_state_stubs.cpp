/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- op_state_stubs.cpp - Mock per-op state constructors --------------===//
//
// Mock-runtime counterparts of the per-op state constructors that the real
// runtime defines in each op's translation unit (e.g. MatmulState +
// hipdnn_ep_op_state_construct_matmul in real/matmul.cpp). The mock has no HIP,
// so each stub builds a trivial MockOpState whose only job is to occupy the
// slot (so the generated op-states init succeeds) and to be freed generically
// by its deletor at cleanup. See docs/design/op-state-slots-design.md.
//
//===----------------------------------------------------------------------===//

#include "../op_state.h"

namespace {
// Trivial state with no device resources; owns nothing beyond the OpState
// deletor wired by OpStateT. Used by every mock construct stub.
struct MockOpState : OpStateT<MockOpState> {};
} // namespace

// MatMulNBits: real runtime owns a zero_points unpack cache (MatmulNbitsState
// in real/matmul_nbits.cpp); the mock owns no device memory.
extern "C" int8_t hipdnn_ep_op_state_construct_matmul_nbits(RuntimeState *state,
                                                            int32_t slot) {
  return MockOpState::create(state, slot);
}

// MatMul: real runtime holds a shared_ptr to a device-wide hipBLASLt algo
// table (MatmulState in real/matmul.cpp); the mock owns no device/hipBLASLt
// resources.
extern "C" int8_t hipdnn_ep_op_state_construct_matmul(RuntimeState *state,
                                                      int32_t slot) {
  return MockOpState::create(state, slot);
}

// CausalConvWithState: real runtime owns a per-shape MIOpen descriptor/algo
// cache (CausalConvState in real/causal_conv_with_state.cpp); the mock owns no
// device/MIOpen resources.
extern "C" int8_t
hipdnn_ep_op_state_construct_causal_conv_with_state(RuntimeState *state,
                                                    int32_t slot) {
  return MockOpState::create(state, slot);
}

// GQA: real runtime owns a per-GEMM-shape hipBLASLt descriptor/algo cache
// (GqaState in real/gqa.cpp); the mock owns no device/hipBLASLt resources.
extern "C" int8_t hipdnn_ep_op_state_construct_gqa(RuntimeState *state,
                                                   int32_t slot) {
  return MockOpState::create(state, slot);
}

// MultiHeadAttention: real runtime owns a per-GEMM-shape hipBLASLt
// descriptor/algo cache (MhaState in real/multi_head_attention.cpp); the mock
// owns no device/hipBLASLt resources.
extern "C" int8_t
hipdnn_ep_op_state_construct_multi_head_attention(RuntimeState *state,
                                                  int32_t slot) {
  return MockOpState::create(state, slot);
}

// Activation (sigmoid/tanh/softplus): real runtime holds a shared_ptr to a
// device-wide MIOpen descriptor table (ActivationState in real/activation.cpp);
// the mock owns no device/MIOpen resources.
extern "C" int8_t hipdnn_ep_op_state_construct_activation(RuntimeState *state,
                                                          int32_t slot) {
  return MockOpState::create(state, slot);
}

// MIOpen OpTensor (miopen.add): real runtime holds a shared_ptr to a
// device-wide MIOpen descriptor table (OpTensorState in real/elementwise.cpp);
// the mock owns no device/MIOpen resources.
extern "C" int8_t hipdnn_ep_op_state_construct_optensor(RuntimeState *state,
                                                        int32_t slot) {
  return MockOpState::create(state, slot);
}

// SimplifiedLayerNorm (rms_norm): real runtime holds a shared_ptr to a
// device-wide MIOpen descriptor table (T5NormState in
// real/simplified_layer_norm.cpp); the mock owns no device/MIOpen resources.
extern "C" int8_t hipdnn_ep_op_state_construct_t5norm(RuntimeState *state,
                                                      int32_t slot) {
  return MockOpState::create(state, slot);
}

// SkipSimplifiedLayerNorm (skip_rms_norm): real runtime holds a shared_ptr to a
// device-wide MIOpen descriptor table (SkipT5NormState in
// real/skip_simplified_layer_norm.cpp); the mock owns no device/MIOpen
// resources.
extern "C" int8_t hipdnn_ep_op_state_construct_skip_t5norm(RuntimeState *state,
                                                           int32_t slot) {
  return MockOpState::create(state, slot);
}

// Gemm: real runtime holds a shared_ptr to a device-wide hipBLASLt algo table
// (GemmState in real/gemm.cpp); the mock owns no device/hipBLASLt resources.
extern "C" int8_t hipdnn_ep_op_state_construct_gemm(RuntimeState *state,
                                                    int32_t slot) {
  return MockOpState::create(state, slot);
}
