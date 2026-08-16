/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gqa_contract.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdio>
#include <limits>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

int invokeMock(void *positionIds, void *outputQk, int64_t qkOutput,
               float softcap) {
  RuntimeState state{};
  int storage = 0;
  void *data = &storage;
  return wrap_group_query_attention(
      &state, /*op_state_slot=*/0,
      /*query=*/data, /*key=*/data, /*value=*/data,
      /*past_key=*/nullptr, /*past_value=*/nullptr,
      /*seqlens_k=*/data, /*total_seq_len=*/data,
      /*cos_cache=*/nullptr, /*sin_cache=*/nullptr, positionIds,
      /*attention_bias=*/nullptr, /*head_sink=*/nullptr,
      /*k_scale=*/nullptr, /*v_scale=*/nullptr,
      /*output=*/data, /*present_key=*/data, /*present_value=*/data, outputQk,
      /*num_heads=*/4, /*kv_num_heads=*/2, /*scale=*/0.5f,
      /*do_rotary=*/0, /*rotary_interleaved=*/0, softcap,
      /*local_window_size=*/-1, /*smooth_softmax=*/0, qkOutput,
      /*k_quant_type=*/0, /*v_quant_type=*/0, /*kv_cache_bit_width=*/8,
      /*no_causal=*/0, /*batch_size=*/1, /*seq_len_q=*/1, /*seq_len_kv=*/1,
      /*past_buf_seq=*/0, /*head_dim=*/8, /*element_size_bytes=*/2,
      /*attn_bias_batch=*/1, /*attn_bias_num_heads=*/1);
}

void testSharedRealAndMockPreflight() {
  CHECK(validateGqaRuntimeContract(false, false, 0, 0.0f) ==
        GqaRuntimeContractViolation::None);
  CHECK(validateGqaRuntimeContract(false, false, 0, -0.0f) ==
        GqaRuntimeContractViolation::None);
  CHECK(validateGqaRuntimeContract(true, false, 0, 0.0f) ==
        GqaRuntimeContractViolation::PositionIds);
  CHECK(validateGqaRuntimeContract(false, true, 0, 0.0f) ==
        GqaRuntimeContractViolation::OutputQk);
  CHECK(validateGqaRuntimeContract(false, false, 1, 0.0f) ==
        GqaRuntimeContractViolation::OutputQk);

  for (float unsupported : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::denorm_min()})
    CHECK(validateGqaRuntimeContract(false, false, 0, unsupported) ==
          GqaRuntimeContractViolation::Softcap);
}

void testMockWrapper() {
  int storage = 0;
  void *present = &storage;
  CHECK(invokeMock(nullptr, nullptr, 0, 0.0f) == 0);
  CHECK(invokeMock(nullptr, nullptr, 0, -0.0f) == 0);
  CHECK(invokeMock(present, nullptr, 0, 0.0f) == -1);
  CHECK(invokeMock(nullptr, present, 0, 0.0f) == -1);
  CHECK(invokeMock(nullptr, nullptr, 1, 0.0f) == -1);
  CHECK(invokeMock(nullptr, nullptr, 0,
                   std::numeric_limits<float>::quiet_NaN()) == -1);
  CHECK(invokeMock(nullptr, nullptr, 0,
                   std::numeric_limits<float>::infinity()) == -1);
  CHECK(invokeMock(nullptr, nullptr, 0,
                   std::numeric_limits<float>::denorm_min()) == -1);
}

} // namespace

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *) { return 0; }

int main() {
  testSharedRealAndMockPreflight();
  testMockWrapper();
  if (failures == 0) {
    std::printf("GQA contract unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "GQA contract unit test: %d FAILURE(S)\n", failures);
  return 1;
}
