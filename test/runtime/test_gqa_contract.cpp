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
int errorFlagWrites = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

struct GqaCase {
  void *positionIds = nullptr;
  void *outputQk = nullptr;
  int64_t qkOutput = 0;
  float softcap = 0.0f;
  int64_t rotaryInterleaved = 0;
  int64_t kQuantType = 0;
  int64_t vQuantType = 0;
  int64_t kvCacheBitWidth = 8;
  bool hasKScale = false;
  bool hasVScale = false;
  int64_t elementSizeBytes = 2;
  int64_t kCacheDataType = HIPDNN_EP_DATATYPE_HALF;
  int64_t vCacheDataType = HIPDNN_EP_DATATYPE_HALF;
  int64_t kScaleDataType = -1;
  int64_t vScaleDataType = -1;
};

GqaRuntimeContractViolation validate(const GqaCase &testCase,
                                     GqaKvCacheMode *mode = nullptr) {
  return validateGqaRuntimeContract(
      testCase.positionIds != nullptr, testCase.outputQk != nullptr,
      testCase.qkOutput, testCase.softcap, testCase.rotaryInterleaved,
      testCase.kQuantType, testCase.vQuantType, testCase.kvCacheBitWidth,
      testCase.hasKScale, testCase.hasVScale, testCase.elementSizeBytes,
      testCase.kCacheDataType, testCase.vCacheDataType, testCase.kScaleDataType,
      testCase.vScaleDataType, mode);
}

int invokeMock(const GqaCase &testCase) {
  RuntimeState state{};
  int storage = 0;
  void *data = &storage;
  void *kScale = testCase.hasKScale ? data : nullptr;
  void *vScale = testCase.hasVScale ? data : nullptr;
  return wrap_group_query_attention(
      &state, /*op_state_slot=*/0,
      /*query=*/data, /*key=*/data, /*value=*/data,
      /*past_key=*/nullptr, /*past_value=*/nullptr,
      /*seqlens_k=*/data, /*total_seq_len=*/data,
      /*cos_cache=*/nullptr, /*sin_cache=*/nullptr, testCase.positionIds,
      /*attention_bias=*/nullptr, /*head_sink=*/nullptr, kScale, vScale,
      /*output=*/data, /*present_key=*/data, /*present_value=*/data,
      testCase.outputQk,
      /*num_heads=*/4, /*kv_num_heads=*/2, /*scale=*/0.5f,
      /*do_rotary=*/0, testCase.rotaryInterleaved, testCase.softcap,
      /*local_window_size=*/-1, /*smooth_softmax=*/0, testCase.qkOutput,
      testCase.kQuantType, testCase.vQuantType, testCase.kvCacheBitWidth,
      /*no_causal=*/0, /*batch_size=*/1, /*seq_len_q=*/1, /*seq_len_kv=*/1,
      /*past_buf_seq=*/0, /*head_dim=*/8, testCase.elementSizeBytes,
      /*attn_bias_batch=*/1, /*attn_bias_num_heads=*/1, testCase.kCacheDataType,
      testCase.vCacheDataType, testCase.kScaleDataType,
      testCase.vScaleDataType);
}

void testSharedRealAndMockPreflight() {
  int storage = 0;
  GqaCase testCase;
  GqaKvCacheMode mode = GqaKvCacheMode::Int8PerChannel;
  CHECK(validate(testCase, &mode) == GqaRuntimeContractViolation::None);
  CHECK(mode == GqaKvCacheMode::Unquantized);
  testCase.softcap = -0.0f;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::None);
  testCase.softcap = 0.0f;
  testCase.positionIds = &storage;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::PositionIds);
  testCase.positionIds = nullptr;
  testCase.outputQk = &storage;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::OutputQk);
  testCase.outputQk = nullptr;
  testCase.qkOutput = 1;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::OutputQk);
  testCase.qkOutput = 0;

  for (float unsupported : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::denorm_min()}) {
    testCase.softcap = unsupported;
    CHECK(validate(testCase) == GqaRuntimeContractViolation::Softcap);
  }

  testCase = {};
  testCase.rotaryInterleaved = 1;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::RotaryInterleaved);
  testCase = {};
  testCase.kQuantType = testCase.vQuantType = 1;
  CHECK(validate(testCase) ==
        GqaRuntimeContractViolation::PerTensorQuantization);
  testCase = {};
  testCase.kQuantType = testCase.vQuantType = 2;
  testCase.kvCacheBitWidth = 4;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::QuantizedBitWidth);
  testCase = {};
  testCase.kQuantType = 2;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::MixedQuantization);
  testCase = {};
  testCase.kQuantType = testCase.vQuantType = 2;
  testCase.hasKScale = testCase.hasVScale = true;
  testCase.kCacheDataType = HIPDNN_EP_DATATYPE_INT8;
  testCase.vCacheDataType = HIPDNN_EP_DATATYPE_HALF;
  testCase.kScaleDataType = testCase.vScaleDataType = HIPDNN_EP_DATATYPE_FLOAT;
  CHECK(validate(testCase) == GqaRuntimeContractViolation::CacheDataType);
  testCase.vCacheDataType = HIPDNN_EP_DATATYPE_INT8;
  mode = GqaKvCacheMode::Unquantized;
  CHECK(validate(testCase, &mode) == GqaRuntimeContractViolation::None);
  CHECK(mode == GqaKvCacheMode::Int8PerChannel);
}

void testMockWrapper() {
  int storage = 0;
  GqaCase supported;
  CHECK(invokeMock(supported) == 0);
  supported.kQuantType = supported.vQuantType = 2;
  supported.hasKScale = supported.hasVScale = true;
  supported.kCacheDataType = supported.vCacheDataType = HIPDNN_EP_DATATYPE_INT8;
  supported.kScaleDataType = supported.vScaleDataType =
      HIPDNN_EP_DATATYPE_FLOAT;
  CHECK(invokeMock(supported) == 0);

  GqaCase positionIds;
  positionIds.positionIds = &storage;
  GqaCase outputQk;
  outputQk.outputQk = &storage;
  GqaCase qkOutput;
  qkOutput.qkOutput = 1;
  GqaCase softcap;
  softcap.softcap = std::numeric_limits<float>::quiet_NaN();
  GqaCase rotary;
  rotary.rotaryInterleaved = 1;
  GqaCase perTensor;
  perTensor.kQuantType = perTensor.vQuantType = 1;
  GqaCase int4;
  int4.kQuantType = int4.vQuantType = 2;
  int4.kvCacheBitWidth = 4;
  GqaCase mixedScheme;
  mixedScheme.kQuantType = 2;
  GqaCase mixedDtype = supported;
  mixedDtype.vCacheDataType = HIPDNN_EP_DATATYPE_HALF;
  for (const GqaCase &unsupported :
       {positionIds, outputQk, qkOutput, softcap, rotary, perTensor, int4,
        mixedScheme, mixedDtype}) {
    int before = errorFlagWrites;
    CHECK(invokeMock(unsupported) == -1);
    CHECK(errorFlagWrites == before + 1);
  }
}

} // namespace

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *) {
  ++errorFlagWrites;
  return 0;
}

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
