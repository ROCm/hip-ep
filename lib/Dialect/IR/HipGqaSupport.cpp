/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipGqaSupport.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#include <limits>

using namespace mlir;
using namespace mlir::hip;

namespace {

Type getElementType(Type type) {
  auto shapedType = dyn_cast_or_null<ShapedType>(type);
  return shapedType ? shapedType.getElementType() : Type{};
}

bool isSignedInt8(Type type) {
  Type elementType = getElementType(type);
  return elementType &&
         (elementType.isSignlessInteger(8) || elementType.isSignedInteger(8));
}

} // namespace

FailureOr<GqaKvCacheMode> mlir::hip::verifyGqaFeatureSupport(
    StringRef kQuantType, StringRef vQuantType, int64_t kvCacheBitWidth,
    int64_t rotaryInterleaved, int64_t kvNumHeads, const GqaFeatureTypes &types,
    function_ref<InFlightDiagnostic()> emitError) {
  if (rotaryInterleaved != 0) {
    emitError() << "rotary_interleaved must be zero; interleaved RoPE is "
                   "unsupported by the runtime";
    return failure();
  }

  auto isKnownQuantType = [](StringRef value) {
    return value == "NONE" || value == "PER_TENSOR" || value == "PER_CHANNEL";
  };
  if (!isKnownQuantType(kQuantType)) {
    emitError() << "unsupported k_quant_type '" << kQuantType << "'";
    return failure();
  }
  if (!isKnownQuantType(vQuantType)) {
    emitError() << "unsupported v_quant_type '" << vQuantType << "'";
    return failure();
  }
  if (kQuantType != vQuantType) {
    emitError() << "K/V quantization schemes must match, got k_quant_type='"
                << kQuantType << "' and v_quant_type='" << vQuantType << "'";
    return failure();
  }
  if (kvCacheBitWidth != 4 && kvCacheBitWidth != 8) {
    emitError() << "kv_cache_bit_width must be 4 or 8, got " << kvCacheBitWidth;
    return failure();
  }

  Type queryElementType = getElementType(types.query);
  if (!queryElementType) {
    emitError() << "GQA data operands must have shaped types";
    return failure();
  }

  if (kQuantType == "NONE") {
    if (types.kScale || types.vScale) {
      emitError() << "unquantized GQA must omit both k_scale and v_scale";
      return failure();
    }
    if (!queryElementType.isF16() && !queryElementType.isF32()) {
      emitError() << "unquantized GQA supports only f16 or f32 data, got "
                  << queryElementType;
      return failure();
    }
    for (auto [name, type] : {std::pair<StringRef, Type>{"key", types.key},
                              {"value", types.value},
                              {"past_key", types.pastKey},
                              {"past_value", types.pastValue},
                              {"output", types.output},
                              {"present_key", types.presentKey},
                              {"present_value", types.presentValue}}) {
      if (type && getElementType(type) != queryElementType) {
        emitError() << "unquantized GQA " << name
                    << " element type must match query element type "
                    << queryElementType << ", got " << getElementType(type);
        return failure();
      }
    }
    return GqaKvCacheMode::Unquantized;
  }

  if (kQuantType == "PER_TENSOR") {
    emitError() << "PER_TENSOR KV quantization is unsupported by the runtime";
    return failure();
  }
  if (kvCacheBitWidth != 8) {
    emitError() << "quantized GQA supports only 8-bit KV caches; "
                   "4-bit KV caches are unsupported";
    return failure();
  }
  if (!types.kScale || !types.vScale) {
    emitError() << "PER_CHANNEL KV quantization requires both k_scale and "
                   "v_scale";
    return failure();
  }

  for (auto [name, type] : {std::pair<StringRef, Type>{"query", types.query},
                            {"key", types.key},
                            {"value", types.value},
                            {"output", types.output}}) {
    if (type && !getElementType(type).isF16()) {
      emitError() << "quantized GQA " << name
                  << " element type must be f16, got " << getElementType(type);
      return failure();
    }
  }
  for (auto [name, type] :
       {std::pair<StringRef, Type>{"past_key", types.pastKey},
        {"past_value", types.pastValue},
        {"present_key", types.presentKey},
        {"present_value", types.presentValue}}) {
    if (type && !isSignedInt8(type)) {
      emitError() << "quantized GQA " << name
                  << " element type must be signed int8, got "
                  << getElementType(type);
      return failure();
    }
  }
  if (!getElementType(types.kScale).isF32() ||
      !getElementType(types.vScale).isF32()) {
    emitError() << "PER_CHANNEL k_scale and v_scale element types must both "
                   "be f32";
    return failure();
  }

  auto presentKeyType = dyn_cast<ShapedType>(types.presentKey);
  auto kScaleType = dyn_cast<ShapedType>(types.kScale);
  auto vScaleType = dyn_cast<ShapedType>(types.vScale);
  if (!presentKeyType || !presentKeyType.hasRank() ||
      presentKeyType.getRank() != 4 ||
      ShapedType::isDynamic(presentKeyType.getDimSize(3)) || !kScaleType ||
      !kScaleType.hasStaticShape() || !vScaleType ||
      !vScaleType.hasStaticShape()) {
    emitError() << "PER_CHANNEL int8 GQA requires a static cache head size "
                   "and statically shaped scales";
    return failure();
  }
  int64_t headSize = presentKeyType.getDimSize(3);
  if (kvNumHeads <= 0 ||
      headSize > std::numeric_limits<int64_t>::max() / kvNumHeads) {
    emitError() << "PER_CHANNEL int8 GQA scale element count overflows i64";
    return failure();
  }
  int64_t expectedScaleElements = kvNumHeads * headSize;
  if (kScaleType.getNumElements() != expectedScaleElements ||
      vScaleType.getNumElements() != expectedScaleElements) {
    emitError() << "PER_CHANNEL int8 GQA scales must each contain "
                << expectedScaleElements << " elements (kv_num_heads * "
                << "head_size)";
    return failure();
  }

  return GqaKvCacheMode::Int8PerChannel;
}
