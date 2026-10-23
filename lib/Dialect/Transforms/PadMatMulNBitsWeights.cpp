/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Store int4 MatMulNBits weights row-padded so their row stride stops
// aliasing into a few cache sets.
//
// A weight row of K/2 bytes with many factors of two in K/2 (K = 4096, 8192,
// 12288, 14336, ...) maps the rows a WMMA tile reads in one K-step onto a
// handful of the 64 cache sets, so they evict each other. Padding each row
// breaks the alignment. The runtime used to do that per prefill call into a
// scratch buffer; doing it once in the stored constant removes the repack and
// leaves a single copy of the weight.
//
// The pad of a weight comes from the `row-pads` table when the table lists it
// (the compiler driver fills the table from the custom kernels library, which
// measured it for the device), and is kRowPadBytes for an unlisted weight
// whose stride aliases. A listed pad of 0 keeps the dense layout.
//
// Before:
//   %w = hip.constant {memory_address = A, size = 8388608}
//            : tensor<4096x32x64xui8>
//   hip.matmul_nbits(%ctx) ins(%a, %w, %s : ...)
//            ... {K = 4096, N = 4096, ...}
//
// After:
//   %w = hip.constant {memory_address = A, size = 8388608, row_bytes = 2048}
//            : tensor<4096x2304xui8>
//   hip.matmul_nbits(%ctx) ins(%a, %w, %s : ...)
//            ... {K = 4096, N = 4096, b_row_bytes = 2304, ...}
//
// `row_bytes` tells hip-externalize-constants (and the runtime loader) that
// the source holds dense rows of that many bytes, to be stored at the result's
// row pitch. Inline carriers are rewritten to padded bytes directly.

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <map>
#include <numeric>
#include <tuple>

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PADMATMULNBITSWEIGHTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Pad for a weight the row-pad table does not list, when its stride aliases:
// two 128-byte cache lines, a whole number of lines so every row stays
// line-aligned. Matches kU4PadRowBytes in
// lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip, the per-call repack's pad.
constexpr int64_t kRowPadBytes = 256;

// The arrival stride K/2 aliases when it is a whole number of 128-byte lines
// whose count shares a factor of at least 8 with the 64 cache sets. Mirrors
// shouldPadRow() in lib/Runtime/real/matmul_nbits.cpp.
bool strideAliases(int64_t k) {
  const int64_t stride = k / 2;
  if (stride % 128 != 0)
    return false;
  return std::gcd(stride / 128, int64_t{64}) >= 8;
}

// The padded layout is defined for whole quant groups of 4-bit weights only,
// which is also what every int4 runtime path accepts with a row pitch.
bool layoutSupported(MatMulNBitsOp op) {
  const int64_t k = op.getK();
  const int64_t bs = op.getBlockSize();
  return op->getNumResults() == 1 && op.getBits() == 4 && bs > 0 &&
         k % bs == 0 && op.getBRowBytes() == 0 && !op.getGIdx();
}

// Key of a row-pad table entry: the weight's geometry and quantization.
using RowPadKey = std::tuple<int64_t, int64_t, int64_t, bool>; // N, K, bs, zp

struct PadMatMulNBitsWeightsPass
    : public impl::PadMatMulNBitsWeightsPassBase<PadMatMulNBitsWeightsPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    // Each entry is `N:K:block_size:zero_points=pad_bytes`. The pad must keep
    // rows 16-byte aligned: every int4 path loads B in 16-byte vectors.
    for (StringRef entry : rowPads) {
      auto [geometry, padText] = entry.split('=');
      SmallVector<StringRef, 4> fields;
      geometry.split(fields, ':');
      int64_t n, k, bs, zp, pad;
      if (fields.size() != 4 || fields[0].getAsInteger(10, n) ||
          fields[1].getAsInteger(10, k) || fields[2].getAsInteger(10, bs) ||
          fields[3].getAsInteger(10, zp) || padText.getAsInteger(10, pad) ||
          n <= 0 || k <= 0 || bs <= 0 || (zp != 0 && zp != 1) || pad < 0 ||
          pad % 16 != 0)
        return emitError(UnknownLoc::get(context))
               << "hip-pad-matmul-nbits-weights: malformed row-pads entry '"
               << entry << "'";
      table[RowPadKey{n, k, bs, zp != 0}] = pad;
    }
    return success();
  }

  void runOnOperation() override {
    SmallVector<ConstantOp> carriers;
    getOperation().walk([&](ConstantOp c) { carriers.push_back(c); });
    for (ConstantOp c : carriers)
      padCarrier(c);
  }

  // The table's pad for this use of the weight, or the default when the
  // stride aliases; 0 keeps the dense layout.
  int64_t padFor(MatMulNBitsOp mm) const {
    auto it = table.find(RowPadKey{mm.getN(), mm.getK(), mm.getBlockSize(),
                                   static_cast<bool>(mm.getZeroPoints())});
    if (it != table.end())
      return it->second;
    return strideAliases(mm.getK()) ? kRowPadBytes : 0;
  }

  void padCarrier(ConstantOp c) {
    // Every user must be a MatMulNBits reading this value as B with the same
    // geometry and pad: a weight shared with another consumer (a tied
    // embedding's GatherBlockQuantized, say) keeps the dense layout.
    SmallVector<MatMulNBitsOp> users;
    int64_t n = -1, k = -1, pad = -1;
    for (OpOperand &use : c.getResult().getUses()) {
      auto mm = dyn_cast<MatMulNBitsOp>(use.getOwner());
      if (!mm || &use != &mm.getBMutable() || !layoutSupported(mm))
        return;
      const int64_t usePad = padFor(mm);
      if (n >= 0 && (mm.getN() != n || mm.getK() != k || usePad != pad))
        return;
      n = mm.getN();
      k = mm.getK();
      pad = usePad;
      users.push_back(mm);
    }
    if (users.empty() || pad == 0)
      return;

    RankedTensorType type = c.getResult().getType();
    if (!type.getElementType().isInteger(8) || !type.hasStaticShape())
      return;
    const int64_t rowBytes = k / 2;
    if (type.getNumElements() != n * rowBytes)
      return;
    if (c.getRowBytesAttr())
      return;
    // A packed-4-bit carrier holds half as many bytes as its element count;
    // only a plain byte-per-element source is padded.
    if (IntegerAttr size = c.getSizeAttr())
      if (size.getInt() != n * rowBytes)
        return;

    const int64_t pitch = rowBytes + pad;
    auto paddedType = RankedTensorType::get({n, pitch}, type.getElementType());
    OpBuilder b(c);
    ConstantOp padded;
    if (auto value = dyn_cast_or_null<DenseElementsAttr>(c.getValueAttr())) {
      ArrayRef<char> src = value.getRawData();
      if (value.isSplat() || static_cast<int64_t>(src.size()) != n * rowBytes)
        return;
      std::vector<char> bytes(static_cast<size_t>(n * pitch), 0);
      for (int64_t r = 0; r < n; ++r)
        std::memcpy(bytes.data() + r * pitch, src.data() + r * rowBytes,
                    static_cast<size_t>(rowBytes));
      padded = ConstantOp::create(
          b, c.getLoc(), paddedType,
          DenseElementsAttr::getFromRawBuffer(paddedType, bytes),
          /*location=*/nullptr, /*offset=*/nullptr, /*memory_address=*/nullptr,
          /*size=*/nullptr, c.getSymbolNameHintAttr(), c.getSourceNameAttr(),
          c.getSerializationOrderAttr(), /*row_bytes=*/nullptr);
    } else {
      padded = ConstantOp::create(
          b, c.getLoc(), paddedType, /*value=*/nullptr, c.getLocationAttr(),
          c.getOffsetAttr(), c.getMemoryAddressAttr(), c.getSizeAttr(),
          c.getSymbolNameHintAttr(), c.getSourceNameAttr(),
          c.getSerializationOrderAttr(), b.getI64IntegerAttr(rowBytes));
    }

    for (MatMulNBitsOp mm : users) {
      mm.getBMutable().set(padded.getResult());
      mm.setBRowBytes(pitch);
    }
    c.erase();
  }

  std::map<RowPadKey, int64_t> table;
};

} // namespace
} // namespace hip
} // namespace mlir
