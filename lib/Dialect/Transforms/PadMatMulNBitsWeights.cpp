/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Store int4 MatMulNBits weights row-padded, or in another byte order, so
// their row stride stops aliasing into a few cache sets.
//
// A weight row of K/2 bytes with many factors of two in K/2 (K = 4096, 8192,
// 12288, 14336, ...) maps the rows a WMMA tile reads in one K-step onto a
// handful of the 64 cache sets, so they evict each other. Padding each row
// breaks the alignment. The runtime used to do that per prefill call into a
// scratch buffer; doing it once in the stored constant removes the repack and
// leaves a single copy of the weight.
//
// The placement of a weight is its `row-pads` entry in the device's autotune
// table, which the compiler driver reads from the custom kernels library: a
// row pad, or a byte order with no extra memory
// (include/hip/matmul_nbits_weight_layout.h): XOR-swizzled 128-byte lines, or
// 512-byte blocks of 16 rows. Only measured weights are listed; an unlisted
// weight keeps the dense layout and the runtime keeps repacking it per call
// when its stride aliases. A listed pad of 0 keeps the weight dense and marks
// its users with b_row_bytes = K/2, so the runtime does not repack a weight
// that was measured faster (or as fast, in less memory) dense.
//
// Before:
//   %w = hip.constant {memory_address = A, size = 8388608}
//            : tensor<4096x32x64xui8>
//   hip.matmul_nbits(%ctx) ins(%a, %w, %s : ...)
//            ... {K = 4096, N = 4096, ...}
//
// After (row-pads=4096:4096:128:0=256, the weight's geometry and pad):
//   %w = hip.constant {memory_address = A, size = 8388608, row_bytes = 2048}
//            : tensor<4096x2304xui8>
//   hip.matmul_nbits(%ctx) ins(%a, %w, %s : ...)
//            ... {K = 4096, N = 4096, b_row_bytes = 2304, ...}
//
// After (row-pads=4096:4096:128:0=blocks, a byte order):
//   %w = hip.constant {memory_address = A, size = 8388608, row_bytes = 2048,
//                      weight_layout = 2} : tensor<4096x2048xui8>
//   hip.matmul_nbits(%ctx) ins(%a, %w, %s : ...)
//            ... {K = 4096, N = 4096, b_row_bytes = 2048, b_layout = 2, ...}
//
// `row_bytes` tells hip-externalize-constants (and the runtime loader) that
// the source holds dense rows of that many bytes, to be stored at the result's
// row pitch and, with `weight_layout`, in that byte order. Inline carriers are
// rewritten to the stored bytes directly.

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/matmul_nbits_weight_layout.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PADMATMULNBITSWEIGHTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

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

// How a listed weight is stored: rows `pad` bytes longer (Rows), or the dense
// rows in another byte order (pad 0).
struct WeightPlacement {
  int64_t pad = 0;
  ::hip::MatmulNBitsWeightLayout layout = ::hip::MatmulNBitsWeightLayout::Rows;
  bool operator==(const WeightPlacement &o) const {
    return pad == o.pad && layout == o.layout;
  }
  bool operator!=(const WeightPlacement &o) const { return !(*this == o); }
};

struct PadMatMulNBitsWeightsPass
    : public impl::PadMatMulNBitsWeightsPassBase<PadMatMulNBitsWeightsPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    // Each entry is `N:K:block_size:zero_points=<pad_bytes|xor|blocks>`. A pad
    // must keep rows 16-byte aligned: every int4 path loads B in 16-byte
    // vectors.
    for (StringRef entry : rowPads) {
      auto [geometry, placementText] = entry.split('=');
      SmallVector<StringRef, 4> fields;
      geometry.split(fields, ':');
      int64_t n, k, bs, zp;
      WeightPlacement placement;
      bool badPlacement = false;
      if (placementText == "xor")
        placement.layout = ::hip::MatmulNBitsWeightLayout::XorLines;
      else if (placementText == "blocks")
        placement.layout = ::hip::MatmulNBitsWeightLayout::Blocks;
      else
        badPlacement = placementText.getAsInteger(10, placement.pad) ||
                       placement.pad < 0 || placement.pad % 16 != 0;
      if (fields.size() != 4 || fields[0].getAsInteger(10, n) ||
          fields[1].getAsInteger(10, k) || fields[2].getAsInteger(10, bs) ||
          fields[3].getAsInteger(10, zp) || badPlacement || n <= 0 || k <= 0 ||
          bs <= 0 || (zp != 0 && zp != 1))
        return emitError(UnknownLoc::get(context))
               << "hip-pad-matmul-nbits-weights: malformed row-pads entry '"
               << entry << "'";
      if (!table.try_emplace(RowPadKey{n, k, bs, zp != 0}, placement).second)
        return emitError(UnknownLoc::get(context))
               << "hip-pad-matmul-nbits-weights: duplicate row-pads entry '"
               << entry << "'";
    }
    return success();
  }

  void runOnOperation() override {
    SmallVector<ConstantOp> carriers;
    getOperation().walk([&](ConstantOp c) { carriers.push_back(c); });
    for (ConstantOp c : carriers)
      padCarrier(c);
  }

  // The placement for this use of the weight: its table entry, else none
  // (nullopt, leave the weight to the runtime). Rows with pad 0 is a listed
  // decision to keep it dense.
  std::optional<WeightPlacement> placementFor(MatMulNBitsOp mm) const {
    auto it = table.find(RowPadKey{mm.getN(), mm.getK(), mm.getBlockSize(),
                                   static_cast<bool>(mm.getZeroPoints())});
    if (it == table.end())
      return std::nullopt;
    return it->second;
  }

  void padCarrier(ConstantOp c) {
    // Every user must be a MatMulNBits reading this value as B with the same
    // geometry and placement: a weight shared with another consumer (a tied
    // embedding's GatherBlockQuantized, say) keeps the dense layout.
    SmallVector<MatMulNBitsOp> users;
    int64_t n = -1, k = -1;
    std::optional<WeightPlacement> placement;
    for (OpOperand &use : c.getResult().getUses()) {
      auto mm = dyn_cast<MatMulNBitsOp>(use.getOwner());
      if (!mm || &use != &mm.getBMutable() || !layoutSupported(mm))
        return;
      const std::optional<WeightPlacement> usePlacement = placementFor(mm);
      if (!usePlacement || (placement && *usePlacement != *placement) ||
          (n >= 0 && (mm.getN() != n || mm.getK() != k)))
        return;
      n = mm.getN();
      k = mm.getK();
      placement = usePlacement;
      users.push_back(mm);
    }
    if (users.empty() || !placement)
      return;
    const int64_t pad = placement->pad;
    const ::hip::MatmulNBitsWeightLayout layout = placement->layout;
    const bool rows = layout == ::hip::MatmulNBitsWeightLayout::Rows;
    if (rows && pad == 0) {
      for (MatMulNBitsOp mm : users)
        mm.setBRowBytes(k / 2);
      return;
    }

    RankedTensorType type = c.getResult().getType();
    if (!type.getElementType().isInteger(8) || !type.hasStaticShape())
      return;
    // The pitch must stay a multiple of the 16-byte B loads and fit the `int`
    // row pitch of the kernel ABI; otherwise the weight keeps its layout.
    const int64_t rowBytes = k / 2;
    int64_t pitch = 0, denseBytes = 0, paddedBytes = 0;
    if (llvm::AddOverflow(rowBytes, pad, pitch) || pitch % 16 != 0 ||
        pitch > std::numeric_limits<int32_t>::max() ||
        llvm::MulOverflow(n, rowBytes, denseBytes) ||
        llvm::MulOverflow(n, pitch, paddedBytes))
      return;
    if (type.getNumElements() != denseBytes)
      return;
    if (c.getRowBytesAttr())
      return;
    if (!::hip::matmulNBitsWeightLayoutFits(layout, n, rowBytes))
      return;
    // A packed-4-bit carrier holds half as many bytes as its element count;
    // only a plain byte-per-element source is padded.
    if (IntegerAttr size = c.getSizeAttr())
      if (size.getInt() != denseBytes)
        return;

    auto paddedType = RankedTensorType::get({n, pitch}, type.getElementType());
    OpBuilder b(c);
    ConstantOp padded;
    if (auto value = dyn_cast_or_null<DenseElementsAttr>(c.getValueAttr())) {
      ArrayRef<char> src = value.getRawData();
      if (value.isSplat() || static_cast<int64_t>(src.size()) != denseBytes)
        return;
      std::vector<char> bytes(static_cast<size_t>(paddedBytes), 0);
      for (int64_t r = 0; r < n; ++r)
        std::memcpy(bytes.data() + r * pitch, src.data() + r * rowBytes,
                    static_cast<size_t>(rowBytes));
      std::vector<uint8_t> scratch(static_cast<size_t>(
          ::hip::matmulNBitsWeightLayoutRowStep(layout) * rowBytes));
      ::hip::applyMatmulNBitsWeightLayout(
          reinterpret_cast<uint8_t *>(bytes.data()), 0,
          static_cast<uint64_t>(n), static_cast<uint64_t>(rowBytes), layout,
          scratch.data());
      padded = ConstantOp::create(
          b, c.getLoc(), paddedType,
          DenseElementsAttr::getFromRawBuffer(paddedType, bytes),
          /*location=*/nullptr, /*offset=*/nullptr, /*memory_address=*/nullptr,
          /*size=*/nullptr, c.getSymbolNameHintAttr(), c.getSourceNameAttr(),
          c.getSerializationOrderAttr(), /*row_bytes=*/nullptr,
          /*weight_layout=*/nullptr);
    } else {
      padded = ConstantOp::create(
          b, c.getLoc(), paddedType, /*value=*/nullptr, c.getLocationAttr(),
          c.getOffsetAttr(), c.getMemoryAddressAttr(), c.getSizeAttr(),
          c.getSymbolNameHintAttr(), c.getSourceNameAttr(),
          c.getSerializationOrderAttr(), b.getI64IntegerAttr(rowBytes),
          rows ? IntegerAttr()
               : b.getI64IntegerAttr(static_cast<int64_t>(layout)));
    }

    for (MatMulNBitsOp mm : users) {
      mm.getBMutable().set(padded.getResult());
      mm.setBRowBytes(pitch);
      mm.setBLayout(static_cast<int64_t>(layout));
    }
    c.erase();
  }

  std::map<RowPadKey, WeightPlacement> table;
};

} // namespace
} // namespace hip
} // namespace mlir
