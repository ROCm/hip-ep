/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlLayout.h - How GGML arranges a tensor's bytes ---------------===//
//
// The GGML storage ABI: how many bytes a block of each type occupies, what
// strides an extent vector implies, and whether a recorded stride vector is the
// dense one. These belong to GGML rather than to any graph, which is why a
// stride can be derived without observing one.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLLAYOUT_H
#define HIP_DIALECT_GGML_IR_GGMLLAYOUT_H

#include "hip/Dialect/Ggml/IR/GgmlProvenance.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeArithmetic.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir {
namespace ggml {

/// The products the module around `scope` declares among its symbols
/// (`ggml.symbols` entries with a `product`: an image's blocks are its
/// block columns times its block rows), which shape arithmetic may then
/// multiply.
inline void declareModuleProducts(Operation *scope,
                                  shape_expr::Arithmetic &arithmetic) {
  auto module =
      scope ? (isa<ModuleOp>(scope) ? cast<ModuleOp>(scope)
                                    : scope->getParentOfType<ModuleOp>())
            : ModuleOp();
  auto symbols =
      module ? module->getAttrOfType<ArrayAttr>("ggml.symbols") : ArrayAttr();
  if (!symbols)
    return;
  for (Attribute entry : symbols) {
    auto described = dyn_cast<DictionaryAttr>(entry);
    auto name = described ? described.getAs<StringAttr>("name") : StringAttr();
    auto factors =
        described ? described.getAs<ArrayAttr>("product") : ArrayAttr();
    if (!name || !factors || factors.size() != 2)
      continue;
    auto a = dyn_cast<StringAttr>(factors[0]),
         b = dyn_cast<StringAttr>(factors[1]);
    if (a && b)
      arithmetic.declareProduct(name.getValue(), a.getValue(), b.getValue());
  }
}

/// The operation a value belongs to: its producer, or the one owning the
/// block it is an argument of.
inline Operation *getScope(Value value) {
  if (Operation *producer = value.getDefiningOp())
    return producer;
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner()->getParentOp();
  return nullptr;
}

/// GGML tensors are rank four, always, with unused dimensions set to one.
constexpr unsigned kGgmlRank = 4;

/// The layout `type` carries, or null when it carries none.
///
/// A tensor outside this dialect has no GGML layout, which is not the same as
/// having a trivial one: the caller decides what to do about not knowing.
inline LayoutAttr getGgmlLayout(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return {};
  return dyn_cast_or_null<LayoutAttr>(tensor.getEncoding());
}

inline LayoutAttr getGgmlLayout(Value value) {
  return getGgmlLayout(value.getType());
}

/// The four extents this value's type carries, innermost first, or empty.
inline ArrayRef<Attribute> getGgmlExtents(Value value) {
  LayoutAttr layout = getGgmlLayout(value);
  return layout ? layout.getExtents() : ArrayRef<Attribute>();
}

/// The four byte strides this value's type carries, in the same order.
inline ArrayRef<Attribute> getGgmlStrides(Value value) {
  LayoutAttr layout = getGgmlLayout(value);
  return layout ? layout.getStrides() : ArrayRef<Attribute>();
}

/// The shape `extents` imply: the same four read the other way round, with a
/// varying extent standing as `?`. This is the relation `#ggml.layout` verifies
/// against the type it encodes.
inline SmallVector<int64_t, kGgmlRank>
getShapeFromGgmlExtents(ArrayRef<Attribute> extents) {
  SmallVector<int64_t, kGgmlRank> shape;
  for (Attribute extent : llvm::reverse(extents)) {
    std::optional<int64_t> constant = getShapeExprConstant(extent);
    shape.push_back(constant ? *constant : ShapedType::kDynamic);
  }
  return shape;
}

/// `type` with `layout` as its encoding, and the shape that layout implies.
inline RankedTensorType withGgmlLayout(RankedTensorType type,
                                       LayoutAttr layout) {
  return RankedTensorType::get(getShapeFromGgmlExtents(layout.getExtents()),
                               type.getElementType(), layout);
}

/// `type` without its GGML layout.
///
/// After `convert-ggml-to-hip` the types belong to dialects that do not know
/// this encoding, so the conversion drops it from what it produces and keeps
/// what it needed as attributes instead.
inline Type dropGgmlLayout(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !isa_and_nonnull<LayoutAttr>(tensor.getEncoding()))
    return type;
  return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
}

/// Bytes per stored block and elements per block.
///
/// Quantized types are block-based, and a block is always a whole number of
/// bytes even when an element is not: q4_K averages four and a half bits, but
/// two hundred and fifty-six of them occupy exactly one hundred and
/// forty-four bytes. That is what lets a stride stay in bytes.
struct BlockLayout {
  int64_t typeSize;
  int64_t blockElements;
};

/// The block geometry of a GGML encoding, named as `ggml_type_name` names it.
///
/// This is a second spelling of what ggml already knows -- `ggml_type_size`
/// and `ggml_blck_size` -- and has to be, because the dialect is linked by a
/// compiler that has no ggml to ask. It is not left to be trusted:
/// `test/unit/hipep_backend` walks every `ggml_type` and holds
/// this against ggml's own answer, so a number that is wrong here, or an
/// encoding ggml gains and this does not, is a failing test rather than a
/// silent fallback.
inline std::optional<BlockLayout> getBlockLayout(StringRef name) {
  return llvm::StringSwitch<std::optional<BlockLayout>>(name)
      // Dense.
      .Case("f32", BlockLayout{4, 1})
      .Case("f16", BlockLayout{2, 1})
      .Case("bf16", BlockLayout{2, 1})
      .Case("f64", BlockLayout{8, 1})
      .Case("i8", BlockLayout{1, 1})
      .Case("i16", BlockLayout{2, 1})
      .Case("i32", BlockLayout{4, 1})
      .Case("i64", BlockLayout{8, 1})
      // Round-number quantization, one scale per block.
      .Case("q4_0", BlockLayout{18, 32})
      .Case("q4_1", BlockLayout{20, 32})
      .Case("q5_0", BlockLayout{22, 32})
      .Case("q5_1", BlockLayout{24, 32})
      .Case("q8_0", BlockLayout{34, 32})
      .Case("q8_1", BlockLayout{36, 32})
      .Case("q1_0", BlockLayout{18, 128})
      .Case("q2_0", BlockLayout{18, 64})
      // K-quants: a super-block of 256 with per-sub-block scales.
      .Case("q2_K", BlockLayout{84, 256})
      .Case("q3_K", BlockLayout{110, 256})
      .Case("q4_K", BlockLayout{144, 256})
      .Case("q5_K", BlockLayout{176, 256})
      .Case("q6_K", BlockLayout{210, 256})
      .Case("q8_K", BlockLayout{292, 256})
      // I-quants: codebook lookups, same super-block.
      .Case("iq1_s", BlockLayout{50, 256})
      .Case("iq1_m", BlockLayout{56, 256})
      .Case("iq2_xxs", BlockLayout{66, 256})
      .Case("iq2_xs", BlockLayout{74, 256})
      .Case("iq2_s", BlockLayout{82, 256})
      .Case("iq3_xxs", BlockLayout{98, 256})
      .Case("iq3_s", BlockLayout{110, 256})
      .Case("iq4_nl", BlockLayout{18, 32})
      .Case("iq4_xs", BlockLayout{136, 256})
      // Ternary.
      .Case("tq1_0", BlockLayout{54, 256})
      .Case("tq2_0", BlockLayout{66, 256})
      // Microscaling floating point.
      .Case("mxfp4", BlockLayout{17, 32})
      .Case("nvfp4", BlockLayout{36, 64})
      // OCP 8-bit float, one byte an element.
      .Case("f8_e4m3", BlockLayout{1, 1})
      .Default(std::nullopt);
}

/// The layout of a dense builtin element type, or nullopt when it is not one.
inline std::optional<BlockLayout> getBlockLayout(Type element) {
  if (element.isF32())
    return getBlockLayout("f32");
  if (element.isF16())
    return getBlockLayout("f16");
  if (element.isBF16())
    return getBlockLayout("bf16");
  if (element.isF64())
    return getBlockLayout("f64");
  if (element.isInteger(8))
    return getBlockLayout("i8");
  if (element.isInteger(16))
    return getBlockLayout("i16");
  if (element.isInteger(32))
    return getBlockLayout("i32");
  if (element.isInteger(64))
    return getBlockLayout("i64");
  return std::nullopt;
}

/// The ggml type a value's bytes are stored as.
///
/// Logical and physical differ for anything packed: a Q4_K value is logically
/// F32 and physically one hundred and forty-four bytes per two hundred and
/// fifty-six elements, and only the second decides its strides. So there are
/// three answers, in order:
///
///   1. the layout's `storage`, which is how a value says it. A quantized KV
///      cache is one, and there is nowhere else for it to say so;
///   2. a `ggml.parameter`'s `source_encoding`, which is how the *file* says
///      it. A parameter leaves the layout's storage absent rather than saying
///      the same thing twice;
///   3. the element type, for everything dense.
inline std::optional<BlockLayout> getStorageLayout(Value value) {
  auto typed = dyn_cast<RankedTensorType>(value.getType());
  if (typed) {
    if (auto layout = dyn_cast_or_null<LayoutAttr>(typed.getEncoding())) {
      if (StringAttr storage = layout.getStorage())
        return getBlockLayout(storage.getValue());
    }
  }
  if (Operation *producer = value.getDefiningOp()) {
    if (auto encoding =
            producer->getAttrOfType<StringAttr>("source_encoding")) {
      return getBlockLayout(encoding.getValue());
    }
  }
  if (!typed)
    return std::nullopt;
  return getBlockLayout(typed.getElementType());
}

/// How far the bytes a layout describes reach, or null when the arithmetic has
/// no affine answer.
///
/// The span runs from the first element to the last: one innermost row, plus
/// one stride for every further step along each outer dimension. A quantized
/// row is counted in blocks, which is why the innermost term divides.
inline Attribute deriveByteSpan(MLIRContext *context, LayoutAttr arrangement,
                                BlockLayout layout,
                                Operation *scope = nullptr) {
  shape_expr::Arithmetic arithmetic;
  declareModuleProducts(scope, arithmetic);
  SmallVector<shape_expr::Expr, kGgmlRank> extents;
  SmallVector<shape_expr::Expr, kGgmlRank> strides;
  for (auto [list, into] :
       {std::make_pair(arrangement.getExtents(), &extents),
        std::make_pair(arrangement.getStrides(), &strides)}) {
    for (Attribute entry : list) {
      std::optional<shape_expr::Expr> view = getShapeExprView(entry);
      if (!view)
        return {};
      into->push_back(*view);
    }
  }
  if (extents.size() != kGgmlRank || strides.size() != kGgmlRank)
    return {};

  shape_expr::Expr total;
  unsigned first = 0;
  if (layout.blockElements == 1) {
    total = shape_expr::constant(layout.typeSize);
  } else {
    total = arithmetic.divide(arithmetic.multiply(extents[0], strides[0]),
                              layout.blockElements);
    first = 1;
  }
  for (unsigned dimension = first; dimension < kGgmlRank; ++dimension) {
    shape_expr::Expr steps =
        arithmetic.add(extents[dimension], shape_expr::constant(-1));
    total =
        arithmetic.add(total, arithmetic.multiply(steps, strides[dimension]));
  }
  if (arithmetic.hasFailure())
    return {};
  return shape_expr::toAttribute(context, total);
}

/// How far the buffer this value occupies reaches.
///
/// A view addresses bytes its root owns, so the span is the root's, not its
/// own: a window on a KV cache may touch four rows of a buffer that holds two
/// hundred and fifty-six.
inline Attribute deriveStorageByteLength(Value value) {
  Value root = getStorageRoot(value);
  if (!root)
    return {};
  LayoutAttr arrangement = getGgmlLayout(root);
  std::optional<BlockLayout> layout = getStorageLayout(root);
  if (!arrangement || !layout)
    return {};
  return deriveByteSpan(value.getContext(), arrangement, *layout,
                        getScope(value));
}

/// Whether a stride vector is the dense one its extents imply.
///
/// Contiguity is a predicate over extents that may themselves vary with the
/// request, so it can vary too: a permuted tensor is contiguous at one token
/// count and not at another. `Varies` says the answer is not the same for every
/// request, which every consumer must read as "not proven dense".
enum class Contiguity { Dense, Strided, Varies };

inline Contiguity deriveContiguity(ArrayRef<Attribute> extents,
                                   ArrayRef<Attribute> strides,
                                   BlockLayout layout,
                                   Operation *scope = nullptr) {
  if (extents.size() != kGgmlRank || strides.size() != kGgmlRank)
    return Contiguity::Varies;

  // The dense stride vector, walked one dimension at a time so that a symbolic
  // extent multiplies the stride that steps over it.
  shape_expr::Arithmetic arithmetic;
  declareModuleProducts(scope, arithmetic);
  SmallVector<shape_expr::Expr, 4> observed;
  SmallVector<shape_expr::Expr, 4> dense;
  for (Attribute stride : strides) {
    std::optional<shape_expr::Expr> view = getShapeExprView(stride);
    if (!view)
      return Contiguity::Varies;
    observed.push_back(*view);
  }
  SmallVector<shape_expr::Expr, 4> extentExprs;
  for (Attribute extent : extents) {
    std::optional<shape_expr::Expr> view = getShapeExprView(extent);
    if (!view)
      return Contiguity::Varies;
    extentExprs.push_back(*view);
  }

  // Dimensions of one element step nowhere, so their strides say nothing
  // (as in GGML's own contiguity): trailing ones are not compared - a view
  // of a cache's first rows keeps the whole cache's outer stride, and past
  // two request-dependent extents (a mask as wide as the session's cache
  // and as tall as the step) the dense stride is no affine expression.
  const auto unit = [&](unsigned dimension) {
    return shape_expr::isConstant(extentExprs[dimension]) &&
           extentExprs[dimension].offset == 1;
  };
  SmallVector<bool, 4> free(kGgmlRank, false);
  dense.push_back(shape_expr::constant(layout.typeSize));
  dense.push_back(
      arithmetic.scale(arithmetic.divide(extentExprs[0], layout.blockElements),
                       layout.typeSize));
  for (unsigned dimension = 2; dimension < kGgmlRank; ++dimension) {
    const bool steps =
        llvm::any_of(llvm::seq<unsigned>(dimension, kGgmlRank),
                     [&](unsigned outer) { return !unit(outer); });
    if (!steps) {
      free[dimension] = true;
      dense.push_back(dense[dimension - 1]);
      continue;
    }
    dense.push_back(
        arithmetic.multiply(dense[dimension - 1], extentExprs[dimension - 1]));
  }
  if (arithmetic.hasFailure())
    return Contiguity::Varies;

  bool allEqual = true;
  bool everyMismatchConstant = true;
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
    if (free[dimension])
      continue;
    const shape_expr::Expr &was = observed[dimension];
    const shape_expr::Expr &want = dense[dimension];
    if (was.hasSameTerms(want) && was.offset == want.offset)
      continue;
    allEqual = false;
    if (!shape_expr::isConstant(was) || !shape_expr::isConstant(want))
      everyMismatchConstant = false;
  }
  if (allEqual)
    return Contiguity::Dense;
  return everyMismatchConstant ? Contiguity::Strided : Contiguity::Varies;
}

/// The contiguity of `value`, derived from what it records about itself.
///
/// Returns nullopt when the value records no GGML layout, which is not the same
/// as being strided: a value produced outside this dialect is dense by
/// construction and its caller decides that.
inline std::optional<Contiguity> deriveContiguity(Value value) {
  LayoutAttr arrangement = getGgmlLayout(value);
  if (!arrangement)
    return std::nullopt;
  std::optional<BlockLayout> layout = getStorageLayout(value);
  if (!layout)
    return Contiguity::Varies;
  return deriveContiguity(arrangement.getExtents(), arrangement.getStrides(),
                          *layout, getScope(value));
}

/// Where this value begins inside the buffer it occupies, when that is a fixed
/// number of bytes. A varying offset returns nullopt, which every caller that
/// requires the start of the buffer must read as "not proven to be zero".
inline std::optional<int64_t> getStorageOffset(Value value) {
  LayoutAttr arrangement = getGgmlLayout(value);
  if (!arrangement)
    return std::nullopt;
  return getShapeExprConstant(arrangement.getOffset());
}

} // namespace ggml
} // namespace mlir

#endif // HIP_DIALECT_GGML_IR_GGMLLAYOUT_H
