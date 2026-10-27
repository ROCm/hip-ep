/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- IfResultDims.cpp - Size dynamic hip.if outputs on the host ---------===//
//
// `onnx-if-outline` sizes each `?` dimension of a hip.if o_init with a
// `hip.if_result_dim` placeholder, because the branch bodies are still ONNX
// and their output sizes are unknown there. Once the bodies are lowered, each
// dynamic output size is host arithmetic on the branch arguments: DPS results
// take their init's size, `tensor.empty` carries its sizes, and those sizes
// are `tensor.dim` of earlier values plus index math.
//
// This pass replaces every placeholder with that arithmetic. For each branch
// it walks back from the returned value to the branch arguments, rebuilds the
// expression in the caller on the hip.if captures, and selects between the
// then and else sizes on the hip.if condition:
//
//   %d = hip.if_result_dim(%c) captures(%ids, %past : ...)
//            then @then else @else {result_index = 1, dim = 2}
// becomes, when @then returns concat(past, new) and @else returns new:
//   %p    = tensor.dim %past, %c2
//   %s    = tensor.dim %ids, %c1
//   %then = arith.addi %p, %s
//   %d    = arith.select %c, %then, %s
//
// A size that depends on tensor data (e.g. NonZero) has no such expression
// and is reported as an error naming the If output.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

#define DEBUG_TYPE "hip-resolve-if-result-dims"

namespace mlir {
namespace hip {
namespace {

constexpr unsigned kMaxDepth = 512;

/// Rebuilds sizes of values in one branch function as caller-side host
/// arithmetic on the hip.if captures. Branch argument 0 is the context;
/// argument i > 0 is capture i - 1.
class BranchDimResolver {
public:
  BranchDimResolver(OpBuilder &callerBuilder, func::FuncOp branch,
                    ValueRange captures)
      : b(callerBuilder), branch(branch), captures(captures) {}

  FailureOr<Value> dim(Value t, int64_t d, unsigned depth = 0) {
    auto ty = dyn_cast<RankedTensorType>(t.getType());
    if (!ty || d < 0 || d >= ty.getRank() || depth > kMaxDepth)
      return failure();
    if (!ty.isDynamicDim(d))
      return constant(ty.getDimSize(d));
    if (!inBranch(t))
      return b.createOrFold<tensor::DimOp>(branch.getLoc(), t, d);

    auto key = std::make_pair(t, d);
    if (auto it = dims.find(key); it != dims.end())
      return it->second;
    FailureOr<Value> v = computeDim(t, d, depth);
    if (succeeded(v))
      dims[key] = *v;
    else if (!blocker)
      blocker = t.getDefiningOp();
    return v;
  }

  /// Deepest op whose size could not be rebuilt, for the diagnostic.
  Operation *blocker = nullptr;

private:
  /// Values outside the branch are caller values that dominate the hip.if
  /// (they reach it as captures), so the rebuilt arithmetic uses them as is.
  bool inBranch(Value v) {
    Region *region = v.getParentRegion();
    return region && branch->isAncestor(region->getParentOp());
  }

  Value constant(int64_t v) {
    return arith::ConstantIndexOp::create(b, branch.getLoc(), v);
  }

  FailureOr<Value> capture(BlockArgument arg) {
    if (arg.getOwner() != &branch.getBody().front() || arg.getArgNumber() == 0)
      return failure();
    unsigned idx = arg.getArgNumber() - 1;
    if (idx >= captures.size())
      return failure();
    return captures[idx];
  }

  FailureOr<Value> computeDim(Value t, int64_t d, unsigned depth) {
    Location loc = branch.getLoc();
    if (auto arg = dyn_cast<BlockArgument>(t)) {
      FailureOr<Value> outer = capture(arg);
      if (failed(outer))
        return failure();
      return b.createOrFold<tensor::DimOp>(loc, *outer, d);
    }

    auto res = cast<OpResult>(t);
    Operation *def = res.getOwner();

    if (auto empty = dyn_cast<tensor::EmptyOp>(def))
      return scalar(empty.getDynamicSize(d), depth + 1);
    if (auto castOp = dyn_cast<tensor::CastOp>(def))
      return dim(castOp.getSource(), d, depth + 1);
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(def)) {
      FailureOr<Value> e = element(reshape.getShape(), d, depth + 1);
      if (failed(e))
        return failure();
      return castIndex(*e, b.getIndexType());
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(def)) {
      if (dps.hasPureTensorSemantics())
        return dim(dps.getTiedOpOperand(res)->get(), d, depth + 1);
    }

    // Upstream tensor ops (expand_shape, collapse_shape, extract_slice, pad,
    // concat, ...) reify their sizes next to the op; rebuild from those.
    if (auto reify = dyn_cast<ReifyRankedShapedTypeOpInterface>(def)) {
      OpBuilder inner(def);
      ReifiedRankedShapedTypeDims reified;
      if (failed(reify.reifyResultShapes(inner, reified)))
        return failure();
      OpFoldResult ofr = reified[res.getResultNumber()][d];
      if (std::optional<int64_t> c = getConstantIntValue(ofr))
        return constant(*c);
      return scalar(cast<Value>(ofr), depth + 1);
    }
    return failure();
  }

  /// Clone the scalar (index / integer) computation of `s` into the caller.
  FailureOr<Value> scalar(Value s, unsigned depth) {
    if (depth > kMaxDepth)
      return failure();
    if (!inBranch(s))
      return s;
    if (auto it = scalars.find(s); it != scalars.end())
      return it->second;
    FailureOr<Value> v = computeScalar(s, depth);
    if (succeeded(v))
      scalars[s] = *v;
    else if (!blocker)
      blocker = s.getDefiningOp();
    return v;
  }

  FailureOr<Value> computeScalar(Value s, unsigned depth) {
    if (auto arg = dyn_cast<BlockArgument>(s)) {
      FailureOr<Value> outer = capture(arg);
      if (failed(outer) || outer->getType() != s.getType())
        return failure();
      return outer;
    }

    Operation *def = s.getDefiningOp();
    if (auto dimOp = dyn_cast<tensor::DimOp>(def)) {
      std::optional<int64_t> idx = dimOp.getConstantIndex();
      if (!idx)
        return failure();
      return dim(dimOp.getSource(), *idx, depth + 1);
    }

    // A shape value read from a small integer tensor: an extract with
    // constant indices, or a readback the EP would otherwise sync on.
    if (auto extract = dyn_cast<tensor::ExtractOp>(def)) {
      Value src = extract.getTensor();
      auto srcTy = cast<RankedTensorType>(src.getType());
      if (!srcTy.hasStaticShape())
        return failure();
      int64_t linear = 0;
      for (auto [i, size] : llvm::zip(extract.getIndices(), srcTy.getShape())) {
        std::optional<int64_t> c = getConstantIntValue(i);
        if (!c)
          return failure();
        linear = linear * size + *c;
      }
      FailureOr<Value> e = element(src, linear, depth + 1);
      if (failed(e))
        return failure();
      return castIndex(*e, s.getType());
    }
    if (auto readback = dyn_cast<ReadbackScalarOp>(def)) {
      FailureOr<Value> e = element(readback.getScalar(), 0, depth + 1);
      if (failed(e))
        return failure();
      return castIndex(*e, s.getType());
    }

    Dialect *dialect = def->getDialect();
    bool hostScalarOp = dialect &&
                        (isa<arith::ArithDialect>(dialect) ||
                         dialect->getNamespace() == "affine") &&
                        def->getNumRegions() == 0 && isMemoryEffectFree(def);
    if (!hostScalarOp)
      return failure();

    IRMapping mapping;
    for (Value operand : def->getOperands()) {
      if (!operand.getType().isIntOrIndexOrFloat())
        return failure();
      FailureOr<Value> mapped = scalar(operand, depth + 1);
      if (failed(mapped))
        return failure();
      mapping.map(operand, *mapped);
    }
    // Both branches' sizes are computed whichever branch runs, so a divisor
    // that is zero only on the untaken branch must not trap on the host.
    bool isDivision = isa<arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
                          arith::CeilDivUIOp, arith::FloorDivSIOp,
                          arith::RemSIOp, arith::RemUIOp>(def);
    std::optional<int64_t> knownDivisor =
        isDivision ? getConstantIntValue(mapping.lookup(def->getOperand(1)))
                   : std::nullopt;
    if (isDivision && (!knownDivisor || *knownDivisor == 0)) {
      Value divisor = mapping.lookup(def->getOperand(1));
      Location loc = def->getLoc();
      Value zero =
          arith::ConstantOp::create(b, loc, b.getZeroAttr(divisor.getType()));
      Value one = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(divisor.getType(), 1));
      Value isZero = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                           divisor, zero);
      mapping.map(
          def->getOperand(1),
          arith::SelectOp::create(b, loc, isZero, one, divisor).getResult());
    }
    Operation *clone = b.clone(*def, mapping);
    return clone->getResult(cast<OpResult>(s).getResultNumber());
  }

  /// Initial value of a small constant hip-externalize-constants left inline
  /// as a `memref.global`. Globals backed by external data have no value here.
  DenseElementsAttr externalizedConstant(Value buffer) {
    auto get = buffer.getDefiningOp<memref::GetGlobalOp>();
    if (!get)
      return {};
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        get, get.getNameAttr());
    if (!global || !global.getConstant())
      return {};
    return dyn_cast_or_null<DenseElementsAttr>(global.getConstantInitValue());
  }

  /// `index` value `v` as the integer or index type `ty`.
  FailureOr<Value> castIndex(Value v, Type ty) {
    if (v.getType() == ty)
      return v;
    if (!ty.isIndex() && !ty.isSignlessInteger())
      return failure();
    return arith::IndexCastOp::create(b, branch.getLoc(), ty, v).getResult();
  }

  /// Element `idx` (row-major) of a small static integer shape tensor, as a
  /// caller-side `index` value. Covers the spellings ONNX shape arithmetic
  /// takes after conversion: constants, `from_elements` of dims, reshapes,
  /// `insert_slice` chains from Concat, and elementwise hip integer ops.
  FailureOr<Value> element(Value t, int64_t idx, unsigned depth) {
    auto ty = dyn_cast<RankedTensorType>(t.getType());
    if (!ty || !ty.hasStaticShape() || !isa<IntegerType>(ty.getElementType()) ||
        idx < 0 || idx >= ty.getNumElements() || depth > kMaxDepth)
      return failure();
    auto key = std::make_pair(t, idx);
    if (auto it = elements.find(key); it != elements.end())
      return it->second;
    FailureOr<Value> v = computeElement(t, ty, idx, depth);
    if (succeeded(v))
      elements[key] = *v;
    else if (!blocker)
      blocker = t.getDefiningOp();
    return v;
  }

  FailureOr<Value> computeElement(Value t, RankedTensorType ty, int64_t idx,
                                  unsigned depth) {
    // A shape tensor computed before the If and captured: evaluate the
    // caller's value instead.
    if (auto arg = dyn_cast<BlockArgument>(t)) {
      FailureOr<Value> outer = capture(arg);
      if (failed(outer) || !inBranch(t))
        return failure();
      return element(*outer, idx, depth + 1);
    }
    Operation *def = t.getDefiningOp();
    Location loc = branch.getLoc();

    DenseElementsAttr dense;
    if (auto cst = dyn_cast<arith::ConstantOp>(def))
      dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    else if (isa<ConstantOp>(def))
      dense = def->getAttrOfType<DenseElementsAttr>("value");
    else if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(def))
      dense = externalizedConstant(toTensor.getBuffer());
    if (dense) {
      if (!isa<IntegerType>(dense.getElementType()))
        return failure();
      APInt v = dense.isSplat() ? dense.getSplatValue<APInt>()
                                : *(dense.value_begin<APInt>() + idx);
      return constant(v.getSExtValue());
    }

    if (auto from = dyn_cast<tensor::FromElementsOp>(def)) {
      Value elem = from.getElements()[idx];
      if (auto cast = elem.getDefiningOp<arith::IndexCastOp>())
        if (cast.getIn().getType().isIndex())
          return scalar(cast.getIn(), depth + 1);
      FailureOr<Value> v = scalar(elem, depth + 1);
      if (failed(v))
        return failure();
      return castIndex(*v, b.getIndexType());
    }

    if (isa<tensor::CastOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
            tensor::ReshapeOp>(def))
      return element(def->getOperand(0), idx, depth + 1);

    // Concat lowers to insert_slice chains; the element comes from the
    // source when it falls inside the inserted box, else from the dest.
    if (auto insert = dyn_cast<tensor::InsertSliceOp>(def)) {
      SmallVector<int64_t> pos(ty.getRank());
      int64_t rem = idx;
      for (int64_t d = ty.getRank() - 1; d >= 0; --d) {
        pos[d] = rem % ty.getDimSize(d);
        rem /= ty.getDimSize(d);
      }
      auto srcTy = dyn_cast<RankedTensorType>(insert.getSourceType());
      if (!srcTy || !srcTy.hasStaticShape() || srcTy.getRank() != ty.getRank())
        return failure();
      bool inside = true;
      int64_t srcLinear = 0;
      for (int64_t d = 0; d < ty.getRank(); ++d) {
        std::optional<int64_t> off =
            getConstantIntValue(insert.getMixedOffsets()[d]);
        std::optional<int64_t> stride =
            getConstantIntValue(insert.getMixedStrides()[d]);
        if (!off || stride != std::optional<int64_t>(1))
          return failure();
        int64_t rel = pos[d] - *off;
        if (rel < 0 || rel >= srcTy.getDimSize(d))
          inside = false;
        srcLinear = srcLinear * srcTy.getDimSize(d) + rel;
      }
      if (inside)
        return element(insert.getSource(), srcLinear, depth + 1);
      return element(insert.getDest(), idx, depth + 1);
    }

    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      auto srcTy = dyn_cast<RankedTensorType>(slice.getSourceType());
      if (!srcTy || !srcTy.hasStaticShape())
        return failure();
      llvm::SmallBitVector dropped = slice.getDroppedDims();
      SmallVector<int64_t> pos(ty.getRank());
      int64_t rem = idx;
      for (int64_t d = ty.getRank() - 1; d >= 0; --d) {
        pos[d] = rem % ty.getDimSize(d);
        rem /= ty.getDimSize(d);
      }
      int64_t srcLinear = 0;
      unsigned resultDim = 0;
      for (int64_t d = 0; d < srcTy.getRank(); ++d) {
        std::optional<int64_t> off =
            getConstantIntValue(slice.getMixedOffsets()[d]);
        std::optional<int64_t> stride =
            getConstantIntValue(slice.getMixedStrides()[d]);
        if (!off || stride != std::optional<int64_t>(1))
          return failure();
        int64_t p = *off + (dropped.test(d) ? 0 : pos[resultDim++]);
        srcLinear = srcLinear * srcTy.getDimSize(d) + p;
      }
      return element(slice.getSource(), srcLinear, depth + 1);
    }

    auto dps = dyn_cast<DestinationStyleOpInterface>(def);
    if (!dps || !isa<HipDialect>(def->getDialect()))
      return failure();
    SmallVector<Value> ins;
    for (Value in : dps.getDpsInputs())
      if (isa<RankedTensorType>(in.getType()))
        ins.push_back(in);

    // An operand with one element broadcasts; otherwise shapes must match.
    auto operandElement = [&](Value operand) -> FailureOr<Value> {
      auto opTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!opTy || !opTy.hasStaticShape())
        return failure();
      if (opTy.getNumElements() == 1)
        return element(operand, 0, depth + 1);
      if (opTy.getShape() != ty.getShape())
        return failure();
      return element(operand, idx, depth + 1);
    };

    if (isa<CastOp, NegOp>(def) && ins.size() == 1) {
      FailureOr<Value> x = operandElement(ins[0]);
      if (failed(x) || isa<CastOp>(def))
        return x;
      return arith::SubIOp::create(b, loc, constant(0), *x).getResult();
    }

    if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp>(def) && ins.size() == 2) {
      FailureOr<Value> l = operandElement(ins[0]);
      FailureOr<Value> r = operandElement(ins[1]);
      if (failed(l) || failed(r))
        return failure();
      if (isa<AddOp>(def))
        return arith::AddIOp::create(b, loc, *l, *r).getResult();
      if (isa<SubOp>(def))
        return arith::SubIOp::create(b, loc, *l, *r).getResult();
      if (isa<MulOp>(def))
        return arith::MulIOp::create(b, loc, *l, *r).getResult();
      if (isa<DivOp>(def))
        return arith::DivSIOp::create(b, loc, *l, *r).getResult();
      if (isa<MaxOp>(def))
        return arith::MaxSIOp::create(b, loc, *l, *r).getResult();
      return arith::MinSIOp::create(b, loc, *l, *r).getResult();
    }

    // Reshape's "0 copies the input dim": Where(Equal(shape, 0), dims, shape).
    // Booleans are carried as index 0 / non-zero.
    if (isa<EqualOp>(def) && ins.size() == 2) {
      FailureOr<Value> l = operandElement(ins[0]);
      FailureOr<Value> r = operandElement(ins[1]);
      if (failed(l) || failed(r))
        return failure();
      Value eq =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, *l, *r);
      return arith::SelectOp::create(b, loc, eq, constant(1), constant(0))
          .getResult();
    }
    if (isa<WhereOp>(def) && ins.size() == 3) {
      FailureOr<Value> c = operandElement(ins[0]);
      FailureOr<Value> x = operandElement(ins[1]);
      FailureOr<Value> y = operandElement(ins[2]);
      if (failed(c) || failed(x) || failed(y))
        return failure();
      Value pick = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, *c,
                                         constant(0));
      return arith::SelectOp::create(b, loc, pick, *x, *y).getResult();
    }

    // Gather(Shape(x), k): 1-D data along axis 0 with constant indices.
    if (isa<GatherOp>(def) && ins.size() == 2) {
      auto dataTy = dyn_cast<RankedTensorType>(ins[0].getType());
      if (!dataTy || dataTy.getRank() != 1 || !dataTy.hasStaticShape())
        return failure();
      if (auto axis = def->getAttrOfType<IntegerAttr>("axis"))
        if (axis.getInt() != 0 && axis.getInt() != -1)
          return failure();
      FailureOr<Value> k = operandElement(ins[1]);
      if (failed(k))
        return failure();
      std::optional<int64_t> kc = getConstantIntValue(*k);
      if (!kc)
        return failure();
      int64_t n = dataTy.getDimSize(0);
      int64_t at = *kc < 0 ? *kc + n : *kc;
      return element(ins[0], at, depth + 1);
    }
    return failure();
  }

  OpBuilder &b;
  func::FuncOp branch;
  ValueRange captures;
  llvm::DenseMap<std::pair<Value, int64_t>, Value> dims;
  llvm::DenseMap<Value, Value> scalars;
  llvm::DenseMap<std::pair<Value, int64_t>, Value> elements;
};

static FailureOr<Value> resolveBranchDim(OpBuilder &b, IfResultDimOp op,
                                         func::FuncOp branch) {
  auto ret = dyn_cast<func::ReturnOp>(branch.getBody().front().getTerminator());
  unsigned resultIdx = op.getResultIndex();
  if (!ret || resultIdx >= ret.getNumOperands())
    return failure();
  BranchDimResolver resolver(b, branch, op.getCaptures());
  FailureOr<Value> size = resolver.dim(ret.getOperand(resultIdx), op.getDim());
  if (failed(size)) {
    Operation *blocker = resolver.blocker;
    if (blocker && blocker->getName().getDialectNamespace() == "onnx") {
      op.emitOpError("cannot size If output ")
          << op.getResultIndex() << " in " << branch.getName() << ": '"
          << blocker->getName() << "' (" << blocker->getLoc()
          << ") was not converted to the hip dialect";
      return failure();
    }
    std::string stop = "<branch argument>";
    if (blocker) {
      stop.clear();
      llvm::raw_string_ostream os(stop);
      OpPrintingFlags flags;
      flags.elideLargeElementsAttrs(8).skipRegions().useLocalScope();
      blocker->print(os, flags);
      for (Value operand : blocker->getOperands()) {
        if (Operation *producer = operand.getDefiningOp()) {
          os << " <- ";
          producer->print(os, flags);
        }
      }
    }
    op.emitOpError("dim ")
        << op.getDim() << " of If output " << op.getResultIndex()
        << " is not host shape arithmetic on the If inputs in "
        << branch.getName()
        << " (data-dependent output shapes are not supported); size stops "
           "at: "
        << stop;
  }
  return size;
}

struct ResolveIfResultDimsPass
    : public PassWrapper<ResolveIfResultDimsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveIfResultDimsPass)

  StringRef getArgument() const override {
    return "hip-resolve-if-result-dims";
  }

  StringRef getDescription() const override {
    return "Replace hip.if_result_dim with host size arithmetic selected "
           "between the then and else branches";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<IfResultDimOp> ops;
    module.walk([&](IfResultDimOp op) { ops.push_back(op); });

    bool failedAny = false;
    for (IfResultDimOp op : ops) {
      auto thenFn = module.lookupSymbol<func::FuncOp>(op.getThenFunc());
      auto elseFn = module.lookupSymbol<func::FuncOp>(op.getElseFunc());
      if (!thenFn || !elseFn) {
        op.emitOpError("then/else function not found");
        failedAny = true;
        continue;
      }

      OpBuilder b(op);
      FailureOr<Value> thenSize = resolveBranchDim(b, op, thenFn);
      FailureOr<Value> elseSize = resolveBranchDim(b, op, elseFn);
      if (failed(thenSize) || failed(elseSize)) {
        failedAny = true;
        continue;
      }

      Value size = *thenSize == *elseSize
                       ? *thenSize
                       : b.createOrFold<arith::SelectOp>(
                             op.getLoc(), op.getCond(), *thenSize, *elseSize);
      op.replaceAllUsesWith(size);
      op.erase();
    }
    if (failedAny)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createResolveIfResultDimsPass() {
  return std::make_unique<ResolveIfResultDimsPass>();
}

} // namespace hip
} // namespace mlir
