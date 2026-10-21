// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "routed_experts_fusion.hpp"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <optional>

namespace hip::fusion_transform {
namespace {
using namespace mlir;

Value peel(Value value) {
  while (auto cast = value.getDefiningOp<tensor::CastOp>())
    value = cast.getSource();
  return value;
}
DenseElementsAttr payload(Value value) {
  auto constant = peel(value).getDefiningOp<mlir::hip::ConstantOp>();
  return constant ? dyn_cast<DenseElementsAttr>(constant->getAttr("value"))
                  : DenseElementsAttr();
}
std::optional<int64_t> integer(Value value) {
  auto data = payload(value);
  if (!data || data.getNumElements() != 1 ||
      !isa<IntegerType>(data.getElementType()))
    return std::nullopt;
  return (*data.getValues<APInt>().begin()).getSExtValue();
}
bool zero(Value value) {
  auto data = payload(value);
  return data && data.isSplat() && isa<FloatType>(data.getElementType()) &&
         data.getSplatValue<APFloat>().isZero();
}
bool lastAxisSum(mlir::hip::ReduceSumOp op) {
  auto axis = integer(op.getAxes());
  return axis && (*axis == -1 || *axis == 2) && op.getKeepdims() == 1;
}

// Prove the leading dimensions from data flow, not from dimension names.
bool leadingFrom(Value value, Value input, int depth = 0) {
  value = peel(value);
  if (value == input)
    return true;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || depth > 20)
    return false;
  if (type.getRank() <= 1 || (type.getRank() == 3 && type.getDimSize(0) == 1 &&
                              type.getDimSize(1) == 1))
    return true;
  auto *op = value.getDefiningOp();
  if (!op)
    return false;
  auto recurse = [&](Value v) { return leadingFrom(v, input, depth + 1); };
  if (auto cast = dyn_cast<mlir::hip::CastOp>(op))
    return recurse(cast.getInput());
  if (isa<mlir::hip::SigmoidOp, mlir::hip::ReciprocalOp>(op))
    return recurse(op->getOperand(1));
  if (auto matmul = dyn_cast<mlir::hip::MatmulOp>(op)) {
    auto b = dyn_cast<RankedTensorType>(matmul.getB().getType());
    return b && b.getRank() == 2 && matmul.getTransA() == 0 &&
           recurse(matmul.getA());
  }
  if (auto top = dyn_cast<mlir::hip::TopKOp>(op))
    return (top.getAxis() == -1 || top.getAxis() == 2) && recurse(top.getX());
  if (auto gather = dyn_cast<mlir::hip::GatherElementsOp>(op))
    return recurse(gather.getIndices());
  if (auto sum = dyn_cast<mlir::hip::ReduceSumOp>(op))
    return lastAxisSum(sum) && recurse(sum.getData());
  if (isa<mlir::hip::AddOp, mlir::hip::MulOp>(op))
    return recurse(op->getOperand(1)) && recurse(op->getOperand(2));
  return false;
}

struct Expert {
  int64_t id;
  Value input, indices, gates, handle;
  mlir::hip::MatMulNBitsOp up, down;
};

std::optional<Expert> matchExpert(Value value) {
  auto product = peel(value).getDefiningOp<mlir::hip::MulOp>();
  if (!product)
    return std::nullopt;
  auto down = peel(product.getLhs()).getDefiningOp<mlir::hip::MatMulNBitsOp>();
  auto gate = peel(product.getRhs()).getDefiningOp<mlir::hip::CastOp>();
  if (!down || !gate)
    return std::nullopt;
  auto gateType = dyn_cast<RankedTensorType>(gate->getResult(0).getType());
  if (!gateType || gateType.getRank() != 3 ||
      !gateType.getElementType().isF16() || gateType.getDimSize(2) != 1)
    return std::nullopt;
  auto sum = peel(gate.getInput()).getDefiningOp<mlir::hip::ReduceSumOp>();
  if (!sum || !lastAxisSum(sum))
    return std::nullopt;
  auto maskProduct = peel(sum.getData()).getDefiningOp<mlir::hip::MulOp>();
  if (!maskProduct)
    return std::nullopt;
  auto mask = peel(maskProduct.getRhs()).getDefiningOp<mlir::hip::CastOp>();
  if (!mask)
    return std::nullopt;
  auto gateInputType = dyn_cast<RankedTensorType>(gate.getInput().getType());
  auto maskType = dyn_cast<RankedTensorType>(mask->getResult(0).getType());
  if (!gateInputType || !gateInputType.getElementType().isF32() || !maskType ||
      !maskType.getElementType().isF32())
    return std::nullopt;
  auto equality = peel(mask.getInput()).getDefiningOp<mlir::hip::EqualOp>();
  if (!equality)
    return std::nullopt;
  auto id = integer(equality.getRhs());
  if (!id)
    return std::nullopt;
  auto square = peel(down.getA()).getDefiningOp<mlir::hip::MulOp>();
  if (!square || square.getLhs() != square.getRhs())
    return std::nullopt;
  auto maximum = peel(square.getLhs()).getDefiningOp<mlir::hip::MaxOp>();
  if (!maximum || !zero(maximum->getOperand(2)))
    return std::nullopt;
  auto up =
      peel(maximum->getOperand(1)).getDefiningOp<mlir::hip::MatMulNBitsOp>();
  if (!up)
    return std::nullopt;
  // The fused kernel preserves FP16 rounding at each expert boundary.
  for (Value result :
       {up->getResult(0), maximum->getResult(0), square->getResult(0),
        down->getResult(0), product->getResult(0)}) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type || type.getRank() != 3 || !type.getElementType().isF16() ||
        type.getEncoding())
      return std::nullopt;
  }
  Value handle = product->getOperand(0);
  for (Operation *op :
       {down.getOperation(), gate.getOperation(), sum.getOperation(),
        maskProduct.getOperation(), mask.getOperation(),
        equality.getOperation(), square.getOperation(), maximum.getOperation(),
        up.getOperation()})
    if (op->getOperand(0) != handle)
      return std::nullopt;
  return Expert{*id,
                peel(up.getA()),
                peel(equality.getLhs()),
                peel(maskProduct.getLhs()),
                handle,
                up,
                down};
}

bool validMatrix(mlir::hip::MatMulNBitsOp op, int64_t n, int64_t reduction,
                 int64_t block) {
  if (op.getBits() != 4 || op.getBlockSize() != block ||
      op.getK() != reduction || op.getN() != n || op.getBias() ||
      !op.getZeroPoints() || op.getZpElemSize() != 1)
    return false;
  int64_t groups = (reduction + block - 1) / block;
  auto w = dyn_cast<RankedTensorType>(op.getB().getType());
  auto s = dyn_cast<RankedTensorType>(op.getScales().getType());
  auto z = dyn_cast<RankedTensorType>(op.getZeroPoints().getType());
  return w && s && z && !w.getEncoding() && !s.getEncoding() &&
         !z.getEncoding() &&
         w.getShape() == ArrayRef<int64_t>({n, groups, block / 2}) &&
         s.getShape() == ArrayRef<int64_t>({n, groups}) &&
         z.getShape() == ArrayRef<int64_t>({n, (groups + 1) / 2}) &&
         s.getElementType().isF16() && isa<IntegerType>(w.getElementType()) &&
         cast<IntegerType>(w.getElementType()).getWidth() == 8 &&
         isa<IntegerType>(z.getElementType()) &&
         cast<IntegerType>(z.getElementType()).getWidth() == 8;
}

struct FuseExperts : OpRewritePattern<mlir::hip::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::hip::AddOp root,
                                PatternRewriter &rewriter) const override {
    SmallVector<Expert> experts;
    Value current = root->getResult(0);
    while (auto add = peel(current).getDefiningOp<mlir::hip::AddOp>()) {
      auto expert = matchExpert(add.getRhs());
      if (!expert || experts.size() >= 128 ||
          add->getOperand(0) != expert->handle)
        return failure();
      experts.push_back(*expert);
      current = add.getLhs();
    }
    auto first = matchExpert(current);
    if (!first)
      return failure();
    experts.push_back(*first);
    std::reverse(experts.begin(), experts.end());
    auto &base = experts.front();
    auto inputType = dyn_cast<RankedTensorType>(base.input.getType());
    auto indicesType = dyn_cast<RankedTensorType>(base.indices.getType());
    auto gatesType = dyn_cast<RankedTensorType>(base.gates.getType());
    auto top = base.indices.getDefiningOp<mlir::hip::TopKOp>();
    if (!top || base.indices != top->getResult(1) || !inputType ||
        inputType.getRank() != 3 || !inputType.getElementType().isF16() ||
        inputType.getEncoding() || !indicesType || indicesType.getRank() != 3 ||
        !indicesType.getElementType().isInteger(64) ||
        indicesType.getEncoding() || !gatesType || gatesType.getEncoding() ||
        !gatesType.getElementType().isF32() ||
        gatesType.getShape() != indicesType.getShape() ||
        root->getResult(0).getType() != inputType)
      return failure();
    auto scoreType = dyn_cast<RankedTensorType>(top.getX().getType());
    int64_t count = experts.size(), k = indicesType.getDimSize(2);
    int64_t hidden = inputType.getDimSize(2), intermediate = base.up.getN();
    int64_t block = base.up.getBlockSize();
    if (!scoreType || scoreType.getRank() != 3 ||
        scoreType.getDimSize(2) != count || count < 2 || count > 128 || k < 1 ||
        k > count || hidden < 1 || intermediate < 1 ||
        (block != 32 && block != 64 && block != 128) ||
        !leadingFrom(base.indices, base.input) ||
        !leadingFrom(base.gates, base.input))
      return failure();
    for (int axis = 0; axis < 2; ++axis)
      if (inputType.getDimSize(axis) != indicesType.getDimSize(axis))
        return failure();
    SmallVector<Value> parameters;
    for (auto [id, expert] : llvm::enumerate(experts)) {
      if (expert.id != static_cast<int64_t>(id) || expert.input != base.input ||
          expert.indices != base.indices || expert.gates != base.gates ||
          expert.handle != base.handle ||
          !validMatrix(expert.up, intermediate, hidden, block) ||
          !validMatrix(expert.down, hidden, intermediate, block))
        return failure();
      parameters.append({expert.up.getB(), expert.up.getScales(),
                         expert.up.getZeroPoints(), expert.down.getB(),
                         expert.down.getScales(), expert.down.getZeroPoints()});
    }
    // Use fresh shape queries. The old destination depends on the dense chain.
    SmallVector<Value> dynamic;
    for (int axis = 0; axis < 3; ++axis)
      if (inputType.isDynamicDim(axis))
        dynamic.push_back(
            tensor::DimOp::create(rewriter, root.getLoc(), base.input, axis));
    Value output =
        tensor::EmptyOp::create(rewriter, root.getLoc(), inputType.getShape(),
                                inputType.getElementType(), dynamic);
    auto fused = mlir::hip::RoutedExpertsOp::create(
        rewriter, root.getLoc(), base.handle, base.input, base.indices,
        base.gates, parameters, output, count, intermediate, k, block);
    rewriter.replaceOp(root, fused->getResults());
    return success();
  }
};
} // namespace
void populateRoutedExpertsPattern(mlir::RewritePatternSet &patterns) {
  patterns.add<FuseExperts>(patterns.getContext());
}
} // namespace hip::fusion_transform
