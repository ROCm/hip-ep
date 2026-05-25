/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "DimSpecMaterializer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {

// Name of the runtime helper called for RuntimeSlot leaves. Must match
// the symbol declared by GenerateInterface and exported from
// hipdnn_ep_runtime_state.cpp.
static constexpr const char *kReadDimFnName = "hipdnn_ep_state_read_dim";

namespace {

// Materialize a single node recursively. Returns an i64 SSA Value.
Value materializeNode(OpBuilder &builder, Location loc, const DimSpec &spec,
                      int32_t idx, const DimSpecMaterializerCallbacks &cbacks) {
  const auto &n = spec.nodes()[idx];
  Type i64 = builder.getI64Type();
  switch (n.kind) {
  case DimSpecKind::Static:
    return arith::ConstantIntOp::create(builder, loc, i64, n.value);
  case DimSpecKind::InputDim: {
    if (!cbacks.readInputDim) {
      // Defensive: caller must provide an emitter for InputDim leaves.
      // Emit a constant -1 to make the IR well-typed but flagged.
      return arith::ConstantIntOp::create(builder, loc, i64, -1);
    }
    Value v =
        cbacks.readInputDim((unsigned)n.input_index, (unsigned)n.dim_index);
    if (v.getType() != i64) {
      v = arith::IndexCastOp::create(builder, loc, i64, v);
    }
    return v;
  }
  case DimSpecKind::InputValueI64: {
    if (!cbacks.readInputValueI64) {
      return arith::ConstantIntOp::create(builder, loc, i64, -1);
    }
    Value v = cbacks.readInputValueI64((unsigned)n.input_index, n.flat_offset);
    if (v.getType() != i64) {
      v = arith::IndexCastOp::create(builder, loc, i64, v);
    }
    return v;
  }
  case DimSpecKind::RuntimeSlot: {
    auto module =
        builder.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();
    if (!module || !cbacks.statePtr) {
      return arith::ConstantIntOp::create(builder, loc, i64, -1);
    }
    Type ptr = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32 = builder.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(i64, {ptr, i32});
    auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(kReadDimFnName);
    if (!fn) {
      OpBuilder declBuilder(module.getRegion());
      declBuilder.setInsertionPointToStart(module.getBody());
      fn = LLVM::LLVMFuncOp::create(declBuilder, module.getLoc(),
                                    kReadDimFnName, funcType);
      fn.setLinkage(LLVM::Linkage::External);
    }
    Value slotIdConst =
        arith::ConstantIntOp::create(builder, loc, i32, (int64_t)n.slot_id);
    auto callOp = LLVM::CallOp::create(
        builder, loc, fn, ValueRange{cbacks.statePtr, slotIdConst});
    return callOp.getResult();
  }
  case DimSpecKind::Add:
  case DimSpecKind::Sub:
  case DimSpecKind::Mul:
  case DimSpecKind::FloorDiv:
  case DimSpecKind::CeilDiv:
  case DimSpecKind::Min:
  case DimSpecKind::Max: {
    Value lhs = materializeNode(builder, loc, spec, n.lhs, cbacks);
    Value rhs = materializeNode(builder, loc, spec, n.rhs, cbacks);
    switch (n.kind) {
    case DimSpecKind::Add:
      return arith::AddIOp::create(builder, loc, lhs, rhs);
    case DimSpecKind::Sub:
      return arith::SubIOp::create(builder, loc, lhs, rhs);
    case DimSpecKind::Mul:
      return arith::MulIOp::create(builder, loc, lhs, rhs);
    case DimSpecKind::FloorDiv:
      return arith::DivSIOp::create(builder, loc, lhs, rhs);
    case DimSpecKind::CeilDiv: {
      // ceildiv(a, b) = floor((a + b - 1) / b), assuming b > 0 (well-formed
      // shape arithmetic).
      Value one = arith::ConstantIntOp::create(builder, loc, lhs.getType(), 1);
      Value bm1 = arith::SubIOp::create(builder, loc, rhs, one);
      Value sum = arith::AddIOp::create(builder, loc, lhs, bm1);
      return arith::DivSIOp::create(builder, loc, sum, rhs);
    }
    case DimSpecKind::Min:
      return arith::MinSIOp::create(builder, loc, lhs, rhs);
    case DimSpecKind::Max:
      return arith::MaxSIOp::create(builder, loc, lhs, rhs);
    default:
      break;
    }
    break;
  }
  }
  return arith::ConstantIntOp::create(builder, loc, i64, -1);
}

} // namespace

Value materializeDimSpec(OpBuilder &builder, Location loc, const DimSpec &spec,
                         const DimSpecMaterializerCallbacks &cbacks) {
  if (spec.nodes().empty()) {
    return arith::ConstantIntOp::create(builder, loc, builder.getI64Type(), -1);
  }
  return materializeNode(builder, loc, spec, 0, cbacks);
}

} // namespace hip
} // namespace mlir
