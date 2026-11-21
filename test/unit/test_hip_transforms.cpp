/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hip/IR/HipDialect.h"
#include "hip/Dialect/Hip/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::hip::HipDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  constexpr char source[] = R"mlir(
    func.func @extent() -> i64 {
      %c1 = arith.constant 1 : index
      %shape = hip.constant {value = dense<[3, 7]> : tensor<2xi64>} : tensor<2xi64>
      %extent = tensor.extract %shape[%c1] : tensor<2xi64>
      return %extent : i64
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hip::createFoldConstantExtractsPass());
  if (mlir::failed(pm.run(*module)))
    return 1;

  auto func = module->lookupSymbol<mlir::func::FuncOp>("extent");
  auto ret = mlir::cast<mlir::func::ReturnOp>(func.front().getTerminator());
  auto extent = ret.getOperand(0).getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!extent || extent.value() != 7) {
    llvm::errs() << "expected the HIP pass to fold the returned extent to 7\n";
    return 1;
  }
  return 0;
}
