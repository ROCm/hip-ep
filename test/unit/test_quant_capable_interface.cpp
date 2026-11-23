/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// HipQuantCapableOpInterface queries and their free-function forms: C++ API,
// so not reachable from LIT. Covers float, quantized and legalized ops, and an
// op without the interface. Plain main() (no GTest).

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipQuantTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond) {
    llvm::outs() << "[ OK ] " << what << "\n";
  } else {
    llvm::errs() << "[FAIL] " << what << "\n";
    ++g_failures;
  }
}

constexpr llvm::StringLiteral kModule = R"mlir(
!qa = !quant.uniform<u16:f32, 0.25:32768>
!qb = !quant.uniform<u16:f32, 0.5:32768>
!qy = !quant.uniform<u16:f32, 0.125:32768>
func.func @float_add(%ctx: !hip.context, %a: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%a, %a : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %s : tensor<8xf32>
}
func.func @quant_add(%ctx: !hip.context, %a: tensor<8x!qa>, %b: tensor<8x!qb>) -> tensor<8x!qy> {
  %e = tensor.empty() : tensor<8x!qy>
  %s = hip.add(%ctx) ins(%a, %b : tensor<8x!qa>, tensor<8x!qb>) outs(%e : tensor<8x!qy>) -> tensor<8x!qy>
  return %s : tensor<8x!qy>
}
func.func @legalized_add(%ctx: !hip.context, %a: memref<8xui16>, %b: memref<8xui16>, %o: memref<8xui16>) {
  hip.add(%ctx) ins(%a, %b : memref<8xui16>, memref<8xui16>) outs(%o : memref<8xui16>)
      {quant_types = [!qa, !qb, !qy]}
  return
}
func.func @legalized_float_add(%ctx: !hip.context, %a: memref<8xf16>, %o: memref<8xf16>) {
  hip.add(%ctx) ins(%a, %a : memref<8xf16>, memref<8xf16>) outs(%o : memref<8xf16>)
      {quant_types = [f16, f16, f16]}
  return
}
func.func @matmul_bias(%ctx: !hip.context, %a: tensor<4x8x!qa>,
                       %w: tensor<8x6x!quant.uniform<i8:f32, 0.0625:3>>,
                       %c: tensor<6x!quant.uniform<i32:f32, 0.015625>>) -> tensor<4x6x!qy> {
  %e = tensor.empty() : tensor<4x6x!qy>
  %m = hip.matmul(%ctx) ins(%a, %w, %c : tensor<4x8x!qa>, tensor<8x6x!quant.uniform<i8:f32, 0.0625:3>>, tensor<6x!quant.uniform<i32:f32, 0.015625>>)
      outs(%e : tensor<4x6x!qy>) : tensor<4x6x!qy>
  return %m : tensor<4x6x!qy>
}
func.func @gemm_no_c(%ctx: !hip.context, %a: tensor<4x8xf16>, %b: tensor<8x6xf16>) -> tensor<4x6xf16> {
  %e = tensor.empty() : tensor<4x6xf16>
  %g = hip.gemm(%ctx) ins(%a, %b : tensor<4x8xf16>, tensor<8x6xf16>) outs(%e : tensor<4x6xf16>) : tensor<4x6xf16>
  return %g : tensor<4x6xf16>
}
func.func @weight_only_matmul(%ctx: !hip.context, %a: tensor<4x8xf16>,
                              %w: tensor<8x6x!quant.uniform<i8:f16, 0.0625:3>>) -> tensor<4x6xf16> {
  %e = tensor.empty() : tensor<4x6xf16>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<4x8xf16>, tensor<8x6x!quant.uniform<i8:f16, 0.0625:3>>)
      outs(%e : tensor<4x6xf16>) : tensor<4x6xf16>
  return %m : tensor<4x6xf16>
}
func.func @quant_gelu(%ctx: !hip.context, %a: tensor<8x!qa>) -> tensor<8x!qy> {
  %e = tensor.empty() : tensor<8x!qy>
  %g = hip.gelu(%ctx) ins(%a : tensor<8x!qa>) outs(%e : tensor<8x!qy>) : tensor<8x!qy>
  return %g : tensor<8x!qy>
}
func.func @float_transpose(%ctx: !hip.context, %a: tensor<2x4xf32>) -> tensor<4x2xf32> {
  %e = tensor.empty() : tensor<4x2xf32>
  %t = hip.transpose(%ctx) ins(%a : tensor<2x4xf32>) outs(%e : tensor<4x2xf32>) {perm = [1, 0]} : tensor<4x2xf32>
  return %t : tensor<4x2xf32>
}
func.func @float_mul(%ctx: !hip.context, %a: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.mul(%ctx) ins(%a, %a : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %s : tensor<8xf32>
}
)mlir";

/// The single op of `kind` in function `name`.
mlir::Operation *findOp(mlir::ModuleOp module, llvm::StringRef name,
                        llvm::StringRef kind) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getSymName() != name)
      return;
    func.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == kind)
        found = op;
    });
  });
  return found;
}

bool isUniform(mlir::Type type, double scale, int64_t zeroPoint) {
  auto quant = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(type);
  return quant && quant.getScale() == scale &&
         quant.getZeroPoint() == zeroPoint;
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::hip::HipDialect, mlir::quant::QuantDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext ctx(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &ctx);
  if (!module) {
    llvm::errs() << "[FAIL] test module did not parse\n";
    return 1;
  }
  mlir::Type f32 = mlir::Float32Type::get(&ctx);
  mlir::Type f16 = mlir::Float16Type::get(&ctx);
  using mlir::hip::HipQuantCapableOpInterface;

  // A float add implements the interface and answers "not quantized".
  {
    auto add =
        mlir::cast<mlir::hip::AddOp>(findOp(*module, "float_add", "hip.add"));
    check(!add.isQuantized(), "float add: isQuantized() is false");
    check(add.getOperandQuantTypes() ==
              llvm::SmallVector<mlir::Type>{f32, f32, f32},
          "float add: getOperandQuantTypes() is [f32, f32, f32]");
    check(add.getResultQuantType() == f32,
          "float add: getResultQuantType() is f32");
    check(!add.getUniformQuantType(0),
          "float add: getUniformQuantType(0) is null");
    check(!mlir::hip::isQuantizedOp(add),
          "float add: isQuantizedOp() is false");
  }

  // A quantized add answers from its tensor element types.
  {
    auto add =
        mlir::cast<mlir::hip::AddOp>(findOp(*module, "quant_add", "hip.add"));
    check(add.isQuantized(), "quant add: isQuantized() is true");
    check(isUniform(add.getInputQuantType(0), 0.25, 32768),
          "quant add: input 0 is u16 0.25:32768");
    check(isUniform(add.getInputQuantType(1), 0.5, 32768),
          "quant add: input 1 is u16 0.5:32768");
    check(isUniform(add.getResultQuantType(), 0.125, 32768),
          "quant add: result is u16 0.125:32768");
    auto out = add.getUniformQuantType(2);
    check(out && out.getScale() == 0.125,
          "quant add: getUniformQuantType(2) gives the result scale");
    check(mlir::hip::getQuantElementTypes(add).size() == 3,
          "quant add: getQuantElementTypes() has 3 entries");
  }

  // After legalization the same answers come from `quant_types`.
  {
    auto add = mlir::cast<mlir::hip::AddOp>(
        findOp(*module, "legalized_add", "hip.add"));
    check(add.isQuantized(), "legalized add: isQuantized() is true");
    check(isUniform(add.getResultQuantType(), 0.125, 32768),
          "legalized add: result read from quant_types");
    auto addF = mlir::cast<mlir::hip::AddOp>(
        findOp(*module, "legalized_float_add", "hip.add"));
    check(!addF.isQuantized(), "legalized float add: isQuantized() is false");
    check(addF.getResultQuantType() == f16,
          "legalized float add: result is f16");
  }

  // The interface counts an optional operand (the bias) when present.
  {
    auto matmul = mlir::cast<mlir::hip::MatmulOp>(
        findOp(*module, "matmul_bias", "hip.matmul"));
    check(matmul.getOperandQuantTypes().size() == 4,
          "matmul with bias: 4 quant types (A, B, bias, out)");
    auto bias = matmul.getUniformQuantType(2);
    check(bias && bias.getStorageTypeIntegralWidth() == 32,
          "matmul with bias: input 2 is the i32 bias");
    check(isUniform(matmul.getInputQuantType(1), 0.0625, 3),
          "matmul with bias: input 1 is the i8 weight");
  }

  // The default getQuantOperands skips !hip.context and an absent optional
  // operand (here gemm's C).
  {
    auto gemm =
        mlir::cast<mlir::hip::GemmOp>(findOp(*module, "gemm_no_c", "hip.gemm"));
    llvm::SmallVector<mlir::Value> operands = gemm.getQuantOperands();
    check(operands.size() == 3 && operands[0] == gemm.getInputA() &&
              operands[1] == gemm.getInputB() &&
              operands[2] == gemm.getOutput(),
          "gemm without C: getQuantOperands() is [A, B, output]");
    check(gemm.getOperandQuantTypes() ==
              llvm::SmallVector<mlir::Type>{f16, f16, f16},
          "gemm without C: getOperandQuantTypes() is [f16, f16, f16]");
  }

  // A mixed op: per-operand answers; getQuantElementTypes() needs all
  // operands quantized, so it is empty.
  {
    auto matmul = mlir::cast<mlir::hip::MatmulOp>(
        findOp(*module, "weight_only_matmul", "hip.matmul"));
    check(matmul.isQuantized(), "weight-only matmul: isQuantized() is true");
    check(matmul.getInputQuantType(0) == f16 && !matmul.getUniformQuantType(0),
          "weight-only matmul: A is f16, no uniform type");
    check(isUniform(matmul.getInputQuantType(1), 0.0625, 3),
          "weight-only matmul: B is the weight's uniform type");
    check(matmul.getResultQuantType() == f16,
          "weight-only matmul: result is f16");
    check(mlir::hip::getQuantElementTypes(matmul).empty(),
          "weight-only matmul: getQuantElementTypes() is empty");
  }

  // An op without the interface in ODS gets it at dialect load; a
  // deny-listed one does not.
  {
    auto gelu = mlir::dyn_cast<HipQuantCapableOpInterface>(
        findOp(*module, "quant_gelu", "hip.gelu"));
    check(gelu && gelu.isQuantized(),
          "hip.gelu: attached interface, isQuantized() is true");
    check(gelu && isUniform(gelu.getResultQuantType(), 0.125, 32768),
          "hip.gelu: result is its uniform type");
    check(!mlir::isa<HipQuantCapableOpInterface>(
              findOp(*module, "float_transpose", "hip.transpose")),
          "hip.transpose is deny-listed");
  }

  // An op without the interface: no cast possible, free functions answer.
  {
    mlir::Operation *mul = findOp(*module, "float_mul", "hip.mul");
    check(!mlir::isa<HipQuantCapableOpInterface>(mul),
          "hip.mul does not implement the interface");
    check(!mlir::hip::isQuantizedOp(mul), "hip.mul: isQuantizedOp() is false");
    check(mlir::hip::getOperandQuantTypes(mul) ==
              llvm::SmallVector<mlir::Type>{f32, f32, f32, f32},
          "hip.mul: getOperandQuantTypes() falls back to its element types");
    check(mlir::hip::getQuantElementTypes(mul).empty(),
          "hip.mul: getQuantElementTypes() is empty");
  }

  if (g_failures) {
    llvm::errs() << g_failures << " check(s) failed\n";
    return 1;
  }
  llvm::outs() << "all checks passed\n";
  return 0;
}
