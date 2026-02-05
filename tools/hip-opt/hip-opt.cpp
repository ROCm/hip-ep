#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "HipDialect.h"
#include "HipPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::hip::HipDialect>();

  mlir::hip::registerHipPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hip-opt: custom compiler driver\n", registry));
}
