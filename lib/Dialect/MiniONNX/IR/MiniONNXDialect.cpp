//===- MiniONNXDialect.cpp - MiniONNX Dialect Implementation --------------===//
//
// Minimal ONNX dialect for hip-ep conversion.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/MiniONNX/IR/MiniONNX.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::onnx;

#include "hip/Dialect/MiniONNX/IR/MiniONNXDialect.cpp.inc"

void MiniONNXDialect::initialize() {
  // CRITICAL: Allow unregistered ONNX operations
  // This handles the ~200 ONNX ops we don't explicitly register
  allowUnknownOperations();
  
  // Register only the operations we need for hip-ep conversion
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/MiniONNX/IR/MiniONNXOps.cpp.inc"
  >();
}
