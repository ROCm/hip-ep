/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Unit test for onnxElementTypeToMlirDenseElementType (morphizen/mlir-imp).
//
// Regression guard for the QDQ UINT16 sign-flip: inline ONNX UINT16 dense
// constants must import as *unsigned* i16 (ui16), not signless i16. The
// downstream HipToLLVM QDQ dtype classifier (added for the Quark ResNet50-INT8
// signed-int16 bias) treats signless i16 as INT16, which sign-flips any UINT16
// value >= 32768 (e.g. google_bert's per-layer attention sqrt(d_k) scaling
// scalar) and corrupts DequantizeLinear output.
//
// This cannot be a LIT test: hip-mlir-opt has no ONNX-protobuf front end and
// only consumes textual MLIR, so the ONNX-dtype -> MLIR-type mapping is only
// reachable at the C++ layer. Plain main() (no GTest): matches the other
// MLIR-side unit tests and avoids a GTest dependency absent from the compiler
// build.

#include "mlir-constants.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

// ONNX TensorProto_DataType codes (subset under test).
enum : int {
  ONNX_FLOAT = 1,
  ONNX_UINT8 = 2,
  ONNX_INT8 = 3,
  ONNX_UINT16 = 4,
  ONNX_INT16 = 5,
  ONNX_INT32 = 6,
  ONNX_FLOAT16 = 10,
};

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

// Returns the IntegerType for `elem` or nullptr if `elem` is not an integer.
mlir::IntegerType asInt(mlir::Type elem) {
  return mlir::dyn_cast<mlir::IntegerType>(elem);
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  mlir::OpBuilder builder(&ctx);
  using morphizen::mlir_impl::onnxElementTypeToMlirDenseElementType;
  using morphizen::mlir_impl::onnxElementTypeToMlirElementType;

  // --- The fix: inline UINT16 dense constant must stay UNSIGNED i16 ---------
  {
    auto t = asInt(onnxElementTypeToMlirDenseElementType(ONNX_UINT16, builder));
    check(t && t.getWidth() == 16 && t.isUnsigned(),
          "dense(UINT16) -> unsigned i16 (ui16), not signless i16");
  }

  // --- INT16 must remain SIGNLESS i16 so the QDQ classifier still reads it as
  // INT16 (Quark ResNet50-INT8 bias path is unaffected by the fix) -----------
  {
    auto t = asInt(onnxElementTypeToMlirDenseElementType(ONNX_INT16, builder));
    check(t && t.getWidth() == 16 && t.isSignless(),
          "dense(INT16) -> signless i16 (ResNet50-INT8 bias path unchanged)");
  }

  // --- 8-bit integers keep the pre-existing signless-flatten behavior -------
  {
    auto u8 = asInt(onnxElementTypeToMlirDenseElementType(ONNX_UINT8, builder));
    check(u8 && u8.getWidth() == 8 && u8.isSignless(),
          "dense(UINT8) -> signless i8 (unchanged)");
    auto i8 = asInt(onnxElementTypeToMlirDenseElementType(ONNX_INT8, builder));
    check(i8 && i8.getWidth() == 8 && i8.isSignless(),
          "dense(INT8) -> signless i8 (unchanged)");
  }

  // --- Wider signless integers pass through untouched ------------------------
  {
    auto i32 =
        asInt(onnxElementTypeToMlirDenseElementType(ONNX_INT32, builder));
    check(i32 && i32.getWidth() == 32 && i32.isSignless(),
          "dense(INT32) -> signless i32 (unchanged)");
  }

  // --- Non-integer element types are returned untouched ----------------------
  {
    auto f16 = onnxElementTypeToMlirDenseElementType(ONNX_FLOAT16, builder);
    check(f16.isF16() && !asInt(f16), "dense(FLOAT16) -> f16 (unchanged)");
    auto f32 = onnxElementTypeToMlirDenseElementType(ONNX_FLOAT, builder);
    check(f32.isF32() && !asInt(f32), "dense(FLOAT) -> f32 (unchanged)");
  }

  // --- Dense mapping for UINT16 now matches the bare element-type mapping,
  // which already preserved ui16; confirm they agree (inline vs external
  // UINT16 consistency the fix establishes) ----------------------------------
  {
    auto dense = onnxElementTypeToMlirDenseElementType(ONNX_UINT16, builder);
    auto elem = onnxElementTypeToMlirElementType(ONNX_UINT16, builder);
    check(dense == elem, "dense(UINT16) == elem(UINT16) (both ui16)");
  }

  if (g_failures == 0) {
    llvm::outs() << "All dtype-mapping checks passed.\n";
    return 0;
  }
  llvm::errs() << g_failures << " dtype-mapping check(s) FAILED.\n";
  return 1;
}
