##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS OnnxDialect.td)
mlir_tablegen(OnnxDialect.h.inc -gen-dialect-decls -dialect=onnx)
mlir_tablegen(OnnxDialect.cpp.inc -gen-dialect-defs -dialect=onnx)

set(LLVM_TARGET_DEFINITIONS OnnxOps.td)
mlir_tablegen(OnnxOps.h.inc -gen-op-decls)
mlir_tablegen(OnnxOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(OnnxDialectIncGen)
