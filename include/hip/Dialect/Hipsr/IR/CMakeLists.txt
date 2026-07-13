##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS HipsrDialect.td)
mlir_tablegen(HipsrDialect.h.inc -gen-dialect-decls -dialect=hipsr)
mlir_tablegen(HipsrDialect.cpp.inc -gen-dialect-defs -dialect=hipsr)

set(LLVM_TARGET_DEFINITIONS HipsrOps.td)
mlir_tablegen(HipsrOps.h.inc -gen-op-decls)
mlir_tablegen(HipsrOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(HipsrDialectIncGen)

set(LLVM_TARGET_DEFINITIONS HipsrAttributes.td)
mlir_tablegen(HipsrEnums.h.inc -gen-enum-decls)
mlir_tablegen(HipsrEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HipsrAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=hipsr)
mlir_tablegen(HipsrAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hipsr)
add_public_tablegen_target(HipsrAttributesIncGen)
