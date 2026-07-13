##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS HipsrDialect.td)
mlir_tablegen(HipsrDialect.h.inc -gen-dialect-decls -dialect=hipsr)
mlir_tablegen(HipsrDialect.cpp.inc -gen-dialect-defs -dialect=hipsr)

set(LLVM_TARGET_DEFINITIONS HipsrInterfaces.td)
mlir_tablegen(HipsrInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HipsrInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS HipsrOps.td)
mlir_tablegen(HipsrOps.h.inc -gen-op-decls)
mlir_tablegen(HipsrOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(HipsrDialectIncGen)
