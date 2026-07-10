##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS HipsrDialect.td)
mlir_tablegen(HipsrDialect.h.inc -gen-dialect-decls -dialect=hipsr)
mlir_tablegen(HipsrDialect.cpp.inc -gen-dialect-defs -dialect=hipsr)
add_public_tablegen_target(HipsrDialectIncGen)
