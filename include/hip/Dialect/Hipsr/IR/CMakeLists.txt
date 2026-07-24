##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS HipsrDialect.td)
mlir_tablegen(HipsrDialect.h.inc -gen-dialect-decls -dialect=hipsr)
mlir_tablegen(HipsrDialect.cpp.inc -gen-dialect-defs -dialect=hipsr)

set(LLVM_TARGET_DEFINITIONS HipsrShapeRegionInterface.td)
mlir_tablegen(HipsrShapeRegionInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(HipsrShapeRegionInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS HipsrStartBarrierInterface.td)
mlir_tablegen(HipsrStartBarrierInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(HipsrStartBarrierInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS HipsrEndBarrierInterface.td)
mlir_tablegen(HipsrEndBarrierInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(HipsrEndBarrierInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS HipsrShapeYieldOp.td)
mlir_tablegen(HipsrShapeYieldOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrShapeYieldOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrOps.td)
mlir_tablegen(HipsrOps.h.inc -gen-op-decls)
mlir_tablegen(HipsrOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrPoolDomainYieldOp.td)
mlir_tablegen(HipsrPoolDomainYieldOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrPoolDomainYieldOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrPoolDomainOp.td)
mlir_tablegen(HipsrPoolDomainOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrPoolDomainOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrEmptyYieldOp.td)
mlir_tablegen(HipsrEmptyYieldOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrEmptyYieldOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrEmptyOp.td)
mlir_tablegen(HipsrEmptyOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrEmptyOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrConstantOp.td)
mlir_tablegen(HipsrConstantOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrConstantOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrCastOp.td)
mlir_tablegen(HipsrCastOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrCastOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrMatMulOp.td)
mlir_tablegen(HipsrMatMulOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrMatMulOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrAddOp.td)
mlir_tablegen(HipsrAddOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrAddOp.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HipsrPlaceholderOp.td)
mlir_tablegen(HipsrPlaceholderOp.h.inc -gen-op-decls)
mlir_tablegen(HipsrPlaceholderOp.cpp.inc -gen-op-defs)

add_public_tablegen_target(HipsrDialectIncGen)

set(LLVM_TARGET_DEFINITIONS HipsrEnums.td)
mlir_tablegen(HipsrEnums.h.inc -gen-enum-decls)
mlir_tablegen(HipsrEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(HipsrEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS HipsrAttrs.td)
mlir_tablegen(HipsrAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hipsr)
mlir_tablegen(HipsrAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hipsr)
add_public_tablegen_target(HipsrAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS HipsrTypes.td)
mlir_tablegen(HipsrTypes.h.inc -gen-typedef-decls -typedefs-dialect=hipsr)
mlir_tablegen(HipsrTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hipsr)
add_public_tablegen_target(HipsrTypesIncGen)
