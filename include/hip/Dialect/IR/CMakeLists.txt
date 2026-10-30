##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS HipDialect.td)
mlir_tablegen(HipDialect.h.inc -gen-dialect-decls -dialect=hip)
mlir_tablegen(HipDialect.cpp.inc -gen-dialect-defs -dialect=hip)
add_public_tablegen_target(HipDialectIncGen)

set(LLVM_TARGET_DEFINITIONS HipTypes.td)
mlir_tablegen(HipTypes.h.inc -gen-typedef-decls -typedefs-dialect=hip)
mlir_tablegen(HipTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hip)
add_public_tablegen_target(HipTypesIncGen)

# HipDpsOpInterface: shared reify default for HIP DPS ops. Generated
# before HipOps so ops can reference the interface in their TableGen
# trait list. The `.h.inc` carries the interface class declaration and
# the `.cpp.inc` carries the trait/concept method bindings; this file
# pairs with `lib/Dialect/IR/HipDpsOpInterface.cpp` which includes the
# `.cpp.inc` and provides the default `reifyResultShapes` body.
set(LLVM_TARGET_DEFINITIONS HipDpsOpInterface.td)
mlir_tablegen(HipDpsOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(HipDpsOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(HipDpsOpInterfaceIncGen)

# OpStateOpInterface: per-op state construction code-gen hook. Generated
# before HipOps so stateful ops can reference the interface in their TableGen
# trait list. Pairs with lib/Dialect/IR/HipOpStateInterface.cpp, which includes
# the `.cpp.inc` and provides the shared `emitOpStateConstruct` helper.
set(LLVM_TARGET_DEFINITIONS HipOpStateInterface.td)
mlir_tablegen(HipOpStateInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(HipOpStateInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(HipOpStateInterfaceIncGen)

# ElementMoveOpInterface: ops that only move or select elements, so a
# per-tensor quantize commutes with them. Pairs with
# lib/Dialect/IR/HipElementMoveInterface.cpp, which also attaches the
# interface to the tensor view ops.
set(LLVM_TARGET_DEFINITIONS HipElementMoveInterface.td)
mlir_tablegen(HipElementMoveInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(HipElementMoveInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(HipElementMoveInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS HipOps.td)
mlir_tablegen(HipOps.h.inc -gen-op-decls)
mlir_tablegen(HipOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HipOpsIncGen)
# HipDialect.h includes the interface header; targets that depend only on
# HipOpsIncGen need it generated too.
add_dependencies(HipOpsIncGen HipElementMoveInterfaceIncGen)
