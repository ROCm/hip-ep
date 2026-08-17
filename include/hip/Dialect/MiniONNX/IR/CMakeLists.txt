##
# MiniONNX TableGen
##

set(LLVM_TARGET_DEFINITIONS MiniONNX.td)
mlir_tablegen(MiniONNXDialect.h.inc -gen-dialect-decls -dialect=onnx)
mlir_tablegen(MiniONNXDialect.cpp.inc -gen-dialect-defs -dialect=onnx)
mlir_tablegen(MiniONNXOps.h.inc -gen-op-decls)
mlir_tablegen(MiniONNXOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MiniONNXIncGen)
