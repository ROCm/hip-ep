##
# MiniONNX Dialect Library
##

add_library(MiniONNXDialectIR STATIC
  MiniONNXDialect.cpp
)

add_dependencies(MiniONNXDialectIR
  MiniONNXIncGen
)

target_link_libraries(MiniONNXDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRFuncDialect
)
