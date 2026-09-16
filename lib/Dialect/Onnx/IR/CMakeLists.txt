##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxDialectIR STATIC
  OnnxDialect.cpp
  OnnxOps.cpp
)

add_dependencies(OnnxDialectIR
  OnnxDialectIncGen
)

target_link_libraries(OnnxDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRSideEffectInterfaces
)

# One translation unit holds every generated operation definition, which exceeds
# MSVC's default section limit.
if(MSVC)
  target_compile_options(OnnxDialectIR PRIVATE /bigobj)
endif()
