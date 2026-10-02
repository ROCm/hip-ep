##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(Interpreter)
add_subdirectory(Bindings)

# Generic Scheme pass runner — not dialect-specific
add_library(SchemePassLib STATIC SchemePass.cpp)
add_dependencies(SchemePassLib SchemePassIncGen HipsrTransformsPassIncGen)
target_include_directories(SchemePassLib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SchemePassLib PUBLIC
  HipsrDialectIR
  HipsrSchemeRuntime
  OnnxDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRShapeDialect
  MLIRIR
  MLIRSupport
)
