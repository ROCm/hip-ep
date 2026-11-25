##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# libhipep_backend.so (hipep_backend.h): a ggml backend's splits compiled and
# run. The split frontend and hip-ep's compiler (its ggml-to-hip pipeline) make
# LLVM bitcode of a split, which is JIT-loaded as the execution provider loads
# its models (LlvmIrJit). One library whose only exported symbols are
# hipep_backend_*; it needs HIP and, at run time, the kernel libraries
# installed beside it.

find_package(hip CONFIG REQUIRED)
if(NOT TARGET morphizen-custom-op-mlir)
  message(FATAL_ERROR "libhipep_backend loads code with the execution provider's "
                      "JIT (backend-mlir-compiler/custom-op-mlir)")
endif()

# Descriptions read and splits captured and compiled: what a test checks
# without a device.
add_library(hipep_backend_graph STATIC
  Graph.cpp
  Ops.cpp
  ${PROJECT_SOURCE_DIR}/lib/Frontend/Ggml/SplitCompiler.cpp)
add_dependencies(hipep_backend_graph HipSchemas)
target_include_directories(hipep_backend_graph
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/include"
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(hipep_backend_graph PUBLIC
  GgmlEmitter
  GgmlTransforms
  GgmlDialectIR
  LibHipCompiler
  HipSchemasLib
  MLIRParser
  MLIRPass)
set_target_properties(hipep_backend_graph PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(hipep_backend SHARED
  Api.cpp
  Engine.cpp)
target_include_directories(hipep_backend PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src")
target_compile_definitions(hipep_backend PRIVATE HIPEP_BACKEND_BUILD)
target_link_libraries(hipep_backend PRIVATE
  hipep_backend_graph
  morphizen-custom-op-mlir
  hip::host
  ${CMAKE_DL_LIBS})
set_target_properties(hipep_backend PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  INSTALL_RPATH "\$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH FALSE)
target_link_options(hipep_backend PRIVATE
  "LINKER:--exclude-libs,ALL" "LINKER:-Bsymbolic"
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_property(TARGET hipep_backend APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")

install(TARGETS hipep_backend LIBRARY DESTINATION lib)
install(FILES include/hipep_backend.h DESTINATION include)
install(FILES cmake/hipep_backendConfig.cmake DESTINATION lib/cmake/hipep_backend)
