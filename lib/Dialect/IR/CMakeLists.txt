##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectIR STATIC
  HipCausalConvCanonicalize.cpp
  HipDialect.cpp
  HipDpsOpInterface.cpp
  HipOpStateInterface.cpp
  HipReifyResultShapesImpl.cpp
  HipShapeUtils.cpp
)

add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipDpsOpInterfaceIncGen
  HipOpStateInterfaceIncGen
  HipOpsIncGen
)

# HipDialect.cpp pulls in the TableGen-generated op definitions for every Hip op
# (each now carrying the DPS/reify/infer interface bodies). On top of the
# shape-inference cascade the single translation unit exceeds MSVC's default
# COFF section limit (fatal error C1128). /bigobj raises it. No-op on non-MSVC.
if(MSVC)
  target_compile_options(HipDialectIR PRIVATE /bigobj)
endif()

target_link_libraries(HipDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  MLIRMemRefDialect
  # OpStateOpInterface's emitOpStateConstruct helper emits LLVM-dialect calls
  # (the per-op state constructor calls woven into the op-state init function).
  MLIRLLVMDialect
)

# HipDialect.cpp aggregates every TableGen-generated op class (GET_OP_CLASSES).
# As the dialect grows, this single translation unit exceeds the MSVC COFF
# section limit (fatal error C1128); /bigobj raises it.
if(MSVC)
  target_compile_options(HipDialectIR PRIVATE /bigobj)
endif()

# --- hip-ir packaging -------------------------------------------------------
# Usage requirements. Previously these came from the enclosing project's global
# include_directories(), which works in-tree but cannot be exported.
get_filename_component(HIP_IR_SOURCE_INCLUDE_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)

target_include_directories(HipDialectIR PUBLIC
  $<BUILD_INTERFACE:${HIP_IR_SOURCE_INCLUDE_ROOT}>
  $<BUILD_INTERFACE:${HIP_IR_GENERATED_INCLUDE_ROOT}>
  $<INSTALL_INTERFACE:include>)

target_compile_features(HipDialectIR PUBLIC cxx_std_17)

# Consumers link the namespaced name whether hip-ir comes from this source tree
# or from the installed package, so they never branch on how it was obtained.
if(NOT TARGET hip::HipDialectIR)
  add_library(hip::HipDialectIR ALIAS HipDialectIR)
endif()

if(HIP_IR_INSTALL)
  get_filename_component(_hip_ir_cmake_dir
    "${CMAKE_CURRENT_LIST_DIR}/../../../cmake" ABSOLUTE)
  include("${_hip_ir_cmake_dir}/hip-ir-install.cmake")
  unset(_hip_ir_cmake_dir)
endif()
