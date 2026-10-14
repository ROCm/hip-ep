##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-rocmlir-compiler hip-rocmlir-compiler.cpp)

add_dependencies(hip-rocmlir-compiler
  HipDialectIncGen
  HipsrDialectIncGen
  OnnxDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

# rocMLIR (rocmlirTriton) is built in-tree against the same LLVM/MLIR as this
# executable (ENABLE_ROCMLIRTRITON), so its pipeline/tuning libraries are
# called in-process -- no dlopen, no serialized-IR hand-off.
#
# LibHipCompiler links the rock/triton closure PRIVATE (so its OBJECT-library
# members are not absorbed into every static archive that links it; see
# lib/Compiler/CMakeLists.txt). PRIVATE drops include-dir propagation, so name
# the closure here too: this executable includes rocMLIR headers directly
# (mlir/Dialect/Rock/*). Using the same ${HIPDNN_ROCMLIR_LINK_LIBS} RESCAN
# group keeps the link feature identical to LibHipCompiler's own use, so CMake
# does not reject the shared items as occurring "with different features".
if(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
  set(_rocmlir_link_closure "$<LINK_GROUP:RESCAN,${HIPDNN_ROCMLIR_LINK_LIBS}>")
else()
  set(_rocmlir_link_closure "${HIPDNN_ROCMLIR_LINK_LIBS}")
endif()
target_link_libraries(hip-rocmlir-compiler PRIVATE
  HipSupport
  LibHipCompiler
  HipDialectIR
  HipsrDialectIR
  OnnxDialectIR
  HipDialectTransforms
  HipToTosa
  OnnxToHip
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRFuncTransforms
  "${_rocmlir_link_closure}"
  # LLVM/MLIR come from ${llvm_libs} -- no bundled second copy.
  ${llvm_libs}
)

if(NOT BUILD_MOCK_RUNTIME)
  find_package(hip REQUIRED)
  target_link_libraries(hip-rocmlir-compiler PRIVATE hip::host)
  target_compile_definitions(hip-rocmlir-compiler
    PRIVATE HIP_ROCMLIR_AUTOTUNE=1)
else()
  target_compile_definitions(hip-rocmlir-compiler
    PRIVATE HIP_ROCMLIR_AUTOTUNE=0)
endif()

target_include_directories(hip-rocmlir-compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/schemas
)

install(TARGETS hip-rocmlir-compiler
  RUNTIME DESTINATION bin
)
