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
# The link closure itself now rides on LibHipCompiler (PUBLIC), which also
# gives it to hipgpu.dll. Naming it here too would link the same targets both
# inside and outside a LINK_GROUP, which CMake rejects as "link item occurred
# with different features".
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
  # LLVM/MLIR come from ${llvm_libs} -- no bundled second copy.
  ${llvm_libs}
)

# Follows what LibHipCompiler actually built, not BUILD_MOCK_RUNTIME: the
# benchmarking code lives there now and stubs itself out when the build does not
# allow HIP host APIs, so a tool flag derived from the mock setting alone would
# offer tuning that cannot run. No hip::host link either -- nothing here
# references a HIP symbol since the autotuner moved, which is why this binary no
# longer carries a HIP import at all.
if(HIPDNN_EP_ROCMLIR_AUTOTUNE_AVAILABLE)
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
