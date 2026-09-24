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
# executable (ENABLE_ROCMLIRTRITON), so we link its C++ pipeline/tuning
# libraries directly and call them in-process -- no dlopen, no serialized-IR
# hand-off. tools/CMakeLists.txt only adds this subdirectory when the flag is
# on, so ENABLE_ROCMLIRTRITON is guaranteed here.
#
# rocMLIR/Triton's per-library static archives have circular dependencies, so a
# single-pass link leaves undefined symbols. GNU/BSD-style linkers need an
# explicit RESCAN link group ($<LINK_GROUP:RESCAN,...> -> --start-group/
# --end-group) to iterate until resolved. MSVC-style linkers (link.exe /
# lld-link, used by the clang-cl toolchain on Windows) rescan archives
# automatically and do not support the RESCAN feature, so pass the closure
# directly there. CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED is set by the active
# platform module exactly when the generator expression is available.
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
  # rocMLIR/Triton reference this MLIR extension (registerInlinerExtension) but
  # it is a leaf target not pulled in transitively by the rock/triton closure
  # or ${llvm_libs} (Support+Core).
  MLIRFuncInlinerExtension
  # rocMLIR/Triton C++ API (in-tree), linked as the curated target closure
  # (HIPDNN_ROCMLIR_LINK_LIBS from cmake/deps.cmake), wrapped in a RESCAN link
  # group where the linker requires it (see _rocmlir_link_closure above).
  # LLVM/MLIR come from ${llvm_libs} above -- no bundled second copy.
  "${_rocmlir_link_closure}"
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
