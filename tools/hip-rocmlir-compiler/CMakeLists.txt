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
  # (HIPDNN_ROCMLIR_LINK_LIBS from cmake/deps.cmake). Wrapped in a RESCAN link
  # group: rocMLIR/Triton's per-library static archives have circular
  # dependencies, so a single-pass link leaves undefined symbols; RESCAN makes
  # the linker iterate the group until resolved (equivalent to
  # --start-group/--end-group). LLVM/MLIR come from ${llvm_libs} above -- no
  # bundled second copy.
  "$<LINK_GROUP:RESCAN,${HIPDNN_ROCMLIR_LINK_LIBS}>"
  ${llvm_libs}
)

target_include_directories(hip-rocmlir-compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/schemas
)

install(TARGETS hip-rocmlir-compiler
  RUNTIME DESTINATION bin
)
