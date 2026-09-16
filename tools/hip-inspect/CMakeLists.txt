##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-inspect prints metadata from a compiled model artifact. The .bc path
# is lightweight (parseBitcodeFile + read the @__metadata_json global, no
# ROCm/JIT) and always built. The native .dll/.so path uses morphizen::Plugin.
add_executable(hip-inspect
  hip-inspect.cpp
)

# LLVM components: Core / BitReader for IR parsing, Support for
# MemoryBuffer + JSON. No ORC / ExecutionEngine / CodeGen needed.
llvm_map_components_to_libnames(llvm_libs
  Core
  BitReader
  Support
)

target_link_libraries(hip-inspect
  PRIVATE
  HipSupport
  ${llvm_libs}
)

target_include_directories(hip-inspect
  PRIVATE
  ${CMAKE_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src  # artifact_format.h
  ${CMAKE_SOURCE_DIR}/include                                   # hip/artifact_abi.h
)

target_compile_definitions(hip-inspect PRIVATE HIPDNN_INSPECT_NATIVE=1)
target_link_libraries(hip-inspect PRIVATE morphizen::morphizen-core-static)
target_include_directories(hip-inspect
  PRIVATE
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-core/include        # morphizen/plugin.hpp
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include  # morphizen-foundation
)

install(TARGETS hip-inspect
  RUNTIME DESTINATION bin
)
