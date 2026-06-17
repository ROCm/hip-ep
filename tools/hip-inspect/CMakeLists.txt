##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-inspect prints metadata from a compiled model artifact. The .bc path
# is lightweight (parseBitcodeFile + read the @__metadata_json global, no
# ROCm/JIT) and always built. The native .dll/.so path needs morphizen::Plugin,
# only available under BUILD_EP, so it is compiled in conditionally.
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

# Native-artifact inspection via morphizen::Plugin. Only when the morphizen
# core target exists (BUILD_EP); the .bc path stays dependency-free otherwise.
if(TARGET morphizen-core-static)
  target_compile_definitions(hip-inspect PRIVATE HIPDNN_INSPECT_NATIVE=1)
  target_link_libraries(hip-inspect PRIVATE morphizen::morphizen-core-static)
  target_include_directories(hip-inspect
    PRIVATE
    ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/morphizen-core/include        # morphizen/plugin.hpp
    ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/morphizen-foundation/include  # morphizen-foundation
  )
endif()

install(TARGETS hip-inspect
  RUNTIME DESTINATION bin
)
