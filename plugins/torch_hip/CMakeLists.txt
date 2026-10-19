##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# torch_hip: Torch-MLIR -> HIP frontend, delivered as a static compiler plugin.
# Co-built into hip-ep (it is NOT a standalone project):
#
#   cmake ... -DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<hip-ep>/plugins/torch_hip
#             -DHIPDNN_EP_COMPILER_PLUGINS=torch_hip
#
# Without these options hip-ep builds exactly as before. See
# docs/01_TORCH_TO_HIP_PLUGIN_DESIGN.md and docs/design/plugin-interface.md.

if(NOT COMMAND hipdnn_ep_compiler_plugin_register)
  message(FATAL_ERROR
    "torch_hip must be co-built into hip-ep via "
    "-DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<this dir>; it is not a standalone "
    "project (static-plugin linkage model).")
endif()

add_library(torch_hip_plugin STATIC
  src/plugin_main.cpp
  src/ConvertTorchToHip.cpp
)

target_include_directories(torch_hip_plugin PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Declared PUBLIC for static link ORDER only (plugin and host are one binary
# sharing one MLIR); same pattern as test/plugin/sample_plugin.
target_link_libraries(torch_hip_plugin PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)

target_compile_features(torch_hip_plugin PRIVATE cxx_std_17)

# Linked into LibHipCompiler, which is linked into the EP DLL (shared library),
# so objects must be position-independent on ELF.
set_target_properties(torch_hip_plugin PROPERTIES
  FOLDER "plugins/torch_hip"
  POSITION_INDEPENDENT_CODE ON
)

# RTTI off on ELF to match LLVM/MLIR's no-RTTI ABI.
if(NOT WIN32)
  target_compile_options(torch_hip_plugin PRIVATE -fno-rtti)
endif()

# Available to the static registrar; linked only when `torch_hip` is listed in
# HIPDNN_EP_COMPILER_PLUGINS (cmake/HipEpPlugins.cmake).
hipdnn_ep_compiler_plugin_register(PLUGIN_ID torch_hip TARGET torch_hip_plugin)
