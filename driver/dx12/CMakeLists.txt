## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.

# DX12 runner -- Windows only. Drives a precompiled GPU kernel ELF through the
# AMD Cross-Compile D3D12 extension API (CreateComputePipelineCrossCompile).
if(NOT WIN32)
  return()
endif()

add_library(hip_ep_dx12 STATIC dx12_runner.cpp)

target_compile_features(hip_ep_dx12 PUBLIC cxx_std_17)

target_include_directories(hip_ep_dx12 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hip_ep_dx12 PUBLIC
  d3d12
  dxgi
)