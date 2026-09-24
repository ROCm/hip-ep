## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.

# Standalone DX12 kernel-execution driver -- Windows only. Not part of the
# ONNX Runtime execution provider; this loads a precompiled GPU kernel ELF
# and dispatches it through Dx12Runner (AMD Cross-Compile D3D12 extension
# API) for smoke-testing HIP kernels on the DX12 backend. See
# dx12/kernels/build_add_kernel.ps1 for how the ELF is produced.
if(NOT WIN32)
  return()
endif()

add_subdirectory(dx12)

add_executable(hip-ep-dx12-driver
  hip_ep_dx12_driver.cpp
)

target_compile_features(hip-ep-dx12-driver PRIVATE cxx_std_17)

target_link_libraries(hip-ep-dx12-driver PRIVATE
  hip_ep_dx12
)

install(TARGETS hip-ep-dx12-driver
  RUNTIME DESTINATION bin
)