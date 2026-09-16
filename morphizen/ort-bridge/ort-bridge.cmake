##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(_ORT_BRIDGE_SOURCES
  src/ort-bridge.cpp
  src/ort-api-version.hpp
  src/ort-api-version.cpp
  src/api-ptrs.hpp
  src/api-ptrs.cpp
  src/ort-status-exception.hpp
  src/ort-status-exception.cpp
  src/ort-graph-wrapper.hpp
  src/ort-graph-wrapper.cpp
  src/morphizen-ep-factory.cpp
  src/morphizen-ep-factory.hpp
  src/morphizen-ep.hpp
  src/morphizen-ep.cpp
  src/ir-converter.hpp
  src/ir-converter.cpp
  src/ir-converter-imp.hpp
  src/ir-converter-imp.cpp
)

if(morphizen_ENABLE_HIP_GPU_ALLOCATOR)
  list(APPEND _ORT_BRIDGE_SOURCES
    src/morphizen-hip-gpu-allocator.hpp
    src/morphizen-hip-gpu-allocator.cpp
  )
endif()

add_library(ort-bridge
  STATIC
  ${_ORT_BRIDGE_SOURCES}
)

target_include_directories(ort-bridge
  PRIVATE
  $<BUILD_INTERFACE:${MORPHIZEN_ORT_API_DIR}>
  $<INSTALL_INTERFACE:include>
)

## enable c++17
target_compile_features(ort-bridge
  PRIVATE
  cxx_std_17)
target_compile_options(ort-bridge
  PRIVATE
  ${MORPHIZEN_COMPILER_OPTIONS})
# Disable C4946: ONNXRuntime uses opaque type pattern with reinterpret_cast by design.
# Opaque types (OrtNodeComputeInfo <-> MorphiZenEP_ComputeInfo) are intentionally cast
# using reinterpret_cast for C API compatibility. This is a valid use case where C4946
# is a false positive.
if(MSVC)
  target_compile_options(ort-bridge PRIVATE /wd4946)
endif()
target_link_libraries(ort-bridge
  PRIVATE
  onnxruntime::onnxruntime
  morphizen::morphizen-graph  # ir-converter-imp.cpp uses graph wrappers
  morphizen-ort-api-ext  # Still needed by other components, also transitive via morphizen-graph
  morphizen-core-static
  protobuf::libprotobuf
)

if(morphizen_ENABLE_HIP_GPU_ALLOCATOR)
  # The hipMalloc-based OrtAllocator + DataTransfer needs the HIP runtime.
  # The parent project (onnx-hipdnn-ep / lib/Runtime) already does
  # find_package(hip QUIET) against the same TheRock-distributed ROCm, so we
  # require it here. Use PUBLIC so the dependency propagates to the SHARED
  # EP DLL (${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}) that links us via
  # WHOLE_ARCHIVE below.
  find_package(hip REQUIRED)
  target_link_libraries(ort-bridge PUBLIC hip::host)
  target_compile_definitions(ort-bridge PUBLIC MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR=1)
endif()

if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,ort-bridge>)
  if(MSVC AND morphizen_ENABLE_HIP_GPU_ALLOCATOR)
    get_target_property(_amdhip64_dll hip::amdhip64 IMPORTED_LOCATION_RELEASE)
    if(NOT _amdhip64_dll)
      get_target_property(_amdhip64_dll hip::amdhip64 IMPORTED_LOCATION)
    endif()
    if(NOT _amdhip64_dll)
      message(FATAL_ERROR "hip::amdhip64 has no IMPORTED_LOCATION; cannot set /DELAYLOAD")
    endif()
    cmake_path(GET _amdhip64_dll FILENAME _amdhip64_name)
    target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE delayimp)
    target_link_options(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE
      "/DELAYLOAD:${_amdhip64_name}")
  endif()
endif(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
