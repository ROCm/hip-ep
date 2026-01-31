##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
add_library(ort-bridge
  STATIC
  src/ort-bridge.cpp
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
  morphizen-ort-api-ext
  morphizen-core-static
  protobuf::libprotobuf
)
if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,ort-bridge>)
endif(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
