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
  src/vitisai-ep-factory.cpp
  src/vitisai-ep-factory.hpp
  src/vitisai-ep.hpp
  src/vitisai-ep.cpp
  src/ir-converter.hpp
  src/ir-converter.cpp
  src/ir-converter-imp.hpp
  src/ir-converter-imp.cpp
)

target_include_directories(ort-bridge
  PRIVATE
  $<BUILD_INTERFACE:${VAIP_ORT_API_DIR}>
  $<INSTALL_INTERFACE:include>
)

## enable c++17
target_compile_features(ort-bridge
  PRIVATE
  cxx_std_17)
target_compile_options(ort-bridge
  PRIVATE
  ${MORPHIZEN_COMPILER_OPTIONS})
target_link_libraries(ort-bridge
  PRIVATE
  onnxruntime::onnxruntime
  vaip-ort-api-ext
  morphizen-core-static
  protobuf::libprotobuf
)
if(TARGET ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,ort-bridge>)
endif(TARGET ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})
