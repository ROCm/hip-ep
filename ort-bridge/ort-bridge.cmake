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
  src/ort-array-span.hpp
  src/graph.hpp
  src/graph.cpp
  src/vitisai-ep-factory.cpp
  src/vitisai-ep-factory.hpp
  src/vitisai-ep.hpp
  src/vitisai-ep.cpp
  src/ir-converter.hpp
  src/ir-converter.cpp
)

target_include_directories(ort-bridge
  # TODO: it is not a good pratice, we should use
  #      #include <core/session/onnxruntime_c_api.h>
  # instead of
  #      #include <onnxruntime_c_api.h>
  #
  # however, too many source code use the later, so we add it to search path
  #
  # if morphizen users need to use pre-installed version of morphizen,
  # it should use the following in their CMakeLists.txt
  #  #    find_package(onnxruntime CONFIG REQUIRED)
  #    target_link_libraries (<YOUR-TARGET> PRIVATE morphizen::morphizen-core-static)
  #
  # then use #include <morphizen/onnxruntime_api.h> instread.
  #
  PRIVATE
  $<BUILD_INTERFACE:${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime>
  $<BUILD_INTERFACE:${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime/core/session>
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
  morphizen-utils
  Microsoft.GSL::GSL
  onnx
  )

if(TARGET ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,ort-bridge>)
endif(TARGET ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})
