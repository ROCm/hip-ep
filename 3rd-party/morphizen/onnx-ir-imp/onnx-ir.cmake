##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(LIB_NAME onnx-ir)
add_library(${LIB_NAME}
  STATIC
  src/model.hpp
  src/model.cpp
  src/graph.hpp
  src/graph.cpp
  src/graph-resolver.hpp
  src/graph-resolver.cpp
  src/node.cpp
  src/node.hpp
  src/node-arg.hpp
  src/node-arg.cpp
  src/node-index.cpp
  src/node-index.hpp
  src/node-arg-index.hpp
  src/node-arg-index.cpp
  src/morphizen-ort-api.cpp
  src/morphizen-ort-api.hpp
  src/onnx-deps.hpp
  src/onnx-deps.cpp
)

target_include_directories(${LIB_NAME}
  PRIVATE
  $<INSTALL_INTERFACE:include>
)
## enable c++17
target_compile_features(${LIB_NAME}
  PRIVATE
  cxx_std_17)
target_compile_options(${LIB_NAME}
  PRIVATE
  ${MORPHIZEN_COMPILER_OPTIONS})
target_link_libraries(${LIB_NAME}
  PRIVATE
  #orphizen-utils
  Microsoft.GSL::GSL
  onnx
  onnx_proto
  morphizen-utils
)
