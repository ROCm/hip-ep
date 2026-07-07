##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(${CMAKE_CURRENT_SOURCE_DIR}/../morphizen-core/cmake/find_onnxruntime.cmake)


set(LIB_NAME onnx-ir)
add_library(${LIB_NAME}
  STATIC
  src/model.hpp
  src/model.cpp
  src/graph.hpp
  src/graph.cpp
  src/graph-resolver.hpp
  src/graph-resolver.cpp
  src/graph-store.hpp
  src/graph-store.cpp
  src/node.cpp
  src/node.hpp
  src/node-arg.hpp
  src/node-arg.cpp
  src/node-index.cpp
  src/node-index.hpp
  src/node-arg-index.hpp
  src/node-arg-index.cpp
  src/node-arg-producer-map.hpp
  src/node-arg-producer-map.cpp
  src/graph-id.hpp
  src/graph-id.cpp
  src/staging-graph.hpp
  src/staging-graph.cpp
  src/onnx-deps.hpp
  src/onnx-deps.cpp
  src/morphizen-ort-api.cpp
  src/morphizen-ort-api.hpp

  src/core/graph/contrib_ops/contrib_defs.cc
)
target_sources(${LIB_NAME}
     INTERFACE
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_protobuf.natvis>)
## compile time dependency
set(ORT_INC  "")
list(APPEND ORT_INC
  ${MORPHIZEN_ORT_API_DIR}
)
set_source_files_properties(src/morphizen-ort-api.cpp
  PROPERTIES INCLUDE_DIRECTORIES "${ORT_INC}")


target_include_directories(${LIB_NAME}
  PRIVATE
  $<INSTALL_INTERFACE:include>
  src
)
## enable c++17
target_compile_features(${LIB_NAME}
  PRIVATE
  cxx_std_17)
target_compile_options(${LIB_NAME}
  PRIVATE
  ${MORPHIZEN_COMPILER_OPTIONS})
# GCC 13+ Linux: onnx_proto's generated *.pb.h accessors return size_t but
# their signatures are `int`, which trips MORPHIZEN_COMPILER_OPTIONS's
# `-Werror -Wconversion`. Keep the warning visible but stop it from being
# a hard build error (mirrors the MSVC /wd4267 pattern in morphizen-pattern).
#
# TODO(linux): this is a temporary per-target opt-out. The root-cause fix
# mirrors morphizen-core/morphizen-pattern: extract onnx_proto's
# INTERFACE_INCLUDE_DIRECTORIES and re-apply them as SYSTEM (see the
# onnx-handling pattern already in morphizen-core/cmake/morphizen-core-static.cmake).
# Tracked separately because morphizen_ENABLE_ONNX_BACKEND defaults to OFF and
# no standard CI flow exercises this target today. Remove this block once the
# SYSTEM-include fix lands.
if(NOT MSVC)
  target_compile_options(${LIB_NAME} PRIVATE -Wno-error=conversion)
endif()
target_link_libraries(${LIB_NAME}
  PRIVATE
  onnxruntime::onnxruntime
  morphizen-ort-api-ext
  Microsoft.GSL::GSL
  onnx
  onnx_proto
  morphizen-utils
)

if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>)
endif(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
