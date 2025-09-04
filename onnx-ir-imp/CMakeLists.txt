##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(${CMAKE_CURRENT_SOURCE_DIR}/../vaip-core/cmake/find_onnxruntime.cmake)

## find_package(onnx)
set(ONNX_USE_MSVC_STATIC_RUNTIME ON CACHE BOOL "use static onnx")
## we have to create our own namespace to avoid name confliction with ORT
set(ONNX_NAMESPACE "morphizen_onnx" CACHE STRING "onnx namespace")
if(TARGET onnx::onnx)
  get_target_property(TMP onnx::onnx INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found onnx at ${TMP}")
else()
  message(STATUS "cannot find_package(onnx), fetch it from ${DEP_URL_onnx}")
  FetchContent_Declare(
    onnx
    URL ${DEP_URL_onnx}
    URL_HASH SHA1=${DEP_SHA1_onnx}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE)
  find_package(onnx REQUIRED)
endif()

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
  #src/graph-shape-infer.hpp
  #src/graph-shape-infer.cpp
  #src/graph-inference-context.hpp
  #src/graph-inference-context.cpp
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
  src/vaip-ort-api.cpp
  src/vaip-ort-api.hpp

  src/core/graph/contrib_ops/contrib_defs.cc
)
target_sources(${LIB_NAME}
     INTERFACE
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_protobuf.natvis>)
## compile time dependency
set(ORT_INC  "")
list(APPEND ORT_INC
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
  ${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime
  ${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime/core/session
  ${VAIP_ORT_API_DIR}
)
set_source_files_properties(src/vaip-ort-api.cpp
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
target_link_libraries(${LIB_NAME}
  PRIVATE
  vaip-ort-api-ext
  Microsoft.GSL::GSL
  onnx
  onnx_proto
  morphizen-utils
)

if(TARGET ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>)
endif(TARGET ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})
