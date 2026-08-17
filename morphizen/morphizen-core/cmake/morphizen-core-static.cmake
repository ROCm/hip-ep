##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/config_json_binary.hpp ${CMAKE_CURRENT_BINARY_DIR}/morphizen_config.json
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../cmake/scripts"
  $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/src/binary/config_json_binary.hpp.py
  "${MORPHIZEN_JSON_CONFIG_FILE}"
  "${TRIM_CONFIG}"
  "${morphizen_WITH_MORPHIZEN_CONFIG_FILE}"
  DEPENDS "${MORPHIZEN_JSON_CONFIG_FILE}"
)

if(morphizen_WITH_MORPHIZEN_CONFIG_FILE)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/morphizen_config.json DESTINATION bin)
endif()
set(LIB_NAME morphizen-core-static)
add_library(${LIB_NAME} STATIC
  ${PROTO_SRCS} ${PROTO_HDRS}
  src/version_info.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/version_info.cpp
  include/morphizen/plugin.hpp
  include/morphizen/morphizen.hpp
  include/morphizen/_sanity_check.hpp
  include/morphizen/cache_identity.hpp
  src/cache_identity.cpp
  src/ort_api_impl.cpp
  include/morphizen/pass_context.hpp
  src/pass_context_imp.hpp
  src/pass_context_imp.cpp
  src/anchor_point.cpp
  src/anchor_point_imp.cpp
  src/anchor_point_imp.hpp
  include/morphizen/anchor_point.hpp
  include/morphizen/node_builder.hpp
  src/node_builder.cpp
  src/pass_imp.hpp
  src/pass_imp.cpp
  src/tar_file.cpp
  src/tar_file.hpp
  src/tar_header.hpp
  src/tar_header.cpp
  src/tar_entry.cpp
  src/tar_entry.hpp
  src/mmap_file.hpp
  src/mmap_file.cpp
  src/mem_stream_buffer.hpp
  include/morphizen/pass.hpp
  src/pass.cpp
  include/morphizen/rewrite_rule.hpp
  src/rewrite_rule.cpp
  include/morphizen/model.hpp
  src/model.cpp
  include/morphizen/tensor_proto.hpp
  src/tensor_proto.cpp
  src/node_arg_const_data.cpp
  src/profile_utils.hpp
  src/profile_utils.cpp
  src/util.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/morphizen_version_info.hpp.inc
  src/config.hpp
  src/config.cpp
  src/custom_op.cpp
  include/morphizen/rewrite_rule.hpp
  src/rewrite_rule.cpp
  src/node_arg_const_data.cpp
  src/file_lock.hpp
  src/file_lock.cpp
  include/morphizen/guess_reshape.hpp
  src/guess_reshape.cpp
  include/morphizen/config_reader.hpp
  src/binary/config_reader.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config_json_binary.hpp
  src/morphizen_compile_model.cpp
  src/profile.cpp
  src/onnxruntime_morphizen_ep.cpp
  src/model_compatibility.cpp
  include/morphizen/file_stream.hpp
  src/file_stream.cpp
  include/morphizen/temp_file_stream.hpp
  src/temp_file_stream.cpp
  src/cleanup.hpp
  src/cleanup.cpp
  src/logger_adapter.hpp
  src/logger_adapter.cpp
  src/ep_shared_context_workspace.cpp
  src/ep_shared_context_workspace.hpp
  include/morphizen/op_invoker.hpp
  src/op_invoker.cpp
)

# ONNX schema files are needed for node_with_named_args feature.
#      This feature allows pattern matching using argument names (e.g., {"X": input, "W": weight})
#      instead of positions, making patterns more readable. The schema provides the mapping
#      from names to positions. Users who don't need this feature can disable it to avoid ONNX dependency.
if(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT)
  target_sources(${LIB_NAME} PRIVATE
    src/binary/onnx_schema_json_binary.hpp
    include/morphizen/onnx_schema.hpp
    src/onnx_schema.cpp
  )
endif()
add_library (morphizen::morphizen-core-static ALIAS morphizen-core-static)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER morphizen)
if(MSVC)
  target_sources(${LIB_NAME} PRIVATE
    src/mmap_file_win.hpp
    src/mmap_file_win.cpp
    src/mmap_file_tmphandle_win.hpp
    src/mmap_file_tmphandle_win.cpp
    src/util_mswin.cpp)
else(MSVC)
  target_sources(${LIB_NAME} PRIVATE
    src/util_linux.cpp)
endif(MSVC)
target_include_directories(${LIB_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${MORPHIZEN_ORT_API_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../3rd-party
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
# Generated protobuf headers (morphizen/{config,anchor_point,capability,
# pass_context,version,model_compatibility}.pb.h) live under
# ${CMAKE_CURRENT_BINARY_DIR}/morphizen/. Mark BINARY_DIR as SYSTEM so the
# inline accessor methods (`Map::size()` returns size_t but accessor signature
# is `int` on protobuf >=22) do NOT trigger -Wconversion in either:
#   - morphizen-core's own private sources (config.cpp, pass_context_imp.cpp,
#     model_compatibility.cpp) that include "morphizen/*.pb.h" directly, OR
#   - PUBLIC consumers (ort-bridge, morphizen-graph, etc.) that inherit
#     INTERFACE_SYSTEM_INCLUDE_DIRECTORIES via target_link_libraries.
# This is the root-cause fix for the per-target -Wno-error=conversion patches
# previously applied across morphizen-core / morphizen-pattern / and that had
# to be supplemented by a parent-project FORCE override in onnx-hipdnn-ep
# because transitive consumers (ort-bridge / morphizen-graph) weren't covered.
# Generator-expression SYSTEM scope is restricted to BUILD_INTERFACE because
# the installed layout puts .pb.h under <install>/include/morphizen/ next to
# the public ABI headers, and we want -Wconversion to fire on those (they're
# header-only, no .pb.cc to silence with /w).
# Side note: BINARY_DIR also contains the configure_file output
# `version_info_config.h` (consumed by src/version_info.cpp). SYSTEM-marking
# it is accepted collateral — that file is generated `#define` macros only,
# no code that -Wconversion could meaningfully flag.
target_include_directories(${LIB_NAME}
  SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Link against onnxruntime::onnxruntime target directly.
# This automatically propagates include directories via INTERFACE_INCLUDE_DIRECTORIES,
# eliminating the need for manual include directory management.
# The target is provided by find_package(onnxruntime) from installed ONNX Runtime package.
set(MorphiZen_DEPS
  onnxruntime::onnxruntime
  protobuf::libprotobuf
  glog::glog
  morphizen::foundation
  Microsoft.GSL::GSL
  morphizen-utils
  morphizen-ort-api-ext
  morphizen::morphizen-graph
)

# Conditional pattern matching support
if(morphizen_ENABLE_PATTERN_MATCHING)
  list(APPEND MorphiZen_DEPS morphizen::morphizen-pattern)
  target_compile_definitions(${LIB_NAME} PUBLIC MORPHIZEN_HAS_PATTERN_MATCHING=1)
else()
  target_compile_definitions(${LIB_NAME} PUBLIC MORPHIZEN_HAS_PATTERN_MATCHING=0)
endif()
target_link_libraries(${LIB_NAME} PUBLIC ${MorphiZen_DEPS})
target_compile_definitions(${LIB_NAME}
  PUBLIC
  "-DONNX_NAMESPACE=onnx"
  PRIVATE
    # Note: MORPHIZEN_USE_DLL is inherited from morphizen-graph for DLL export control
    "-DMORPHIZEN_EXPORT_DLL=1"
    "-DHAVE_VERSION_INFO_CONFIG=1"  # Enable generated version header
  )
target_compile_features(morphizen-core-static PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(morphizen-core-static PUBLIC "/Zc:__cplusplus")
else(MSVC)
  target_compile_options(morphizen-core-static PUBLIC "-fPIC")
endif(MSVC)
target_compile_options(morphizen-core-static PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_COMPILER_OPTIONS}>")
# Disable C4946: Protobuf uses opaque type pattern with reinterpret_cast by design.
# Opaque types (MessageLite <-> concrete proto types) are intentionally cast using
# reinterpret_cast because they have the same memory layout but no inheritance relationship.
# This is a valid use case where C4946 is a false positive.
if(MSVC)
  target_compile_options(morphizen-core-static PRIVATE /wd4946)
endif()
# Suppress C4267 (size_t → int) from protobuf-generated .pb.h headers
# included by morphizen-core private sources. MSVC has no "system header"
# concept (the /external:I family exists but is configuration-dependent),
# so the warning still needs a per-target opt-out. The GCC/Clang equivalent
# is handled by marking ${CMAKE_CURRENT_BINARY_DIR} as SYSTEM PUBLIC above —
# the .pb.h headers are then processed as system headers and -Wconversion
# is suppressed inside them automatically (and the SYSTEM marking propagates
# to PUBLIC consumers via INTERFACE_SYSTEM_INCLUDE_DIRECTORIES).
if(MSVC)
  target_compile_options(morphizen-core-static PRIVATE /wd4267)
endif()
if(WIN24_BUILD)
  target_compile_definitions(${LIB_NAME} PUBLIC "-DWIN24_BUILD=ON")
endif()

if(BUILD_PYTHON)
  target_link_libraries(${LIB_NAME} PRIVATE Python3::Python)
  target_compile_definitions(${LIB_NAME} PRIVATE "ENABLE_PYTHON=1")
endif(BUILD_PYTHON)

# ONNX uses custom namespace "morphizen_onnx" to avoid conflicts with ORT's ONNX.
# We link ONNX privately to prevent namespace pollution in dependent targets.
if(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT)
  message(STATUS "ONNX schema support enabled, linking ONNX to morphizen-core-static")
  set(onnx_targets onnx onnx_proto)
  foreach(tgt IN LISTS onnx_targets)
      target_include_directories(${LIB_NAME} PRIVATE $<TARGET_PROPERTY:${tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_link_libraries(${LIB_NAME} PRIVATE $<TARGET_FILE:${tgt}>)
      add_dependencies(${LIB_NAME} ${tgt})
  endforeach()
  target_compile_definitions(${LIB_NAME} PRIVATE ONNX_ML=1)
  target_compile_definitions(${LIB_NAME} PUBLIC MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT=1)
else()
  message(STATUS "ONNX schema support disabled in morphizen-core-static")
  target_compile_definitions(${LIB_NAME} PUBLIC MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT=0)
endif()
