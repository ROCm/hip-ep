##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
vaip_add_remote_target(
  FILE ${CMAKE_CURRENT_BINARY_DIR}/tar.h
  URL https://raw.githubusercontent.com/freebsd/freebsd-src/refs/heads/stable/12/bin/pax/tar.h
  EXPECTED_MD5 eab89f86c63edb8e9dec2ea8faf5ebe2
  PATCH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/patches/tar.h.force_align_1.patch
)

vaip_add_remote_target(
  FILE ${CMAKE_CURRENT_BINARY_DIR}/xclbin.h
  URL https://raw.githubusercontent.com/Xilinx/XRT/347acbd5e2b2d658ecd21d024547703fccc5572c/src/runtime_src/core/include/xclbin.h
  EXPECTED_MD5 a3eb400c4836d01a59b7c5bb3ddd83e3
)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin_file.hpp.inc
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
  $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/src/xclbin/compress_xclbin.py
  ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin_file.hpp.inc
  "${VAIP_XCLBIN_DIR}"
)

add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/config_json_binary.hpp ${CMAKE_CURRENT_BINARY_DIR}/vaip_config.json
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
  $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/src/xclbin/config_json_binary.hpp.py
  "${VAIP_JSON_CONFIG_FILE}"
  "${TRIM_CONFIG}"
  "${VAIP_XCLBIN_DIR}"
  "${morphizen_WITH_VAIP_CONFIG_FILE}"
)

if(morphizen_WITH_VAIP_CONFIG_FILE)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vaip_config.json DESTINATION bin)
endif()
set(LIB_NAME morphizen-core-static)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/version_info.cpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/src/version_info.cpp
  @ONLY)

add_library(${LIB_NAME} STATIC
  ${PROTO_SRCS} ${PROTO_HDRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../vaip-core/src/getenv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../vaip-core/src/getenv.c
  src/version_info.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/version_info.cpp
  include/morphizen/vaip_plugin.hpp
  src/vaip_plugin.cpp
  include/morphizen/vaip.hpp
  include/morphizen/_sanity_check.hpp
  include/morphizen/vaip_io.hpp
  src/vaip_io.cpp
  src/vaip_ort_api.cpp
  include/morphizen/pass_context.hpp
  src/pass_context_imp.hpp
  src/pass_context_imp.cpp
  src/anchor_point.cpp
  src/anchor_point_imp.cpp
  src/anchor_point_imp.hpp
  include/morphizen/anchor_point.hpp
  src/pass_imp.hpp
  src/pass_imp.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/tar.h
  src/tar_ball.cpp
  src/tar_ball.hpp
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
  include/morphizen/model.hpp
  src/model.cpp
  include/morphizen/graph.hpp
  src/graph.cpp
  include/morphizen/node_attr.hpp
  src/node_attr.cpp
  include/morphizen/node_arg.hpp
  src/node_arg.cpp
  include/morphizen/tensor_proto.hpp
  src/tensor_proto.cpp
  include/morphizen/node_input.hpp
  src/node_input.cpp
  include/morphizen/node.hpp
  src/node.cpp
  src/profile_utils.hpp
  src/profile_utils.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin_file.hpp.inc
  include/morphizen/mem_xclbin.hpp
  src/xclbin/mem_xclbin.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/xclbin.h
  include/morphizen/xclbin_file.hpp
  src/xclbin/xclbin_file.cpp
  src/util.cpp
  src/cache_dir.cpp
  src/cache_dir.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/vaip_version_info.hpp.inc
  src/config.hpp
  src/config.cpp
  src/custom_op.cpp
  include/morphizen/pattern.hpp
  src/pattern/pattern.cpp
  src/pattern/pattern_node.cpp
  src/pattern/pattern_node.hpp
  src/pattern/pattern_commutable_node.cpp
  src/pattern/pattern_commutable_node.hpp
  src/pattern/pattern_or.cpp
  src/pattern/pattern_or.hpp
  src/pattern/pattern_wildcard.cpp
  src/pattern/pattern_wildcard.hpp
  src/pattern/pattern_commutable_node.hpp
  src/pattern/pattern_sequence.cpp
  src/pattern/pattern_sequence.hpp
  src/pattern/pattern_constant.cpp
  src/pattern/pattern_constant.hpp
  src/pattern/pattern_graph_input.cpp
  src/pattern/pattern_graph_input.hpp
  src/pattern/pattern_where.cpp
  src/pattern/pattern_where.hpp
  include/morphizen/rewrite_rule.hpp
  src/rewrite_rule.cpp
  src/xir_ops/xir_ops_defs.hpp
  src/xir_ops/xir_ops_defs.cpp
  src/xir_ops/xir_ops.cpp
  include/morphizen/xir_ops.hpp
  src/stat.cpp
  src/stat.hpp
  src/file_lock.hpp
  src/file_lock.cpp
  include/morphizen/transpose.hpp
  src/transpose.cpp
  include/morphizen/guess_reshape.hpp
  src/guess_reshape.cpp
  include/morphizen/config_reader.hpp
  src/xclbin/config_reader.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config_json_binary.hpp
  src/vitisai_compile_model.cpp
  src/vaip_profile.cpp
  src/onnxruntime_vitisai_ep.cpp
  src/file_stream.hpp
  src/file_stream.cpp
)
add_library (morphizen::morphizen-core-static ALIAS morphizen-core-static)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER morphizen)
if(MSVC)
  target_sources(${LIB_NAME} PRIVATE
    src/mmap_file_win.hpp
    src/mmap_file_win.cpp
    src/vaip_plugin_win.cpp
    src/util_mswin.cpp)
else(MSVC)
  target_sources(${LIB_NAME} PRIVATE
    src/vaip_plugin_lnx.cpp
    src/util_linux.cpp)
endif(MSVC)
target_include_directories(${LIB_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime>
  # TODO: it is not a good pratice, we should use
  #      #include <core/session/onnxruntime_c_api.h>
  # instead of
  #      #include <onnxruntime_c_api.h>
  #
  # however, too many source code use the later, so we add it to search path
  #
  # if morphizen users need to use pre-installed version of morphizen,
  # it should use the following in their CMakeLists.txt
  #
  #    find_package(onnxruntime CONFIG REQUIRED)
  #    target_link_libraries (<YOUR-TARGET> PRIVATE morphizen::morphizen-core-static)
  #
  # then use #include <morphizen/onnxruntime_api.h> instread.
  #
  $<BUILD_INTERFACE:${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime/core/session>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${VAIP_ORT_API_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../3rd-party
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  ${zlib_SOURCE_DIR}
  ${zlib_BINARY_DIR}
)

set(MorphiZen_DEPS protobuf::libprotobuf
  glog::glog morphizen::encryption ZLIB::ZLIB Microsoft.GSL::GSL)
target_link_libraries(${LIB_NAME} PUBLIC ${MorphiZen_DEPS})
target_compile_definitions(${LIB_NAME}
  PRIVATE "-DVAIP_USE_DLL=1" "-DVAIP_EXPORT_DLL=1"
  PUBLIC "-DONNX_NAMESPACE=onnx")
target_compile_features(morphizen-core-static PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(morphizen-core-static PUBLIC "/Zc:__cplusplus")
else(MSVC)
  target_compile_options(morphizen-core-static PUBLIC "-fPIC")
endif(MSVC)
target_compile_options(morphizen-core-static PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_COMPILER_OPTIONS}>")
if(WIN24_BUILD)
  target_compile_definitions(${LIB_NAME} PUBLIC "-DWIN24_BUILD=ON")
endif()

if(BUILD_PYTHON)
  target_link_libraries(${LIB_NAME} PRIVATE Python3::Python)
  target_compile_definitions(${LIB_NAME} PRIVATE "ENABLE_PYTHON=1")
endif(BUILD_PYTHON)
