#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC.
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#
set(VAIP_XCLBIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vaip_xclbin CACHE PATH "Path to the directory containing xclbin files")
set(VAIP_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/vaip_config.json" CACHE PATH "Path to the file containing vaip_config.json files")

message(STATUS "VAIP_XCLBIN_DIR : ${VAIP_XCLBIN_DIR}")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dynamic_vaip_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto.cmake)
# MorphiZen must be build along with onnxruntime source tree
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_onnxruntime.cmake)
add_library(vaip_core_objects OBJECT
  ${PROTO_SRCS} ${PROTO_HDRS})
set_target_properties(vaip_core_objects PROPERTIES FOLDER vaip)

vaip_add_remote_target(
  FILE ${CMAKE_CURRENT_BINARY_DIR}/tar.h
  URL https://raw.githubusercontent.com/freebsd/freebsd-src/refs/heads/stable/12/bin/pax/tar.h
  EXPECTED_MD5 eab89f86c63edb8e9dec2ea8faf5ebe2
  PATCH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/patches/tar.h.force_align_1.patch
)
if(NOT TARGET Python3::Interpreter)
  message(FATAL_ERROR "Python3::Interpreter not found")
endif(NOT TARGET Python3::Interpreter)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin_file.hpp.inc
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
  $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/src/xclbin/compress_xclbin.py
  ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin_file.hpp.inc
  "${VAIP_XCLBIN_DIR}"
)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vaip_version_info.hpp.inc
  COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/src/vaip_version_info.hpp.inc.py "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/config_json_binary.hpp
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
  $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/src/xclbin/config_json_binary.hpp.py
  "${VAIP_JSON_CONFIG_FILE}" ON "${VAIP_XCLBIN_DIR}"
)

target_sources(vaip_core_objects PRIVATE
  src/getenv.cpp
  src/getenv.c
  include/vaip/vaip_plugin.hpp
  src/vaip_plugin.cpp
  include/vaip/vaip.hpp
  include/vaip/_sanity_check.hpp
  include/vaip/vaip_io.hpp
  src/vaip_io.cpp
  src/vaip_ort_api.cpp
  include/vaip/pass_context.hpp
  src/pass_context_imp.hpp
  src/pass_context_imp.cpp
  src/anchor_point.cpp
  src/anchor_point_imp.cpp
  src/anchor_point_imp.hpp
  include/vaip/anchor_point.hpp
  src/pass_imp.hpp
  src/pass_imp.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/tar.h
  src/tar_ball.cpp
  include/vaip/pass.hpp
  src/pass.cpp
  include/vaip/model.hpp
  src/model.cpp
  include/vaip/graph.hpp
  src/graph.cpp
  include/vaip/node_attr.hpp
  src/node_attr.cpp
  include/vaip/node_arg.hpp
  src/node_arg.cpp
  include/vaip/tensor_proto.hpp
  src/tensor_proto.cpp
  include/vaip/node_input.hpp
  src/node_input.cpp
  include/vaip/node.hpp
  src/node.cpp
  src/profile_utils.hpp
  src/profile_utils.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin_file.hpp.inc
  include/vaip/mem_xclbin.hpp
  src/xclbin/mem_xclbin.cpp
  src/util.cpp
  src/cache_dir.cpp
  src/cache_dir.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/vaip_version_info.hpp.inc
  src/config.hpp
  src/config.cpp
  src/custom_op.cpp
  include/vaip/pattern.hpp
  src/pattern/pattern.cpp
  src/pattern/pattern_node.cpp
  src/pattern/pattern_node.hpp
  src/pattern/pattern_commutable_node.cpp
  src/pattern/pattern_commutable_node.hpp
  src/pattern/pattern_or.cpp
  src/pattern/pattern_or.hpp
  src/pattern/pattern_wildcard.cpp
  src/pattern/pattern_wildcard.hpp
  src/pattern/pattern_sequence.cpp
  src/pattern/pattern_sequence.hpp
  src/pattern/pattern_constant.cpp
  src/pattern/pattern_constant.hpp
  src/pattern/pattern_graph_input.cpp
  src/pattern/pattern_graph_input.hpp
  src/pattern/pattern_where.cpp
  src/pattern/pattern_where.hpp
  include/vaip/rewrite_rule.hpp
  src/rewrite_rule.cpp
  src/xir_ops/xir_ops_defs.hpp
  src/xir_ops/xir_ops_defs.cpp
  src/xir_ops/ort_custom_ops.hpp
  src/stat.cpp
  src/stat.hpp
  src/file_lock.hpp
  src/file_lock.cpp
  include/vaip/transpose.hpp
  src/transpose.cpp
  include/vaip/guess_reshape.hpp
  src/guess_reshape.cpp
  ../3rd-party/hash-library/crc32.cpp
  ../3rd-party/hash-library/crc32.h
  ../3rd-party/hash-library/hash.h
  ../3rd-party/hash-library/hmac.h
  ../3rd-party/hash-library/keccak.cpp
  ../3rd-party/hash-library/keccak.h
  ../3rd-party/hash-library/md5.cpp
  ../3rd-party/hash-library/md5.h
  ../3rd-party/hash-library/sha1.cpp
  ../3rd-party/hash-library/sha1.h
  ../3rd-party/hash-library/sha256.cpp
  ../3rd-party/hash-library/sha256.h
  ../3rd-party/hash-library/sha3.cpp
  ../3rd-party/hash-library/sha3.h
  include/vaip/config_reader.hpp
  src/xclbin/config_reader.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config_json_binary.hpp
  src/vitisai_compile_model.cpp
  src/vaip_profile.cpp
  src/vaip.cpp
  src/onnxruntime_vitisai_ep.cpp
)
if(MSVC)
  target_sources(vaip_core_objects PRIVATE
    src/vaip_plugin_win.cpp
    src/util_mswin.cpp)
else(MSVC)
  target_sources(vaip_core_objects PRIVATE
    src/vaip_plugin_lnx.cpp
    src/util_linux.cpp)
endif(MSVC)
target_include_directories(vaip_core_objects
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${VAIP_ORT_API_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../3rd-party
)
add_library(vaip_core_static STATIC $<TARGET_OBJECTS:vaip_core_objects>)
set_target_properties(vaip_core_static PROPERTIES FOLDER vaip)
add_library(vaip_core_dynamic SHARED $<TARGET_OBJECTS:vaip_core_objects>)
set_target_properties(vaip_core_dynamic PROPERTIES FOLDER vaip)
# set output name of vaip_core_dynamic, it is required by VitisAI EP.
set_target_properties(vaip_core_dynamic PROPERTIES OUTPUT_NAME onnxruntime_vitisai_ep)

if(MSVC)
  find_program(DUMPBIN_EXECUTABLE dumpbin REQUIRED)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt"
    COMMAND "${DUMPBIN_EXECUTABLE}" "/symbols" "$<TARGET_FILE:vaip::core>" > "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/dump_symbol.py
      ${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt
      ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_vitisai_ep.def.in
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
      $<TARGET_LINKER_LIBRARY_FILE:vaip_core_dynamic>
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt" ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_vitisai_ep.def.in
  )
  target_sources(vaip_core_dynamic PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def)
endif(MSVC)
if(BUILD_SHARED_LIBS)
  add_library (vaip::core ALIAS vaip_core_dynamic)
else(BUILD_SHARED_LIBS)
  add_library (vaip::core ALIAS vaip_core_static)
endif(BUILD_SHARED_LIBS)
set(MorphiZen_DEPS protobuf::libprotobuf glog::glog ZLIB::ZLIB Eigen3::Eigen encryption)
target_link_libraries(vaip_core_objects PUBLIC ${MorphiZen_DEPS})
target_compile_definitions(vaip_core_objects PUBLIC "-DVAIP_USE_DLL=1" "-DVAIP_EXPORT_DLL=1" "-DONNX_NAMESPACE=onnx")
target_link_libraries(vaip_core_dynamic PRIVATE  ${MorphiZen_DEPS})
target_link_libraries(vaip_core_static PUBLIC  ${MorphiZen_DEPS})
target_include_directories(vaip_core_dynamic PUBLIC $<TARGET_PROPERTY:vaip_core_objects,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(vaip_core_static PUBLIC $<TARGET_PROPERTY:vaip_core_objects,INTERFACE_INCLUDE_DIRECTORIES>)

install(
  TARGETS vaip_core_static vaip_core_objects
  EXPORT vaip-core-targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  EXPORT vaip-core-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})
