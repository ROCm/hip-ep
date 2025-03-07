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


cmake_minimum_required(VERSION 3.26)
project(
  MorphiZen
  VERSION 1.0.0
  LANGUAGES C CXX)
cmake_policy(SET CMP0115 NEW)
set(CMAKE_CXX_STANDARD 17)
if(MSVC)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_options.msvc.cmake)
else()
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_options.linux.cmake)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dynamic_vaip_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto.cmake)
add_library(vaip_core STATIC
  src/main.cpp ${PROTO_SRCS} ${PROTO_HDRS})
# vai_add_library(
#   NAME
#   core
#   INCLUDE_DIR
#   include
#   SRCS
#   ${PROTO_SRCS}
#   ${PROTO_HDRS}
#   ${CMAKE_CURRENT_BINARY_DIR}/config_reader.cpp
#   src/xclbin/xclbin_file.cpp
#   ${CMAKE_CURRENT_BINARY_DIR}/mem_xclbin.cpp
#   ${CMAKE_CURRENT_LIST_DIR}/../vaip_summary_report/reporter.cpp
#   ${CMAKE_CURRENT_LIST_DIR}/../vaip_summary_report/reporter.hpp
#   include/vaip/vaip_plugin.hpp
#   src/vaip_plugin.cpp
#   include/vaip/plugin.hpp
#   src/plugin.cpp
#   src/vaip.cpp
#   src/vaip_profile.cpp
#   src/tensor_proto.cpp
#   include/vaip/tensor_proto.hpp
#   include/vaip/vaip.hpp
#   include/vaip/_sanity_check.hpp
#   include/vaip/custom_op_imp.hpp
#   src/custom_op.cpp
#   include/vaip/xir_headers.hpp
#   include/vaip/util.hpp
#   src/vitisai_compile_model.cpp
#   src/pass_context_imp.hpp
#   src/pass_context_imp.cpp
#   src/pass_imp.hpp
#   src/pass_imp.cpp
#   src/tar_ball.cpp
#   src/pass.cpp
#   include/vaip/pass.hpp
#   src/graph.cpp
#   include/vaip/graph.hpp
#   src/model.cpp
#   include/vaip/model.hpp
#   src/anchor_point.cpp
#   src/anchor_point_imp.cpp
#   src/anchor_point_imp.hpp
#   include/vaip/anchor_point.hpp
#   include/vaip/node_attr.hpp
#   include/vaip/node_arg.hpp
#   src/node_arg.cpp
#   include/vaip/node_input.hpp
#   src/node_input.cpp
#   include/vaip/node.hpp
#   src/node.cpp
#   src/util.cpp
#   src/file_lock.cpp
#   include/vaip/pattern.hpp
#   src/pattern/pattern.cpp
#   src/pattern/pattern_node.cpp
#   src/pattern/pattern_node.hpp
#   src/pattern/pattern_commutable_node.cpp
#   src/pattern/pattern_commutable_node.hpp
#   src/pattern/pattern_or.cpp
#   src/pattern/pattern_or.hpp
#   src/pattern/pattern_wildcard.cpp
#   src/pattern/pattern_wildcard.hpp
#   src/pattern/pattern_sequence.cpp
#   src/pattern/pattern_sequence.hpp
#   src/pattern/pattern_constant.cpp
#   src/pattern/pattern_constant.hpp
#   src/pattern/pattern_graph_input.cpp
#   src/pattern/pattern_graph_input.hpp
#   src/pattern/pattern_where.cpp
#   src/pattern/pattern_where.hpp
#   include/vaip/rewrite_rule.hpp
#   src/xir_ops/xir_ops_defs.hpp
#   src/xir_ops/xir_ops_defs.cpp
#   src/xir_ops/xir_ops.hpp
#   src/rewrite_rule.cpp
#   src/node_attrs.cpp
#   src/config.hpp
#   src/config.cpp
#   src/cache_dir.hpp
#   src/cache_dir.cpp
#   src/stat.cpp
#   src/stat.hpp
#   src/file_lock.hpp
#   src/profile_utils.hpp
#   src/profile_utils.cpp
#   ${CMAKE_CURRENT_BINARY_DIR}/version_info.cpp
#   src/version_info.cpp.in
#   include/vaip/transpose.hpp
#   src/transpose.cpp
#   include/vaip/guess_reshape.hpp
#   src/guess_reshape.cpp
#   include/vaip/dd/coeffs.hpp
#   src/dd/coeffs.cpp
#   include/vaip/vaiml/vaiml_logging.hpp
#   include/vaip/vaiml/vaiml_config.hpp
#   include/vaip/vaiml/vaiml_utils.hpp
#   include/vaip/vaiml/vaiml_config_rule.hpp
#   src/vaiml/vaiml_logging.cpp
#   src/vaiml/vaiml_utils.cpp
#   src/vaiml/vaiml_config_rule.cpp
#   ${CMAKE_CURRENT_BINARY_DIR}/version.c
#   ${vaip_lib_file}
#   include/vaip/vaip_io.hpp
#   src/vaip_io.cpp
#   )

# include(CheckIncludeFile)
# check_include_file(dlfcn.h DLFCN_H_FOUND)
# if(DLFCN_H_FOUND)
#   target_sources(core PRIVATE src/util_linux.cpp)
# elseif(MSVC)
#   target_sources(core PRIVATE src/util_mswin.cpp)
# else()
#   target_sources(core PRIVATE src/util_unknown.cpp)
# endif()
target_link_libraries(
  vaip_core
  PRIVATE
  PUBLIC protobuf::libprotobuf)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(vaip_core PRIVATE "-DVAIP_EXPORT_DLL=1")
else()
    target_compile_definitions(vaip_core PUBLIC "-DVAIP_EXPORT_DLL=1")
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_core PUBLIC "-DONNX_NAMESPACE=onnx")
# install(DIRECTORY include/vaip DESTINATION include)

# install(
#   FILES ${CMAKE_CURRENT_BINARY_DIR}/config.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/anchor_point.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/capability.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/pass_context.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/version.pb.h
#   DESTINATION include/vaip)

# foreach(CONFIG_FILE IN LISTS CONFIG_FILE_LIST)
#   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE} DESTINATION bin)
# endforeach()

# configure_file(cmake/config.cmake.in
#                "${CMAKE_CURRENT_BINARY_DIR}/vaip-core-config.cmake" @ONLY)

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#   "${CMAKE_CURRENT_BINARY_DIR}/vaip-core-config-version.cmake"
#   VERSION ${PROJECT_VERSION}
#   COMPATIBILITY AnyNewerVersion)
# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vaip-core-config.cmake"
#               "${CMAKE_CURRENT_BINARY_DIR}/vaip-core-config-version.cmake"
#         DESTINATION share/cmake/${PROJECT_NAME})

# vai_add_debug_command(core
#     COMMAND "${CMAKE_INSTALL_PREFIX}\\bin\\test_onnx_runner.exe"
#     ARGUMENTS "${CMAKE_CURRENT_SOURCE_DIR}\\..\\..\\test_onnx_runner\\data\\pt_resnet50.onnx"
#     ENVIRONMENT DEBUG_DD_MERGE_QMHAGRPB=1
#   )
