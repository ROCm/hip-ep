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
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dynamic_vaip_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto.cmake)
# MorphiZen must be build along with onnxruntime source tree
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_onnxruntime.cmake)
add_library(vaip_core_objects OBJECT
  ${PROTO_SRCS} ${PROTO_HDRS})
target_sources(vaip_core_objects PRIVATE
  include/vaip/vaip_plugin.hpp
  src/vaip_plugin.cpp
)
if(MSVC)
  target_sources(vaip_core_objects PRIVATE src/vaip_plugin_win.cpp)
else(MSVC)
  target_sources(vaip_core_objects PRIVATE src/vaip_plugin_lnx.cpp)
endif(MSVC)
target_include_directories(vaip_core_objects PRIVATE include ${VAIP_ORT_API_DIR})
add_library(vaip_core_static STATIC $<TARGET_OBJECTS:vaip_core_objects>)
add_library(vaip_core_dynamic SHARED $<TARGET_OBJECTS:vaip_core_objects>)
if(BUILD_SHARED_LIBS)
  add_library (vaip::core ALIAS vaip_core_dynamic)
else(BUILD_SHARED_LIBS)
  add_library (vaip::core ALIAS vaip_core_static)
endif(BUILD_SHARED_LIBS)
set(MorphiZen_DEPS  protobuf::libprotobuf glog::glog)
target_link_libraries(vaip_core_objects PUBLIC ${MorphiZen_DEPS})
target_compile_definitions(vaip_core_objects PUBLIC "-DVAIP_EXPORT_DLL=1")
target_compile_definitions(vaip_core_objects PUBLIC "-DONNX_NAMESPACE=onnx")
target_link_libraries(vaip_core_dynamic PRIVATE  ${MorphiZen_DEPS})
target_link_libraries(vaip_core_static PUBLIC  ${MorphiZen_DEPS})
