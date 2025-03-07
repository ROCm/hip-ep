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
add_library(vaip_core_objects OBJECT
  ${PROTO_SRCS} ${PROTO_HDRS})
add_library(vaip_core_static STATIC $<TARGET_OBJECTS:vaip_core_objects>)
add_library(vaip_core_dynamic SHARED $<TARGET_OBJECTS:vaip_core_objects>)
if(BUILD_SHARED_LIBS)
  add_library (vaip::core ALIAS vaip_core_dynamic)
else(BUILD_SHARED_LIBS)
  add_library (vaip::core ALIAS vaip_core_static)
endif(BUILD_SHARED_LIBS)
target_link_libraries(vaip_core_objects PUBLIC protobuf::libprotobuf)
target_compile_definitions(vaip_core_objects PUBLIC "-DVAIP_EXPORT_DLL=1")
target_compile_definitions(vaip_core_objects PUBLIC "-DONNX_NAMESPACE=onnx")
target_link_libraries(vaip_core_dynamic PRIVATE protobuf::libprotobuf)
