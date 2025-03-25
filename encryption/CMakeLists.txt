#
# The Xilinx Vitis AI Vaip in this distribution are provided under the following
# free and permissive binary-only license, but are not provided in source code
# form.  While the following free and permissive license is similar to the BSD
# open source license, it is NOT the BSD open source license nor other
# OSI-approved open source license.
#
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in binary form only, without modification, is permitted
# provided that the following conditions are met:
#
# 1. Redistributions must reproduce the above copyright notice, this list of
#   conditions and the following disclaimer in the documentation and/or other
#   materials provided with the distribution.
#
# 1. The name of Xilinx, Inc. may not be used to endorse or promote products
#   redistributed with this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE
#
find_package(OpenSSL QUIET)
if(TARGET OpenSSL::SSL)
  message(STATUS "found OpenSSL at ${OpenSSL_DIR} build morephizen-encryption with OpenSSL")
else()
  message(STATUS "cannot find_package(OpenSSL), build morephizen-encryption without OpenSSL")
endif()
set(LIB_NAME morphizen-encryption)
add_library(
  ${LIB_NAME}
  STATIC
  include/morphizen/encryption.hpp
  src/encryption.cpp
  )
add_library(morphizen::encryption ALIAS ${LIB_NAME})
## add an alias to keep it compatible with VAIP
add_library(vaip::encryption ALIAS ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES FOLDER morphizen/encryption)
target_include_directories(${LIB_NAME}
  PRIVATE
    # compile time dependency on vaip_io.hpp
    # todo: use std::ostream and std::istream instead of vaip_io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vaip-core/include
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
if(TARGET OpenSSL::SSL)
  target_compile_definitions(${LIB_NAME} PUBLIC -DWITH_OPENSSL)
  target_link_libraries(${LIB_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto ${CMAKE_DL_LIBS} glog::glog)
  if (MSVC)
    # TODO: why we need these windows specific libraries
    target_link_libraries(${LIB_NAME} PRIVATE WS2_32.LIB GDI32.LIB ADVAPI32.LIB CRYPT32.LIB USER32.LIB)
  endif()
else()
  target_link_libraries(${LIB_NAME} PRIVATE glog::glog)
endif()

install(
  TARGETS ${LIB_NAME}
  EXPORT ${LIB_NAME}-targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  EXPORT ${LIB_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})
