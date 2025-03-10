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


option(WITH_OPENSSL "include openssl or not" OFF)
if(WITH_OPENSSL)
  find_package(OpenSSL REQUIRED)
endif(WITH_OPENSSL)

add_library(
  encryption
  STATIC
  src/encryption.cpp
  src/encryption.hpp
  )

target_include_directories(encryption PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vaip_core/include
)
if(WITH_OPENSSL)
  target_compile_definitions(encryption PUBLIC -DWITH_OPENSSL)
  target_link_libraries(encryption PRIVATE OpenSSL::SSL OpenSSL::Crypto ${CMAKE_DL_LIBS} glog::glog)
  if (MSVC)
    target_link_libraries(encryption PRIVATE WS2_32.LIB GDI32.LIB ADVAPI32.LIB CRYPT32.LIB USER32.LIB)
  endif()
else()
target_link_libraries(encryption PRIVATE glog::glog)
endif(WITH_OPENSSL)
