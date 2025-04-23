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

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto.cmake)
# MorphiZen must be build along with onnxruntime source tree
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_onnxruntime.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphizen-core-static.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hash-library.cmake)
if(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphizen-core-dynamic.cmake)
endif(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/install.cmake)
