/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#pragma once

#include "onnxruntime_api.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
namespace onnxruntime {
using ProviderOptions = std::unordered_map<std::string, std::string>;
}
namespace vaip_core {
/**
 * Precedence: "config_file" option > "xclbin" option > XLNX_VART_FIRMWARE
 * environment variable At least one of above must be provided.
 *
 * Please note XLNX_VART_FIRMWARE can be set as a directory. But this would
 * leads to ambiguity. So, at VAIP, this would be treated as not set!
 *
 * Graph engine could find xclbin without the user setting XLNX_VART_FIRMWARE.
 * But, VAIP could not determine which one is actually used. So there are a few
 * possible conflicts: 1 User set the XLNX_VART_FIRMWARE as a directory and VAIP
 * couldn't find the any config. 2 Graph engine automatically found a different
 * xclbin file which is conflicting with xclbin and config_file option.
 *
 */
std::string get_config_json_str(const onnxruntime::ProviderOptions& options);
Ort::SessionOptions*
get_session_option(const onnxruntime::ProviderOptions& options);
} // namespace vaip_core
