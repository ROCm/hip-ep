/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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
#include <vector>

namespace vaip_core {
struct OpDefInfo {
  void (*get_domains)(std::vector<Ort::CustomOpDomain>& domains);
};

template <typename T> struct ProcessorOpDefInfo {
  static OpDefInfo* op_fef_info() { return &info; }
  static OpDefInfo info;
};

template <typename T> OpDefInfo ProcessorOpDefInfo<T>::info = {T::process};
} // namespace vaip_core

#ifndef _WIN32
#  include <vaip/export.h>
#  define DEFINE_VAIP_OPDEF(cls, id)                                           \
    extern "C" VAIP_PASS_ENTRY vaip_core::OpDefInfo* vaip_op_def_info() {      \
      return ProcessorOpDefInfo<cls>::op_fef_info();                           \
    }                                                                          \
    extern "C" {                                                               \
    void* /* a hook var*/ id##__hook = nullptr;                                \
    }
#else
#  include <vitis/ai/plugin.hpp>
#  define DEFINE_VAIP_OPDEF(cls, id)                                           \
    static vaip_core::OpDefInfo* vaip_op_def_info() {                          \
      return ProcessorOpDefInfo<cls>::op_fef_info();                           \
    }                                                                          \
    namespace {                                                                \
    static vitis::ai::StaticPluginRegister                                     \
        __register(OUTPUT_NAME, "vaip_op_def_info", (void*)&vaip_op_def_info); \
    }                                                                          \
    extern "C" {                                                               \
    void* /* a hook var*/ id##__hook = &__register;                            \
    }
#endif
