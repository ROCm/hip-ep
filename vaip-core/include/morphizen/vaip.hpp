/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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
/// basically there are 3 types client if VAIP.
///
/// 1. ORT VITISAI Execution Provider.
///     USE_VITISAI is defined.
/// 2. VAIP pass
/// 3. Vaip Custom OP
///
/// different user see different part of vaip.
#define VAIP_USER__ORT_VITIS_AI_EP 1
#define VAIP_USER__PASS 2
#define VAIP_USER__CUSTOM_OP 3
#define VAIP_USER__INTERNAL 4

#ifndef VAIP_USER
#  if defined(USE_VITISAI)
#    define VAIP_USER VAIP_USER__ORT_VITIS_AI_EP
#  elif defined(VAIP_CUSTOM_OP)
#    define VAIP_USER VAIP_USER__CUSTOM_OP
#  else
#    define VAIP_USER VAIP_USER__PASS
#  endif
#endif

#if VAIP_USER == VAIP_USER__PASS
#  include "./graph.hpp"
#  include "./guess_reshape.hpp"
#  include "./model.hpp"
#  include "./node_arg.hpp"
#  include "./pass.hpp"
#  include "./rewrite_rule.hpp"
#  include "./tensor_proto.hpp"
#  include "./util.hpp"
#  include "./vaip_plugin.hpp"
#endif

#include "./vaip_core.hpp"
#include "./with_current_graph.hpp"
#include "./xclbin_file.hpp"
#include <vaip/my_ort.h>
#include <vaip/vaip_ort_api.h>

#if VAIP_USER == VAIP_USER__ORT_VITIS_AI_EP
#  include "./vaip_ort.hpp"
#endif

#if VAIP_USER == VAIP_USER__CUSTOM_OP
#  include "./anchor_point.hpp"
#  include "./pass_context.hpp"
#endif

#if VAIP_USER == VAIP_USER__CUSTOM_OP || VAIP_USER == VAIP_USER__ORT_VITIS_AI_EP
#  include "./custom_op_imp.hpp"
#endif

#if VAIP_USER == VAIP_USER__CUSTOM_OP || VAIP_USER == VAIP_USER__PASS
#  include "./transpose.hpp"
#endif
