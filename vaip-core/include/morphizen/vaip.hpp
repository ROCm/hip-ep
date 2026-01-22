/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
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
#include "./provider_option_keys.hpp"
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

#include "./op_def.hpp"
#include "./vaip_core.hpp"
#include "./with_current_graph.hpp"
#include "./xir_ops.hpp"
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
