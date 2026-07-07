/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
/// basically there are 3 types of MorphiZen clients.
///
/// 1. ORT MORPHIZEN Execution Provider.
///     USE_MORPHIZEN is defined.
/// 2. MorphiZen pass
/// 3. MorphiZen Custom OP
///
/// different user see different part of morphizen.
#define MORPHIZEN_USER__ORT_VITIS_AI_EP 1
#define MORPHIZEN_USER__PASS 2
#define MORPHIZEN_USER__CUSTOM_OP 3
#define MORPHIZEN_USER__INTERNAL 4

#ifndef MORPHIZEN_USER
#if defined(USE_MORPHIZEN)
#define MORPHIZEN_USER MORPHIZEN_USER__ORT_VITIS_AI_EP
#elif defined(MORPHIZEN_CUSTOM_OP)
#define MORPHIZEN_USER MORPHIZEN_USER__CUSTOM_OP
#else
#define MORPHIZEN_USER MORPHIZEN_USER__PASS
#endif
#endif
#include "morphizen/provider_option_keys.hpp"
#if MORPHIZEN_USER == MORPHIZEN_USER__PASS
#include "morphizen/graph.hpp"
#include "morphizen/graph_extensions.hpp"
#include "morphizen/guess_reshape.hpp"
#include "morphizen/model.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
#include "morphizen/node_builder.hpp"
#include "morphizen/pass.hpp"
#if MORPHIZEN_HAS_PATTERN_MATCHING
#include "morphizen/rewrite_rule.hpp"
#endif
#include "./plugin.hpp"
#include "./tensor_proto.hpp"
#include "./util.hpp"
#endif

#include "morphizen/morphizen_core.hpp"
#include "morphizen/op_def.hpp"
#include "morphizen/with_current_graph.hpp"
#include <morphizen/morphizen_ort_api.h>
#include <morphizen/my_ort.h>
#if MORPHIZEN_USER == MORPHIZEN_USER__ORT_VITIS_AI_EP
#include "./ort_api_wrapper.hpp"
#endif

#if MORPHIZEN_USER == MORPHIZEN_USER__CUSTOM_OP
#include "./anchor_point.hpp"
#include "./pass_context.hpp"
#endif

#if MORPHIZEN_USER == MORPHIZEN_USER__CUSTOM_OP ||                             \
    MORPHIZEN_USER == MORPHIZEN_USER__ORT_VITIS_AI_EP
#include "./custom_op_imp.hpp"
#endif
