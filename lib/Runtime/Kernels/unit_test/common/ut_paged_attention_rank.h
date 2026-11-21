/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "gqa_paged_attention.h"
extern "C" int hipdnn_ut_paged_config_count(const hip_paged_attention_f16_desc*);
extern "C" int hipdnn_ut_paged_config(const hip_paged_attention_f16_desc*,int,GqaPagedConfig*);
