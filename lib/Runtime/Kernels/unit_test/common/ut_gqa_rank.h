/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
struct UtGqaShape { int b, h, g, d, sq, skv, window, kv_dtype, max_splits; };
struct UtGqaConfig { int wmma = 0, splits = 0, bkv = 0, hb = 0, qw = 0, mt = 0, nd = 0; };
extern "C" int hipdnn_ut_gqa_config_count(const UtGqaShape*);
extern "C" int hipdnn_ut_gqa_config(const UtGqaShape*, int id, UtGqaConfig*);
extern "C" int hip_gqa_flash_decode_configured(
    void*, const void*, const void*, const void*, void*, void*, int, int, int, int,
    int, int, int, float, const void*, int, const void*, int, int, const void*,
    const void*, int, int, int, const void*, int);
extern "C" int hip_gqa_flash_prefill_v3_configured(
    void*, const void*, const void*, const void*, void*, int, int, int, int, int,
    int, int, int, float, int, const void*, int, int, int, int, int, int, int,
    const void*, int);
