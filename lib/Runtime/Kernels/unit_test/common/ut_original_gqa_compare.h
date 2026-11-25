/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
extern "C" int hipdnn_original_hip_gqa_rope(void*,const void*,void*,const void*,const void*,int,int,int,int,int,int,const void*,int);
extern "C" int hipdnn_original_hip_gqa_kv_cache_append(void*,const void*,void*,int,int,int,int,int,int,const void*,int,int,const void*);
extern "C" int hipdnn_original_hip_gqa_flash_decode(void*,const void*,const void*,const void*,void*,void*,int,int,int,int,int,int,int,float,const void*,int,const void*,int,int,const void*,const void*,const void*,int);
extern "C" int hipdnn_original_hip_gqa_flash_prefill(void*,const void*,const void*,const void*,void*,int,int,int,int,int,int,int,int,float,int,const void*,int,int,const void*,int);
extern "C" const char* hipdnn_ut_original_config();

extern "C" int hipdnn_ut_original_prefill_configured(const void*,const void*,const void*,void*,int,int,int,int,int,int,int,int,float,int,const GqaPagedConfig*);
extern "C" int hipdnn_ut_original_decode_configured(const void*,const void*,const void*,void*,void*,int,int,int,int,int,int,float,const void*,int,const GqaPagedConfig*);

// UT-only native descriptors retain the original Flash candidate space.
struct HipdnnUtOriginalConfig {
  GqaPagedConfig geometry;
  int wmma;
};
extern "C" int hipdnn_ut_original_candidates(int,int,int,int,int,int,HipdnnUtOriginalConfig*,int);
extern "C" int hipdnn_ut_original_selected(int,int,int,int,int,int,int,int,HipdnnUtOriginalConfig*);
extern "C" int hipdnn_ut_original_native_decode(const void*,const void*,const void*,void*,void*,int,int,int,int,int,int,float,const void*,int,const HipdnnUtOriginalConfig*);
