/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../hip/gqa_kernel.hip"
#include "ut_paged_attention_rank.h"
extern "C" int hipdnn_ut_paged_config_count(const hip_paged_attention_f16_desc* d) {
  return gqa_paged_candidate_count(*d);
}
extern "C" int hipdnn_ut_paged_config(const hip_paged_attention_f16_desc* d,int id,GqaPagedConfig* out) {
  if(!out||id<0||id>=gqa_paged_candidate_count(*d))return -1;
  *out=gqa_paged_candidate(*d,id);return gqa_paged_config_valid(*d,*out)?0:-1;
}
