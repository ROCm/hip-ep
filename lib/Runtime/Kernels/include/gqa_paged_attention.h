/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "hip_custom_kernels.h"

// Internal view shared by the paged entry and the existing GQA math TU.
struct GqaPagedTile { size_t base; bool continuous; };
inline int gqa_paged_page_shift(int size) {
  if(size<=0||(size&(size-1)))return -1;
  int shift=0;while((1u<<shift)!=static_cast<unsigned>(size))++shift;return shift;
}
struct GqaPagedArgs {
  const int32_t *table, *cum, *past, *post, *status;
  int page_size, columns, kv_heads;
  int page_shift=-1;
  bool adaptive_decode=false;
  int decode_window=0;
  int decode_min_splits=1;
#if defined(__HIPCC__)
  __host__ __device__
#endif
  static int effective_splits(int length,int window,int cap,int minimum=1) {
    const int active=window>0&&window<length?window:length;
    int splits=1;
    while(splits<cap && int64_t(splits)*128<active) splits*=2;
    while(splits<cap && splits<minimum)splits*=2;
    // At most one split per 16 keys, with one for empty/short ranges.
    const int useful=active>0?int((int64_t(active)+15)/16):1;
    while(splits>useful && splits>1)splits/=2;
    return splits;
  }
#if defined(__HIPCC__)
  __device__ __forceinline__ size_t offset(int b, int pos, int h, int d) const {
    const unsigned position=static_cast<unsigned>(pos);
    const unsigned logical_page=page_shift>=0?position>>page_shift:position/static_cast<unsigned>(page_size);
    const unsigned row=page_shift>=0?position&static_cast<unsigned>(page_size-1):position%static_cast<unsigned>(page_size);
    const int page = table[static_cast<size_t>(b) * columns + logical_page];
    return ((static_cast<size_t>(page) * page_size + row) * kv_heads + h) * d;
  }
  __device__ __forceinline__ GqaPagedTile tile(int b,int h,int pos,int rows,int d,int len) const {
    const unsigned position=static_cast<unsigned>(pos);
    const unsigned row=page_shift>=0?position&static_cast<unsigned>(page_size-1):position%static_cast<unsigned>(page_size);
    const bool continuous=int64_t(pos)+rows<=len && rows<=page_size-int(row);
    if(!continuous)return {0,false};
    // tile() is called only at wave-uniform KV loop origins, unlike offset()
    // which also serves divergent append and fallback accesses.
    const unsigned logical_page=page_shift>=0?position>>page_shift:position/static_cast<unsigned>(page_size);
    const int page=__builtin_amdgcn_readfirstlane(table[static_cast<size_t>(b)*columns+logical_page]);
    return {((static_cast<size_t>(page)*page_size+row)*kv_heads+h)*d,true};
  }
  __device__ __forceinline__ bool active(int b) const {
    return *status == 0 && cum[b + 1] > cum[b];
  }
#endif
};

// Configurations describe final launches; native IDs never enter the ABI.
using GqaPagedConfig = hip_paged_attention_f16_config_v2;
constexpr int kPagedDecodeSplits = 64;
// Native host drivers link these helpers from the production kernel library.
HIP_KERNEL_API GqaPagedConfig gqa_paged_default(const hip_paged_attention_f16_desc&);
HIP_KERNEL_API GqaPagedConfig gqa_paged_legacy(const hip_paged_attention_f16_desc&, hip_paged_attention_f16_config);
HIP_KERNEL_API bool gqa_paged_config_valid(const hip_paged_attention_f16_desc&, GqaPagedConfig);
HIP_KERNEL_API int gqa_paged_candidate_count(const hip_paged_attention_f16_desc&);
HIP_KERNEL_API GqaPagedConfig gqa_paged_candidate(const hip_paged_attention_f16_desc&, int);

HIP_KERNEL_API int gqa_paged_attention_f16_launch(void *stream,
    const hip_paged_attention_f16_desc &desc, const void *q_rotated,
    float *partials, GqaPagedArgs view, GqaPagedConfig config);

// Explicit preparation only; calls the existing production GQA tuner.
int gqa_paged_attention_f16_tune(void* stream,const hip_paged_attention_f16_desc&,
    const void* q_rotated,float* partials,GqaPagedArgs,GqaPagedConfig* selected);
