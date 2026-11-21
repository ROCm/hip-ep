/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Exact original production source is supplied as FLASH_SOURCE at build.
// Namespace/symbol isolation and read-only config diagnostics change no math.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <sstream>
#define hip_gqa_rope hipdnn_original_hip_gqa_rope
#define hip_gqa_transpose_mid_dims hipdnn_original_hip_gqa_transpose_mid_dims
#define hip_gqa_kv_cache_append hipdnn_original_hip_gqa_kv_cache_append
#define hip_gqa_kv_cache_concat hipdnn_original_hip_gqa_kv_cache_concat
#define hip_gqa_dequant_kv_i8_to_fp16 hipdnn_original_hip_gqa_dequant_kv_i8_to_fp16
#define hip_gqa_expand_kv hipdnn_original_hip_gqa_expand_kv
#define hip_gqa_split_qkv hipdnn_original_hip_gqa_split_qkv
#define hip_gqa_causal_mask hipdnn_original_hip_gqa_causal_mask
#define hip_gqa_causal_mask_f32 hipdnn_original_hip_gqa_causal_mask_f32
#define hip_gqa_add_attention_bias_f32 hipdnn_original_hip_gqa_add_attention_bias_f32
#define hip_gqa_bias_key_extent hipdnn_original_hip_gqa_bias_key_extent
#define hip_gqa_softmax_f32_to_f16 hipdnn_original_hip_gqa_softmax_f32_to_f16
#define hip_gqa_softmax_f32_to_f32 hipdnn_original_hip_gqa_softmax_f32_to_f32
#define hip_gqa_softmax_f32_to_out_biased hipdnn_original_hip_gqa_softmax_f32_to_out_biased
#define hip_gqa_softmax_inplace hipdnn_original_hip_gqa_softmax_inplace
#define hip_gqa_flash_decode hipdnn_original_hip_gqa_flash_decode
#define hip_gqa_flash_decode_configured hipdnn_original_hip_gqa_flash_decode_configured
#define hip_gqa_flash_decode_fused_copy hipdnn_original_hip_gqa_flash_decode_fused_copy
#define hip_gqa_flash_prefill_v2 hipdnn_original_hip_gqa_flash_prefill_v2
#define hip_gqa_flash_prefill hipdnn_original_hip_gqa_flash_prefill
#define hip_gqa_flash_prefill_v3_configured hipdnn_original_hip_gqa_flash_prefill_v3_configured
#define hip_gqa_fused_decode hipdnn_original_hip_gqa_fused_decode
#include "hip_custom_kernels.h"
#include "gqa_paged_attention.h"
#include "hip_arch_compat.h"
#include "debug_log.h"
#include "original_gqa_source.h"
namespace original_flash {
#include HIPDNN_ORIGINAL_GQA_SOURCE
}
static thread_local std::string original_forced_config;
extern "C" const char* hipdnn_ut_original_config() {
  if(!original_forced_config.empty())return original_forced_config.c_str();
  static thread_local std::string value;std::ostringstream out;
  for(const auto& p:original_flash::g_flash_decode_cache){const auto& c=p.second;out<<"decode:key="<<p.first<<";wmma="<<int(c.use_wmma)<<";splits="<<c.splits<<";bkv="<<c.bkv<<"|";}
  for(const auto& p:original_flash::g_v6x_cfg_cache){const auto& c=p.second;out<<"v6x:key="<<p.first<<";hb="<<c.hb<<";qw="<<c.qw<<";mt="<<c.mt<<";bkv="<<c.bkv<<"|";}
  for(const auto& p:original_flash::g_v8x_cfg_cache){const auto& c=p.second;out<<"v8x:key="<<p.first<<";nd="<<c.nd<<";hb="<<c.hb<<";bkv="<<c.bkv<<"|";}
  value=out.str();return value.c_str();
}

extern "C" int hipdnn_ut_original_prefill_configured(const void* Q,const void* K,const void* V,void* O,
    int B,int H,int G,int sq,int skv,int D,int capacity,int past,float scale,int window,const GqaPagedConfig* c) {
  original_forced_config="fixed:original:splits="+std::to_string(c->splits)+";bkv="+std::to_string(c->bkv)+";hb="+std::to_string(c->hb)+";qw="+std::to_string(c->qw)+";mt="+std::to_string(c->mt)+";nd="+std::to_string(c->nd);
  return original_flash::hipdnn_original_hip_gqa_flash_prefill_v3_configured(nullptr,Q,K,V,O,B,H,G,sq,skv,D,capacity,past,scale,window,nullptr,H,0,c->hb,c->qw,c->mt,c->bkv,c->nd,nullptr,0);
}
extern "C" int hipdnn_ut_original_decode_configured(const void* Q,const void* K,const void* V,void* O,void* partial,
    int B,int H,int G,int D,int skv,int capacity,float scale,const void* lengths,int window,const GqaPagedConfig* c) {
  const int bkv=D>=256?8:D==128||D==96?32:16;
  original_forced_config="fixed:original:wmma=0;splits="+std::to_string(c->splits)+";bkv="+std::to_string(bkv);
  return original_flash::hipdnn_original_hip_gqa_flash_decode_configured(nullptr,Q,K,V,O,partial,B,H,G,D,skv,capacity,64,scale,lengths,window,nullptr,0,HIP_KV_DTYPE_FP16,nullptr,nullptr,0,c->splits,bkv,nullptr,0);
}
