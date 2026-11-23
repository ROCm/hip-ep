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

#include "ut_original_gqa_compare.h"
extern "C" int hipdnn_ut_original_selected(int decode,int B,int H,int G,int D,int sq,int skv,int window,HipdnnUtOriginalConfig* out) {
  if(!out)return -1;
  if(decode){
    const int len=window>0?std::min(skv,window):skv;
    const auto it=original_flash::g_flash_decode_cache.find(original_flash::flashDecodeKey(B,H,G,D,len,false));
    if(it==original_flash::g_flash_decode_cache.end())return -1;
    *out={{it->second.splits,it->second.bkv,0,0,0,0},int(it->second.use_wmma)};
  }else{
    const auto key=original_flash::prefillTuneKey(D,sq,skv,H,G,window,false);
    if(D==256){auto it=original_flash::g_v8x_cfg_cache.find(key);if(it==original_flash::g_v8x_cfg_cache.end())return -1;const auto& c=it->second;*out={{0,c.bkv,c.hb,0,0,c.nd},0};}
    else{auto it=original_flash::g_v6x_cfg_cache.find(key);if(it==original_flash::g_v6x_cfg_cache.end())return -1;const auto& c=it->second;*out={{0,c.bkv,c.hb,c.qw,c.mt,0},0};}
  }
  return 0;
}
extern "C" int hipdnn_ut_original_candidates(int decode,int H,int G,int D,int skv,int window,
    HipdnnUtOriginalConfig* output,int capacity) {
  std::vector<HipdnnUtOriginalConfig> values;
  if(decode) {
    int splits[7],n=0;
    const int len=window>0?std::min(skv,window):skv;
    original_flash::flashDecodeCandidateSplits(len,64,splits,n);
    const bool wmma_ok=original_flash::flash_decode_wmma_supported(D,H/G);
    for(int wmma : {0,1}) {
      if(wmma&&!wmma_ok)continue;
      const int nb=wmma&&D==64?2:1;
      for(int bi=0;bi<nb;++bi)for(int i=0;i<n;++i) {
        const int bkv=bi?32:16;
        if(!original_flash::flashDecodeTallTileUseful(bkv,len,splits[i]))continue;
        values.push_back({{splits[i],bkv,0,0,0,0},wmma});
      }
    }
  } else if(D==256) {
    original_flash::PrefillV8xCfg configs[original_flash::kPrefillV8xMaxCandidates];
    const int n=original_flash::prefillV8xCandidates(D,H,G,configs);
    for(int i=0;i<n;++i)values.push_back({{0,configs[i].bkv,configs[i].hb,0,0,configs[i].nd},0});
  } else {
    original_flash::PrefillV6xCfg configs[original_flash::kPrefillV6xMaxCandidates];
    const int n=original_flash::prefillV6xCandidates(D,H,G,window,configs);
    for(int i=0;i<n;++i)values.push_back({{0,configs[i].bkv,configs[i].hb,configs[i].qw,configs[i].mt,0},0});
  }
  if(!output)return int(values.size());
  if(capacity<int(values.size()))return -1;
  std::copy(values.begin(),values.end(),output);return int(values.size());
}
extern "C" int hipdnn_ut_original_native_decode(const void* Q,const void* K,const void* V,void* O,void* partial,
    int B,int H,int G,int D,int skv,int capacity,float scale,const void* lengths,int window,const HipdnnUtOriginalConfig* c) {
  const auto& g=c->geometry;
  original_forced_config="fixed:original:wmma="+std::to_string(c->wmma)+";splits="+std::to_string(g.splits)+";bkv="+std::to_string(g.bkv);
  return original_flash::hipdnn_original_hip_gqa_flash_decode_configured(nullptr,Q,K,V,O,partial,B,H,G,D,skv,capacity,64,scale,lengths,window,nullptr,0,HIP_KV_DTYPE_FP16,nullptr,nullptr,c->wmma,g.splits,g.bkv,nullptr,0);
}
