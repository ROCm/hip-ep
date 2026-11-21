/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_custom_kernels.h"
#include "gqa_paged_attention.h"
#include "paged_attention_autotune.h"
#include "../../../common/ut_rank_input.h"
#include "../../../common/ut_paged_attention_rank.h"
#include "../../../common/hip_adaptive_timing.h"
#include "../../../common/ut_gpu_execution_scope.h"
#include "../../../common/ut_case_input.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


#ifndef HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS
#ifdef HIPDNN_PAGED_LUT_EMBED
#include "lut_fb_path.h"
const unsigned char paged_lut[] = {
#embed HIPDNN_PAGED_LUT_FB
};
extern "C" const unsigned char* const kPagedAttentionLutBlobs[1]={paged_lut};
extern "C" const size_t kPagedAttentionLutBlobSizes[1]={sizeof(paged_lut)};
extern "C" const size_t kPagedAttentionLutBlobCount=1;
#else
extern "C" const unsigned char* const kPagedAttentionLutBlobs[1]={nullptr};
extern "C" const size_t kPagedAttentionLutBlobSizes[1]={0};
extern "C" const size_t kPagedAttentionLutBlobCount=0;
#endif
#endif

namespace timing = hipdnn_ep_test::timing;
namespace input = hipdnn_ep_test::input;
static constexpr bool kDecode = 1;
static const char* kLeaf = 1 ? "paged_attention/decode/Xfp16_Wfp16_Yfp16" : "paged_attention/prefill/Xfp16_Wfp16_Yfp16";
static void check(hipError_t rc) { if(rc!=hipSuccess) throw std::runtime_error(hipGetErrorString(rc)); }
struct Case {
  int B=3,H=8,G=2,D=128,P=64,S=17,M=8,rotary=0,max_position=4096,window=-1,rotary_dim=0,interleaved=0;
  std::vector<int> q = kDecode ? std::vector<int>{1,0,1} : std::vector<int>{3,0,17};
  std::vector<int> past{16,17,18};
  float scale=0,softcap=0;
  unsigned seed=1119;
  int shared_prefix=0,compare=0,compare_order=0;
  std::string map="permuted",error="none";
};
static std::string list(const std::vector<int>& v,char separator='+') {
  std::string s; for(int x:v){if(!s.empty())s+=separator;s+=std::to_string(x);} return s;
}
static std::string number(float x) { char s[40];std::snprintf(s,sizeof(s),"%.9g",x);return s; }
static std::string describe_case(const Case& c) {
  return std::string(kDecode ? "decode" : "prefill")+"_B"+std::to_string(c.B)+"_H"+std::to_string(c.H)+"_G"+std::to_string(c.G)+"_D"+std::to_string(c.D)+
      "_P"+std::to_string(c.P)+"_S"+std::to_string(c.S)+"_M"+std::to_string(c.M)+"_q"+list(c.q)+"_past"+list(c.past)+"_rope"+std::to_string(c.rotary)+
      "_rD"+std::to_string(c.rotary_dim)+"_inter"+std::to_string(c.interleaved)+"_pos"+std::to_string(c.max_position)+"_scale"+number(c.scale)+
      "_win"+std::to_string(c.window)+"_cap"+number(c.softcap)+"_map"+c.map+"_shared"+std::to_string(c.shared_prefix)+"_seed"+std::to_string(c.seed)+"_err"+c.error;
}
static std::vector<int> array(const std::string& s) {
  std::vector<int> v;std::stringstream in(s);std::string item;
  while(std::getline(in,item,','))v.push_back(int(input::integer(item,"array",0)));
  if(v.empty() || s.back()==',')throw std::invalid_argument("empty array item");return v;
}
static float floating(const std::string& s) { size_t n;float x=std::stof(s,&n);if(n!=s.size())throw std::invalid_argument("invalid float");return x; }
static const std::vector<std::string> errors={"none","cum_start","cum_end","cum_order","negative_past","decode_length","post_overflow","table_small","bad_page","position","workspace","workspace_alignment","alias","pool_alias","softcap","negative_scale","nan_scale","inf_scale","nan_softcap","interleaved","partial_rotary","d256_window","geometry","write_conflict","null_rotary","size_overflow","tensor_alignment"};
static Case parse_single(int argc,char** argv,bool& single,int& coverage) {
  Case c;coverage=3;
  if(const char* e=std::getenv("HIPDNN_UT_COVERAGE"))coverage=int(input::integer(e,"coverage",1,3));
  for(int i=1;i<argc;++i){const std::string f=argv[i];const std::string s=input::value(argc,argv,i);
    if(f=="--coverage"){coverage=int(input::integer(s,"coverage",1,3));continue;}
    single=true;
    if(f=="--b")c.B=int(input::integer(s,"B"));
    else if(f=="--h")c.H=int(input::integer(s,"H"));
    else if(f=="--g")c.G=int(input::integer(s,"G"));
    else if(f=="--d")c.D=int(input::integer(s,"D"));
    else if(f=="--pages")c.P=int(input::integer(s,"P"));
    else if(f=="--page-size")c.S=int(input::integer(s,"S"));
    else if(f=="--table-columns")c.M=int(input::integer(s,"M"));
    else if(f=="--q-lens")c.q=array(s);
    else if(f=="--past")c.past=array(s);
    else if(f=="--rotary")c.rotary=input::boolean(s,"rotary");
    else if(f=="--rotary-dim")c.rotary_dim=int(input::integer(s,"rotary_dim",0));
    else if(f=="--interleaved")c.interleaved=input::boolean(s,"interleaved");
    else if(f=="--max-position")c.max_position=int(input::integer(s,"max_position"));
    else if(f=="--window")c.window=int(input::integer(s,"window",INT32_MIN));
    else if(f=="--scale")c.scale=floating(s);
    else if(f=="--softcap")c.softcap=floating(s);
    else if(f=="--shared-prefix")c.shared_prefix=int(input::integer(s,"shared_prefix",0));
    else if(f=="--compare")c.compare=input::boolean(s,"compare");
    else if(f=="--compare-order")c.compare_order=input::boolean(s,"compare_order");
    else if(f=="--map")c.map=s;
    else if(f=="--seed")c.seed=unsigned(input::integer(s,"seed",0,UINT32_MAX));
    else if(f=="--error")c.error=s;
    else if(f=="--phase"){if(s!=(kDecode ? "decode":"prefill"))throw std::invalid_argument("phase must match leaf");}
    else throw std::invalid_argument("unknown flag: "+f);
  }
  if(c.H%c.G)throw std::invalid_argument("H must be a multiple of G");
  if(int(c.q.size())!=c.B || int(c.past.size())!=c.B)throw std::invalid_argument("q-lens/past must have B items");
  if(c.map!="ordered"&&c.map!="permuted"&&c.map!="shared")throw std::invalid_argument("unknown page map");
  if(std::find(errors.begin(),errors.end(),c.error)==errors.end())throw std::invalid_argument("unknown error case");
  return c;
}

// A rejection fixture starts from a valid base. Check trigger prerequisites
// before allocation/reference work, so an inapplicable injection is a CLI error.
static void validate_case(const Case& c) {
  if (c.B<=0 || c.H<=0 || c.G<=0 || c.H%c.G || c.P<=0 || c.S<=0 || c.M<=0 || c.max_position<=0)
    throw std::invalid_argument("base geometry must be positive; H must divide by G");
  if (int(c.q.size())!=c.B || int(c.past.size())!=c.B)
    throw std::invalid_argument("q-lens/past must have B items");
  if (c.D!=64 && c.D!=96 && c.D!=128 && c.D!=256)
    throw std::invalid_argument("base D must be 64,96,128,256");
  if (!std::isfinite(c.scale) || c.scale<0 || !std::isfinite(c.softcap) || c.softcap!=0)
    throw std::invalid_argument("base scale must be finite/nonnegative and softcap zero; use ERROR for rejection");
  if (c.rotary && (c.interleaved || (c.rotary_dim && c.rotary_dim!=c.D)))
    throw std::invalid_argument("base RoPE must be non-interleaved full-D");
  if (!kDecode && c.D==256 && c.window>0)
    throw std::invalid_argument("prefill base D256+window unsupported; use ERROR=d256_window");
  const int hpg=c.H/c.G;
  const bool decode_geometry=hpg==1||hpg==2||hpg==3||hpg==4||hpg==5||hpg==6||hpg==8||hpg==16||(hpg==9&&c.D==128);
  if ((kDecode||c.error=="decode_length")&&!decode_geometry)
    throw std::invalid_argument("base geometry must support decode for this case");
  int64_t tokens=0,pages=0;int active=0;bool position_trigger=false;
  for(int b=0;b<c.B;++b) {
    if(c.q[b]<0||c.past[b]<0||(kDecode&&c.q[b]>1))throw std::invalid_argument("invalid base q/past; decode q must be 0/1");
    tokens+=c.q[b];const int64_t post=int64_t(c.past[b])+c.q[b];
    if(tokens>INT32_MAX||post>INT32_MAX||(post+c.S-1)/c.S>c.M)throw std::invalid_argument("base T/post/table capacity insufficient");
    pages+=(post+c.S-1)/c.S;
    if(c.q[b]){++active;position_trigger=position_trigger||post>1;
      if(c.rotary&&post>c.max_position)throw std::invalid_argument("base RoPE position exceeds max-position");
      if(c.map=="shared"&&c.past[b]<c.S)throw std::invalid_argument("shared map needs one readonly prefix page");}
  }
  const int prefix_pages=c.shared_prefix?c.shared_prefix/c.S:1;
  if(c.shared_prefix && (c.map!="shared"||c.shared_prefix%c.S))throw std::invalid_argument("shared-prefix must be page-aligned and MAP=shared");
  if(c.map=="shared") {for(int b=0;b<c.B;++b){const int n=int((int64_t(c.past[b])+c.q[b]+c.S-1)/c.S);pages-=std::min(n,prefix_pages);if(c.q[b]&&c.past[b]<prefix_pages*c.S)throw std::invalid_argument("shared-prefix must be readonly for active requests");}pages+=prefix_pages;}
  if(pages>c.P)throw std::invalid_argument("base physical pool insufficient");
  if(c.map!="ordered"&&c.map!="permuted"&&c.map!="shared")throw std::invalid_argument("unknown page map");
  if(std::find(errors.begin(),errors.end(),c.error)==errors.end())throw std::invalid_argument("unknown error case");
  if(kDecode&&c.error=="d256_window")throw std::invalid_argument("ERROR=d256_window is prefill-only; decode D256+window is legal");
  const std::set<std::string> needs_tokens={"cum_start","cum_end","cum_order","negative_past","decode_length","post_overflow","table_small","bad_page","position","write_conflict","workspace","workspace_alignment","alias","pool_alias","tensor_alignment","null_rotary"};
  if(!tokens&&needs_tokens.count(c.error))throw std::invalid_argument("ERROR="+c.error+" requires T>0; all-empty entry is a no-op");
  if(c.error=="write_conflict"&&active<2)throw std::invalid_argument("ERROR=write_conflict requires at least two active requests (q>0)");
  if(c.error=="decode_length"&&tokens<2)throw std::invalid_argument("ERROR=decode_length requires T>=2 and decode geometry");
  if(c.error=="table_small"&&int64_t(c.M)*c.S>=INT32_MAX)throw std::invalid_argument("ERROR=table_small requires M*S<INT32_MAX to avoid post overflow");
  if(c.error=="position"&&!position_trigger)throw std::invalid_argument("ERROR=position requires an active post>1");
  // Check allocation arithmetic without allocating. The production workspace
  // query separately enforces descriptor indexing bounds.
  input::bytes({unsigned(c.P),unsigned(c.S),unsigned(c.G),unsigned(c.D)},2);
  if(tokens)input::bytes({unsigned(tokens),unsigned(c.H),unsigned(c.D)},2);
}

static void compact(Case& c) {
  int64_t pages=0;c.M=1;
  for(int b=0;b<c.B;++b){const int64_t n=(int64_t(c.past[b])+c.q[b]+c.S-1)/c.S;if(n>=INT32_MAX)throw std::invalid_argument("page count overflow");c.M=std::max(c.M,int(n+1));pages+=n;}
  if(pages>INT32_MAX-2)throw std::invalid_argument("physical page count overflow");
  c.P=int(std::max<int64_t>(3,pages+2));
}
static std::vector<Case> build_cases(int coverage) {
  std::vector<Case> out;
  const int hpgs[]={1,2,3,4,5,6,8,16,9};const int dims[]={64,96,128,256};
  const int contexts[]={31,513,2049};
  for(int tier=0;tier<coverage;++tier)for(int d:dims)for(int hpg:hpgs){
    if(kDecode&&hpg==9&&d!=128)continue;
    Case c;c.D=d;c.G=1;c.H=hpg;c.S=tier==0 ? 17:tier==1 ? 31:96;
    const int base=contexts[tier];c.past={base-1,base,base+1};
    c.q=kDecode ? std::vector<int>{1,0,1} : std::vector<int>{tier==0 ? 3:17,0,tier==0 ? 17:33};
    c.rotary=(hpg+d/32)%2;c.window=d==256&&!kDecode ? -1 : (hpg%3==0 ? 1:hpg%3==1 ? c.S+3:-1);
    c.scale=hpg%2 ? 0:0.17f;c.map=hpg%3==0 ? "shared":hpg%3==1 ? "ordered":"permuted";
    compact(c);c.seed+=unsigned(out.size());out.push_back(c);
  }
  for(const std::string& map:{"ordered","permuted","shared"})for(int rotary:{0,1})for(int window:{-1,1,41}){
    Case c;c.D=64;c.H=4;c.G=1;c.past={34,51,68};c.q=kDecode ? std::vector<int>{1,0,1}:std::vector<int>{17,0,33};
    c.rotary=rotary;c.window=window;c.map=map;compact(c);c.seed+=unsigned(out.size());out.push_back(c);
  }
  for(float scale:{0.0f,0.125f}){Case c;c.D=64;c.H=4;c.G=1;c.scale=scale;c.seed=1119;c.window=0;compact(c);out.push_back(c);}
  {Case c;c.D=64;c.H=4;c.G=1;c.window=8192;compact(c);out.push_back(c);}
  for(int pattern=0;pattern<4;++pattern){Case c;c.D=64;c.H=1;c.G=1;c.B=pattern==0?1:3;
    c.q.assign(c.B,pattern==3?0:1);c.past.assign(c.B,pattern==1?0:pattern==2?17:18);
    compact(c);c.seed+=unsigned(out.size());out.push_back(c);}
  for(const auto& error:errors){if(error=="none"||kDecode&&error=="d256_window")continue;
    Case c;c.D=64;c.H=4;c.G=1;c.error=error;compact(c);c.seed+=unsigned(out.size());out.push_back(c);}
  return out;
}
static Case rank_case(const std::string& shape) {
  Case c;std::stringstream in(shape);std::string token;
  std::vector<std::string> values;while(std::getline(in,token,'_'))values.push_back(token);
  if(values.size()!=21||values[0]!=(kDecode?"decode":"prefill"))throw std::invalid_argument("UT rank: malformed Page phase/key");
  size_t at=1;
  auto value=[&](const char* name){const std::string prefix=name;if(values.at(at).rfind(prefix,0)!=0)throw std::invalid_argument("UT rank: Page key field missing");return values[at++].substr(prefix.size());};
  auto integer=[&](const char* name,int minimum=1){return int(input::integer(value(name),name,minimum));};
  c.B=integer("B");c.H=integer("H");c.G=integer("G");c.D=integer("D");c.P=integer("P");c.S=integer("S");c.M=integer("M");
  auto q=value("q"),past=value("past");std::replace(q.begin(),q.end(),'+',',');std::replace(past.begin(),past.end(),'+',',');c.q=array(q);c.past=array(past);
  c.rotary=integer("rope",0);c.rotary_dim=integer("rD",0);c.interleaved=integer("inter",0);c.max_position=integer("pos");c.scale=floating(value("scale"));c.window=integer("win",INT32_MIN);c.softcap=floating(value("cap"));c.map=value("map");c.shared_prefix=integer("shared",0);c.seed=unsigned(input::integer(value("seed"),"seed",0,UINT32_MAX));c.error=value("err");
  validate_case(c);if(c.error!="none"||describe_case(c)!=shape)throw std::invalid_argument("UT rank: Page key must be canonical normal case");return c;
}
static std::string config_string(GqaPagedConfig c){return "splits="+std::to_string(c.splits)+";bkv="+std::to_string(c.bkv)+";hb="+std::to_string(c.hb)+";qw="+std::to_string(c.qw)+";mt="+std::to_string(c.mt)+";nd="+std::to_string(c.nd);}
static bool public_entry(){const char* value=std::getenv("HIPDNN_PAGED_ATTENTION_PUBLIC");return value&&std::string(value)=="1";}
static int page_complete_entry(void* stream,const hip_paged_attention_f16_desc* d,void* workspace,size_t bytes,const GqaPagedConfig* config,size_t size) {
  if(!public_entry())return hip_paged_attention_f16_configured_v2(stream,d,workspace,bytes,config,size);
  return hip_paged_attention_f16(stream,d,workspace,bytes);
}
static GqaPagedConfig requested_config(const hip_paged_attention_f16_desc& d,const std::string& request) {
  if(request=="default")return gqa_paged_default(d);
  GqaPagedConfig c;
  if(request.rfind("id=",0)==0){const int id=timing::rankConfigInt(request,"id");
    if(id<0||id>=gqa_paged_candidate_count(d)||request!="id="+std::to_string(id))throw std::invalid_argument("UT rank: invalid Page candidate ID");
    c=gqa_paged_candidate(d,id);
  }else{
    const int splits=timing::rankConfigInt(request,"splits"),bkv=timing::rankConfigInt(request,"bkv");
    if(request=="splits="+std::to_string(splits)+";bkv="+std::to_string(bkv))c=gqa_paged_legacy(d,{splits,bkv});
    else {c={splits,bkv,timing::rankConfigInt(request,"hb"),timing::rankConfigInt(request,"qw"),timing::rankConfigInt(request,"mt"),timing::rankConfigInt(request,"nd")};
      if(config_string(c)!=request)throw std::invalid_argument("UT rank: config must be splits=N;bkv=N;hb=N;qw=N;mt=N;nd=N");}
  }
  if(!gqa_paged_config_valid(d,c))throw std::invalid_argument("UT rank: incompatible Page config");return c;
}
#ifdef HIPDNN_UT_RANK_PAGED_ADAPTER
static void print_catalog(const Case& c) {
  validate_case(c);hip_paged_attention_f16_desc d{};d.phase=kDecode?HIP_PAGED_ATTENTION_DECODE:HIP_PAGED_ATTENTION_PREFILL;d.head_dim=c.D;d.local_window_size=c.window;d.batch=c.B;d.num_heads=c.H;d.kv_num_heads=c.G;d.tokens=0;for(int q:c.q)d.tokens+=q;
  std::printf("UT_CANDIDATES {\"leaf\":%s,\"shape\":%s,\"default\":%s,\"configs\":[",timing::jsonString(kLeaf).c_str(),timing::jsonString(describe_case(c)).c_str(),timing::jsonString(config_string(gqa_paged_default(d))).c_str());
  for(int id=1;id<hipdnn_ut_paged_config_count(&d);++id){GqaPagedConfig v;if(hipdnn_ut_paged_config(&d,id,&v))throw std::runtime_error("Page catalog failed");std::printf("%s{\"request\":\"id=%d\",\"final\":%s}",id>1?",":"",id,timing::jsonString(config_string(v)).c_str());}
  std::printf("]}\n");
}
#endif
static void set_env(const char* name,const std::string& value){
#ifdef _WIN32
  if(_putenv_s(name,value.c_str()))throw std::runtime_error("environment update failed");
#else
  if(value.empty()?unsetenv(name):setenv(name,value.c_str(),1))throw std::runtime_error("environment update failed");
#endif
}
struct Host {
  hip_paged_attention_f16_desc desc{};
  std::vector<__half> q,k,v,kp,vp,co,si,expected_k,expected_v;
  std::vector<float> reference;
  std::vector<__half> history_k,history_v,q_rotated;
  std::vector<int32_t> cum,past,table;
  int host_status=0,device_status=0;
};
static size_t offset(const Case& c,const std::vector<int32_t>& table,int b,int p,int h,int d) {
  return ((size_t(table[size_t(b)*c.M+p/c.S])*c.S+p%c.S)*c.G+h)*c.D+d;
}
static __half rotate_component(float x,float y,float co,float si,bool second) {
  const float first=(second?y:x)*co;
  return __float2half(std::fma(second?x:-y,si,first));
}
static Host prepare(const Case& c) {
  validate_case(c);
  Host h;h.cum={0};h.past.assign(c.past.begin(),c.past.end());
  for(int q:c.q){if(q>INT32_MAX-h.cum.back())throw std::invalid_argument("T overflow");h.cum.push_back(h.cum.back()+q);}
  const int T=h.cum.back();
  h.desc={};auto& d=h.desc;d.batch=c.B;d.tokens=T;d.num_heads=c.H;d.kv_num_heads=c.G;d.head_dim=c.D;
  d.num_pages=c.P;d.page_size=c.S;d.table_columns=c.M;d.max_position=c.max_position;
  d.phase=kDecode?HIP_PAGED_ATTENTION_DECODE:HIP_PAGED_ATTENTION_PREFILL;
  d.do_rotary=c.rotary;d.rotary_dim=c.rotary_dim?c.rotary_dim:c.D;d.rotary_interleaved=c.interleaved;
  d.scale=c.scale;d.softcap=c.softcap;d.local_window_size=c.window;
  h.q.resize(T ? input::bytes({unsigned(T),unsigned(c.H),unsigned(c.D)},2)/2 : 0);h.k.resize(T ? input::bytes({unsigned(T),unsigned(c.G),unsigned(c.D)},2)/2 : 0);h.v.resize(h.k.size());
  h.kp.resize(input::bytes({unsigned(c.P),unsigned(c.S),unsigned(c.G),unsigned(c.D)},2)/2);
  h.vp.resize(h.kp.size());
  h.co.resize(input::bytes({unsigned(c.max_position),unsigned(c.D/2)},2)/2);h.si.resize(h.co.size());
  std::mt19937 rng(c.seed);std::uniform_real_distribution<float> random(-0.7f,0.7f);
  for(auto* v:{&h.q,&h.k,&h.v,&h.kp,&h.vp})for(auto& x:*v)x=__float2half(random(rng));
  for(int p=0;p<c.max_position;++p)for(int j=0;j<c.D/2;++j){const float angle=float(p)/std::pow(10000.0f,float(2*j)/c.D);
    h.co[size_t(p)*c.D/2+j]=__float2half(std::cos(angle));h.si[size_t(p)*c.D/2+j]=__float2half(std::sin(angle));}
  h.table.assign(size_t(c.B)*c.M,-999);
  std::vector<int> order(c.P);for(int i=0;i<c.P;++i)order[i]=i;
  if(c.map!="ordered")std::shuffle(order.begin(),order.end(),rng);
  const int prefix_pages=c.shared_prefix?c.shared_prefix/c.S:1;
  int next=c.map=="shared"?prefix_pages:0;
  for(int b=0;b<c.B;++b){const int64_t post=int64_t(c.past[b])+c.q[b];
    if(post>INT32_MAX || (post+c.S-1)/c.S>c.M)throw std::invalid_argument("case capacity insufficient; use --error table_small/post_overflow");
    if(c.map=="shared"&&c.q[b]&&c.past[b]<c.S)throw std::invalid_argument("shared map needs one readonly prefix page");
    const int pages=int((post+c.S-1)/c.S);
    for(int j=0;j<pages;++j){if(j<prefix_pages&&c.map=="shared")h.table[size_t(b)*c.M+j]=order[j];
      else{if(next>=c.P)throw std::invalid_argument("physical pool insufficient");h.table[size_t(b)*c.M+j]=order[next++];}}
  }
  // Canonical logical history is independent of physical pool traversal/map.
  // Shared pages receive identical prefix bytes in every logical request.
  std::mt19937 logical_rng(c.seed^0x517cc1b7u),prefix_rng(c.seed^0x9e3779b9u);
  const size_t prefix_tokens=c.map=="shared"?size_t(prefix_pages)*c.S:0;
  std::vector<__half> prefix_k(prefix_tokens*c.G*c.D),prefix_v(prefix_k.size());
  for(size_t i=0;i<prefix_k.size();++i){prefix_k[i]=__float2half(random(prefix_rng));prefix_v[i]=__float2half(random(prefix_rng));}
  for(int b=0;b<c.B;++b)for(int pos=0;pos<c.past[b];++pos)for(int head=0;head<c.G;++head)for(int d=0;d<c.D;++d){
    __half key=__float2half(random(logical_rng)),value=__float2half(random(logical_rng));
    if(size_t(pos)<prefix_tokens){const size_t index=(size_t(pos)*c.G+head)*c.D+d;key=prefix_k[index];value=prefix_v[index];}
    h.history_k.push_back(key);h.history_v.push_back(value);const size_t index=offset(c,h.table,b,pos,head,d);
    h.kp[index]=key;h.vp[index]=value;
  }
  h.expected_k=h.kp;h.expected_v=h.vp;
  // Independent logical reference: rotate to FP16, scatter, then FP32 softmax.
  std::vector<__half> qr=h.q;
  for(int b=0;b<c.B;++b)for(int j=0;j<c.q[b];++j){const int t=h.cum[b]+j,p=c.past[b]+j;
    if(c.rotary&&p>=c.max_position)throw std::invalid_argument("RoPE position outside table; use --error position");
    for(int head=0;head<c.H+c.G;++head)for(int channel=0;channel<c.D/2;++channel){
      const bool query=head<c.H;const int hh=query?head:head-c.H;
      auto& in=query?h.q:h.k;const size_t src=(size_t(t)*(query?c.H:c.G)+hh)*c.D;
      __half x=in[src+channel],y=in[src+channel+c.D/2];
      if(c.rotary){const size_t a=size_t(p)*c.D/2+channel;
        const float co=__half2float(h.co[a]),si=__half2float(h.si[a]);
        const float fx=__half2float(x),fy=__half2float(y);
        x=rotate_component(fx,fy,co,si,false);y=rotate_component(fx,fy,co,si,true);}
      auto& out=query?qr:h.expected_k;const size_t dst=query?src:offset(c,h.table,b,p,hh,0);
      out[dst+channel]=x;out[dst+channel+c.D/2]=y;
      if(!query){h.expected_v[dst+channel]=h.v[src+channel];h.expected_v[dst+channel+c.D/2]=h.v[src+channel+c.D/2];}
    }
  }
  h.q_rotated=qr;
  const float scale=c.scale==0?1.0f/std::sqrt(float(c.D)):c.scale;
  h.reference.resize(h.q.size());
  const size_t jobs=size_t(T)*c.H;const int threads=int(std::min<size_t>(8,std::max<size_t>(1,jobs)));std::vector<std::thread> workers;
  for(int worker=0;worker<threads;++worker)workers.emplace_back([&,worker]{
    for(size_t job=worker;job<jobs;job+=threads){const int t=int(job/c.H),head=int(job%c.H);int b=0;while(h.cum[b+1]<=t)++b;
      const int p=c.past[b]+t-h.cum[b],lo=c.window>0?std::max(0,p-c.window+1):0;
      const int kv=head/(c.H/c.G);std::vector<float> logits(p-lo+1);float maximum=-INFINITY;
      for(int k=lo;k<=p;++k){float dot=0;for(int j=0;j<c.D;++j)dot+=__half2float(qr[(size_t(t)*c.H+head)*c.D+j])*__half2float(h.expected_k[offset(c,h.table,b,k,kv,j)]);
        logits[k-lo]=dot*scale;maximum=std::max(maximum,logits[k-lo]);}
      float sum=0;for(auto& x:logits){x=std::exp(x-maximum);sum+=x;}
      for(int j=0;j<c.D;++j){float out=0;for(int k=lo;k<=p;++k)out+=logits[k-lo]*__half2float(h.expected_v[offset(c,h.table,b,k,kv,j)]);
        h.reference[(size_t(t)*c.H+head)*c.D+j]=out/sum;}
    }
  });for(auto& t:workers)t.join();
  // Invalid values are introduced AFTER the independent valid input/reference.
  const auto& e=c.error;
  int active=0;while(active<c.B&&!c.q[active])++active;
  if(e=="cum_start"){h.cum[0]=1;h.device_status=HIP_PAGED_ATTENTION_BAD_CUM;}
  else if(e=="cum_end"){h.cum.back()++;h.device_status=HIP_PAGED_ATTENTION_BAD_CUM;}
  else if(e=="cum_order"){h.cum[1]=T+1;h.device_status=HIP_PAGED_ATTENTION_BAD_CUM;}
  else if(e=="negative_past"){h.past[0]=-1;h.device_status=HIP_PAGED_ATTENTION_BAD_PAST;}
  else if(e=="decode_length"){d.phase=HIP_PAGED_ATTENTION_DECODE;for(int b=1;b<=c.B;++b)h.cum[b]=T;h.device_status=HIP_PAGED_ATTENTION_BAD_DECODE_LENGTH;}
  else if(e=="post_overflow"){h.past[active]=INT32_MAX;h.device_status=HIP_PAGED_ATTENTION_POST_OVERFLOW;}
  else if(e=="table_small"){h.past[active]=int32_t(int64_t(c.M)*c.S+1-c.q[active]);h.device_status=HIP_PAGED_ATTENTION_TABLE_TOO_SMALL;}
  else if(e=="bad_page"){h.table[size_t(active)*c.M]=c.P;h.device_status=HIP_PAGED_ATTENTION_BAD_PAGE;}
  else if(e=="position"){d.do_rotary=1;d.rotary_dim=c.D;d.rotary_interleaved=0;d.max_position=1;h.device_status=HIP_PAGED_ATTENTION_BAD_POSITION;}
  else if(e=="write_conflict"){int source=active,target=active+1;while(!c.q[target])++target;
    if(h.past[source]>h.past[target])std::swap(source,target);
    h.past[target]=h.past[source];
    h.table[size_t(target)*c.M+h.past[target]/c.S]=h.table[size_t(source)*c.M+h.past[source]/c.S];h.device_status=HIP_PAGED_ATTENTION_WRITE_CONFLICT;}
  else if(e=="softcap"){d.softcap=1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="negative_scale"){d.scale=-1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="nan_scale"){d.scale=NAN;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="inf_scale"){d.scale=INFINITY;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="nan_softcap"){d.softcap=NAN;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="interleaved"){d.do_rotary=1;d.rotary_interleaved=1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="partial_rotary"){d.do_rotary=1;d.rotary_dim=c.D/2;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="d256_window"){d.head_dim=256;d.local_window_size=1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="geometry"){d.num_heads=0;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="null_rotary"){d.do_rotary=1;d.rotary_dim=c.D;d.rotary_interleaved=0;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="size_overflow"){d.num_heads=d.kv_num_heads=d.num_pages=d.page_size=INT32_MAX;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="tensor_alignment")h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  else if(e=="workspace")h.host_status=HIP_PAGED_ATTENTION_WORKSPACE_TOO_SMALL;
  else if(e=="workspace_alignment"||e=="alias"||e=="pool_alias"||e=="null_rotary")h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  return h;
}
struct Buffer {
  unsigned char* allocation=nullptr;size_t size,padding;
  Buffer(size_t n,size_t align=16):size(n),padding(align){if(size>SIZE_MAX-padding*2)throw std::invalid_argument("buffer size overflow");check(hipMalloc(&allocation,size+padding*2));check(hipMemset(allocation,0xa5,size+padding*2));}
  ~Buffer(){if(allocation&&hipFree(allocation)!=hipSuccess)std::terminate();}
  void* ptr()const{return allocation+padding;}
  void upload(const void* p){if(size)check(hipMemcpy(ptr(),p,size,hipMemcpyHostToDevice));}
  std::vector<unsigned char> download()const{std::vector<unsigned char> out(size+padding*2);check(hipMemcpy(out.data(),allocation,out.size(),hipMemcpyDeviceToHost));return out;}
};
struct Bundle {
  Buffer q,k,v,kp,vp,co,si,cum,past,table,o,workspace;
  hip_paged_attention_f16_desc desc;
  Bundle(const Host& h,size_t bytes):q(h.q.size()*2),k(h.k.size()*2),v(h.v.size()*2),kp(h.kp.size()*2),vp(h.vp.size()*2),
      co(h.co.size()*2),si(h.si.size()*2),cum(h.cum.size()*4),past(h.past.size()*4),table(h.table.size()*4),o(h.q.size()*2),workspace(bytes,256),desc(h.desc){
    q.upload(h.q.data());k.upload(h.k.data());v.upload(h.v.data());kp.upload(h.kp.data());vp.upload(h.vp.data());co.upload(h.co.data());si.upload(h.si.data());
    cum.upload(h.cum.data());past.upload(h.past.data());table.upload(h.table.data());
    desc.q=q.ptr();desc.k_new=k.ptr();desc.v_new=v.ptr();desc.k_pool=kp.ptr();desc.v_pool=vp.ptr();desc.o=o.ptr();
    desc.cos_cache=co.ptr();desc.sin_cache=si.ptr();desc.cum=static_cast<int32_t*>(cum.ptr());desc.past_seqlens=static_cast<int32_t*>(past.ptr());desc.block_table=static_cast<int32_t*>(table.ptr());
  }
};
static bool matches(const std::vector<unsigned char>& raw,const void* expected,size_t n,size_t padding=16){
  if(raw.size()!=n+2*padding)return false;
  for(size_t i=0;i<padding;++i)if(raw[i]!=0xa5||raw[padding+n+i]!=0xa5)return false;
  return !n||!std::memcmp(raw.data()+padding,expected,n);
}
static bool guards(const std::vector<unsigned char>& raw,size_t padding){
  for(size_t i=0;i<padding;++i)if(raw[i]!=0xa5||raw[raw.size()-1-i]!=0xa5)return false;return true;
}
__global__ void hold_stream(const int* flag){volatile const int* p=flag;while(*p==0){}}
struct Copies { int n;size_t bytes_per_copy;int count()const{return n;}size_t bytes()const{return bytes_per_copy;} };
struct CaseResult { bool passed=false;double rel_l2=0,time_ms=0;std::string config; };
static CaseResult run_case(const Case& c,const timing::RankRow* rank_row=nullptr) {
  const std::string shape=describe_case(c);
  Host h=prepare(c);CaseResult result;result.config=kDecode?"fixed:gqa-scalar splits=16":c.D==256?"fixed:gqa-v8x BKV=16 ND=2 HB=1":"fixed:gqa-v6x MT=1 BKV=32 HB=1 QW=2";
  GqaPagedConfig selected=gqa_paged_default(h.desc);std::string source="fixed:default";
  const char* pin=std::getenv("HIPDNN_PAGED_ATTENTION_CFG");
  if(public_entry()&&pin&&*pin)throw std::invalid_argument("public entry conflicts with config pin");
  if(pin&&pin[0]){if(rank_row)throw std::invalid_argument("UT rank: conflicting Page config pin");selected=requested_config(h.desc,pin);source="fixed:pin";}
  bool online=false;
  if(rank_row){
    if(std::find(rank_row->configs.begin(),rank_row->configs.end(),"default")==rank_row->configs.end())throw std::invalid_argument("UT rank: candidate CSV must include default");
    for(const auto& request:rank_row->configs)requested_config(h.desc,request);
  }
  // Query a valid allocation layout even for deliberately invalid host options.
  auto allocation_desc=h.desc;allocation_desc.num_heads=c.H;allocation_desc.head_dim=c.D;
  allocation_desc.batch=c.B;allocation_desc.kv_num_heads=c.G;allocation_desc.num_pages=c.P;allocation_desc.page_size=c.S;allocation_desc.table_columns=c.M;
  allocation_desc.scale=c.scale;allocation_desc.softcap=c.softcap;allocation_desc.do_rotary=c.rotary;
  allocation_desc.rotary_interleaved=c.interleaved;allocation_desc.rotary_dim=c.rotary_dim?c.rotary_dim:c.D;
  allocation_desc.local_window_size=c.window;allocation_desc.phase=h.desc.phase;
  size_t bytes=0;const int query=hip_paged_attention_f16_workspace_size(&allocation_desc,&bytes);
  if(query){throw std::invalid_argument("custom options unsupported; select explicit --error for rejection tests");}
  if(h.desc.do_rotary&&!c.rotary){allocation_desc.do_rotary=1;allocation_desc.rotary_interleaved=0;allocation_desc.rotary_dim=c.D;allocation_desc.max_position=c.max_position;
    if(hip_paged_attention_f16_workspace_size(&allocation_desc,&bytes))throw std::runtime_error("workspace query failed");}
  int rc=0,device=0;std::vector<unsigned char> output,kpool,vpool,workspace;bool immutable=true;
  {
    hipdnn_ep_test::GpuExecutionScope admission([]{check(hipDeviceSynchronize());});
    {
      if(public_entry()){selected=gqa_paged_default(h.desc);source="public:default";}
      else if(c.error=="none"&&!rank_row&&!(pin&&pin[0])) {
        const char* mode=std::getenv("HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE");
        if(mode&&std::string(mode)=="online"){online=true;source="autotune:production-prepare";}
        else {auto resolution=hipdnn_ep::resolvePagedAttention(h.desc,shape);selected=resolution.config;source=resolution.source;online=source.rfind("fallback:",0)==0;}
      }
      Bundle b(h,bytes);void* ws=b.workspace.ptr();size_t capacity=bytes;
      if(c.error=="tensor_alignment")b.desc.q=static_cast<const unsigned char*>(b.desc.q)+2;
      if(c.error=="workspace")capacity=bytes?bytes-1:0;
      if(c.error=="workspace_alignment")ws=static_cast<unsigned char*>(ws)+1;
      if(c.error=="alias")b.desc.o=b.desc.q==nullptr?nullptr:const_cast<void*>(b.desc.q);
      if(c.error=="pool_alias")b.desc.v_pool=b.desc.k_pool;
      if(c.error=="null_rotary"){b.desc.do_rotary=1;b.desc.cos_cache=nullptr;}
      rc=page_complete_entry(nullptr,&b.desc,ws,capacity,&selected,sizeof(selected));check(hipDeviceSynchronize());
      if(public_entry()&&!rc){GqaPagedConfig actual{};if(hip_paged_attention_f16_last_config_v2(&actual,sizeof(actual))||config_string(actual)!=config_string(selected))throw std::runtime_error("public Page actual geometry differs");}
      if(!rc&&h.desc.tokens)check(hipMemcpy(&device,b.workspace.ptr(),4,hipMemcpyDeviceToHost));
      output=b.o.download();kpool=b.kp.download();vpool=b.vp.download();workspace=b.workspace.download();
      immutable=matches(b.q.download(),h.q.data(),h.q.size()*2)&&matches(b.k.download(),h.k.data(),h.k.size()*2)&&matches(b.v.download(),h.v.data(),h.v.size()*2)&&
          matches(b.cum.download(),h.cum.data(),h.cum.size()*4)&&matches(b.past.download(),h.past.data(),h.past.size()*4)&&matches(b.table.download(),h.table.data(),h.table.size()*4)&&
          matches(b.co.download(),h.co.data(),h.co.size()*2)&&matches(b.si.download(),h.si.data(),h.si.size()*2);
      check(hipDeviceSynchronize());
    }
  }
  const bool rejected=h.host_status||h.device_status;
  bool valid=rc==h.host_status&&(!rc?device==h.device_status:true)&&immutable&&guards(workspace,256);
  valid=valid&&matches(kpool,(rejected?h.kp:h.expected_k).data(),h.kp.size()*2)&&matches(vpool,(rejected?h.vp:h.expected_v).data(),h.vp.size()*2)&&guards(output,16);
  if(rejected||!h.desc.tokens){for(size_t i=16;i+16<output.size();++i)valid=valid&&output[i]==0xa5;
    if(!h.desc.tokens)for(auto x:workspace)valid=valid&&x==0xa5;
    result.config=rejected?"declined":"fixed:no-op";result.passed=valid;
    std::printf("%s %s host=%d device=%d expected=%d/%d immutable=%d\n",valid?"PASS":"FAIL",shape.c_str(),rc,device,h.host_status,h.device_status,int(immutable));return result;}
  double squared=0,reference=0;size_t worst=0;double worst_error=0;
  for(size_t i=0;i<h.reference.size();++i){__half x;std::memcpy(&x,output.data()+16+i*2,2);const float f=__half2float(x);const double e=f-h.reference[i];
    if(!std::isfinite(f))valid=false;squared+=e*e;reference+=double(h.reference[i])*h.reference[i];if(std::abs(e)>worst_error){worst_error=std::abs(e);worst=i;}}
  result.rel_l2=std::sqrt(squared/std::max(reference,1e-30));valid=valid&&result.rel_l2<(kDecode?2e-2:2e-3);
  if(!valid){
    auto mismatch=[&](const char* name,const std::vector<unsigned char>& raw,const std::vector<__half>& want){
      for(size_t i=0;i<want.size();++i){uint16_t got,expected;std::memcpy(&got,raw.data()+16+2*i,2);std::memcpy(&expected,want.data()+i,2);
        if(got!=expected){std::printf("STORAGE_MISMATCH %s index=%zu got=0x%04x expected=0x%04x\n",name,i,unsigned(got),unsigned(expected));
          for(int b=0;b<c.B;++b)for(int j=0;j<c.q[b];++j)for(int head=0;head<c.G;++head){const int pos=c.past[b]+j;const size_t start=offset(c,h.table,b,pos,head,0);
            if(i>=start&&i<start+size_t(c.D)){const int channel=int(i-start),paired=channel%c.D%(c.D/2);const size_t src=(size_t(h.cum[b]+j)*c.G+head)*c.D;const size_t angle=size_t(pos)*(c.D/2)+paired;
              std::printf("ROTARY_INPUT b=%d j=%d h=%d channel=%d x=%.9g y=%.9g cos=%.9g sin=%.9g\n",b,j,head,channel,__half2float(h.k[src+paired]),__half2float(h.k[src+paired+c.D/2]),__half2float(h.co[angle]),__half2float(h.si[angle]));}}
          break;}}
    };
    mismatch("K_pool",kpool,h.expected_k);mismatch("V_pool",vpool,h.expected_v);
    std::printf("GUARDS output=%d workspace=%d K=%d V=%d\n",int(guards(output,16)),int(guards(workspace,256)),int(guards(kpool,16)),int(guards(vpool,16)));
    std::printf("FAIL %s host=%d device=%d relL2=%.9g worst=%zu absolute=%.9g immutable=%d\n",shape.c_str(),rc,device,result.rel_l2,worst,worst_error,int(immutable));return result;}
  result.config=source+":"+config_string(selected);
  // Rotate complete coherent bundles. Each timed call appends the SAME new K/V
  // at the SAME past positions; no length/cache accumulation occurs on reuse.
  // Count read slots rather than unused physical pool capacity as working bytes.
  std::set<size_t> read_slots;
  for(int b=0;b<c.B;++b)if(c.q[b]){const int hi=c.past[b]+c.q[b],lo=c.window>0?std::max(0,c.past[b]-c.window+1):0;
    for(int pos=lo;pos<hi;++pos)read_slots.insert(offset(c,h.table,b,pos,0,0)/(size_t(c.G)*c.D));}
  const size_t working=read_slots.size()*c.G*c.D*4;
  const bool warm=std::getenv("HIPDNN_KERNEL_UT_WARM")&&std::getenv("HIPDNN_KERNEL_UT_WARM")[0]=='1';
  const size_t bundle_bytes=2*h.kp.size()*2+64;
  const int desired=warm?1:int(std::min<size_t>(4096,1+((size_t(256)<<20)+working-1)/working));
  const int copies=std::max(1,std::min(desired,int((size_t(512)<<20)/std::max<size_t>(1,bundle_bytes))));
  if(copies<desired)std::printf("UT_ROTATION_LIMIT requested=%d actual=%d working_bytes=%zu: bounded allocation, cold residency not certified\n",desired,copies,working);
  {
    hipdnn_ep_test::GpuExecutionScope admission([]{check(hipDeviceSynchronize());});
    {
      Bundle base(h,bytes);
      struct Cache {
        Buffer k,v;
        explicit Cache(const Host& h):k(h.kp.size()*2),v(h.vp.size()*2){k.upload(h.kp.data());v.upload(h.vp.data());}
      };
      std::vector<std::unique_ptr<Cache>> caches;
      for(int i=1;i<copies;++i)caches.emplace_back(new Cache(h));
      int* flag=nullptr;check(hipHostMalloc(&flag,sizeof(int),hipHostMallocMapped|hipHostMallocCoherent));
      try {
        Copies rotation{copies,working};
        auto launch=[&](int i){auto desc=base.desc;if(i){desc.k_pool=caches[size_t(i-1)]->k.ptr();desc.v_pool=caches[size_t(i-1)]->v.ptr();}return page_complete_entry(nullptr,&desc,base.workspace.ptr(),bytes,&selected,sizeof(selected));};
        auto validate=[&](){
          if(launch(0))throw std::runtime_error("Page candidate host rejection");check(hipDeviceSynchronize());
          auto raw=base.o.download();auto kp=base.kp.download(),vp=base.vp.download();int status=0;check(hipMemcpy(&status,base.workspace.ptr(),4,hipMemcpyDeviceToHost));
          if(status||!guards(raw,16)||!guards(base.workspace.download(),256)||!matches(base.co.download(),h.co.data(),h.co.size()*2)||!matches(base.si.download(),h.si.data(),h.si.size()*2)||!matches(kp,h.expected_k.data(),h.expected_k.size()*2)||!matches(vp,h.expected_v.data(),h.expected_v.size()*2)||
             !matches(base.q.download(),h.q.data(),h.q.size()*2)||!matches(base.k.download(),h.k.data(),h.k.size()*2)||!matches(base.v.download(),h.v.data(),h.v.size()*2)||
             !matches(base.cum.download(),h.cum.data(),h.cum.size()*4)||!matches(base.past.download(),h.past.data(),h.past.size()*4)||!matches(base.table.download(),h.table.data(),h.table.size()*4))throw std::runtime_error("Page candidate storage/metadata check failed");
          double sum=0,norm=0;for(size_t i=0;i<h.reference.size();++i){__half x;std::memcpy(&x,raw.data()+16+2*i,2);double e=__half2float(x)-h.reference[i];sum+=e*e;norm+=double(h.reference[i])*h.reference[i];}
          double error=std::sqrt(sum/std::max(norm,1e-30));if(!(error<(kDecode?2e-2:2e-3)))throw std::runtime_error("Page candidate CPU reference FAIL");return error;
        };
        auto measure=[&](const timing::RankOptions* options=nullptr){return timing::adaptiveRotatedMeanMs(nullptr,rotation,launch,
          flag,[&](const int* p){hold_stream<<<1,1>>>(p);},warm,size_t(128)<<20,"paged_attention_f16",(public_entry()?"hip_paged_attention_f16":"hip_paged_attention_f16_configured_v2"),timing::NoPreparation{},options);};
        if(rank_row){
#ifndef HIPDNN_UT_RANK_PAGED_ADAPTER
          throw std::invalid_argument("UT rank: Page requires RANK=1 build");
#else
          timing::RankOptions options;std::vector<GqaPagedConfig> configs{gqa_paged_default(h.desc)};std::set<std::string> seen{config_string(configs[0])};bool have_default=false;
          for(const auto& request:rank_row->configs){if(request=="default"){have_default=true;continue;}auto candidate=requested_config(h.desc,request);if(seen.insert(config_string(candidate)).second)configs.push_back(candidate);}
          if(!have_default)throw std::invalid_argument("UT rank: candidate CSV must include default");
          for(auto cfg:configs)options.configs.push_back(config_string(cfg));options.priority=rank_row->priority;options.sequential_confirm=std::getenv("HIPDNN_UT_RANK_REFERENCE")&&std::string(std::getenv("HIPDNN_UT_RANK_REFERENCE"))=="1";
          options.select=[&](size_t i){selected=configs.at(i);};options.validate=[&](size_t i){options.select(i);return validate();};
          measure(&options);result.config="rank:UT_RANK";result.time_ms=0;
#endif
        }else{
          if(online){
            hipdnn_ep::PagedAttentionResult prepared;
            const int rc=hipdnn_ep::preparePagedAttention(nullptr,base.desc,base.workspace.ptr(),bytes,shape,source=="autotune:production-prepare",&prepared);
            if(rc)throw std::runtime_error("production Page preparation failed");
            selected=prepared.config;source=prepared.source;result.rel_l2=validate();
          }
          result.config=source+":"+config_string(selected);std::printf("PAGED_CONFIG shape=%s config=%s boundary=complete\n",shape.c_str(),result.config.c_str());
          result.time_ms=measure();
        }
        int state=0;check(hipMemcpy(&state,base.workspace.ptr(),4,hipMemcpyDeviceToHost));if(state)throw std::runtime_error("timed entry metadata failure");
        check(hipDeviceSynchronize());check(hipHostFree(flag));flag=nullptr;
      }catch(...){if(flag){check(hipDeviceSynchronize());check(hipHostFree(flag));}throw;}
    }
  }
  result.passed=true;std::printf("PASS %s relL2=%.9g ms=%.9g config=%s\n",shape.c_str(),result.rel_l2,result.time_ms,result.config.c_str());return result;
}
static const char* env(const char* key,const char* fallback){const char* p=std::getenv(key);return p&&p[0]?p:fallback;}
static void csv(const Case& c,const CaseResult& r){
  const auto shape=describe_case(c);timing::emitCsvBinding(shape,r.config,r.passed?"PASS":"FAIL",r.rel_l2,r.time_ms);
  const char* path=std::getenv("HIPDNN_RESULTS_CSV");if(!path||!path[0])return;
  bool header=true;if(FILE* f=std::fopen(path,"rb")){std::fseek(f,0,SEEK_END);header=std::ftell(f)==0;std::fclose(f);}
  FILE* f=std::fopen(path,"a");if(!f)throw std::runtime_error("CSV open failed");
  if(header)std::fprintf(f,"op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
  std::fprintf(f,"%s,%s,%s,%s,%s,%s,%.9g,%.9g,%s\n",
      env("HIPDNN_RESULTS_OP",kDecode?"paged_attention/decode":"paged_attention/prefill"),env("HIPDNN_RESULTS_LEAF","Xfp16_Wfp16_Yfp16"),
      env("HIPDNN_RESULTS_ARCH","unknown"),env("HIPDNN_RESULTS_MODE","lookup"),shape.c_str(),r.config.c_str(),r.time_ms,r.rel_l2,r.passed?"PASS":"FAIL");
  if(std::fclose(f))throw std::runtime_error("CSV close failed");
}
#ifdef HIPDNN_UT_COMPARE_ORIGINAL_GQA
#include "../../../common/ut_original_gqa_compare.h"
struct Accuracy { double rel=0,absolute=0;size_t nonfinite=0; };
static Accuracy accuracy(const std::vector<float>& got,const std::vector<float>& reference){
  if(got.size()!=reference.size())throw std::runtime_error("compare output shape mismatch");
  Accuracy out;double squared=0,norm=0;for(size_t i=0;i<got.size();++i){double e=double(got[i])-reference[i];squared+=e*e;norm+=double(reference[i])*reference[i];out.absolute=std::max(out.absolute,std::abs(e));out.nonfinite+=!std::isfinite(got[i]);}out.rel=std::sqrt(squared/std::max(norm,1e-30));return out;
}
static std::vector<float> output_floats(const Buffer& b){auto raw=b.download();if(!guards(raw,16))throw std::runtime_error("compare output guards changed");std::vector<float> out(b.size/2);for(size_t i=0;i<out.size();++i){__half x;std::memcpy(&x,raw.data()+16+2*i,2);out[i]=__half2float(x);}return out;}
static std::string logical_hash(const Case& c,const Host& h){
  uint64_t value=14695981039346656037ull;auto update=[&](const void* data,size_t n){auto p=static_cast<const unsigned char*>(data);while(n--){value^=*p++;value*=1099511628211ull;}};
  for(const auto* v:{&h.q,&h.k,&h.v,&h.history_k,&h.history_v})update(v->data(),v->size()*2);
  if(c.rotary){update(h.co.data(),h.co.size()*2);update(h.si.data(),h.si.size()*2);}
  update(h.cum.data(),h.cum.size()*4);update(h.past.data(),h.past.size()*4);
  char s[32];std::snprintf(s,sizeof(s),"%016llx",(unsigned long long)value);return s;
}
static void run_compare(const Case& c){
  if(c.error!="none")throw std::invalid_argument("compare requires normal case");
  Host h=prepare(c);std::vector<int> active;for(int b=0;b<c.B;++b)if(c.q[b])active.push_back(b);
  if(active.empty())throw std::invalid_argument("compare requires at least one active request");
  const int A=int(active.size()),T=h.desc.tokens;
  int maximum=0;for(int b:active)maximum=std::max(maximum,c.past[b]+c.q[b]);
  const char* capacity_request=std::getenv("HIPDNN_UT_DENSE_CAPACITY");
  const int capacity=capacity_request&&*capacity_request?int(input::integer(capacity_request,"dense_capacity")):(maximum+31)/32*32+32;
  if(capacity<maximum)throw std::invalid_argument("dense capacity must hold the effective context");
  const bool batched=kDecode||std::all_of(active.begin(),active.end(),[&](int b){return c.q[b]==c.q[active[0]]&&c.past[b]==c.past[active[0]];});
  std::vector<int32_t> seq(A),post(c.B);for(int i=0;i<A;++i)seq[i]=c.past[active[i]]+c.q[active[i]]-1;for(int b=0;b<c.B;++b)post[b]=c.past[b]+c.q[b];
  const size_t dense_elems=size_t(A)*c.G*capacity*c.D;
  std::vector<__half> dk(dense_elems,__float2half(0)),dv=dk,common_k=dk,common_v=dv;
  for(int a=0;a<A;++a){const int b=active[a];for(int head=0;head<c.G;++head)for(int pos=0;pos<post[b];++pos)for(int d=0;d<c.D;++d){
    const size_t dst=((size_t(a)*c.G+head)*capacity+pos)*c.D+d,src=offset(c,h.table,b,pos,head,d);
    if(pos<c.past[b]){dk[dst]=h.kp[src];dv[dst]=h.vp[src];}
    common_k[dst]=h.expected_k[src];common_v[dst]=h.expected_v[src];
  }}
  const auto shape=describe_case(c);GqaPagedConfig config=gqa_paged_default(h.desc);std::string config_source="fixed:default";
  const char* pin=std::getenv("HIPDNN_PAGED_ATTENTION_CFG");
  if(public_entry()&&pin&&*pin)throw std::invalid_argument("public entry conflicts with config pin");
  if(pin&&*pin){config=requested_config(h.desc,pin);config_source="fixed:pin";}
  size_t workspace=0;if(hip_paged_attention_f16_workspace_size(&h.desc,&workspace))throw std::runtime_error("compare workspace query failed");
  std::vector<float> page_output,dense_output,page_core_output,dense_common_output;
  double page_ms=0,dense_ms=0,bare_ms=0,page_core_ms=0,dense_common_ms=0;
  size_t rope_k_diff=0,rope_q_diff=0;std::string original_config;
  size_t logical_working=0;for(int b:active){const int lo=c.window>0?std::max(0,c.past[b]-c.window+1):0;logical_working+=size_t(post[b]-lo)*c.G*c.D*4;}
  const bool warm=std::getenv("HIPDNN_KERNEL_UT_WARM")&&std::getenv("HIPDNN_KERNEL_UT_WARM")[0]=='1';
  const size_t max_bundle=std::max(h.kp.size()*4,dense_elems*4);
  const int desired=warm?1:int(std::min<size_t>(4096,1+((size_t(256)<<20)+logical_working-1)/logical_working));
  const int copies=std::max(1,std::min(desired,int((size_t(512)<<20)/std::max<size_t>(1,max_bundle))));
  std::printf("COMPARE_INPUT {\"shape\":%s,\"logical_hash_algorithm\":\"fnv1a64-fp16-and-metadata\",\"logical_hash\":%s,\"dense_route\":%s,\"logical_B\":%d,\"active_B\":%d,\"active_ids\":%s,\"dense_capacity\":%d,\"page_pool_bytes\":%zu,\"dense_pool_bytes\":%zu,\"rotation_copies\":%d,\"rotation_requested\":%d,\"rotation_working_bytes\":%zu,\"shared_prefix_tokens\":%d}\n",timing::jsonString(shape).c_str(),timing::jsonString(logical_hash(c,h)).c_str(),timing::jsonString(kDecode?(A==c.B?"dense_batched":"dense_active_compacted_batched"):batched?"dense_batched":"dense_per_request_loop").c_str(),c.B,A,timing::jsonString(list(active)).c_str(),capacity,h.kp.size()*4,dense_elems*4,copies,desired,logical_working,c.shared_prefix);
  {
    hipdnn_ep_test::GpuExecutionScope admission([]{check(hipDeviceSynchronize());});
    {
      if(public_entry()){config=gqa_paged_default(h.desc);config_source="public:default";}
      else if(!(pin&&*pin)){auto resolution=hipdnn_ep::resolvePagedAttention(h.desc,shape);config=resolution.config;config_source=resolution.source;
        // Comparison's no-table baseline is explicit, rather than an online
        // retune that would obscure the default configuration comparison.
        if(config_source.rfind("lookup:",0)!=0)config_source="fixed:no-table-default("+config_source+")";
      }
      Bundle page(h,workspace);
      const char* preparation=std::getenv("HIPDNN_PAGED_ATTENTION_PRODUCTION_PREPARE");
      if(preparation&&std::string(preparation)=="1"){
        if(public_entry()||(pin&&*pin))throw std::invalid_argument("production preparation conflicts with public/pinned route");
        hipdnn_ep::PagedAttentionResult prepared;
        const char* mode=std::getenv("HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE");
        const int rc=hipdnn_ep::preparePagedAttention(nullptr,page.desc,page.workspace.ptr(),workspace,shape,mode&&std::string(mode)=="online",&prepared);
        if(rc)throw std::runtime_error("comparison production Page preparation failed");
        config=prepared.config;config_source=prepared.source;
      }
      Buffer q(h.q.size()*2),k(h.k.size()*2),v(h.v.size()*2),qr(h.q.size()*2),kr(h.k.size()*2),out(h.q.size()*2),keys(dense_elems*2),values(dense_elems*2),lengths(seq.size()*4),partial(size_t(A)*c.H*64*(c.D+2)*4,256);
      q.upload(h.q.data());k.upload(h.k.data());v.upload(h.v.data());keys.upload(dk.data());values.upload(dv.data());lengths.upload(seq.data());
      Buffer common_q(h.q.size()*2),posts(post.size()*4),status(4),page_part(size_t(c.B)*c.H*64*(c.D+2)*4,256);
      int zero=0;common_q.upload(h.q_rotated.data());posts.upload(post.data());status.upload(&zero);
      const float scale=c.scale==0?1.0f/std::sqrt(float(c.D)):c.scale;
      const char* dense_pin=std::getenv("HIPDNN_UT_FLASH_CFG");
      const bool force_dense=dense_pin&&*dense_pin;
      const auto dense_geometry=force_dense?requested_config(h.desc,dense_pin):GqaPagedConfig{};
      auto dense_flash=[&](const void* Q,const void* K,const void* V)->int{
        if(force_dense){
          if(kDecode)return hipdnn_ut_original_decode_configured(Q,K,V,out.ptr(),partial.ptr(),A,c.H,c.G,c.D,maximum,capacity,scale,lengths.ptr(),c.window,&dense_geometry);
          if(batched)return hipdnn_ut_original_prefill_configured(Q,K,V,out.ptr(),A,c.H,c.G,c.q[active[0]],maximum,c.D,capacity,c.past[active[0]],scale,c.window,&dense_geometry);
          for(int a=0;a<A;++a){const int b=active[a];const size_t query_offset=size_t(h.cum[b])*c.H*c.D*2,bank_offset=size_t(a)*c.G*capacity*c.D*2;
            const int rc=hipdnn_ut_original_prefill_configured(static_cast<const unsigned char*>(Q)+query_offset,static_cast<const unsigned char*>(K)+bank_offset,static_cast<const unsigned char*>(V)+bank_offset,static_cast<unsigned char*>(out.ptr())+query_offset,1,c.H,c.G,c.q[b],post[b],c.D,capacity,c.past[b],scale,c.window,&dense_geometry);if(rc)return rc;}
          return 0;
        }
        if(kDecode)return hipdnn_original_hip_gqa_flash_decode(nullptr,Q,K,V,out.ptr(),partial.ptr(),A,c.H,c.G,c.D,maximum,capacity,64,scale,lengths.ptr(),c.window,nullptr,0,HIP_KV_DTYPE_FP16,nullptr,nullptr,nullptr,0);
        if(batched)return hipdnn_original_hip_gqa_flash_prefill(nullptr,Q,K,V,out.ptr(),A,c.H,c.G,c.q[active[0]],maximum,c.D,capacity,c.past[active[0]],scale,c.window,nullptr,c.H,0,nullptr,0);
        for(int a=0;a<A;++a){const int b=active[a];const size_t query_offset=size_t(h.cum[b])*c.H*c.D*2,bank_offset=size_t(a)*c.G*capacity*c.D*2;
          int rc=hipdnn_original_hip_gqa_flash_prefill(nullptr,static_cast<const unsigned char*>(Q)+query_offset,static_cast<const unsigned char*>(K)+bank_offset,static_cast<const unsigned char*>(V)+bank_offset,static_cast<unsigned char*>(out.ptr())+query_offset,1,c.H,c.G,c.q[b],post[b],c.D,capacity,c.past[b],scale,c.window,nullptr,c.H,0,nullptr,0);if(rc)return rc;}
        return 0;
      };
      auto dense_complete=[&](void* K,void* V)->int{
        const int loops=batched?1:A;
        for(int a=0;a<loops;++a){const int b=active[a],B=batched?A:1,sq=c.q[b];const size_t qoff=batched?0:size_t(h.cum[b])*c.H*c.D*2,koff=batched?0:size_t(h.cum[b])*c.G*c.D*2,bankoff=batched?0:size_t(a)*c.G*capacity*c.D*2;
          auto input_q=static_cast<const unsigned char*>(q.ptr())+qoff,input_k=static_cast<const unsigned char*>(k.ptr())+koff,input_v=static_cast<const unsigned char*>(v.ptr())+koff;
          auto output_q=static_cast<unsigned char*>(qr.ptr())+qoff,output_k=static_cast<unsigned char*>(kr.ptr())+koff;
          auto sl=static_cast<const int32_t*>(lengths.ptr())+(batched?0:a);
          if(c.rotary){int rc=hipdnn_original_hip_gqa_rope(nullptr,input_q,output_q,page.co.ptr(),page.si.ptr(),B,sq,c.H,c.D,c.D/2,c.past[b],sl,2);if(rc)return rc;rc=hipdnn_original_hip_gqa_rope(nullptr,input_k,output_k,page.co.ptr(),page.si.ptr(),B,sq,c.G,c.D,c.D/2,c.past[b],sl,2);if(rc)return rc;}
          int rc=hipdnn_original_hip_gqa_kv_cache_append(nullptr,c.rotary?output_k:input_k,static_cast<unsigned char*>(K)+bankoff,B,sq,c.G,c.D,capacity,c.past[b],sl,2,HIP_KV_DTYPE_FP16,nullptr);if(rc)return rc;
          rc=hipdnn_original_hip_gqa_kv_cache_append(nullptr,input_v,static_cast<unsigned char*>(V)+bankoff,B,sq,c.G,c.D,capacity,c.past[b],sl,2,HIP_KV_DTYPE_FP16,nullptr);if(rc)return rc;
        }
        return dense_flash(c.rotary?qr.ptr():q.ptr(),K,V);
      };
      if(page_complete_entry(nullptr,&page.desc,page.workspace.ptr(),workspace,&config,sizeof(config)))throw std::runtime_error("compare Page launch failed");
      if(public_entry()){GqaPagedConfig actual{};if(hip_paged_attention_f16_last_config_v2(&actual,sizeof(actual))||config_string(actual)!=config_string(config))throw std::runtime_error("public Page actual geometry differs");}check(hipDeviceSynchronize());int device_status=0;check(hipMemcpy(&device_status,page.workspace.ptr(),4,hipMemcpyDeviceToHost));
      page_output=output_floats(page.o);
      if(device_status||!matches(page.kp.download(),h.expected_k.data(),h.expected_k.size()*2)||!matches(page.vp.download(),h.expected_v.data(),h.expected_v.size()*2))throw std::runtime_error("compare Page status/cache FAIL");
      if(dense_complete(keys.ptr(),values.ptr()))throw std::runtime_error("original Flash rejects shape");check(hipDeviceSynchronize());dense_output=output_floats(out);original_config=hipdnn_ut_original_config();
      if(original_config.empty())throw std::runtime_error("original Flash configuration unavailable");
      auto original_k=keys.download();auto original_v=values.download();auto original_q=c.rotary?qr.download():q.download();auto original_new_k=c.rotary?kr.download():k.download();
      auto expected_original_k=dk,expected_original_v=dv;
      for(int a=0;a<A;++a){int b=active[a];for(int j=0;j<c.q[b];++j)for(int head=0;head<c.G;++head)for(int d=0;d<c.D;++d){size_t src=(size_t(h.cum[b]+j)*c.G+head)*c.D+d,dst=((size_t(a)*c.G+head)*capacity+c.past[b]+j)*c.D+d;std::memcpy(&expected_original_k[dst],original_new_k.data()+16+2*src,2);expected_original_v[dst]=h.v[src];}}
      if(!matches(original_k,expected_original_k.data(),dense_elems*2)||!matches(original_v,expected_original_v.data(),dense_elems*2)||!matches(q.download(),h.q.data(),h.q.size()*2)||!matches(k.download(),h.k.data(),h.k.size()*2)||!matches(v.download(),h.v.data(),h.v.size()*2)||!matches(lengths.download(),seq.data(),seq.size()*4))throw std::runtime_error("original append/history/input guards FAIL");
      for(size_t i=0;i<common_k.size();++i)rope_k_diff+=std::memcmp(&common_k[i],&expected_original_k[i],2)!=0;
      for(size_t i=0;i<h.q_rotated.size();++i)rope_q_diff+=std::memcmp(&h.q_rotated[i],original_q.data()+16+2*i,2)!=0;
      auto pa=accuracy(page_output,h.reference),da=accuracy(dense_output,h.reference);
      if(pa.nonfinite||da.nonfinite||!(pa.rel<(kDecode?2e-2:2e-3))||!(da.rel<(kDecode?2e-2:2e-3)))throw std::runtime_error("compare CPU precision FAIL");
      if(!matches(page.q.download(),h.q.data(),h.q.size()*2)||!matches(page.k.download(),h.k.data(),h.k.size()*2)||!matches(page.v.download(),h.v.data(),h.v.size()*2)||!matches(page.cum.download(),h.cum.data(),h.cum.size()*4)||!matches(page.past.download(),h.past.data(),h.past.size()*4)||!matches(page.table.download(),h.table.data(),h.table.size()*4))throw std::runtime_error("compare Page readonly guards FAIL");
      struct CachePair{Buffer pk,pv,dk,dv;CachePair(const Host& h,const std::vector<__half>& k,const std::vector<__half>& v):pk(h.kp.size()*2),pv(h.vp.size()*2),dk(k.size()*2),dv(v.size()*2){pk.upload(h.expected_k.data());pv.upload(h.expected_v.data());dk.upload(k.data());dv.upload(v.data());}};
      std::vector<std::unique_ptr<CachePair>> cache;for(int i=1;i<copies;++i)cache.emplace_back(new CachePair(h,expected_original_k,expected_original_v));
      int* flag=nullptr;check(hipHostMalloc(&flag,4,hipHostMallocMapped|hipHostMallocCoherent));
      try{
        Copies rotation{copies,logical_working};
        auto measure=[&](const char* name,const char* entry,const std::string& identity,double rel,const std::function<int(int)>& launch){
          const std::string entry_shape=shape+"_entry"+name;timing::CaseScope scope(kLeaf,entry_shape,entry);
          double ms=timing::adaptiveRotatedMeanMs(nullptr,rotation,launch,flag,[&](const int* p){hold_stream<<<1,1>>>(p);},warm,size_t(128)<<20,"paged_flash_compare",entry);
          timing::emitCsvBinding(entry_shape,identity,"PASS",rel,ms);
          if(const char* path=std::getenv("HIPDNN_RESULTS_CSV")){bool header=true;if(FILE* f=std::fopen(path,"rb")){std::fseek(f,0,SEEK_END);header=std::ftell(f)==0;std::fclose(f);}FILE* f=std::fopen(path,"a");if(!f)throw std::runtime_error("compare CSV open failed");if(header)std::fprintf(f,"op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");std::fprintf(f,"paged_flash_compare,%s,%s,%s,%s,\"%s\",%.17g,%.17g,PASS\n",kLeaf,std::getenv("HIPDNN_RESULTS_ARCH")?std::getenv("HIPDNN_RESULTS_ARCH"):"unknown",std::getenv("HIPDNN_RESULTS_MODE")?std::getenv("HIPDNN_RESULTS_MODE"):"lookup",entry_shape.c_str(),identity.c_str(),ms,rel);std::fclose(f);}
          return ms;
        };
        auto page_launch=[&](int i){auto d=page.desc;if(i){d.k_pool=cache[i-1]->pk.ptr();d.v_pool=cache[i-1]->pv.ptr();}return page_complete_entry(nullptr,&d,page.workspace.ptr(),workspace,&config,sizeof(config));};
        auto dense_launch=[&](int i){return dense_complete(i?cache[i-1]->dk.ptr():keys.ptr(),i?cache[i-1]->dv.ptr():values.ptr());};
        // Alternate side order by the logical query/history dimensions.
        const bool dense_first=c.compare_order!=0;
        if(dense_first){dense_ms=measure("dense_complete","original_RoPE_append_Flash",original_config,da.rel,dense_launch);page_ms=measure("page_complete",(public_entry()?"hip_paged_attention_f16":"hip_paged_attention_f16_configured_v2"),config_source+":"+config_string(config),pa.rel,page_launch);}
        else{page_ms=measure("page_complete",(public_entry()?"hip_paged_attention_f16":"hip_paged_attention_f16_configured_v2"),config_source+":"+config_string(config),pa.rel,page_launch);dense_ms=measure("dense_complete","original_RoPE_append_Flash",original_config,da.rel,dense_launch);}
        bare_ms=measure("dense_bare","original_Flash_attention",original_config,da.rel,[&](int i){return dense_flash(c.rotary?qr.ptr():q.ptr(),i?cache[i-1]->dk.ptr():keys.ptr(),i?cache[i-1]->dv.ptr():values.ptr());});
        keys.upload(common_k.data());values.upload(common_v.data());for(auto& x:cache){x->dk.upload(common_k.data());x->dv.upload(common_v.data());}
        auto page_core=[&](int i){auto d=page.desc;if(i){d.k_pool=cache[i-1]->pk.ptr();d.v_pool=cache[i-1]->pv.ptr();}GqaPagedArgs pg{d.block_table,d.cum,d.past_seqlens,static_cast<const int32_t*>(posts.ptr()),static_cast<const int32_t*>(status.ptr()),c.S,c.M,c.G,gqa_paged_page_shift(c.S)};return gqa_paged_attention_f16_launch(nullptr,d,common_q.ptr(),static_cast<float*>(page_part.ptr()),pg,config);};
        if(page_core(0))throw std::runtime_error("Page core launch failed");check(hipDeviceSynchronize());page_core_output=output_floats(page.o);auto pca=accuracy(page_core_output,h.reference);
        if(dense_flash(common_q.ptr(),keys.ptr(),values.ptr()))throw std::runtime_error("dense common core failed");check(hipDeviceSynchronize());dense_common_output=output_floats(out);auto dca=accuracy(dense_common_output,h.reference);
        if(pca.nonfinite||dca.nonfinite||!(pca.rel<(kDecode?2e-2:2e-3))||!(dca.rel<(kDecode?2e-2:2e-3)))throw std::runtime_error("common post-RoPE core precision FAIL");
        page_core_ms=measure("page_core","gqa_paged_attention_f16_launch",config_string(config),pca.rel,page_core);
        dense_common_ms=measure("dense_common_core","original_Flash_common_FP16",original_config,dca.rel,[&](int i){return dense_flash(common_q.ptr(),i?cache[i-1]->dk.ptr():keys.ptr(),i?cache[i-1]->dv.ptr():values.ptr());});
        check(hipDeviceSynchronize());check(hipHostFree(flag));flag=nullptr;
      }catch(...){if(flag){check(hipDeviceSynchronize());check(hipHostFree(flag));}throw;}
    }
  }
  auto pa=accuracy(page_output,h.reference),da=accuracy(dense_output,h.reference),pd=accuracy(page_output,dense_output),pca=accuracy(page_core_output,h.reference),dca=accuracy(dense_common_output,h.reference),pcdc=accuracy(page_core_output,dense_common_output);
  std::printf("COMPARE_RESULT {\"shape\":%s,\"page_config\":%s,\"dense_config\":%s,\"page_complete_us\":%.17g,\"dense_complete_us\":%.17g,\"dense_bare_us\":%.17g,\"page_core_us\":%.17g,\"dense_common_core_us\":%.17g,\"ratio\":%.17g,\"percent\":%.17g,\"delta_us\":%.17g,\"page_cpu_relL2\":%.17g,\"page_cpu_max_abs\":%.17g,\"dense_cpu_relL2\":%.17g,\"dense_cpu_max_abs\":%.17g,\"page_dense_relL2\":%.17g,\"page_dense_max_abs\":%.17g,\"precision_increment\":%.17g,\"page_core_cpu_relL2\":%.17g,\"dense_common_core_cpu_relL2\":%.17g,\"core_page_dense_relL2\":%.17g,\"core_page_dense_max_abs\":%.17g,\"rope_K_storage_differences\":%zu,\"rope_Q_storage_differences\":%zu,\"nonfinite\":0,\"storage_guards\":true,\"baseline\":\"a762-original-public-online-no-LUT\",\"verdict\":\"PASS\"}\n",timing::jsonString(shape).c_str(),timing::jsonString(config_source+":"+config_string(config)).c_str(),timing::jsonString(original_config).c_str(),page_ms*1000,dense_ms*1000,bare_ms*1000,page_core_ms*1000,dense_common_ms*1000,page_ms/dense_ms,(page_ms/dense_ms-1)*100,(page_ms-dense_ms)*1000,pa.rel,pa.absolute,da.rel,da.absolute,pd.rel,pd.absolute,pa.rel-da.rel,pca.rel,dca.rel,pcdc.rel,pcdc.absolute,rope_k_diff,rope_q_diff);
}
#endif

int main(int argc,char** argv){
  try {bool single=false;int coverage=3;Case c=parse_single(argc,argv,single,coverage);if(single)validate_case(c);
    if(c.compare){
#ifdef HIPDNN_UT_COMPARE_ORIGINAL_GQA
      if(!single)throw std::invalid_argument("compare requires custom case");run_compare(c);return 0;
#else
      throw std::invalid_argument("compare requires COMPARE=1 FLASH_SOURCE=<original-gqa.hip> build");
#endif
    }
    const std::vector<Case> cases=single?std::vector<Case>{c}:build_cases(coverage);
    if(timing::rankRequested()) {
#ifndef HIPDNN_UT_RANK_PAGED_ADAPTER
      throw std::invalid_argument("UT rank: Page requires RANK=1 build");
#else
      const auto rows=timing::readRankRows(kLeaf);int failed=0;
      for(const auto& row:rows){const auto test=rank_case(row.shape);timing::CaseScope scope(kLeaf,row.shape,"hip_paged_attention_f16_configured_v2");
        if(std::getenv("HIPDNN_UT_RANK_CATALOG")){print_catalog(test);continue;}
        auto result=run_case(test,&row);failed+=!result.passed;
      }
      std::printf("%s rank (%d failures)\n",failed?"SOME FAIL":"ALL PASS",failed);return failed?1:0;
#endif
    }
    std::printf("coverage=%d -> running %zu paged_attention cases; phase=%s; real Page lookup/online; shared ut-adaptive-v3\n",coverage,cases.size(),kDecode?"decode":"prefill");
    int failed=0;for(const auto& test:cases){timing::CaseScope scope(kLeaf,describe_case(test),"hip_paged_attention_f16");
      try{auto result=run_case(test);csv(test,result);failed+=!result.passed;}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s: %s\n",describe_case(test).c_str(),e.what());CaseResult failure;failure.config="declined";csv(test,failure);++failed;}}
    std::printf("%s (%d failing case(s))\n",failed?"SOME FAIL":"ALL PASS",failed);return failed?1:0;
  }catch(const std::exception& e){std::fprintf(stderr,"FAIL arguments: %s\n",e.what());return 2;}
}
