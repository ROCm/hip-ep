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
static constexpr bool kDecode = 0;
static const char* kLeaf = 0 ? "paged_attention/decode/Xfp32_Wfp16_Yfp16_mrope" : "paged_attention/prefill/Xfp32_Wfp16_Yfp16_mrope";
static void check(hipError_t rc) { if(rc!=hipSuccess) throw std::runtime_error(hipGetErrorString(rc)); }
struct Case {
  int B=3,H=8,G=2,D=128,P=64,S=17,M=8,rotary=0,max_position=4096,window=-1,rotary_dim=64,interleaved=0;
  std::vector<int> q = kDecode ? std::vector<int>{1,0,1} : std::vector<int>{3,0,17};
  std::vector<int> past{16,17,18};
  float scale=0,softcap=0,mrope_scale=1;
  int mrope_layout=1; std::vector<int> sections{11,11,10}; std::string positions="distinct";
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
      "_win"+std::to_string(c.window)+"_cap"+number(c.softcap)+"_map"+c.map+"_shared"+std::to_string(c.shared_prefix)+"_seed"+std::to_string(c.seed)+"_err"+c.error+"_mr"+std::to_string(c.mrope_layout)+"_sec"+list(c.sections)+"_rs"+number(c.mrope_scale)+"_ids"+c.positions;
}
static std::vector<int> array(const std::string& s) {
  std::vector<int> v;std::stringstream in(s);std::string item;
  while(std::getline(in,item,','))v.push_back(int(input::integer(item,"array",0)));
  if(v.empty() || s.back()==',')throw std::invalid_argument("empty array item");return v;
}
static float floating(const std::string& s) { size_t n;float x=std::stof(s,&n);if(n!=s.size())throw std::invalid_argument("invalid float");return x; }
static const std::vector<std::string> errors={"none","cum_start","cum_end","cum_order","negative_past","decode_length","post_overflow","table_small","bad_page","workspace","workspace_alignment","alias","pool_alias","softcap","negative_scale","nan_scale","inf_scale","nan_softcap","d256_window","geometry","write_conflict","size_overflow","tensor_alignment","output_alignment","pool_alignment","mrope_layout","mrope_sections","mrope_dim","mrope_scale","mrope_null"};
static Case parse_single(int argc,char** argv,bool& single,int& coverage,bool& list_cases) {
  Case c;coverage=3;
  if(const char* e=std::getenv("HIPDNN_UT_COVERAGE"))coverage=int(input::integer(e,"coverage",1,3));
  for(int i=1;i<argc;++i){const std::string f=argv[i];const std::string s=input::value(argc,argv,i);
    if(f=="--coverage"){coverage=int(input::integer(s,"coverage",1,3));continue;}
    if(f=="--list-cases"){list_cases=input::boolean(s,"list-cases");continue;}
    single=true;
    if(f=="--mrope-layout")c.mrope_layout=int(input::integer(s,"mrope_layout",0,1));
    else if(f=="--mrope-sections")c.sections=array(s);
    else if(f=="--mrope-scale")c.mrope_scale=floating(s);
    else if(f=="--position-mode")c.positions=s;
    else if(f=="--b")c.B=int(input::integer(s,"B"));
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
  if(c.rotary || c.interleaved)throw std::invalid_argument("fused MRoPE uses explicit positions and non-interleaved rotation");
  if(c.rotary_dim<=0 || c.rotary_dim>c.D || c.rotary_dim%2 || c.sections.size()!=3 ||
     c.sections[0]<0 || c.sections[1]<0 || c.sections[2]<0 || c.sections[0]+c.sections[1]+c.sections[2]!=c.rotary_dim/2 ||
     c.mrope_layout<0 || c.mrope_layout>1 || !std::isfinite(c.mrope_scale))throw std::invalid_argument("invalid MRoPE attributes");
  if(c.positions!="distinct"&&c.positions!="same"&&c.positions!="out-of-cache")throw std::invalid_argument("unknown position mode");
  if (!kDecode && c.D==256 && c.window>0)
    throw std::invalid_argument("prefill base D256+window unsupported; use ERROR=d256_window");
  const int hpg=c.H/c.G;
  const bool decode_geometry=hpg==1||hpg==2||hpg==3||hpg==4||hpg==5||hpg==6||hpg==8||hpg==16||(hpg==9&&c.D==128);
  if ((kDecode||c.error=="decode_length")&&!decode_geometry)
    throw std::invalid_argument("base geometry must support decode for this case");
  int64_t tokens=0,pages=0;int active=0;
  for(int b=0;b<c.B;++b) {
    if(c.q[b]<0||c.past[b]<0||(kDecode&&c.q[b]>1))throw std::invalid_argument("invalid base q/past; decode q must be 0/1");
    tokens+=c.q[b];const int64_t post=int64_t(c.past[b])+c.q[b];
    if(tokens>INT32_MAX||post>INT32_MAX||(post+c.S-1)/c.S>c.M)throw std::invalid_argument("base T/post/table capacity insufficient");
    pages+=(post+c.S-1)/c.S;
    if(c.q[b]){++active;
      if(c.map=="shared"&&c.past[b]<c.S)throw std::invalid_argument("shared map needs one readonly prefix page");}
  }
  const int prefix_pages=c.shared_prefix?c.shared_prefix/c.S:1;
  if(c.shared_prefix && (c.map!="shared"||c.shared_prefix%c.S))throw std::invalid_argument("shared-prefix must be page-aligned and MAP=shared");
  if(c.map=="shared") {for(int b=0;b<c.B;++b){const int n=int((int64_t(c.past[b])+c.q[b]+c.S-1)/c.S);pages-=std::min(n,prefix_pages);if(c.q[b]&&c.past[b]<prefix_pages*c.S)throw std::invalid_argument("shared-prefix must be readonly for active requests");}pages+=prefix_pages;}
  if(pages>c.P)throw std::invalid_argument("base physical pool insufficient");
  if(c.map!="ordered"&&c.map!="permuted"&&c.map!="shared")throw std::invalid_argument("unknown page map");
  if(std::find(errors.begin(),errors.end(),c.error)==errors.end())throw std::invalid_argument("unknown error case");
  if(kDecode&&c.error=="d256_window")throw std::invalid_argument("ERROR=d256_window is prefill-only; decode D256+window is legal");
  const std::set<std::string> needs_tokens={"cum_start","cum_end","cum_order","negative_past","decode_length","post_overflow","table_small","bad_page","write_conflict","workspace","workspace_alignment","alias","pool_alias","tensor_alignment","mrope_null","output_alignment","pool_alignment"};
  if(!tokens&&needs_tokens.count(c.error))throw std::invalid_argument("ERROR="+c.error+" requires T>0; all-empty entry is a no-op");
  if(c.error=="write_conflict"&&active<2)throw std::invalid_argument("ERROR=write_conflict requires at least two active requests (q>0)");
  if(c.error=="decode_length"&&tokens<2)throw std::invalid_argument("ERROR=decode_length requires T>=2 and decode geometry");
  if(c.error=="table_small"&&int64_t(c.M)*c.S>=INT32_MAX)throw std::invalid_argument("ERROR=table_small requires M*S<INT32_MAX to avoid post overflow");
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
  // Every tier retains both layouts, full/partial rotation, three position
  // patterns, physical maps, head dimensions, inactive/ragged requests.
  for(int d:{64,96,128,256})for(int layout:{0,1})for(int full:{0,1})for(int window:{-1,0,1,20,153}){
    if(!kDecode&&d==256&&window>0)continue;
    Case c;c.D=d;c.H=6;c.G=1;c.mrope_layout=layout;c.window=window;c.past={34,51,68};c.seed=0x11250000u;c.rotary_dim=full?d:32;
    const int n=c.rotary_dim/2;c.sections={n/3,n/3,n-2*(n/3)};
    c.positions=full?"distinct":"out-of-cache";c.mrope_scale=full?1:0.5f;
    c.map=layout?"permuted":"ordered";compact(c);out.push_back(c);
  }
  // Initial requests retain both stream layouts and rotary widths too.
  for(int layout:{0,1})for(int full:{0,1})for(int window:{-1,0,1,20,153}){
    Case c;c.D=128;c.H=6;c.G=1;c.mrope_layout=layout;c.rotary_dim=full?128:32;
    const int n=c.rotary_dim/2;c.sections={n/3,n/3,n-2*(n/3)};
    c.positions=full?"distinct":"out-of-cache";c.mrope_scale=full?1:0.5f;
    c.window=window;c.past={0,0,0};c.map=layout?"permuted":"ordered";
    c.q=kDecode?std::vector<int>{1,0,1}:std::vector<int>{3,0,33};
    c.seed=0x11250001u;compact(c);out.push_back(c);
  }
  for(const std::string& map:{"ordered","permuted","shared"})for(const std::string& ids:{"same","distinct","out-of-cache"}){
    Case c;c.D=64;c.H=4;c.G=1;c.past={34,51,68};c.map=map;c.positions=ids;c.window=19;
    compact(c);out.push_back(c);
  }
  for(int empty:{0,1}){Case c;c.q.assign(c.B,empty?0:1);c.past.assign(c.B,0);compact(c);out.push_back(c);}
  for(int tier=1;tier<coverage;++tier)for(int page:{16,31,32,128,256}){
    Case c;c.H=24;c.G=4;c.D=256;c.S=page;c.past={tier*257,0,tier*513};
    c.q=kDecode?std::vector<int>{1,0,1}:std::vector<int>{tier*33,0,tier*65};compact(c);out.push_back(c);
  }
  for(const auto& error:errors){if(error=="none"||(kDecode&&error=="d256_window"))continue;
    Case c;c.D=64;c.H=4;c.G=1;c.error=error;compact(c);out.push_back(c);}
  return out;
}
static Case rank_case(const std::string& shape) {
  Case c;std::stringstream in(shape);std::string token;
  std::vector<std::string> values;while(std::getline(in,token,'_'))values.push_back(token);
  if(values.size()!=25||values[0]!=(kDecode?"decode":"prefill"))throw std::invalid_argument("UT rank: malformed Page phase/key");
  size_t at=1;
  auto value=[&](const char* name){const std::string prefix=name;if(values.at(at).rfind(prefix,0)!=0)throw std::invalid_argument("UT rank: Page key field missing");return values[at++].substr(prefix.size());};
  auto integer=[&](const char* name,int minimum=1){return int(input::integer(value(name),name,minimum));};
  c.B=integer("B");c.H=integer("H");c.G=integer("G");c.D=integer("D");c.P=integer("P");c.S=integer("S");c.M=integer("M");
  auto q=value("q"),past=value("past");std::replace(q.begin(),q.end(),'+',',');std::replace(past.begin(),past.end(),'+',',');c.q=array(q);c.past=array(past);
  c.rotary=integer("rope",0);c.rotary_dim=integer("rD",0);c.interleaved=integer("inter",0);c.max_position=integer("pos");c.scale=floating(value("scale"));c.window=integer("win",INT32_MIN);c.softcap=floating(value("cap"));c.map=value("map");c.shared_prefix=integer("shared",0);c.seed=unsigned(input::integer(value("seed"),"seed",0,UINT32_MAX));c.error=value("err");c.mrope_layout=integer("mr",0);auto sec=value("sec");std::replace(sec.begin(),sec.end(),'+',',');c.sections=array(sec);c.mrope_scale=floating(value("rs"));c.positions=value("ids");
  validate_case(c);if(c.error!="none"||describe_case(c)!=shape)throw std::invalid_argument("UT rank: Page key must be canonical normal case");return c;
}
static std::string config_string(GqaPagedConfig c){return "splits="+std::to_string(c.splits)+";bkv="+std::to_string(c.bkv)+";hb="+std::to_string(c.hb)+";qw="+std::to_string(c.qw)+";mt="+std::to_string(c.mt)+";nd="+std::to_string(c.nd);}
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
struct Host {
  hip_paged_attention_f16_desc desc{};
  std::vector<float> q,k,co,si;
  std::vector<int64_t> positions;
  hip_paged_attention_mrope_f16_desc mrope{};
  std::vector<__half> v,kp,vp,expected_k,expected_v;
  std::vector<float> reference;
  std::vector<__half> history_k,history_v,q_rotated;
  std::vector<int32_t> cum,past,table;
  int host_status=0,device_status=0;
};
static size_t offset(const Case& c,const std::vector<int32_t>& table,int b,int p,int h,int d) {
  return ((size_t(table[size_t(b)*c.M+p/c.S])*c.S+p%c.S)*c.G+h)*c.D+d;
}
static __half rotate_component(float x,float y,float co,float si,bool second) {
  const float product=second?y*co:y*si;
  return __float2half(second?std::fma(x,si,product):std::fma(x,co,-product));
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
  h.co.resize(size_t(c.max_position)*(c.rotary_dim/2));h.si.resize(h.co.size());
  std::mt19937 rng(c.seed);std::uniform_real_distribution<float> random(-0.7f,0.7f);
  for(auto* v:{&h.q,&h.k})for(auto& x:*v)x=random(rng);
  for(auto* v:{&h.v,&h.kp,&h.vp})for(auto& x:*v)x=__float2half(random(rng));
  for(int p=0;p<c.max_position;++p)for(int j=0;j<c.rotary_dim/2;++j){
    const float angle=float(p)/std::pow(10000.0f,float(2*j)/c.rotary_dim);
    h.co[size_t(p)*c.rotary_dim/2+j]=std::cos(angle);h.si[size_t(p)*c.rotary_dim/2+j]=std::sin(angle);}
  h.positions.resize(size_t(3)*T);
  for(int axis=0;axis<3;++axis)for(int t=0;t<T;++t){int64_t p=(t*7+(c.positions=="same"?0:axis*13)+5)%c.max_position;
    if(c.positions=="out-of-cache"&&(t+axis)%3!=0)p=(t+axis)%2?-1:c.max_position+3;
    h.positions[size_t(axis)*T+t]=p;}
  h.mrope.rotary_dim=c.rotary_dim;h.mrope.max_position=c.max_position;h.mrope.section_t=c.sections[0];
  h.mrope.section_h=c.sections[1];h.mrope.section_w=c.sections[2];h.mrope.mrope_layout=c.mrope_layout;h.mrope.rotary_scale=c.mrope_scale;
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
  // Independent frequency-to-axis table: contiguous sections for layout 0;
  // scatter H/W slots for layout 1, all remaining columns belong to T.
  const int half=c.rotary_dim/2;std::vector<int> axes(half,0);
  if(c.mrope_layout==0){for(int i=c.sections[0];i<c.sections[0]+c.sections[1];++i)axes[i]=1;
    for(int i=c.sections[0]+c.sections[1];i<half;++i)axes[i]=2;}
  else {for(int i=0;i<c.sections[1]&&3*i+1<half;++i)axes[3*i+1]=1;
    for(int i=0;i<c.sections[2]&&3*i+2<half;++i)axes[3*i+2]=2;}
  auto rotate=[&](const std::vector<float>& in,int heads){std::vector<__half> out(in.size());
    for(size_t i=0;i<in.size();++i)out[i]=__float2half(in[i]);
    for(int t=0;t<T;++t)for(int head=0;head<heads;++head)for(int j=0;j<half;++j){
      const auto p=h.positions[size_t(axes[j])*T+t];if(p<0||p>=c.max_position)continue;
      const size_t start=(size_t(t)*heads+head)*c.D;const float x=in[start+j],y=in[start+j+half];
      const float co=h.co[size_t(p)*half+j]*c.mrope_scale,si=h.si[size_t(p)*half+j]*c.mrope_scale;
      out[start+j]=rotate_component(x,y,co,si,false);out[start+j+half]=rotate_component(x,y,co,si,true);
    }return out;};
  auto qr=rotate(h.q,c.H),kr=rotate(h.k,c.G);
  for(int b=0;b<c.B;++b)for(int j=0;j<c.q[b];++j)for(int head=0;head<c.G;++head)for(int channel=0;channel<c.D;++channel){
    const size_t src=(size_t(h.cum[b]+j)*c.G+head)*c.D+channel,dst=offset(c,h.table,b,c.past[b]+j,head,channel);
    h.expected_k[dst]=kr[src];h.expected_v[dst]=h.v[src];}
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
  else if(e=="write_conflict"){int source=active,target=active+1;while(!c.q[target])++target;
    if(h.past[source]>h.past[target])std::swap(source,target);
    h.past[target]=h.past[source];
    h.table[size_t(target)*c.M+h.past[target]/c.S]=h.table[size_t(source)*c.M+h.past[source]/c.S];h.device_status=HIP_PAGED_ATTENTION_WRITE_CONFLICT;}
  else if(e=="softcap"){d.softcap=1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="negative_scale"){d.scale=-1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="nan_scale"){d.scale=NAN;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="inf_scale"){d.scale=INFINITY;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="nan_softcap"){d.softcap=NAN;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="d256_window"){d.head_dim=256;d.local_window_size=1;h.host_status=HIP_PAGED_ATTENTION_UNSUPPORTED;}
  else if(e=="geometry"){d.num_heads=0;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="size_overflow"){d.num_heads=d.kv_num_heads=d.num_pages=d.page_size=INT32_MAX;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  else if(e=="tensor_alignment")h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  else if(e=="workspace")h.host_status=HIP_PAGED_ATTENTION_WORKSPACE_TOO_SMALL;
  else if(e=="workspace_alignment"||e=="alias"||e=="pool_alias")h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if(e=="mrope_layout"){h.mrope.mrope_layout=2;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  if(e=="mrope_sections"){h.mrope.section_h++;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  if(e=="mrope_dim"){h.mrope.rotary_dim=c.D+2;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  if(e=="mrope_scale"){h.mrope.rotary_scale=NAN;h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;}
  if(e=="output_alignment"||e=="pool_alignment")h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if(e=="mrope_null")h.host_status=HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  h.mrope.page=d;
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
  Buffer q,k,v,kp,vp,co,si,positions,cum,past,table,o,workspace,qr,kr,q16,k16;
  hip_paged_attention_mrope_f16_desc mrope;
  hip_paged_attention_f16_desc desc;
  Bundle(const Host& h,size_t bytes):q(h.q.size()*4),k(h.k.size()*4),v(h.v.size()*2),kp(h.kp.size()*2),vp(h.vp.size()*2),
      co(h.co.size()*4),si(h.si.size()*4),positions(h.positions.size()*8),cum(h.cum.size()*4),past(h.past.size()*4),table(h.table.size()*4),o(h.q.size()*2),workspace(bytes,256),qr(h.q.size()*4),kr(h.k.size()*4),q16(h.q.size()*2),k16(h.k.size()*2),mrope(h.mrope),desc(h.desc){
    mrope.position_ids=reinterpret_cast<const int64_t*>(1);
    q.upload(h.q.data());k.upload(h.k.data());v.upload(h.v.data());kp.upload(h.kp.data());vp.upload(h.vp.data());co.upload(h.co.data());si.upload(h.si.data());
    positions.upload(h.positions.data());cum.upload(h.cum.data());past.upload(h.past.data());table.upload(h.table.data());
    desc.q=q.ptr();desc.k_new=k.ptr();desc.v_new=v.ptr();desc.k_pool=kp.ptr();desc.v_pool=vp.ptr();desc.o=o.ptr();
    desc.cos_cache=co.ptr();desc.sin_cache=si.ptr();desc.cum=static_cast<int32_t*>(cum.ptr());desc.past_seqlens=static_cast<int32_t*>(past.ptr());desc.block_table=static_cast<int32_t*>(table.ptr());
  }
  hip_paged_attention_mrope_f16_desc native(){auto d=mrope;d.page=desc;d.q=static_cast<const float*>(q.ptr());d.k_new=static_cast<const float*>(k.ptr());d.cos_cache=static_cast<const float*>(co.ptr());d.sin_cache=static_cast<const float*>(si.ptr());d.position_ids=static_cast<const int64_t*>(positions.ptr());return d;}
  int fused(const GqaPagedConfig& config,void* ws,size_t bytes){auto d=native();if(!mrope.position_ids)d.position_ids=nullptr;if(mrope.q)d.q=reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(d.q)+2);return hip_paged_attention_mrope_f16_configured_v1(nullptr,&d,ws,bytes,&config,sizeof(config));}
  int unfused(const GqaPagedConfig& config,size_t bytes){auto d=native();const auto& p=d.page;
    int rc=hip_mrope_forward(nullptr,d.q,d.position_ids,d.cos_cache,d.sin_cache,qr.ptr(),1,p.tokens,p.num_heads,p.head_dim,d.rotary_dim,d.max_position,d.section_t,d.section_h,d.section_w,d.mrope_layout,d.rotary_scale,4);if(rc)return rc;
    rc=hip_mrope_forward(nullptr,d.k_new,d.position_ids,d.cos_cache,d.sin_cache,kr.ptr(),1,p.tokens,p.kv_num_heads,p.head_dim,d.rotary_dim,d.max_position,d.section_t,d.section_h,d.section_w,d.mrope_layout,d.rotary_scale,4);if(rc)return rc;
    rc=hip_cast(nullptr,qr.ptr(),q16.ptr(),int64_t(p.tokens)*p.num_heads*p.head_dim,HIP_DTYPE_FLOAT32,HIP_DTYPE_FLOAT16);if(rc)return rc;
    rc=hip_cast(nullptr,kr.ptr(),k16.ptr(),int64_t(p.tokens)*p.kv_num_heads*p.head_dim,HIP_DTYPE_FLOAT32,HIP_DTYPE_FLOAT16);if(rc)return rc;
    auto old=p;old.q=q16.ptr();old.k_new=k16.ptr();old.do_rotary=0;
    return hip_paged_attention_f16_configured_v2(nullptr,&old,workspace.ptr(),bytes,&config,sizeof(config));
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
struct CaseResult { bool passed=false;double rel_l2=0,time_ms=0,unfused_ms=0;std::string config,unfused_config; };
static CaseResult run_case(const Case& c,const timing::RankRow* rank_row=nullptr) {
  const std::string shape=describe_case(c);
  Host h=prepare(c);CaseResult result;result.config="pending";
  GqaPagedConfig selected=gqa_paged_default(h.desc);std::string source="fixed:default";
  const char* pin=std::getenv("HIPDNN_PAGED_ATTENTION_CFG");
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
  auto allocation_mrope=h.mrope;allocation_mrope.page=allocation_desc;
  allocation_mrope.rotary_dim=c.rotary_dim;allocation_mrope.mrope_layout=c.mrope_layout;allocation_mrope.section_t=c.sections[0];allocation_mrope.section_h=c.sections[1];allocation_mrope.section_w=c.sections[2];allocation_mrope.rotary_scale=c.mrope_scale;
  size_t bytes=0;const int query=hip_paged_attention_mrope_f16_workspace_size(&allocation_mrope,&bytes);
  if(query){throw std::invalid_argument("custom options unsupported; select explicit --error for rejection tests");}
  int rc=0,device=0;std::vector<unsigned char> output,kpool,vpool,workspace;bool immutable=true;
  {
    hipdnn_ep_test::GpuExecutionScope admission([]{check(hipDeviceSynchronize());});
    {
      if(c.error=="none"&&!rank_row&&!(pin&&pin[0])) {
        const char* mode=std::getenv("HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE");
        if(mode&&std::string(mode)=="online"){online=true;source="autotune:production-prepare";}
        else {auto resolution=hipdnn_ep::resolvePagedAttention(h.desc,shape);selected=resolution.config;source=resolution.source;online=source.rfind("fallback:",0)==0;}
      }
      Bundle b(h,bytes);void* ws=b.workspace.ptr();size_t capacity=bytes;
      if(c.error=="tensor_alignment")b.mrope.q=reinterpret_cast<const float*>(1);
      if(c.error=="workspace")capacity=bytes?bytes-1:0;
      if(c.error=="workspace_alignment")ws=static_cast<unsigned char*>(ws)+1;
      if(c.error=="alias")b.desc.o=b.desc.q==nullptr?nullptr:const_cast<void*>(b.desc.q);
      if(c.error=="pool_alias")b.desc.v_pool=b.desc.k_pool;
      if(c.error=="mrope_null")b.mrope.position_ids=nullptr;
      if(c.error=="output_alignment")b.desc.o=static_cast<unsigned char*>(b.desc.o)+2;
      if(c.error=="pool_alignment")b.desc.k_pool=static_cast<unsigned char*>(b.desc.k_pool)+2;
      rc=b.fused(selected,ws,capacity);check(hipDeviceSynchronize());
      if(!rc&&h.desc.tokens)check(hipMemcpy(&device,b.workspace.ptr(),4,hipMemcpyDeviceToHost));
      output=b.o.download();kpool=b.kp.download();vpool=b.vp.download();workspace=b.workspace.download();
      immutable=matches(b.q.download(),h.q.data(),h.q.size()*4)&&matches(b.k.download(),h.k.data(),h.k.size()*4)&&matches(b.v.download(),h.v.data(),h.v.size()*2)&&
          matches(b.cum.download(),h.cum.data(),h.cum.size()*4)&&matches(b.past.download(),h.past.data(),h.past.size()*4)&&matches(b.table.download(),h.table.data(),h.table.size()*4)&&
          matches(b.co.download(),h.co.data(),h.co.size()*4)&&matches(b.si.download(),h.si.data(),h.si.size()*4)&&matches(b.positions.download(),h.positions.data(),h.positions.size()*8);
      if(c.error=="none"&&h.desc.tokens){if(b.unfused(selected,bytes))throw std::runtime_error("unfused production chain failed");check(hipDeviceSynchronize());
        auto uo=b.o.download(),uk=b.kp.download(),uv=b.vp.download();auto unfused_fp32=b.kr.download();
        if(uo!=output||uk!=kpool||uv!=vpool){
          auto differences=[&](const char* name,const std::vector<unsigned char>& fused,const std::vector<unsigned char>& unfused){size_t count=0;for(size_t i=16;i+16<fused.size();i+=2)if(std::memcmp(fused.data()+i,unfused.data()+i,2)){uint16_t a,b;std::memcpy(&a,fused.data()+i,2);std::memcpy(&b,unfused.data()+i,2);if(!count){std::printf("MROPE_BYTE_MISMATCH %s element=%zu fused=%04x unfused=%04x\n",name,(i-16)/2,unsigned(a),unsigned(b));
            if(std::string(name)=="K")for(int batch=0;batch<c.B;++batch)for(int local=0;local<c.q[batch];++local)for(int head=0;head<c.G;++head){const int t=h.cum[batch]+local;const size_t start=offset(c,h.table,batch,c.past[batch]+local,head,0),index=(i-16)/2;
              if(index>=start&&index<start+size_t(c.rotary_dim)){const int channel=int(index-start),j=channel%(c.rotary_dim/2),half=c.rotary_dim/2;int axis=0;
                if(c.mrope_layout==0)axis=j<c.sections[0]?0:j<c.sections[0]+c.sections[1]?1:2;else if(j%3==1&&j<3*c.sections[1])axis=1;else if(j%3==2&&j<3*c.sections[2])axis=2;
                const int64_t position=h.positions[size_t(axis)*h.desc.tokens+t];if(position>=0&&position<c.max_position){const size_t src=(size_t(t)*c.G+head)*c.D;float x=h.k[src+j],y=h.k[src+j+half],co=h.co[size_t(position)*half+j]*c.mrope_scale,si=h.si[size_t(position)*half+j]*c.mrope_scale;
                  auto raw=unfused_fp32;float actual;std::memcpy(&actual,raw.data()+16+(src+channel)*4,4);float a1=channel<half?std::fma(-y,si,x*co):std::fma(x,si,y*co),a2=channel<half?std::fma(x,co,-y*si):std::fma(y,co,x*si);__half h1=__float2half(a1),h2=__float2half(a2);uint16_t u1,u2;std::memcpy(&u1,&h1,2);std::memcpy(&u2,&h2,2);
                  std::printf("MROPE_FMA_INPUT t=%d channel=%d x=%.9g y=%.9g co=%.9g si=%.9g fma1=%.9g/%04x fma2=%.9g/%04x unfused_f32=%.9g\n",t,channel,x,y,co,si,a1,unsigned(u1),a2,unsigned(u2),actual);}}
            }}++count;}std::printf("MROPE_BYTE_DIFFERENCES %s %zu\n",name,count);};
          differences("O",output,uo);differences("K",kpool,uk);differences("V",vpool,uv);throw std::runtime_error("fused/unfused production bytes differ");}}
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
  if(!valid){std::printf("FAIL %s host=%d device=%d expected=%d/%d relL2=%.9g immutable=%d K=%d V=%d\n",shape.c_str(),rc,device,h.host_status,h.device_status,result.rel_l2,int(immutable),int(matches(kpool,h.expected_k.data(),h.kp.size()*2)),int(matches(vpool,h.expected_v.data(),h.vp.size()*2)));return result;}
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
        auto launch=[&](int i){auto d=base.native();if(i){d.page.k_pool=caches[size_t(i-1)]->k.ptr();d.page.v_pool=caches[size_t(i-1)]->v.ptr();}return hip_paged_attention_mrope_f16_configured_v1(nullptr,&d,base.workspace.ptr(),bytes,&selected,sizeof(selected));};
        auto validate=[&](){
          if(launch(0))throw std::runtime_error("Page candidate host rejection");check(hipDeviceSynchronize());
          auto raw=base.o.download();auto kp=base.kp.download(),vp=base.vp.download();int status=0;check(hipMemcpy(&status,base.workspace.ptr(),4,hipMemcpyDeviceToHost));
          if(status||!guards(raw,16)||!guards(base.workspace.download(),256)||!matches(base.positions.download(),h.positions.data(),h.positions.size()*8)||!matches(base.co.download(),h.co.data(),h.co.size()*4)||!matches(base.si.download(),h.si.data(),h.si.size()*4)||!matches(kp,h.expected_k.data(),h.expected_k.size()*2)||!matches(vp,h.expected_v.data(),h.expected_v.size()*2)||
             !matches(base.q.download(),h.q.data(),h.q.size()*4)||!matches(base.k.download(),h.k.data(),h.k.size()*4)||!matches(base.v.download(),h.v.data(),h.v.size()*2)||
             !matches(base.cum.download(),h.cum.data(),h.cum.size()*4)||!matches(base.past.download(),h.past.data(),h.past.size()*4)||!matches(base.table.download(),h.table.data(),h.table.size()*4))throw std::runtime_error("Page candidate storage/metadata check failed");
          double sum=0,norm=0;for(size_t i=0;i<h.reference.size();++i){__half x;std::memcpy(&x,raw.data()+16+2*i,2);double e=__half2float(x)-h.reference[i];sum+=e*e;norm+=double(h.reference[i])*h.reference[i];}
          double error=std::sqrt(sum/std::max(norm,1e-30));if(!(error<(kDecode?2e-2:2e-3)))throw std::runtime_error("Page candidate CPU reference FAIL");return error;
        };
        auto measure=[&](const timing::RankOptions* options=nullptr){return timing::adaptiveRotatedMeanMs(nullptr,rotation,launch,
          flag,[&](const int* p){hold_stream<<<1,1>>>(p);},warm,size_t(128)<<20,"paged_mrope_attention", "hip_paged_attention_mrope_f16_configured_v1",timing::NoPreparation{},options);};
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
            auto native=base.native();const int rc=hip_paged_attention_mrope_f16_prepare_v1(nullptr,&native,base.workspace.ptr(),bytes,&prepared.config,sizeof(prepared.config));prepared.source="autotune:mrope-production-prepare";
            if(rc)throw std::runtime_error("production Page preparation failed");
            selected=prepared.config;source=prepared.source;result.rel_l2=validate();
          }
          result.config=source+":"+config_string(selected);std::printf("PAGED_CONFIG shape=%s config=%s boundary=complete\n",shape.c_str(),result.config.c_str());
          if(c.compare){
            auto chain=[&](int i){auto saved=base.desc;if(i){base.desc.k_pool=caches[size_t(i-1)]->k.ptr();base.desc.v_pool=caches[size_t(i-1)]->v.ptr();}int rc=base.unfused(selected,bytes);base.desc=saved;return rc;};
            const std::string baseline_shape=shape+"_unfused";
            auto measure_baseline=[&](){timing::CaseScope baseline(kLeaf,baseline_shape,"MRoPE_Cast_Page_complete");
            double baseline_ms=timing::adaptiveRotatedMeanMs(nullptr,rotation,chain,flag,[&](const int* p){hold_stream<<<1,1>>>(p);},warm,size_t(128)<<20,"paged_mrope_unfused","MRoPE_Cast_Page_complete",timing::NoPreparation{});
            timing::emitCsvBinding(baseline_shape,config_string(selected),"PASS",result.rel_l2,baseline_ms);
            return baseline_ms;};
            // Validate the selected config on both actual complete entries.
            validate();auto expected=base.o.download();if(base.unfused(selected,bytes))throw std::runtime_error("selected unfused config failed");check(hipDeviceSynchronize());
            if(base.o.download()!=expected)throw std::runtime_error("selected fused/unfused output differs");
            double baseline_ms=0;
            if(c.compare_order){baseline_ms=measure_baseline();result.time_ms=measure();}else{result.time_ms=measure();baseline_ms=measure_baseline();}
            result.unfused_ms=baseline_ms;result.unfused_config=config_string(selected);
            std::printf("MROPE_COMPARE {\"shape\":%s,\"config\":%s,\"fused_us\":%.17g,\"unfused_us\":%.17g,\"ratio\":%.17g,\"relL2\":%.17g,\"storage_exact\":true}\n",timing::jsonString(shape).c_str(),timing::jsonString(config_string(selected)).c_str(),result.time_ms*1000,baseline_ms*1000,result.time_ms/baseline_ms,result.rel_l2);
          }else result.time_ms=measure();
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
      env("HIPDNN_RESULTS_OP",kDecode?"paged_attention/decode":"paged_attention/prefill"),env("HIPDNN_RESULTS_LEAF","Xfp32_Wfp16_Yfp16_mrope"),
      env("HIPDNN_RESULTS_ARCH","unknown"),env("HIPDNN_RESULTS_MODE","lookup"),shape.c_str(),r.config.c_str(),r.time_ms,r.rel_l2,r.passed?"PASS":"FAIL");
  if(c.compare&&r.passed&&r.unfused_ms>0)std::fprintf(f,"%s,%s,%s,%s,%s,%s,%.9g,%.9g,PASS\n",
      env("HIPDNN_RESULTS_OP",kDecode?"paged_attention/decode":"paged_attention/prefill"),env("HIPDNN_RESULTS_LEAF","Xfp32_Wfp16_Yfp16_mrope"),
      env("HIPDNN_RESULTS_ARCH","unknown"),env("HIPDNN_RESULTS_MODE","lookup"),(shape+"_unfused").c_str(),r.unfused_config.c_str(),r.unfused_ms,r.rel_l2);
  if(std::fclose(f))throw std::runtime_error("CSV close failed");
}
int main(int argc,char** argv){
  try {bool single=false,list_cases=false;int coverage=3;Case c=parse_single(argc,argv,single,coverage,list_cases);if(single)validate_case(c);
    const std::vector<Case> cases=single?std::vector<Case>{c}:build_cases(coverage);
    if(list_cases){for(const auto& test:cases){validate_case(test);std::printf("UT_CASE %s\n",timing::jsonString(describe_case(test)).c_str());}return 0;}
    if(timing::rankRequested()) {
#ifndef HIPDNN_UT_RANK_PAGED_ADAPTER
      throw std::invalid_argument("UT rank: Page requires RANK=1 build");
#else
      const auto rows=timing::readRankRows(kLeaf);int failed=0;
      for(const auto& row:rows){const auto test=rank_case(row.shape);timing::CaseScope scope(kLeaf,row.shape,"hip_paged_attention_mrope_f16_configured_v1");
        if(std::getenv("HIPDNN_UT_RANK_CATALOG")){print_catalog(test);continue;}
        auto result=run_case(test,&row);failed+=!result.passed;
      }
      std::printf("%s rank (%d failures)\n",failed?"SOME FAIL":"ALL PASS",failed);return failed?1:0;
#endif
    }
    std::printf("coverage=%d -> running %zu paged_attention cases; phase=%s; real Page lookup/online; shared ut-adaptive-v3\n",coverage,cases.size(),kDecode?"decode":"prefill");
    int failed=0;for(const auto& test:cases){timing::CaseScope scope(kLeaf,describe_case(test),"hip_paged_attention_mrope_f16_configured_v1");
      try{auto result=run_case(test);csv(test,result);failed+=!result.passed;}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s: %s\n",describe_case(test).c_str(),e.what());CaseResult failure;failure.config="declined";csv(test,failure);++failed;}}
    std::printf("%s (%d failing case(s))\n",failed?"SOME FAIL":"ALL PASS",failed);return failed?1:0;
  }catch(const std::exception& e){std::fprintf(stderr,"FAIL arguments: %s\n",e.what());return 2;}
}
