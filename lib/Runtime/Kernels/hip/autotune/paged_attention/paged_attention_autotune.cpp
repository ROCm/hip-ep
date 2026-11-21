/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <hip/hip_runtime.h>
#include "paged_attention_autotune.h"
#include "paged_attention_autotune_generated.h"
#include <cstdio>
#include <map>
#include <set>
extern "C" const unsigned char* const kPagedAttentionLutBlobs[];
extern "C" const size_t kPagedAttentionLutBlobSizes[];
extern "C" const size_t kPagedAttentionLutBlobCount;
namespace hipdnn_ep {
namespace {
namespace fbs=hipdnn_ep::paged_attention_autotune::fbs;
struct Loaded { std::map<std::string,GqaPagedConfig> configs; std::string status="absent"; };
bool equal(const flatbuffers::String* s,const char* value){return s&&s->str()==value;}
Loaded load() {
 Loaded out;int device=0;hipDeviceProp_t props{};
 if(hipGetDevice(&device)!=hipSuccess||hipGetDeviceProperties(&props,device)!=hipSuccess){out.status="device-query-failed";return out;}
 std::string arch=props.gcnArchName;arch=arch.substr(0,arch.find(':'));
 for(size_t i=0;i<kPagedAttentionLutBlobCount;++i){
  flatbuffers::Verifier verifier(kPagedAttentionLutBlobs[i],kPagedAttentionLutBlobSizes[i]);
  if(!fbs::VerifyPagedAttentionLutBuffer(verifier)){out.status="corrupt";continue;}
  const auto* doc=fbs::GetPagedAttentionLut(kPagedAttentionLutBlobs[i]);
  if(doc->schema_version()!=2||!equal(doc->kernel_abi(),"paged_fp16_pshd_v2")||!equal(doc->op(),"paged_attention")||
     !equal(doc->layout(),"PSHD")||!equal(doc->dtype(),"Fp16")||!equal(doc->gpu_arch(),arch.c_str())){out.status="incompatible";continue;}
  std::map<std::string,GqaPagedConfig> configs;bool duplicate=false;
  if(doc->points())for(const auto* p:*doc->points()){
   if(!p->shape()||!p->phase()||p->shape()->str().rfind(p->phase()->str()+"_",0)!=0||
      !doc->configs_v2()||p->config()>=doc->configs_v2()->size()){out.status="invalid-point";continue;}
   const auto* c=doc->configs_v2()->Get(p->config());
   if(!configs.emplace(p->shape()->str(),GqaPagedConfig{c->splits(),c->bkv(),c->hb(),c->qw(),c->mt(),c->nd()}).second)duplicate=true;
  }
  if(duplicate){out.status="duplicate-key";continue;}
  out.configs=std::move(configs);out.status="compatible";break;
 }
 std::printf("[paged-lut] arch=%s status=%s points=%zu schema=2 abi=paged_fp16_pshd_v2 layout=PSHD dtype=Fp16\n",arch.c_str(),out.status.c_str(),out.configs.size());
 return out;
}
}
PagedAttentionResult resolvePagedAttention(const hip_paged_attention_f16_desc& d,const std::string& shape) {
 static const Loaded table=load();
 const auto it=table.configs.find(shape);
 if(it!=table.configs.end()&&gqa_paged_config_valid(d,it->second))return {it->second,"lookup:exact"};
 return {gqa_paged_default(d),it==table.configs.end()?"fallback:"+table.status:"fallback:invalid-config"};
}
int preparePagedAttention(void* stream,const hip_paged_attention_f16_desc& d,
    void* workspace,size_t bytes,const std::string& shape,bool online_only,PagedAttentionResult* result) {
 if(!result)return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
 auto resolution=online_only?PagedAttentionResult{gqa_paged_default(d),"online"}:resolvePagedAttention(d,shape);
 if(!online_only&&resolution.source.rfind("lookup:",0)==0){*result=resolution;return 0;}
 GqaPagedConfig selected{};
 const int rc=hip_paged_attention_f16_prepare_v2(stream,&d,workspace,bytes,&selected,sizeof(selected));
 if(rc)return rc;
 *result={selected,resolution.source+"->autotune:production-prepare"};return 0;
}

}
