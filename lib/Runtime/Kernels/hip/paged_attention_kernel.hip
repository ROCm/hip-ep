/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include "gqa_paged_attention.h"

namespace {
constexpr size_t kAlignment = 256;
thread_local GqaPagedConfig last_config{};
thread_local bool last_config_valid=false;
struct Layout {
  size_t bytes = 0, post = 0, query = 0, partial = 0;
  size_t query_bytes = 0, new_bytes = 0, pool_bytes = 0, table_bytes = 0, append_pairs = 0;
};
bool mul(size_t a, size_t b, size_t &v) {
  if (b && a > SIZE_MAX / b) return false;
  v = a * b; return true;
}
bool region(size_t bytes, size_t &cursor, size_t &offset) {
  if (cursor > SIZE_MAX - (kAlignment - 1)) return false;
  cursor = (cursor + kAlignment - 1) & ~(kAlignment - 1);
  offset = cursor;
  if (bytes > SIZE_MAX - cursor) return false;
  cursor += bytes; return true;
}
int layout(const hip_paged_attention_f16_desc &d, Layout &l) {
  if (d.batch <= 0 || d.tokens < 0 || d.num_heads <= 0 || d.kv_num_heads <= 0 ||
      d.num_heads % d.kv_num_heads || d.num_pages <= 0 || d.page_size <= 0 ||
      d.table_columns <= 0 || (d.do_rotary != 0 && d.do_rotary != 1) ||
      (d.phase != HIP_PAGED_ATTENTION_PREFILL && d.phase != HIP_PAGED_ATTENTION_DECODE))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if (!std::isfinite(d.scale) || !std::isfinite(d.softcap))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if (d.head_dim != 64 && d.head_dim != 96 && d.head_dim != 128 && d.head_dim != 256)
    return HIP_PAGED_ATTENTION_UNSUPPORTED;
  if (d.scale < 0 || d.softcap != 0 ||
      (d.phase == HIP_PAGED_ATTENTION_PREFILL && d.head_dim == 256 && d.local_window_size > 0))
    return HIP_PAGED_ATTENTION_UNSUPPORTED;
  if (d.do_rotary) {
    if (d.rotary_interleaved || d.rotary_dim != d.head_dim)
      return HIP_PAGED_ATTENTION_UNSUPPORTED;
    if (d.max_position <= 0) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  }
  const int hpg = d.num_heads / d.kv_num_heads;
  if (d.phase == HIP_PAGED_ATTENTION_DECODE &&
      hpg != 1 && hpg != 2 && hpg != 3 && hpg != 4 && hpg != 5 && hpg != 6 &&
      hpg != 8 && hpg != 16 && !(hpg == 9 && d.head_dim == 128))
    return HIP_PAGED_ATTENTION_UNSUPPORTED;
  size_t n;
  if (!mul(size_t(d.tokens), size_t(d.num_heads)+size_t(d.kv_num_heads), n) ||
      !mul(n, size_t(d.head_dim/2), l.append_pairs) || l.append_pairs > SIZE_MAX-255)
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if (!mul(size_t(d.tokens), size_t(d.num_heads), n) || !mul(n, size_t(d.head_dim)*2, l.query_bytes) ||
      !mul(size_t(d.tokens), size_t(d.kv_num_heads), n) || !mul(n, size_t(d.head_dim)*2, l.new_bytes) ||
      !mul(size_t(d.num_pages), size_t(d.page_size), n) || !mul(n, size_t(d.kv_num_heads), n) ||
      !mul(n, size_t(d.head_dim)*2, l.pool_bytes) ||
      !mul(size_t(d.batch), size_t(d.table_columns)*sizeof(int32_t), l.table_bytes))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  // Existing split-partial indexing multiplies these int dimensions.
  if (!mul(size_t(d.batch),size_t(d.num_heads),n) || n > size_t(INT32_MAX/kPagedDecodeSplits))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  const size_t rows = 16; // smallest supported query tile, before HB grouping
  if (!mul((size_t(d.tokens)+rows-1)/rows+size_t(d.batch)-1, size_t(d.num_heads), n) || n > INT32_MAX)
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if (!d.tokens) return HIP_PAGED_ATTENTION_SUCCESS;
  size_t cursor = sizeof(int32_t), ignored;
  if (!region(size_t(d.batch)*sizeof(int32_t), cursor, l.post) ||
      !region(d.do_rotary ? l.query_bytes : 0, cursor, l.query))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  size_t partial_bytes = 0;
  if (d.phase == HIP_PAGED_ATTENTION_DECODE &&
      !mul(size_t(d.batch)*d.num_heads*kPagedDecodeSplits, size_t(d.head_dim+2)*sizeof(float), partial_bytes))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if (!region(partial_bytes, cursor, l.partial) || !region(0, cursor, ignored))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  l.bytes = cursor;
  return HIP_PAGED_ATTENTION_SUCCESS;
}

struct Range { uintptr_t begin, end; };
bool range(const void *p, size_t bytes, Range &r) {
  r.begin = reinterpret_cast<uintptr_t>(p);
  if (!p || bytes > UINTPTR_MAX-r.begin) return false;
  r.end = r.begin+bytes; return true;
}
bool overlaps(const Range &a, const Range &b) {
  return a.begin < b.end && b.begin < a.end;
}
int storage(const hip_paged_attention_f16_desc &d, const Layout &l, void *workspace, size_t bytes) {
  if (!workspace || bytes < l.bytes) return HIP_PAGED_ATTENTION_WORKSPACE_TOO_SMALL;
  if (reinterpret_cast<uintptr_t>(workspace) % kAlignment) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  Range writes[4], reads[8];
  if (!range(d.k_pool,l.pool_bytes,writes[0]) || !range(d.v_pool,l.pool_bytes,writes[1]) ||
      !range(d.o,l.query_bytes,writes[2]) || !range(workspace,l.bytes,writes[3]))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  const void *ptrs[] = {d.q,d.k_new,d.v_new,d.block_table,d.cum,d.past_seqlens,d.cos_cache,d.sin_cache};
  size_t rope_bytes=0;
  if (d.do_rotary && !mul(size_t(d.max_position),size_t(d.head_dim),rope_bytes))
    return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  const size_t sizes[] = {l.query_bytes,l.new_bytes,l.new_bytes,l.table_bytes,
      (size_t(d.batch)+1)*sizeof(int32_t),size_t(d.batch)*sizeof(int32_t),rope_bytes,rope_bytes};
  const int count = d.do_rotary ? 8 : 6;
  for (int i=0;i<count;++i) {
    if (!range(ptrs[i],sizes[i],reads[i])) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
    const size_t alignment = i<3 ? 16 : i<6 ? 4 : 2;
    if (reads[i].begin%alignment) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  }
  for (int i=0;i<4;++i) {
    if (i<3 && writes[i].begin%16) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
    for (int j=0;j<i;++j) if (overlaps(writes[i],writes[j])) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
    for (int j=0;j<count;++j) if (overlaps(writes[i],reads[j])) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  }
  return HIP_PAGED_ATTENTION_SUCCESS;
}

__global__ void validate_metadata(hip_paged_attention_f16_desc d, int32_t *post, int32_t *status) {
  if (threadIdx.x || blockIdx.x) return;
  *status=0;
  if (d.cum[0]!=0 || d.cum[d.batch]!=d.tokens) { *status=HIP_PAGED_ATTENTION_BAD_CUM; return; }
  for (int b=0;b<d.batch;++b) {
    const int begin=d.cum[b], end=d.cum[b+1], past=d.past_seqlens[b];
    if (begin<0 || end<begin || end>d.tokens) { *status=HIP_PAGED_ATTENTION_BAD_CUM; return; }
    if (past<0) { *status=HIP_PAGED_ATTENTION_BAD_PAST; return; }
    const int q=end-begin;
    if (d.phase==HIP_PAGED_ATTENTION_DECODE && q>1) { *status=HIP_PAGED_ATTENTION_BAD_DECODE_LENGTH; return; }
    if (past>INT32_MAX-q) { *status=HIP_PAGED_ATTENTION_POST_OVERFLOW; return; }
    const int len=past+q; post[b]=len;
    const int pages=len ? (len-1)/d.page_size+1 : 0;
    if (pages>d.table_columns) { *status=HIP_PAGED_ATTENTION_TABLE_TOO_SMALL; return; }
    if (!q) continue;
    if (d.do_rotary && len>d.max_position) { *status=HIP_PAGED_ATTENTION_BAD_POSITION; return; }
    for (int j=0;j<pages;++j) {
      const int p=d.block_table[size_t(b)*d.table_columns+j];
      if (p<0 || p>=d.num_pages) { *status=HIP_PAGED_ATTENTION_BAD_PAGE; return; }
    }
  }
  // Detect overlapping new-write slots without scanning the physical pool.
  // Read sharing is legal; ownership/COW of historical readers is caller-owned.
  for (int b=0;b<d.batch;++b) {
    if (d.cum[b]==d.cum[b+1]) continue;
    for (int j=d.past_seqlens[b]/d.page_size;j<=(post[b]-1)/d.page_size;++j) {
      const int lo=max(0,d.past_seqlens[b]-j*d.page_size);
      const int hi=min(d.page_size,post[b]-j*d.page_size);
      const int page=d.block_table[size_t(b)*d.table_columns+j];
      for (int c=0;c<=b;++c) {
        if (d.cum[c]==d.cum[c+1]) continue;
        for (int k=d.past_seqlens[c]/d.page_size;k<=(post[c]-1)/d.page_size;++k) {
          if (c==b && k>=j) break;
          if (page==d.block_table[size_t(c)*d.table_columns+k] &&
              max(lo,max(0,d.past_seqlens[c]-k*d.page_size)) < min(hi,min(d.page_size,post[c]-k*d.page_size))) {
            *status=HIP_PAGED_ATTENTION_WRITE_CONFLICT; return;
          }
        }
      }
    }
  }
}

__global__ void rotate_append(hip_paged_attention_f16_desc d, GqaPagedArgs pg, __half *q_rotated) {
  if (*pg.status) return;
  const int half=d.head_dim/2;
  const size_t heads=(d.do_rotary?size_t(d.num_heads):0)+d.kv_num_heads;
  const size_t n=size_t(d.tokens)*heads*half;
  for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
    const int c=int(i%half), h=int((i/half)%heads), t=int(i/(heads*half));
    const bool is_q=d.do_rotary && h<d.num_heads;
    int b=0; while (d.cum[b+1]<=t) ++b;
    const int p=d.past_seqlens[b]+(t-d.cum[b]);
    const int head=is_q ? h : h-(d.do_rotary?d.num_heads:0);
    const size_t source=(size_t(t)*(is_q ? d.num_heads : d.kv_num_heads)+head)*d.head_dim;
    const __half *input=static_cast<const __half*>(is_q ? d.q : d.k_new);
    __half a=input[source+c], z=input[source+c+half];
    if (d.do_rotary) {
      const size_t angle=size_t(p)*half+c;
      const float co=__half2float(static_cast<const __half*>(d.cos_cache)[angle]);
      const float si=__half2float(static_cast<const __half*>(d.sin_cache)[angle]);
      const float x=__half2float(a), y=__half2float(z);
      float a32=__fmaf_rn(-y,si,__fmul_rn(x,co));
      float z32=__fmaf_rn(x,si,__fmul_rn(y,co));
      // Materialize the FP32 rounding boundary before FP16 storage. Without
      // this barrier the backend can fold fma + trunc into a mixed FP16 FMA.
      asm("" : "+v"(a32), "+v"(z32));
      a=__float2half(a32);
      z=__float2half(z32);
    }
    __half *output=is_q ? q_rotated : static_cast<__half*>(d.k_pool);
    const size_t dest=is_q ? source : pg.offset(b,p,head,d.head_dim);
    output[dest+c]=a; output[dest+c+half]=z;
    if (!is_q) {
      const __half *v=static_cast<const __half*>(d.v_new);
      static_cast<__half*>(d.v_pool)[dest+c]=v[source+c];
      static_cast<__half*>(d.v_pool)[dest+c+half]=v[source+c+half];
    }
  }
}
} // namespace

extern "C" HIP_KERNEL_API int hip_paged_attention_f16_workspace_size(
    const hip_paged_attention_f16_desc *desc, size_t *bytes) {
  if (!desc || !bytes) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  *bytes=0; Layout l; const int rc=layout(*desc,l);
  if (!rc) *bytes=l.bytes;
  return rc;
}

extern "C" HIP_KERNEL_API int hip_paged_attention_f16_configured_v2(void *stream_ptr,
    const hip_paged_attention_f16_desc *desc, void *workspace, size_t workspace_bytes, const GqaPagedConfig *config, size_t config_size) {
  if (!desc) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  Layout l; int rc=layout(*desc,l); if(rc) return rc;
  if(config && config_size!=sizeof(GqaPagedConfig)) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  const GqaPagedConfig selected=config ? *config : gqa_paged_default(*desc);
  if(!gqa_paged_config_valid(*desc,selected)) return HIP_PAGED_ATTENTION_UNSUPPORTED;
  last_config=selected;last_config_valid=true;
  if(!desc->tokens) return HIP_PAGED_ATTENTION_SUCCESS;
  rc=storage(*desc,l,workspace,workspace_bytes); if (rc) return rc;
  const auto stream=static_cast<hipStream_t>(stream_ptr);
  auto *base=static_cast<unsigned char*>(workspace);
  auto *post=reinterpret_cast<int32_t*>(base+l.post);
  auto *status=reinterpret_cast<int32_t*>(base);
  auto *q_rotated=reinterpret_cast<__half*>(base+l.query);
  const GqaPagedArgs pg={desc->block_table,desc->cum,desc->past_seqlens,post,status,
                        desc->page_size,desc->table_columns,desc->kv_num_heads,gqa_paged_page_shift(desc->page_size)};
  (void)hipGetLastError();
  validate_metadata<<<1,1,0,stream>>>(*desc,post,status);
  if (hipGetLastError()!=hipSuccess) return HIP_PAGED_ATTENTION_HIP_ERROR;
  const size_t pairs=desc->do_rotary?l.append_pairs:size_t(desc->tokens)*desc->kv_num_heads*(desc->head_dim/2);
  const unsigned blocks=unsigned(std::min<size_t>(65535,(pairs+255)/256));
  rotate_append<<<blocks,256,0,stream>>>(*desc,pg,q_rotated);
  if (hipGetLastError()!=hipSuccess) return HIP_PAGED_ATTENTION_HIP_ERROR;
  rc=gqa_paged_attention_f16_launch(stream_ptr,*desc,desc->do_rotary ? q_rotated : desc->q,
                                  reinterpret_cast<float*>(base+l.partial),pg,selected);
  return rc ? HIP_PAGED_ATTENTION_HIP_ERROR : HIP_PAGED_ATTENTION_SUCCESS;
}

extern "C" HIP_KERNEL_API int hip_paged_attention_f16_configured(void *stream_ptr,
    const hip_paged_attention_f16_desc *desc, void *workspace, size_t workspace_bytes,
    const hip_paged_attention_f16_config *config) {
  if(!desc) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  if(!config) return hip_paged_attention_f16_configured_v2(stream_ptr,desc,workspace,workspace_bytes,nullptr,0);
  if(desc->phase==HIP_PAGED_ATTENTION_DECODE&&config->splits==0)return HIP_PAGED_ATTENTION_UNSUPPORTED;
  const auto converted=gqa_paged_legacy(*desc,*config);
  return hip_paged_attention_f16_configured_v2(stream_ptr,desc,workspace,workspace_bytes,&converted,sizeof(converted));
}
extern "C" HIP_KERNEL_API int hip_paged_attention_f16(void *stream_ptr,
    const hip_paged_attention_f16_desc *desc, void *workspace, size_t workspace_bytes) {
  return hip_paged_attention_f16_configured_v2(stream_ptr,desc,workspace,workspace_bytes,nullptr,0);
}

extern "C" HIP_KERNEL_API int hip_paged_attention_f16_last_config_v2(GqaPagedConfig* config,size_t config_size) {
  if(!config||config_size!=sizeof(GqaPagedConfig)||!last_config_valid) return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  *config=last_config;return HIP_PAGED_ATTENTION_SUCCESS;
}

extern "C" HIP_KERNEL_API int hip_paged_attention_f16_prepare_v2(void* stream_ptr,
    const hip_paged_attention_f16_desc* desc,void* workspace,size_t workspace_bytes,
    GqaPagedConfig* config,size_t config_size) {
  if(!desc||!config||config_size!=sizeof(GqaPagedConfig))return HIP_PAGED_ATTENTION_INVALID_ARGUMENT;
  Layout l;int rc=layout(*desc,l);if(rc)return rc;
  GqaPagedConfig selected=gqa_paged_default(*desc);
  rc=hip_paged_attention_f16_configured_v2(stream_ptr,desc,workspace,workspace_bytes,&selected,sizeof(selected));
  if(rc)return rc;
  if(!desc->tokens){*config=selected;return HIP_PAGED_ATTENTION_SUCCESS;}
  const auto stream=static_cast<hipStream_t>(stream_ptr);
  int32_t status=0;
  if(hipMemcpyAsync(&status,workspace,sizeof(status),hipMemcpyDeviceToHost,stream)!=hipSuccess||
     hipStreamSynchronize(stream)!=hipSuccess)return HIP_PAGED_ATTENTION_HIP_ERROR;
  if(!status){
    auto* base=static_cast<unsigned char*>(workspace);
    const GqaPagedArgs pg={desc->block_table,desc->cum,desc->past_seqlens,
      reinterpret_cast<int32_t*>(base+l.post),reinterpret_cast<int32_t*>(base),
      desc->page_size,desc->table_columns,desc->kv_num_heads,gqa_paged_page_shift(desc->page_size)};
    rc=gqa_paged_attention_f16_tune(stream_ptr,*desc,desc->do_rotary?base+l.query:desc->q,
      reinterpret_cast<float*>(base+l.partial),pg,&selected);
    if(rc)return rc;
  }
  *config=selected;return HIP_PAGED_ATTENTION_SUCCESS;
}
