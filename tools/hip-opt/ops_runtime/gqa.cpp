//===- gqa.cpp - Grouped Query Attention (stub) ---------------------------===//
//
// Stub implementation: zeroes the output buffer.
// TODO: implement full GQA with KV-cache update, RoPE is applied upstream.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>

extern "C" void hip_gqa(void * /*handle*/, void * /*q*/, void * /*k*/,
                        void * /*v*/, void * /*kv_cache*/, void *output,
                        int64_t /*layer*/, int64_t /*start_pos*/,
                        int64_t /*seq_len*/) {
  (void)output;
  fprintf(stderr, "[hip_gqa] stub called -- output not zeroed (no size info)\n");
}
