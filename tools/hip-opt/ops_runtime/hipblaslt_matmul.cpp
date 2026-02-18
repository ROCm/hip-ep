//===- hipblaslt_matmul.cpp - hip.hipblaslt.matmul runtime -----------------===//
//
// Runtime for hip.hipblaslt.matmul(handle, A, B, C).
// The MLIR lowering passes raw device pointers (extracted from memref
// descriptors).  Shape information is not passed -- the caller is responsible
// for ensuring the pointers and shapes are consistent.
//
// NOTE: This is a simplified implementation that hardcodes f32 and assumes
// 2-D row-major inputs.  A production version would receive shape metadata.
//
//===----------------------------------------------------------------------===//

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>

#define HIPBLASLT_CHECK(call)                                             \
  do {                                                                    \
    hipblasStatus_t status = (call);                                      \
    if (status != HIPBLAS_STATUS_SUCCESS) {                               \
      fprintf(stderr, "hipBLAS-LT error at %s:%d (status=%d)\n",         \
              __FILE__, __LINE__, status);                                \
      return;                                                             \
    }                                                                     \
  } while (0)

extern "C" void hip_hipblaslt_matmul(void * /*handle*/,
                                      void *A_ptr, void *B_ptr,
                                      void *C_ptr) {
  // Without shape metadata we cannot perform the actual GEMM.
  // This stub prints a message; a full implementation would mirror
  // hip_gemm_f32() in hip_gemm_runtime.cpp with M/K/N passed in.
  fprintf(stderr, "[hip_hipblaslt_matmul] called (A=%p, B=%p, C=%p)\n",
          A_ptr, B_ptr, C_ptr);
}
