/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_attention.cpp - ORT-vs-MLIR attention test driver -------------===//
//
// Usage: attention_test.exe <attention.onnx>
//
// 1. Loads the ONNX model and runs inference on CPU via ONNX Runtime
//    to produce reference output.
// 2. Calls the compiled MLIR DLL (main_graph) on GPU.
// 3. Compares GPU output against the ORT reference.
//
// The MLIR DLL (attention.dll) exports main_graph with the signature
// produced by compiling attention.hip.mlir through convert-hip-to-llvm.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <onnxruntime_c_api.h>

// --------------------------------------------------------------------------
// DLL import: compiled from attention.hip.mlir
//
// func.func @main_graph(
//     %arg0: memref<2x64x64xf32, strided<[?,?,?], offset:?>>,
//     %arg1: memref<2x64x64xf32>)
//
// After convert-func-to-llvm each memref arg is flattened to:
//   (alloc_ptr, aligned_ptr, offset, s0, s1, s2, st0, st1, st2)
// --------------------------------------------------------------------------
extern "C" __declspec(dllimport) void main_graph(
    float* X_a, float* X_al, int64_t X_o,
    int64_t X_s0, int64_t X_s1, int64_t X_s2,
    int64_t X_st0, int64_t X_st1, int64_t X_st2,
    float* out_a, float* out_al, int64_t out_o,
    int64_t out_s0, int64_t out_s1, int64_t out_s2,
    int64_t out_st0, int64_t out_st1, int64_t out_st2);

// --------------------------------------------------------------------------
// Error-checking macros
// --------------------------------------------------------------------------
#define HIP_CHECK(call)                                            \
  do {                                                             \
    hipError_t e = (call);                                         \
    if (e != hipSuccess) {                                         \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
              hipGetErrorString(e));                               \
      exit(1);                                                     \
    }                                                              \
  } while (0)

#define ORT_ABORT_ON_ERROR(ort, expr)                              \
  do {                                                             \
    OrtStatus* _s = (expr);                                        \
    if (_s) {                                                      \
      fprintf(stderr, "ORT error %s:%d: %s\n", __FILE__, __LINE__, \
              (ort)->GetErrorMessage(_s));                         \
      (ort)->ReleaseStatus(_s);                                    \
      exit(1);                                                     \
    }                                                              \
  } while (0)

// --------------------------------------------------------------------------
// Widen a narrow path to wchar_t (Windows CreateSession needs wchar_t*)
// --------------------------------------------------------------------------
static std::wstring to_wstring(const char* s) {
  size_t len = strlen(s);
  std::wstring ws(len, L'\0');
  for (size_t i = 0; i < len; ++i)
    ws[i] = static_cast<wchar_t>(static_cast<unsigned char>(s[i]));
  return ws;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <attention.onnx>\n", argv[0]);
    return 1;
  }
  const char* onnx_path = argv[1];

  const int64_t B = 2, S = 64, D = 64;
  const int64_t total = B * S * D;

  printf("=== Attention ORT-vs-MLIR test ===\n");
  printf("Shape: [%lld, %lld, %lld]  (%lld floats)\n",
         (long long)B, (long long)S, (long long)D, (long long)total);

  // ------------------------------------------------------------------
  // Generate deterministic input
  // ------------------------------------------------------------------
  std::vector<float> h_X(total);
  srand(42);
  for (auto& v : h_X)
    v = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;

  // ==================================================================
  //  ORT: run on CPU as reference
  // ==================================================================
  printf("\n[ORT] Loading model: %s\n", onnx_path);

  const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  OrtEnv* env = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "attn_test", &env));

  OrtSessionOptions* opts = nullptr;
  ORT_ABORT_ON_ERROR(ort, ort->CreateSessionOptions(&opts));

  std::wstring wpath = to_wstring(onnx_path);
  OrtSession* session = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->CreateSession(env, wpath.c_str(), opts, &session));

  OrtAllocator* alloc = nullptr;
  ORT_ABORT_ON_ERROR(ort, ort->GetAllocatorWithDefaultOptions(&alloc));

  // Query input name
  char* raw_input_name = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->SessionGetInputName(session, 0, alloc, &raw_input_name));
  std::string input_name(raw_input_name);
  ORT_ABORT_ON_ERROR(ort, ort->AllocatorFree(alloc, raw_input_name));

  // Query output name
  char* raw_output_name = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->SessionGetOutputName(session, 0, alloc, &raw_output_name));
  std::string output_name(raw_output_name);
  ORT_ABORT_ON_ERROR(ort, ort->AllocatorFree(alloc, raw_output_name));

  printf("[ORT] input: \"%s\"  output: \"%s\"\n",
         input_name.c_str(), output_name.c_str());

  // Create input tensor
  OrtMemoryInfo* mem_info = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                              &mem_info));

  int64_t shape[] = {B, S, D};
  OrtValue* input_tensor = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->CreateTensorWithDataAsOrtValue(
                         mem_info, h_X.data(), total * sizeof(float),
                         shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

  // Run
  const char* in_names[] = {input_name.c_str()};
  const char* out_names[] = {output_name.c_str()};
  OrtValue* output_tensor = nullptr;
  printf("[ORT] Running inference on CPU...\n");
  ORT_ABORT_ON_ERROR(ort,
                     ort->Run(session, nullptr, in_names,
                              (const OrtValue* const*)&input_tensor, 1,
                              out_names, 1, &output_tensor));

  float* ort_data = nullptr;
  ORT_ABORT_ON_ERROR(ort,
                     ort->GetTensorMutableData(output_tensor, (void**)&ort_data));

  std::vector<float> h_ref(ort_data, ort_data + total);
  printf("[ORT] Done. First values: %.6f %.6f %.6f ...\n",
         h_ref[0], h_ref[1], h_ref[2]);

  ort->ReleaseValue(output_tensor);
  ort->ReleaseValue(input_tensor);
  ort->ReleaseMemoryInfo(mem_info);
  ort->ReleaseSession(session);
  ort->ReleaseSessionOptions(opts);
  ort->ReleaseEnv(env);

  // ==================================================================
  //  GPU: run the compiled MLIR DLL
  // ==================================================================
  printf("\n[GPU] Allocating device memory...\n");

  float *d_X = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_X, total * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, total * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_X, h_X.data(), total * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, total * sizeof(float)));

  printf("[GPU] Calling main_graph...\n");
  main_graph(
      d_X, d_X, 0, B, S, D, S * D, D, 1,
      d_out, d_out, 0, B, S, D, S * D, D, 1);

  std::vector<float> h_out(total);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, total * sizeof(float),
                      hipMemcpyDeviceToHost));
  printf("[GPU] Done. First values: %.6f %.6f %.6f ...\n",
         h_out[0], h_out[1], h_out[2]);

  // ==================================================================
  //  Compare
  // ==================================================================
  printf("\n--- Comparison ---\n");
  printf("  GPU output (first 8): ");
  for (int i = 0; i < 8 && i < total; ++i)
    printf(" %8.5f", h_out[i]);
  printf("\n  ORT ref    (first 8): ");
  for (int i = 0; i < 8 && i < total; ++i)
    printf(" %8.5f", h_ref[i]);
  printf("\n");

  float max_diff = 0;
  int64_t worst_idx = 0;
  for (int64_t i = 0; i < total; ++i) {
    float d = std::fabs(h_out[i] - h_ref[i]);
    if (d > max_diff) {
      max_diff = d;
      worst_idx = i;
    }
  }
  printf("\n  Max abs diff: %e  (at index %lld: gpu=%.6f ort=%.6f)\n",
         max_diff, (long long)worst_idx, h_out[worst_idx], h_ref[worst_idx]);
  printf("  Result: %s\n\n", max_diff < 1e-2f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_X));
  HIP_CHECK(hipFree(d_out));
  return (max_diff < 1e-2f) ? 0 : 1;
}
