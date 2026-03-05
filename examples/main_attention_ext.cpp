/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_attention_ext.cpp - ORT-vs-MLIR attention test (ext consts) ---===//
//
// Usage: attention_ext_test.exe <attention.onnx> <model.constants.bin>
//
// Same as main_attention.cpp but for the externalized-constants variant.
//
// The compiled MLIR DLL exposes main_graph with a 23-argument ABI:
//   - arg0: memref<2x64x64xf32, strided<[?,?,?], offset:?>>  (X input)
//   - arg1: memref<2x64x64xf32>                               (output)
//   - arg2: memref<?xi8>                                       (constants blob)
//
// 1. Loads the ONNX model and runs inference on CPU via ONNX Runtime
//    to produce reference output.
// 2. Loads model.constants.bin, uploads it to GPU.
// 3. Calls the compiled MLIR DLL (main_graph) on GPU.
// 4. Compares GPU output against the ORT reference.
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
#include <onnxruntime_cxx_api.h>

// --------------------------------------------------------------------------
// DLL import: compiled from attention_ext.mlir
//
// After convert-func-to-llvm each memref arg is flattened:
//   strided 3D: (alloc, aligned, offset, s0, s1, s2, st0, st1, st2)  = 9
//   static  3D: (alloc, aligned, offset, s0, s1, s2, st0, st1, st2)  = 9
//   dynamic 1D: (alloc, aligned, offset, s0, st0)                     = 5
//                                                             Total   = 23
// --------------------------------------------------------------------------
extern "C" __declspec(dllimport) void main_graph(
    // arg0: X input  memref<2x64x64xf32, strided>
    float *X_a, float *X_al, int64_t X_o, int64_t X_s0, int64_t X_s1,
    int64_t X_s2, int64_t X_st0, int64_t X_st1, int64_t X_st2,
    // arg1: output   memref<2x64x64xf32>
    float *out_a, float *out_al, int64_t out_o, int64_t out_s0, int64_t out_s1,
    int64_t out_s2, int64_t out_st0, int64_t out_st1, int64_t out_st2,
    // arg2: constants blob  memref<?xi8>
    int8_t *c_a, int8_t *c_al, int64_t c_o, int64_t c_s0, int64_t c_st0);

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t e = (call);                                                     \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,             \
              hipGetErrorString(e));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// --------------------------------------------------------------------------
// Load a binary file into a host buffer
// --------------------------------------------------------------------------
static std::vector<int8_t> load_binary_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open constants file: %s\n", path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<int8_t> buf(size);
  if (fread(buf.data(), 1, size, f) != static_cast<size_t>(size)) {
    fprintf(stderr, "Short read on %s\n", path);
    exit(1);
  }
  fclose(f);
  return buf;
}

// --------------------------------------------------------------------------
// ORT: run ONNX model on CPU, return output as host vector
// --------------------------------------------------------------------------
static std::vector<float> run_ort_reference(const char *onnx_path,
                                            const float *input_data, int64_t B,
                                            int64_t S, int64_t D) {
  const int64_t total = B * S * D;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "attn_ext_test");
  Ort::SessionOptions opts;
  std::wstring wpath(onnx_path, onnx_path + strlen(onnx_path));
  Ort::Session session(env, wpath.c_str(), opts);

  auto in_name =
      session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name =
      session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  printf("[ORT] input: \"%s\"  output: \"%s\"\n", in_name.get(),
         out_name.get());

  auto mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t shape[] = {B, S, D};
  auto input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, const_cast<float *>(input_data), total, shape, 3);

  const char *in_names[] = {in_name.get()};
  const char *out_names[] = {out_name.get()};
  printf("[ORT] Running inference on CPU...\n");
  auto outputs =
      session.Run(Ort::RunOptions{}, in_names, &input_tensor, 1, out_names, 1);

  const float *ort_data = outputs[0].GetTensorData<float>();
  std::vector<float> result(ort_data, ort_data + total);
  printf("[ORT] Done. First values: %.6f %.6f %.6f ...\n", result[0], result[1],
         result[2]);
  return result;
}

// --------------------------------------------------------------------------
// GPU: run the compiled MLIR DLL, return output as host vector
// --------------------------------------------------------------------------
static std::vector<float> run_gpu(const float *input_data, int64_t B, int64_t S,
                                  int64_t D, const int8_t *h_constants,
                                  int64_t constants_size) {
  const int64_t total = B * S * D;
  printf("[GPU] Allocating device memory...\n");

  float *d_X = nullptr, *d_out = nullptr;
  int8_t *d_consts = nullptr;
  HIP_CHECK(hipMalloc(&d_X, total * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, total * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_consts, constants_size));

  HIP_CHECK(
      hipMemcpy(d_X, input_data, total * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, total * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_consts, h_constants, constants_size, hipMemcpyHostToDevice));

  printf("[GPU] Calling main_graph (ext constants, %lld bytes)...\n",
         (long long)constants_size);
  main_graph(d_X, d_X, 0, B, S, D, S * D, D, 1,         // X input (strided)
             d_out, d_out, 0, B, S, D, S * D, D, 1,     // output
             d_consts, d_consts, 0, constants_size, 1); // constants blob

  std::vector<float> result(total);
  HIP_CHECK(hipMemcpy(result.data(), d_out, total * sizeof(float),
                      hipMemcpyDeviceToHost));
  printf("[GPU] Done. First values: %.6f %.6f %.6f ...\n", result[0], result[1],
         result[2]);

  HIP_CHECK(hipFree(d_X));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_consts));
  return result;
}

// --------------------------------------------------------------------------
// Compare two float vectors element-wise
// --------------------------------------------------------------------------
static bool compare_results(const std::vector<float> &gpu,
                            const std::vector<float> &ref, float tolerance) {
  int64_t total = static_cast<int64_t>(gpu.size());

  printf("\n--- Comparison ---\n");
  printf("  GPU output (first 8): ");
  for (int i = 0; i < 8 && i < total; ++i)
    printf(" %8.5f", gpu[i]);
  printf("\n  ORT ref    (first 8): ");
  for (int i = 0; i < 8 && i < total; ++i)
    printf(" %8.5f", ref[i]);
  printf("\n");

  float max_diff = 0;
  int64_t worst_idx = 0;
  for (int64_t i = 0; i < total; ++i) {
    float d = std::fabs(gpu[i] - ref[i]);
    if (d > max_diff) {
      max_diff = d;
      worst_idx = i;
    }
  }

  bool pass = (max_diff < tolerance);
  printf("\n  Max abs diff: %e  (at index %lld: gpu=%.6f ort=%.6f)\n", max_diff,
         (long long)worst_idx, gpu[worst_idx], ref[worst_idx]);
  printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <attention.onnx> <model.constants.bin>\n",
            argv[0]);
    return 1;
  }

  const int64_t B = 2, S = 64, D = 64;
  const int64_t total = B * S * D;

  printf("=== Attention ORT-vs-MLIR test (externalized constants) ===\n");
  printf("Shape: [%lld, %lld, %lld]  (%lld floats)\n", (long long)B,
         (long long)S, (long long)D, (long long)total);

  printf("[LOAD] Reading constants from %s\n", argv[2]);
  auto h_constants = load_binary_file(argv[2]);
  printf("[LOAD] Constants: %lld bytes\n", (long long)h_constants.size());

  std::vector<float> h_X(total);
  srand(42);
  for (auto &v : h_X)
    v = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;

  auto h_ref = run_ort_reference(argv[1], h_X.data(), B, S, D);
  auto h_out = run_gpu(h_X.data(), B, S, D, h_constants.data(),
                       static_cast<int64_t>(h_constants.size()));

  return compare_results(h_out, h_ref, 1e-2f) ? 0 : 1;
}
