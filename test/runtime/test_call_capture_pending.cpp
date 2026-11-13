/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Real GPU marker transport with synthetic diagnostic descriptors. This does
// not qualify actual model/session or production-wrapper operand capture.
#include "../../lib/Runtime/op_profile.h"
#include <algorithm>
#include <cstdlib>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

static bool copy_scope(OpProfileState *profile, hipStream_t stream,
                       void *source, void *output, const char *stage,
                       const char *call, bool annotated) {
  using Json = CallCaptureJson;
  OpCallCapture capture;
  capture.exampleOnly = true;
  capture.callId = call;
  capture.caseId = "synthetic-copy-case";
  capture.graphId = "synthetic-marker-graph";
  capture.windowId = stage;
  capture.stage = stage;
  capture.graph = Json::object(
      {{"name", "synthetic-marker-graph"}, {"sha256", std::string(64, 'a')}});
  capture.window = Json::object({{"graph_id", capture.graphId},
                                 {"stage", stage},
                                 {"phase", "transport-fixture"}});
  auto operand = [](const char *role) {
    return Json::object({{"slot", 0},
                         {"role", role},
                         {"present", true},
                         {"dtype", "uint8"},
                         {"shape", Json::dimensions({65536})}});
  };
  capture.descriptor = Json::object(
      {{"graph_id", capture.graphId},
       {"op_type", "SyntheticDeviceCopy"},
       {"execution", Json::object({{"backend", "HIP-EP"},
                                   {"entry", "synthetic-copy-scope"},
                                   {"variant", "fixture"},
                                   {"config", Json::object({})}})},
       {"inputs", Json::array({operand("source")})},
       {"outputs", Json::array({operand("destination")})},
       {"attributes", Json::object({})},
       {"parameters", Json::object({})}});
  op_profile_ensure_epoch(profile, stream);
  const int marker = op_profile_acquire_marker(profile);
  hipError_t result;
  {
    OpProfileScope scope(profile, annotated ? "synthetic-copy" : "legacy-copy",
                         "65536", stream, marker, 131072,
                         annotated ? &capture : nullptr);
    // Mutate the caller's descriptor while the scope owns its copy.
    capture.descriptor = Json::object({{"bad", "caller mutated"}});
    result =
        hipMemcpyAsync(output, source, 65536, hipMemcpyDeviceToDevice, stream);
  }
  // Both scope and caller die before pending records are resolved.
  return result == hipSuccess;
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
#ifdef _WIN32
  _putenv_s("HIPDNN_EP_TRACE_FILE", argv[1]);
#else
  setenv("HIPDNN_EP_TRACE_FILE", argv[1], 1);
#endif
  hipStream_t stream = nullptr;
  void *source = nullptr, *output = nullptr;
  hipDeviceProp_t device = {};
  int runtimeVersion = 0, driverVersion = 0;
  bool ok = hipGetDeviceProperties(&device, 0) == hipSuccess;
  ok = ok && hipRuntimeGetVersion(&runtimeVersion) == hipSuccess;
  ok = ok && hipDriverGetVersion(&driverVersion) == hipSuccess;
  ok = ok && hipStreamCreate(&stream) == hipSuccess;
  ok = ok && hipMalloc(&source, 65536) == hipSuccess;
  ok = ok && hipMalloc(&output, 65536) == hipSuccess;
  OpProfileState *profile = op_profile_create();
  if (ok) {
    ok = hipMemsetAsync(source, 42, 65536, stream) == hipSuccess;
    ok = ok && hipStreamSynchronize(stream) == hipSuccess;
    op_profile_reset(profile);
    ok = ok && copy_scope(profile, stream, source, output, "warmup",
                          "synthetic-warmup", true);
    ok = ok && copy_scope(profile, stream, source, output, "measured",
                          "synthetic-measured", true);
    ok = ok && copy_scope(profile, stream, source, output, "probe",
                          "synthetic-legacy", false);
    ok = ok && hipStreamSynchronize(stream) == hipSuccess;
    if (ok) {
      op_profile_resolve_and_print(profile);
      std::vector<unsigned char> host(65536);
      ok = hipMemcpy(host.data(), output, host.size(), hipMemcpyDeviceToHost) ==
           hipSuccess;
      ok = ok && std::all_of(host.begin(), host.end(),
                             [](unsigned char value) { return value == 42; });
    }
  }
  if (stream)
    ok = (hipStreamSynchronize(stream) == hipSuccess) && ok;
  if (source)
    ok = (hipFree(source) == hipSuccess) && ok;
  if (output)
    ok = (hipFree(output) == hipSuccess) && ok;
  if (stream)
    ok = (hipStreamDestroy(stream) == hipSuccess) && ok;
  op_profile_destroy(profile);
  printf("HIP_DEVICE_ARCH=%s\nHIP_DEVICE_NAME=%s\n", device.gcnArchName,
         device.name);
  printf("HIP_RUNTIME_VERSION=%d\nHIP_DRIVER_VERSION=%d\n", runtimeVersion,
         driverVersion);
#ifdef _WIN32
  char runtimePath[32768] = {};
  GetModuleFileNameA(GetModuleHandleA("amdhip64_7.dll"), runtimePath,
                     sizeof(runtimePath));
  printf("HIP_RUNTIME_PATH=%s\n", runtimePath);
#endif
  printf("SYNTHETIC_PENDING_GPU_TRANSPORT=%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
