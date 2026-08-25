/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// T0.2 spike (Phase 0, throwaway -- see docs/design/hybrid-npu-gpu-tasks.md).
// Proves three toolchains coexist in one process: this executable links HIP
// and ORT directly (hip-ep's own /MT toolchain), loads this repository's
// already-built EP DLL through ORT's plugin-EP mechanism, obtains the EP's
// *real* allocator (morphizen::HipGpuAllocator, via the standard
// Env::CreateSharedAllocator plugin-EP path -- no session needed, so this
// does not depend on the session/target-auto-discovery issue noted in
// morphizen/ort-bridge/test/src/test-hello-ep.cpp), and repeatedly hands
// EP-allocated buffers across a separate /MD DLL built against a prebuilt
// DynamicDispatch to run one DD operator. See ../README.md for what actually
// happened on this machine (no NPU) and what remains for the gfx1151 host.

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "onnxruntime_cxx_api.h"

#include "../dd_op_dll/dd_linkage_op.h"

namespace {

using DdLinkageRunTrivialOpFn = int (*)(void *, size_t, char *, size_t);

// Command-line overrides so this can be pointed at whatever install tree is
// current without editing the source -- see ../README.md for the paths used
// on this machine.
struct Args {
  std::string ep_dll_path = "onnxruntime_morphizen_ep.dll";
  std::string dd_dll_path = "dd_linkage_op.dll";
  int iterations = 200;
};

Args ParseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char *flag) -> const char * {
      return (a == flag && i + 1 < argc) ? argv[++i] : nullptr;
    };
    if (const char *v = next("--ep-dll")) {
      args.ep_dll_path = v;
    } else if (const char *v = next("--dd-dll")) {
      args.dd_dll_path = v;
    } else if (const char *v = next("--iterations")) {
      args.iterations = std::atoi(v);
    }
  }
  return args;
}

std::wstring ToWide(const std::string &s) {
  if (s.empty()) {
    return std::wstring();
  }
  int len = MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(len, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0],
                      len);
  return w;
}

// Direct HIP touch, independent of anything ORT/EP does internally --
// guarantees amdhip64*.dll is loaded by this process directly (not just
// transitively through the EP DLL's own dependency chain), matching the
// task text's "a process that has already loaded HIP" literally.
bool TouchHip() {
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  std::printf("[harness] hipGetDeviceCount -> %s, count=%d\n",
              hipGetErrorString(err), count);
  return err == hipSuccess && count > 0;
}

} // namespace

int main(int argc, char **argv) {
  // Unbuffered so a crash mid-run doesn't swallow the progress trace that
  // would otherwise pinpoint exactly which call it happened in.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  Args args = ParseArgs(argc, argv);

  if (!TouchHip()) {
    std::fprintf(stderr, "[harness] no HIP-visible AMD GPU on this machine -- "
                         "MorphiZenEP's factory returns zero EP devices in "
                         "production mode without one (see "
                         "morphizen-ep-factory.cpp). Cannot exercise the EP "
                         "allocator without one; stopping here.\n");
    return 2;
  }

  // Deliberately non-fatal: if this DLL (or one of ITS dependencies -- e.g.
  // DynamicDispatch's own runtime deps) can't load on this machine, that is
  // exactly the kind of environment/hardware wall this spike is supposed to
  // hit and report rather than route around. The rest of the harness (real
  // ORT + EP DLL + EP allocator loop) still runs so the parts that CAN be
  // proven locally are, instead of the whole run aborting on the one part
  // that can't. See ../README.md for what actually happened here.
  DdLinkageRunTrivialOpFn dd_run_trivial_op = nullptr;
  HMODULE dd_module = LoadLibraryA(args.dd_dll_path.c_str());
  if (dd_module == nullptr) {
    std::fprintf(stderr,
                 "[harness] LoadLibrary(%s) failed: %lu -- continuing "
                 "WITHOUT the DD leg (EP allocator loop still runs).\n",
                 args.dd_dll_path.c_str(), GetLastError());
  } else {
    dd_run_trivial_op = reinterpret_cast<DdLinkageRunTrivialOpFn>(
        GetProcAddress(dd_module, "dd_linkage_run_trivial_op"));
    if (dd_run_trivial_op == nullptr) {
      std::fprintf(stderr,
                   "[harness] GetProcAddress(dd_linkage_run_trivial_op) "
                   "failed: %lu -- continuing WITHOUT the DD leg.\n",
                   GetLastError());
    } else {
      std::printf("[harness] loaded %s\n", args.dd_dll_path.c_str());
    }
  }

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "npu_linkage_harness");

    const std::string registration_name = "MorphiZenExecutionProvider";
    // RegisterExecutionProviderLibrary is the point at which this repo's EP
    // DLL is actually loaded into the process -- everything before this line
    // runs with only HIP + ORT loaded, matching the task's ordering.
    std::wstring ep_dll_path_w = ToWide(args.ep_dll_path);
    Ort::GetApi().RegisterExecutionProviderLibrary(
        env, registration_name.c_str(), ep_dll_path_w.c_str());
    std::printf("[harness] registered EP library %s\n",
                args.ep_dll_path.c_str());

    auto ep_devices = env.GetEpDevices();
    Ort::ConstEpDevice target_device{nullptr};
    for (const auto &device : ep_devices) {
      std::printf("[harness] EP device: %s\n", device.EpName());
      if (registration_name == device.EpName()) {
        target_device = device;
      }
    }
    if (target_device == nullptr) {
      std::fprintf(stderr,
                   "[harness] no EP device named %s (no AMD GPU visible to "
                   "the factory in production mode?) -- stopping.\n",
                   registration_name.c_str());
      Ort::GetApi().UnregisterExecutionProviderLibrary(
          env, registration_name.c_str());
      return 4;
    }

    // The EP allocator: obtained the standard plugin-EP way
    // (Env::CreateSharedAllocator -> OrtApi::CreateSharedAllocator ->
    // MorphiZenEpFactory::CreateAllocatorImpl -> `new
    // morphizen::HipGpuAllocator(...)`), not a session's. No session is
    // created here -- this repo's own EP-vs-ORT session wiring is exercised
    // elsewhere; this spike only needs the allocator that the shim boundary
    // will actually be handed buffers from.
    Ort::UnownedAllocator allocator = env.CreateSharedAllocator(
        target_device, OrtDeviceMemoryType_HOST_ACCESSIBLE, OrtDeviceAllocator,
        nullptr);
    std::printf(
        "[harness] obtained HipGpuAllocator via CreateSharedAllocator\n");

    int ok = 0, failed = 0;
    for (int i = 0; i < args.iterations; ++i) {
      void *buf = allocator.Alloc(DD_LINKAGE_OUTPUT_BYTES);
      if (buf == nullptr) {
        std::fprintf(stderr, "[harness] allocator.Alloc failed at iter %d\n",
                     i);
        failed++;
        continue;
      }
      // Canary so a heap corruption that stomps this buffer, rather than the
      // allocator's own bookkeeping, is at least visible in principle --
      // not a substitute for the application-verifier / heap-check run this
      // gate actually requires (see ../README.md).
      std::memset(buf, 0xCD, DD_LINKAGE_OUTPUT_BYTES);

      if (dd_run_trivial_op != nullptr) {
        char err_msg[512];
        int rc = dd_run_trivial_op(buf, DD_LINKAGE_OUTPUT_BYTES, err_msg,
                                   sizeof(err_msg));
        if (rc == DD_LINKAGE_OK) {
          ok++;
        } else {
          failed++;
          if (i < 3 || i == args.iterations - 1) {
            std::printf(
                "[harness] iter %d: dd_linkage_run_trivial_op -> %d: %s\n", i,
                rc, err_msg);
          }
        }
      }

      allocator.Free(buf);
    }

    std::printf(
        "[harness] loop done: %d iterations, %d DD-op ok, %d DD-op failed "
        "(dd_run_trivial_op %s), no crash\n",
        args.iterations, ok, failed,
        dd_run_trivial_op != nullptr ? "loaded" : "NOT loaded -- see above");

    Ort::GetApi().UnregisterExecutionProviderLibrary(env,
                                                     registration_name.c_str());
  } catch (const Ort::Exception &e) {
    std::fprintf(stderr, "[harness] ORT exception: %s\n", e.what());
    return 5;
  }

  std::printf("[harness] HARNESS COMPLETED, NO CRASH\n");
  return 0;
}
