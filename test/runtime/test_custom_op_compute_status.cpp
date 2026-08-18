/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free unit test for the CustomOp::Compute -> OrtStatus boundary.

#include "custom-op-compute-status.hpp"

#include <cstdio>
#include <string>

namespace {

int gFailures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++gFailures;                                                             \
    }                                                                          \
  } while (0)

struct FakeStatus {
  OrtErrorCode code;
  std::string message;
};

OrtStatus *ORT_API_CALL createStatus(OrtErrorCode code,
                                     const char *message) noexcept {
  return reinterpret_cast<OrtStatus *>(new FakeStatus{code, message});
}

FakeStatus *unwrap(OrtStatus *status) {
  return reinterpret_cast<FakeStatus *>(status);
}

void release(OrtStatus *status) { delete unwrap(status); }

OrtApi makeApi() {
  OrtApi api{};
  api.CreateStatus = &createStatus;
  return api;
}

void testSuccessReturnsNull() {
  const OrtApi api = makeApi();
  bool called = false;
  OrtStatus *status = morphizen::detail::translateComputeExceptions(
      api, [&] { called = true; });
  CHECK(called);
  CHECK(status == nullptr);
}

void testGeneratedFailureReturnsStatus() {
  const OrtApi api = makeApi();
  OrtStatus *status = morphizen::detail::translateComputeExceptions(api, [] {
    morphizen::detail::throwComputeFailure("inference_compute", 37);
  });
  CHECK(status != nullptr);
  if (status) {
    CHECK(unwrap(status)->code == ORT_RUNTIME_EXCEPTION);
    CHECK(unwrap(status)->message == "inference_compute failed with code: 37");
    release(status);
  }
}

void testUnknownExceptionReturnsStatus() {
  const OrtApi api = makeApi();
  OrtStatus *status =
      morphizen::detail::translateComputeExceptions(api, [] { throw 42; });
  CHECK(status != nullptr);
  if (status) {
    CHECK(unwrap(status)->code == ORT_RUNTIME_EXCEPTION);
    CHECK(unwrap(status)->message ==
          "CustomOp::Compute failed with unknown exception");
    release(status);
  }
}

void testDeferredCallbackFailureReturnsStatusAfterCleanup() {
  const OrtApi api = makeApi();
  morphizen::detail::DeferredComputeException deferred;
  bool cleanupRan = false;

  try {
    throw std::runtime_error("output allocator failed");
  } catch (...) {
    deferred.captureCurrent();
  }
  // The first callback failure is authoritative if later callbacks also fail.
  try {
    throw std::runtime_error("secondary failure");
  } catch (...) {
    deferred.captureCurrent();
  }

  OrtStatus *status = morphizen::detail::translateComputeExceptions(api, [&] {
    cleanupRan = true;
    deferred.rethrow();
  });
  CHECK(cleanupRan);
  CHECK(status != nullptr);
  if (status) {
    CHECK(unwrap(status)->code == ORT_RUNTIME_EXCEPTION);
    CHECK(unwrap(status)->message == "output allocator failed");
    release(status);
  }
}

} // namespace

int main() {
  testSuccessReturnsNull();
  testGeneratedFailureReturnsStatus();
  testUnknownExceptionReturnsStatus();
  testDeferredCallbackFailureReturnsStatusAfterCleanup();

  if (gFailures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", gFailures);
    return 1;
  }
  std::puts("CustomOp compute status tests passed");
  return 0;
}
