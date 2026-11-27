/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hipep_backend.h's functions: a context is an Engine (Engine.h).

#include "hipep_backend.h"

#include "Engine.h"
#include "Ops.h"

#include <exception>
#include <memory>
#include <string>

struct hipep_backend_ctx {
  std::unique_ptr<hipep_backend::Engine> engine;
};

namespace {

thread_local std::string lastError;

template <typename T> T *failed(const std::string &error) {
  lastError = error;
  return nullptr;
}

void tell(int *status, hipep_backend_status value) {
  if (status != nullptr)
    *status = value;
}

} // namespace

extern "C" {

HIPEP_BACKEND_API int hipep_backend_abi_version(void) {
  return HIPEP_BACKEND_ABI_VERSION;
}

HIPEP_BACKEND_API hipep_backend_ctx *hipep_backend_open(int device,
                                                        const char *cache_dir) {
  try {
    auto ctx = std::make_unique<hipep_backend_ctx>();
    ctx->engine = std::make_unique<hipep_backend::Engine>(device, cache_dir);
    lastError.clear();
    return ctx.release();
  } catch (const std::exception &e) {
    return failed<hipep_backend_ctx>(e.what());
  }
}

HIPEP_BACKEND_API int
hipep_backend_supports(hipep_backend_ctx *ctx,
                       const hipep_backend_split_desc *node) {
  return ctx != nullptr && node != nullptr && hipep_backend::supports(*node)
             ? 1
             : 0;
}

HIPEP_BACKEND_API hipep_backend_code *
hipep_backend_find(hipep_backend_ctx *ctx,
                   const hipep_backend_split_desc *split, int *status) {
  if (ctx == nullptr || split == nullptr) {
    tell(status, HIPEP_BACKEND_STATUS_INVALID);
    return failed<hipep_backend_code>("no context or split");
  }
  hipep_backend_status found = HIPEP_BACKEND_STATUS_INVALID;
  std::string error;
  hipep_backend_code *code = ctx->engine->find(*split, found, error);
  tell(status, found);
  lastError = code ? "" : error;
  return code;
}

HIPEP_BACKEND_API hipep_backend_code *
hipep_backend_compile(hipep_backend_ctx *ctx,
                      const hipep_backend_split_desc *split, int wait,
                      int *status) {
  if (ctx == nullptr || split == nullptr) {
    tell(status, HIPEP_BACKEND_STATUS_INVALID);
    return failed<hipep_backend_code>("no context or split");
  }
  hipep_backend_status found = HIPEP_BACKEND_STATUS_INVALID;
  std::string error;
  hipep_backend_code *code =
      ctx->engine->compile(*split, wait != 0, found, error);
  tell(status, found);
  lastError = code ? "" : error;
  return code;
}

HIPEP_BACKEND_API int hipep_backend_run(hipep_backend_code *code,
                                        void *const *tensor_data,
                                        void *stream) {
  if (code == nullptr || tensor_data == nullptr || stream != nullptr) {
    lastError = "no code or tensors, or a stream (reserved)";
    return 1;
  }
  std::string error;
  if (!code->engine->run(*code, tensor_data, error)) {
    lastError = error;
    return 1;
  }
  return 0;
}

HIPEP_BACKEND_API void hipep_backend_close(hipep_backend_ctx *ctx) {
  delete ctx;
}

HIPEP_BACKEND_API const char *hipep_backend_last_error(hipep_backend_ctx *) {
  return lastError.c_str();
}

} // extern "C"
