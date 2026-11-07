/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// corelib_model.h for builds without the RyzenAI CoreLib headers (or mock
// builds): every model is refused, so graphs compile as usual.

#include "corelib_model.h"

namespace hipdnn_ep::corelib {

std::shared_ptr<ModelSession> create_model_session(const std::string &,
                                                   const std::string &, int64_t,
                                                   std::string *why) {
  *why = "hip-ep was built without RyzenAI CoreLib";
  return nullptr;
}

int64_t vocab_size(const ModelSession &) { return 0; }

bool forward(ModelSession &, const int32_t *, size_t, int64_t, float *,
             std::string *why) {
  *why = "hip-ep was built without RyzenAI CoreLib";
  return false;
}

} // namespace hipdnn_ep::corelib
