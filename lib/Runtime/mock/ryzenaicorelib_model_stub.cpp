/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// corelib_model.h for builds without the RyzenAI CoreLib headers (or mock
// builds): every model is refused, so graphs compile as usual.

#include "corelib_model.h"

namespace hipdnn_ep::corelib {

namespace {
const std::string kNone;
const std::vector<std::string> kNoInputs;
} // namespace

std::shared_ptr<ModelSession> create_model_session(const std::string &,
                                                   const std::string &, int64_t,
                                                   const GraphSignature &,
                                                   std::string *why) {
  *why = "hip-ep was built without RyzenAI CoreLib";
  return nullptr;
}

const std::string &architecture(const ModelSession &) { return kNone; }

int64_t vocab_size(const ModelSession &) { return 0; }

const std::vector<std::string> &forward_inputs(const ModelSession &) {
  return kNoInputs;
}

bool forward(ModelSession &, const std::vector<HostTensor> &, int64_t, float *,
             std::string *why) {
  *why = "hip-ep was built without RyzenAI CoreLib";
  return false;
}

} // namespace hipdnn_ep::corelib
