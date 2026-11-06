/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- corelib_model.h - Whole models through RyzenAI CoreLib -------------===//
//
// A whole decoder model handed to CoreLib's model API
// (<ryzenai/corelib/model.h>) instead of being compiled: CoreLib reads the
// ONNX file's initializers itself, keeps the KV cache and runs every layer.
// Native code in hipdnn-ep-corelib; the stub build always refuses. See
// docs/design/ryzenai-corelib-model.md.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_CORELIB_MODEL_H
#define HIPDNN_EP_CORELIB_MODEL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace hipdnn_ep::corelib {

class ModelSession;

// Creates a CoreLib model of architecture `arch` (corelib's name, e.g.
// "qwen3") from the ONNX file at `path`, and a session holding
// `max_sequence_length` positions (0: the model's own). Null with `*why` set
// when CoreLib is not built in or not loadable, there is no usable AIE4 NPU,
// or CoreLib refuses the file.
std::shared_ptr<ModelSession> create_model_session(const std::string &arch,
                                                   const std::string &path,
                                                   int64_t max_sequence_length,
                                                   std::string *why);

// The model's vocabulary size: the width of the logits forward writes.
int64_t vocab_size(const ModelSession &session);

// Runs `rows` tokens starting at `position` and writes the last row's logits
// (vocab_size floats). False with `*why` set on failure.
bool forward(ModelSession &session, const int32_t *tokens, size_t rows,
             int64_t position, float *logits, std::string *why);

// Hands a session created while compiling to the op that runs it, keyed by
// the fused node's id; take returns null when none was left (e.g. a session
// loaded from an EPContext model, which creates its own).
void put_session(const std::string &key, std::shared_ptr<ModelSession> s);
std::shared_ptr<ModelSession> take_session(const std::string &key);

} // namespace hipdnn_ep::corelib

#endif // HIPDNN_EP_CORELIB_MODEL_H
