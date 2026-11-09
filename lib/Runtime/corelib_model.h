/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- corelib_model.h - Whole models through RyzenAI CoreLib -------------===//
//
// A whole decoder model handed to CoreLib's model API
// (<ryzenai/corelib/model.h>) instead of being compiled: CoreLib reads the
// ONNX file's initializers itself, keeps the KV cache and runs every layer.
// Nothing here names a model: the models tried are the ones CoreLib ships,
// and the inputs passed are the ones its `forward` asks for, so a CoreLib
// with new or updated models needs no change here. Native code in
// hipdnn-ep-corelib; the stub build always refuses. See
// docs/design/ryzenai-corelib-model.md.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_CORELIB_MODEL_H
#define HIPDNN_EP_CORELIB_MODEL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_ep::corelib {

class ModelSession;

// What the graph offers a model: its input names, the width of its logits
// (-1 when not static), and the model type onnxruntime-genai records for it
// (genai_config.json's model.type), if any.
struct GraphSignature {
  std::vector<std::string> inputs;
  int64_t vocab_size = -1;
  std::string model_type;
};

// Creates a session of the first CoreLib model that takes the ONNX file at
// `path` and fits `graph`: its vocabulary is the logits' width, and every
// input its `forward` requires is a graph input (or `position`, which the
// caller supplies). `arch` names the one model to try, as recorded when a
// session was first created; empty tries each model CoreLib ships: first the
// one named like `graph.model_type`, then the rest in name order. The session
// holds `max_sequence_length` positions (0: the model's own). Null with `*why`
// set when CoreLib is not built in or not loadable, there is no usable NPU, or
// no model takes the file.
std::shared_ptr<ModelSession> create_model_session(const std::string &arch,
                                                   const std::string &path,
                                                   int64_t max_sequence_length,
                                                   const GraphSignature &graph,
                                                   std::string *why);

// The model's name (CoreLib's, e.g. "qwen3") and vocabulary size.
const std::string &architecture(const ModelSession &session);
int64_t vocab_size(const ModelSession &session);

// The graph inputs `forward` takes, by name, besides `position`.
const std::vector<std::string> &forward_inputs(const ModelSession &session);

// A graph input in host memory, without its batch dimension.
enum class ElementType { Int32, Int64, Float16, Float32 };
struct HostTensor {
  std::string name;
  ElementType type;
  std::vector<int64_t> dims;
  const void *data;
};

// Runs `forward` on `inputs` (forward_inputs' tensors) from `position` and
// writes the last row's logits (vocab_size floats). Inputs are converted to a
// type the model accepts. False with `*why` set on failure.
bool forward(ModelSession &session, const std::vector<HostTensor> &inputs,
             int64_t position, float *logits, std::string *why);

// Hands a session created while compiling to the op that runs it, keyed by
// the fused node's id; take returns null when none was left (e.g. a session
// loaded from an EPContext model, which creates its own).
void put_session(const std::string &key, std::shared_ptr<ModelSession> s);
std::shared_ptr<ModelSession> take_session(const std::string &key);

} // namespace hipdnn_ep::corelib

#endif // HIPDNN_EP_CORELIB_MODEL_H
