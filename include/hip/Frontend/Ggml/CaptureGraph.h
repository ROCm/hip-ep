/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- CaptureGraph.h - Observing one ggml split -----------------------===//
//
// Walking the nodes of a split a ggml backend is handed and writing down what
// they are. This is the whole of the GGML side of the conversion, and it needs
// nothing from llama.cpp: a graph described by hand (GgmlTensor.h) is
// observed the same way, which is what lets a test build one and check what
// the capture made of it.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_CAPTUREGRAPH_H
#define HIP_FRONTEND_GGML_CAPTUREGRAPH_H

#include "hip/Frontend/Ggml/CapturedGraph.h"
#include "hip/Frontend/Ggml/GgmlTensor.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace hip::ggml::capture {

/// Walks a split a ggml backend is handed, while its tensors are live: its
/// computed `nodes` in order, and which of them are read outside
/// (`exported`; SplitGraph.h's splitNodes says both). What they read from
/// outside is not walked past: a cache they write is a state, anything else
/// an input of the split (a weight a constant); a view of an outside tensor
/// is a view the split takes. `ids`, when given, gets each value's id.
CapturedGraph
observeSplit(const std::vector<const Tensor *> &nodes,
             const std::vector<bool> &exported, const std::string &name,
             std::unordered_map<const Tensor *, int64_t> *ids = nullptr);

/// What `tensor`'s operation was configured with, read out of its op_params
/// as GgmlOps.td declares - UnsupportedParameters for an operation the capture
/// has no semantics for. Throws Error for an enumerator the dialect does not
/// have.
NodeParameters parametersOf(const Tensor &tensor);

} // namespace hip::ggml::capture

#endif // HIP_FRONTEND_GGML_CAPTUREGRAPH_H
