/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- CaptureGraph.cpp - Observing one ggml split ----------------------===//
//
// GGML says what a graph computes with pointers, a fixed operand array and
// sixty-four bytes of configuration. This says the same thing with numbered
// values, named configuration, explicit storage and an explicit ABI -- and
// nothing else. What varies with the request, and what the operations mean,
// are decided later and elsewhere.
//
//===----------------------------------------------------------------------===//

#include "hip/Frontend/Ggml/CaptureGraph.h"

#include "hip/Frontend/Ggml/Error.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hip::ggml::capture {
namespace {

using hip::ggml::fail;

struct WalkedGraph {
  std::vector<const Tensor *> values;
  std::vector<const Tensor *> nodes;
  std::unordered_map<const Tensor *, int32_t> ids;
  std::unordered_set<const Tensor *> visited;
  /// What the split reads from outside.
  std::unordered_set<const Tensor *> external;

  bool isExternal(const Tensor *tensor) const {
    return external.count(tensor) != 0;
  }
};

/// A split's values: its nodes, the views of outside tensors they read, and
/// what those read from outside - not walked past.
void collectSplitValue(WalkedGraph &graph,
                       const std::unordered_set<const Tensor *> &inside,
                       const Tensor *tensor) {
  if (tensor == nullptr || graph.visited.count(tensor) != 0)
    return;
  graph.visited.insert(tensor);
  const bool node = inside.count(tensor) != 0 ||
                    (tensor->viewSrc != nullptr && isView(*tensor));
  if (node) {
    if (tensor->viewSrc != nullptr)
      collectSplitValue(graph, inside, tensor->viewSrc);
    for (const Tensor *source : tensor->src)
      collectSplitValue(graph, inside, source);
  } else {
    graph.external.insert(tensor);
  }
  graph.ids.emplace(tensor, static_cast<int32_t>(graph.values.size()));
  graph.values.push_back(tensor);
  if (node)
    graph.nodes.push_back(tensor);
}

const Tensor *storageRoot(const Tensor *tensor) {
  std::unordered_set<const Tensor *> visited;
  while (tensor != nullptr && tensor->viewSrc != nullptr) {
    if (!visited.insert(tensor).second) {
      fail("cycle in GGML tensor view chain");
    }
    tensor = tensor->viewSrc;
  }
  return tensor;
}

std::optional<int> writeDestinationSlot(const Tensor &tensor,
                                        const Tensor *outputRoot) {
  if (tensor.op == "SET_ROWS" && tensor.src[2] != nullptr &&
      storageRoot(tensor.src[2]) == outputRoot) {
    return 2;
  }
  for (int slot = 0; slot < kMaxSources; ++slot) {
    if (tensor.src[slot] != nullptr && tensor.src[slot] != &tensor &&
        storageRoot(tensor.src[slot]) == outputRoot) {
      return slot;
    }
  }
  return std::nullopt;
}

/// The four GGML extents or byte strides, read out of the C arrays.
template <typename Element> Extents extentsOf(const Element *values) {
  Extents extents{};
  for (unsigned index = 0; index < kGgmlRank; ++index)
    extents[index] = static_cast<int64_t>(values[index]);
  return extents;
}

// GGML's enumerations and the dialect's agree on names rather than on
// numbers, and the correspondence is declared beside each dialect value.
#include "hip/Dialect/Ggml/IR/GgmlEnumNames.inc"

NodeParameters typedParameters(const Tensor &tensor) {
  const auto i32 = [&](int index) {
    return static_cast<int64_t>(opParamI32(tensor, index));
  };
  const auto f32 = [&](int index) { return opParamF32(tensor, index); };
#include "hip/Dialect/Ggml/IR/GgmlDecode.inc"
  return UnsupportedParameters{};
}

std::string
classifyExternal(const Tensor &tensor,
                 const std::unordered_set<const Tensor *> &stateRoots) {
  if (stateRoots.count(&tensor) != 0) {
    return "state";
  }
  if ((tensor.flags & kFlagInput) != 0) {
    return "input";
  }
  return "constant";
}

} // namespace

CapturedGraph observeSplit(const std::vector<const Tensor *> &nodes,
                           const std::vector<bool> &exported,
                           const std::string &name,
                           std::unordered_map<const Tensor *, int64_t> *ids) {
  WalkedGraph graph;
  const std::unordered_set<const Tensor *> inside(nodes.begin(), nodes.end());
  for (const Tensor *node : nodes)
    collectSplitValue(graph, inside, node);
  std::unordered_set<const Tensor *> outputs;
  for (size_t i = 0; i < nodes.size(); ++i)
    if (exported[i])
      outputs.insert(nodes[i]);
  if (ids != nullptr)
    for (const auto &[tensor, id] : graph.ids)
      ids->emplace(tensor, id);

  // Where a value's storage begins: its view chain's root, no further than
  // what the split reads from outside.
  const auto rootOf = [&](const Tensor *tensor) {
    while (tensor->viewSrc != nullptr && !graph.isExternal(tensor))
      tensor = tensor->viewSrc;
    return tensor;
  };
  std::unordered_set<const Tensor *> mutatingNodes;
  std::unordered_map<const Tensor *, std::vector<int32_t>> stateWrites;
  std::unordered_set<const Tensor *> stateRoots;
  for (const Tensor *tensor : graph.nodes) {
    const Tensor *root = rootOf(tensor);
    if (root == tensor || isView(*tensor)) {
      continue;
    }
    const bool aliasesOperandStorage =
        std::any_of(tensor->src, tensor->src + kMaxSources,
                    [root, &rootOf](const Tensor *operand) {
                      return operand != nullptr && rootOf(operand) == root;
                    });
    if (!aliasesOperandStorage) {
      continue;
    }
    mutatingNodes.insert(tensor);
    // A write into storage the split reads from outside is persistent graph
    // state. Input flags and tensor names are only annotations; neither
    // participates in this classification.
    if (!graph.isExternal(root)) {
      continue;
    }
    stateRoots.insert(root);
    stateWrites[root].push_back(graph.ids.at(tensor));
  }

  std::unordered_map<const Tensor *, std::string> roles;
  for (const Tensor *tensor : graph.values) {
    roles.emplace(tensor, graph.isExternal(tensor)
                              ? classifyExternal(*tensor, stateRoots)
                              : "intermediate");
  }

  CapturedGraph observed;
  observed.name = name;
  for (const Tensor *tensor : graph.values) {
    const Tensor *root = rootOf(tensor);
    const int64_t byteCount = static_cast<int64_t>(nbytes(*tensor));
    const int64_t storageByteLength = static_cast<int64_t>(nbytes(*root));
    // A view a split reads from outside is storage of its own.
    const int64_t viewOffset =
        root == tensor ? 0 : static_cast<int64_t>(tensor->viewOffs);
    if (viewOffset > storageByteLength ||
        byteCount > storageByteLength - viewOffset) {
      fail("tensor view exceeds its storage root: " + tensor->name);
    }
    // GGML normalizes a chain of views to its root, so `viewSrc` is the
    // storage the value already names and is not recorded again.
    if (tensor->viewSrc != nullptr && root != tensor &&
        tensor->viewSrc != root) {
      fail("tensor view is not rooted at its storage: " + tensor->name);
    }
    Value value;
    value.id = graph.ids.at(tensor);
    value.name = tensor->name;
    value.ggmlType = typeName(tensor->type).value_or("");
    value.ne = extentsOf(tensor->ne);
    value.nb = extentsOf(tensor->nb);
    value.byteCount = byteCount;
    value.storageId = graph.ids.at(root);
    value.storageOffset = viewOffset;
    observed.values.push_back(std::move(value));
  }

  for (const Tensor *tensor : graph.nodes) {
    const Tensor *outputRoot = rootOf(tensor);
    const std::optional<int> destinationSlot =
        mutatingNodes.count(tensor) != 0
            ? writeDestinationSlot(*tensor, outputRoot)
            : std::nullopt;

    Node node;
    node.id = graph.ids.at(tensor);
    node.name = tensor->name;
    node.operation = tensor->op;
    node.parameters = typedParameters(*tensor);
    if (destinationSlot)
      node.writeSlot = *destinationSlot;
    if (std::holds_alternative<UnsupportedParameters>(node.parameters))
      observed.untypedNodes.push_back({node.id, node.operation});

    for (int slot = 0; slot < kMaxSources; ++slot) {
      const Tensor *sourceTensor = tensor->src[slot];
      if (sourceTensor == nullptr) {
        continue;
      }
      node.operands.push_back({slot, graph.ids.at(sourceTensor)});
    }
    observed.nodes.push_back(std::move(node));
  }

  for (const Tensor *tensor : graph.values) {
    const std::string &role = roles.at(tensor);
    if (role != "input" && role != "state" && role != "constant") {
      continue;
    }
    Input input;
    input.id = graph.ids.at(tensor);
    input.name = tensor->name;
    input.role = role;
    observed.inputs.push_back(std::move(input));
  }

  for (const Tensor *tensor : graph.nodes) {
    const bool isExported = outputs.count(tensor) != 0;
    const Tensor *root = rootOf(tensor);
    const auto writes = stateWrites.find(root);
    const bool stateUpdate = writes != stateWrites.end() &&
                             !writes->second.empty() &&
                             writes->second.back() == graph.ids.at(tensor);
    const bool stateStorage = stateRoots.count(root) != 0;
    if ((stateStorage && !stateUpdate) || (!stateStorage && !isExported)) {
      continue;
    }
    Output output;
    output.id = graph.ids.at(tensor);
    output.name = tensor->name;
    output.kind = stateUpdate ? "state" : "model";
    observed.outputs.push_back(std::move(output));
  }

  for (const Tensor *tensor : graph.values) {
    if (roles.at(tensor) != "state" || rootOf(tensor) != tensor) {
      continue;
    }
    StateBinding binding;
    binding.storageId = graph.ids.at(tensor);
    const auto found = stateWrites.find(tensor);
    if (found != stateWrites.end())
      binding.writes.assign(found->second.begin(), found->second.end());
    observed.stateBindings.push_back(std::move(binding));
  }

  return observed;
}

} // namespace hip::ggml::capture
