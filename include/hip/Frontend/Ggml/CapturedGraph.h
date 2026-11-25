/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- CapturedGraph.h - One observation of a ggml split ----------------===//
//
// What the capture saw: the tensors, the operations between them, and which
// tensors the runtime binds. This is the capture's own vocabulary, not a
// serialization of it -- a field named here exists, has a type, and a reader
// that asks for the wrong one does not compile.
//
// A field is here because nothing else here says it. What one field already
// determines is not spelled a second time: a count is the length of the list
// it counts, a node's result is the node, and a write's extent is the extent
// of the value it produces. The exception is what only the type's block
// layout could recompute -- how many bytes a quantized tensor occupies --
// which the capture works out (GgmlTensor.h) and keeps.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_CAPTUREDGRAPH_H
#define HIP_FRONTEND_GGML_CAPTUREDGRAPH_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace hip::ggml::capture {

/// GGML tensors are rank four, always, with unused dimensions set to one.
inline constexpr unsigned kGgmlRank = 4;

using Extents = std::array<int64_t, kGgmlRank>;

struct Value {
  int64_t id = 0;
  std::string name;
  std::string ggmlType;
  Extents ne{};
  Extents nb{};
  /// The bytes these extents and strides occupy, which do not follow from
  /// `ne` and `nb` alone, because a quantized type packs several elements into
  /// one block.
  int64_t byteCount = 0;
  /// The buffer this value occupies and where in it it begins. A value that
  /// owns its buffer names itself.
  int64_t storageId = 0;
  int64_t storageOffset = 0;
};

struct Operand {
  int64_t slot = 0;
  int64_t value = 0;
};

/// An operation the capture has no typed semantics for. Its presence stops the
/// build, so this exists to say which one rather than to be tolerated.
struct UnsupportedParameters {
  bool operator==(const UnsupportedParameters &) const { return true; }
};

// Declared beside the operations they configure, in `GgmlOps.td`.
#include "hip/Dialect/Ggml/IR/GgmlParameters.inc"

struct Node {
  /// A node is its result: the value it produces carries the same id, and the
  /// buffer it writes is the one that value names.
  int64_t id = 0;
  std::string name;
  /// The GGML operation's own name, as `ggml_op_name` gives it.
  std::string operation;
  std::vector<Operand> operands;
  /// The operand slot this node writes through, when the operation hands back
  /// one of its operands' buffers rather than a buffer of its own. Which slot
  /// that is, is GGML's rule for the operation and not something the operand
  /// list says, so it is recorded. Everything else about the write -- what it
  /// covers, and whether the buffer outlives the step -- is the result value
  /// and the buffer's role.
  std::optional<int64_t> writeSlot;
  NodeParameters parameters;
};

struct Input {
  int64_t id = 0;
  std::string name;
  /// The role the capture classified it as: `input`, `state`, or `constant`.
  /// Whether the runtime binds it once and carries it forward, or hands it a
  /// fresh buffer each step, is whether that role is `state`.
  std::string role;
};

struct Output {
  int64_t id = 0;
  std::string name;
  /// `state` or `model`.
  std::string kind;
};

/// A buffer the graph carries across steps: the value the runtime binds, and
/// the nodes that leave it holding something new. The binding's name is that
/// value's, and the state the step ends with is the last of those writes.
struct StateBinding {
  int64_t storageId = 0;
  std::vector<int64_t> writes;
};

/// An operation the capture could not describe. Its presence stops the build,
/// so this exists to say which one rather than to be tolerated.
struct UntypedNode {
  int64_t id = 0;
  std::string operation;
};

struct CapturedGraph {
  std::string name;
  std::vector<UntypedNode> untypedNodes;
  std::vector<Input> inputs;
  std::vector<Output> outputs;
  std::vector<StateBinding> stateBindings;
  std::vector<Value> values;
  /// In execution order, which is the order GGML listed them in.
  std::vector<Node> nodes;
};

} // namespace hip::ggml::capture

#endif // HIP_FRONTEND_GGML_CAPTUREDGRAPH_H
