/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SplitGraph.h - A piece of a ggml graph, as a graph of its own ------===//
//
// What a ggml backend is handed is a split: a run of the scheduler's graph,
// its nodes pointing at tensors other splits made, the model's weights, the
// KV cache, llama.cpp's inputs. libhipep_backend rebuilds it from the
// backend's description (hipep_backend.h) as tensors of its own
// (GgmlTensor.h), with what ggml's tensors do not say of themselves
// (SplitTensors), and the compiler captures it from there (SplitCompiler.h).
//
// The description is the split's identity: its tensors' metadata alone,
// never their data. Two splits of one shape (two layers of a model) describe
// the same graph but for the tensors' names, and `signature` says so; splits
// of one structure at other sizes have one `dynamicSignature`.
//
// Sizes: a split's extents are those of one step. Its free numbers - the
// extents of the tensors it reads and of the views it takes, which the host
// chose - are what may vary; every other extent follows from them by what the
// operations do. Which are one size, and which are the model's (of a weight,
// of a cache), the split's operations say (SplitCompiler.h); `SplitShapes` is
// the guard a step's split is checked against before code compiled with them
// runs it.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_SPLITGRAPH_H
#define HIP_FRONTEND_GGML_SPLITGRAPH_H

#include "hip/Frontend/Ggml/GgmlTensor.h"

#include "llvm/Support/JSON.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hip::ggml::capture {

/// A split's computed nodes, in order, and which of them are read outside
/// it.
struct SplitNodes {
  std::vector<const Tensor *> nodes;
  std::vector<bool> exported;
};

/// What ggml's tensors do not say of themselves: which are the model's
/// weights (of the sizes they were loaded at, whatever the step) and its
/// caches (of the sizes the context made).
struct SplitTensors {
  std::unordered_set<const Tensor *> weights;
  std::unordered_set<const Tensor *> caches;
};

/// A split, described: its tensors in an order every source precedes its
/// user - the leaves (what it reads from outside) and its nodes, in the
/// split's order -, by index.
struct SplitDescription {
  struct Tensor {
    std::string name;
    int32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    size_t nb[4] = {0, 0, 0, 0};
    /// ggml's operation name; "NONE" for a leaf.
    std::string op = "NONE";
    int32_t opParams[kOpParams] = {};
    int32_t flags = 0;
    int32_t src[kMaxSources];
    int32_t viewSrc = -1;
    size_t viewOffs = 0;
    /// A node whose value may be read outside the split.
    bool output = false;
    /// A leaf in a buffer of the model's weights: of the sizes it was
    /// loaded at, whatever the step.
    bool weight = false;
    /// A leaf of memory kept across steps (a cache): of the sizes the
    /// context made.
    bool cache = false;
    Tensor() {
      for (int32_t &s : src)
        s = -1;
    }
  };
  std::vector<Tensor> tensors;
  /// The original tensor each describes.
  std::vector<const hip::ggml::Tensor *> originals;
};

/// Cuts the split out: the sources of its nodes outside it become leaves - a
/// view of an outside tensor a view of that tensor's own leaf -, an exported
/// node an output. Its tensors are numbered sources first: for each node, in
/// order, the tensors it reads not numbered yet (depth first), then the node.
SplitDescription describeSplit(const SplitNodes &split,
                               const SplitTensors &tensors);

/// For a look at it (the cache keeps it beside the code).
llvm::json::Value toJson(const SplitDescription &description);

/// What two descriptions of one compiled graph share: everything but the
/// tensors' names, hashed.
std::string signature(const SplitDescription &description);

/// What descriptions of one graph at other sizes share: the signature
/// without the tensors' extents, strides and view offsets - but for those of
/// the weights and of the caches, which one context keeps (a context of
/// another size has another).
std::string dynamicSignature(const SplitDescription &description);

/// The caches the split writes: the leaves its SET_ROWS write views of.
std::vector<bool> caches(const SplitDescription &description);

/// The symbols a split's dynamic-shape code is compiled with: each `scale`
/// times the symbol at the extent `dimension` of described tensor `tensor` (a
/// leaf, or a view), every other extent the compiler's to derive; and the
/// products of two, which the strides of a tensor of two symbolic extents need.
struct SymbolSpec {
  struct Place {
    int32_t tensor = 0;
    int dimension = 0;
    int64_t scale = 1;
  };
  struct Symbol {
    std::string name;
    int64_t minimum = 1;
    int64_t maximum = 1;
    std::vector<Place> places;
    /// A product: the two symbols' names.
    std::vector<std::string> product;
  };
  std::vector<Symbol> symbols;
  bool empty() const { return symbols.empty(); }
};
llvm::json::Value toJson(const SymbolSpec &spec);
/// Throws when `document` is not a SymbolSpec.
SymbolSpec symbolSpecFromJson(const llvm::json::Value &document);

/// A compile's shapes (shapes.json): each described tensor's extents,
/// strides and view offset as expressions of its symbols, null for one the
/// code does not hold (a view nothing compiled reads).
class SplitShapes {
public:
  /// Throws when `document` is not shapes.json.
  explicit SplitShapes(const llvm::json::Value &document);
  /// Whether `description` is the compiled graph at some sizes: each symbol
  /// bound where an extent is it alone, within its bounds, every expression
  /// then equal to the described number; `error` says what does not.
  bool fits(const SplitDescription &description,
            std::string *error = nullptr) const;

private:
  struct Expr {
    int64_t offset = 0;
    std::vector<std::pair<int, int64_t>> terms;
  };
  struct Shape {
    bool present = false;
    Expr ne[4], nb[4], viewOffs;
  };
  Expr parse(const llvm::json::Value &e);
  /// fits, with the symbols' values.
  bool bind(const SplitDescription &description, std::vector<int64_t> &values,
            std::string *error = nullptr) const;
  std::vector<std::string> symbols_;
  std::vector<std::pair<int64_t, int64_t>> bounds_;
  /// Per symbol: the two factors it is the product of, or -1.
  std::vector<std::pair<int, int>> products_;
  std::vector<Shape> tensors_;
};

} // namespace hip::ggml::capture

#endif // HIP_FRONTEND_GGML_SPLITGRAPH_H
