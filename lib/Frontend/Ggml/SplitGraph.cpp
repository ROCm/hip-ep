/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SplitGraph.cpp - A piece of a ggml graph, as a graph of its own ----===//

#include "hip/Frontend/Ggml/SplitGraph.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace hip::ggml::capture {
namespace {

constexpr int kOpParams = GGML_MAX_OP_PARAMS / sizeof(int32_t);

class DescriptionBuilder {
public:
  DescriptionBuilder(const std::vector<const ggml_tensor *> &nodes,
                     const SplitTensors &tensors)
      : nodes_(nodes.begin(), nodes.end()), tensors_(tensors) {}

  int32_t addSource(const ggml_tensor *t) {
    if (t == nullptr)
      return -1;
    if (auto found = index_.find(t); found != index_.end())
      return found->second;
    if (nodes_.count(t) != 0)
      throw std::runtime_error(std::string("split node '") + t->name +
                               "' is read before it is made");
    // From outside: a view of an outside tensor stays a view of that
    // tensor's own leaf (a cache read through a window), anything else is a
    // leaf.
    if (t->view_src != nullptr && t->op != GGML_OP_NONE)
      return addTensor(t, /*node=*/false, false);
    SplitDescription::Tensor d = tensorMetadata(t);
    d.op = GGML_OP_NONE;
    d.flags = t->flags & GGML_TENSOR_FLAG_INPUT;
    d.weight = tensors_.weights.count(t) != 0;
    d.cache = tensors_.caches.count(t) != 0;
    return push(t, std::move(d));
  }

  /// A view of storage the split does not compute (a cache's window another
  /// backend attends over): ggml points it at that storage, so the split has
  /// nothing to write for it.
  bool aliasesOutside(const ggml_tensor *t) const {
    switch (t->op) {
    case GGML_OP_VIEW:
    case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
      break;
    default:
      return false;
    }
    const ggml_tensor *root = t;
    while (root->view_src != nullptr)
      root = root->view_src;
    return nodes_.count(root) == 0;
  }

  int32_t addTensor(const ggml_tensor *t, bool node, bool output) {
    output = output && !aliasesOutside(t);
    int32_t src[GGML_MAX_SRC];
    for (int i = 0; i < GGML_MAX_SRC; ++i)
      src[i] = t->src[i] == t ? -2 : addSource(t->src[i]);
    const int32_t viewSrc = addSource(t->view_src);
    SplitDescription::Tensor d = tensorMetadata(t);
    d.op = t->op;
    std::memcpy(d.opParams, t->op_params, sizeof(d.opParams));
    d.flags = t->flags & GGML_TENSOR_FLAG_INPUT;
    if (node && output)
      d.flags |= GGML_TENSOR_FLAG_OUTPUT;
    d.output = node && output;
    for (int i = 0; i < GGML_MAX_SRC; ++i)
      d.src[i] = src[i];
    d.viewSrc = viewSrc;
    d.viewOffs = t->view_offs;
    return push(t, std::move(d));
  }

  SplitDescription result;

private:
  SplitDescription::Tensor tensorMetadata(const ggml_tensor *t) const {
    SplitDescription::Tensor d;
    d.name = t->name;
    d.type = t->type;
    for (int i = 0; i < 4; ++i) {
      d.ne[i] = t->ne[i];
      d.nb[i] = t->nb[i];
    }
    return d;
  }

  int32_t push(const ggml_tensor *t, SplitDescription::Tensor d) {
    const int32_t i = static_cast<int32_t>(result.tensors.size());
    // An operation that is its own operand (CPY's destination) names itself.
    for (int32_t &s : d.src)
      if (s == -2)
        s = i;
    result.tensors.push_back(std::move(d));
    result.originals.push_back(t);
    index_.emplace(t, i);
    return i;
  }

  std::unordered_set<const ggml_tensor *> nodes_;
  const SplitTensors &tensors_;
  std::unordered_map<const ggml_tensor *, int32_t> index_;
};

/// FNV-1a, 64 bits: stable across processes and builds, which a cache on
/// disk needs.
struct Fnv {
  uint64_t h = 1469598103934665603ull;
  void bytes(const void *p, size_t n) {
    const auto *b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 1099511628211ull;
    }
  }
  template <typename T> void value(const T &v) { bytes(&v, sizeof(v)); }
};

} // namespace

SplitDescription describeSplit(const SplitNodes &split,
                               const SplitTensors &tensors) {
  DescriptionBuilder builder(split.nodes, tensors);
  for (size_t i = 0; i < split.nodes.size(); ++i)
    builder.addTensor(split.nodes[i], /*node=*/true, split.exported[i]);
  SplitDescription result = std::move(builder.result);
  // A state is updated in the shape it is written in: a cache of several
  // streams ([rows, slots, streams]) is written through one view of all its
  // slots ([rows, slots * streams]), so the leaf takes the shape of that
  // write (its views name their own extents and strides).
  for (size_t i = 0; i < result.tensors.size(); ++i) {
    const SplitDescription::Tensor &write = result.tensors[i];
    if (write.op != GGML_OP_SET_ROWS || write.viewSrc < 0)
      continue;
    SplitDescription::Tensor &state =
        result.tensors[static_cast<size_t>(write.viewSrc)];
    const ggml_tensor *original = result.originals[i];
    const ggml_tensor *storage =
        result.originals[static_cast<size_t>(write.viewSrc)];
    if (state.op != GGML_OP_NONE || write.viewOffs != 0 ||
        ggml_nbytes(original) != ggml_nbytes(storage) ||
        !ggml_is_contiguous(original) || !ggml_is_contiguous(storage))
      continue;
    std::memcpy(state.ne, write.ne, sizeof(state.ne));
    std::memcpy(state.nb, write.nb, sizeof(state.nb));
  }
  return result;
}

nlohmann::json toJson(const SplitDescription &description) {
  nlohmann::json tensors = nlohmann::json::array();
  for (const SplitDescription::Tensor &t : description.tensors) {
    nlohmann::json o;
    o["name"] = t.name;
    o["type"] = t.type;
    o["ne"] = std::vector<int64_t>(t.ne, t.ne + 4);
    o["nb"] = std::vector<size_t>(t.nb, t.nb + 4);
    o["op"] = t.op;
    if (t.op != GGML_OP_NONE) {
      o["op_params"] = std::vector<int32_t>(t.opParams, t.opParams + kOpParams);
      o["src"] = std::vector<int32_t>(t.src, t.src + GGML_MAX_SRC);
    }
    o["flags"] = t.flags;
    if (t.viewSrc >= 0) {
      o["view_src"] = t.viewSrc;
      o["view_offs"] = t.viewOffs;
    }
    if (t.output)
      o["output"] = true;
    if (t.weight)
      o["weight"] = true;
    if (t.cache)
      o["cache"] = true;
    tensors.push_back(std::move(o));
  }
  return nlohmann::json{{"format", "hip-ggml split"},
                        {"version", 1},
                        {"tensors", std::move(tensors)}};
}

std::string signature(const SplitDescription &description) {
  // Two halves of different seeds: 128 bits.
  Fnv a, b;
  b.h ^= 0x9e3779b97f4a7c15ull;
  for (Fnv *f : {&a, &b}) {
    f->bytes("hip-ggml split 1", 16);
    for (const SplitDescription::Tensor &t : description.tensors) {
      f->value(t.type);
      f->bytes(t.ne, sizeof(t.ne));
      f->bytes(t.nb, sizeof(t.nb));
      f->value(t.op);
      if (t.op != GGML_OP_NONE) {
        f->bytes(t.opParams, sizeof(t.opParams));
        f->bytes(t.src, sizeof(t.src));
      }
      f->value(t.flags);
      f->value(t.viewSrc);
      f->value(t.viewOffs);
      f->value(t.output);
      f->value(t.weight);
      f->value(t.cache);
    }
  }
  char text[33];
  std::snprintf(text, sizeof(text), "%016llx%016llx",
                static_cast<unsigned long long>(a.h),
                static_cast<unsigned long long>(b.h));
  return text;
}

std::string dynamicSignature(const SplitDescription &description) {
  SplitDescription withoutSizes = description;
  const std::vector<bool> written = caches(description);
  for (size_t i = 0; i < withoutSizes.tensors.size(); ++i) {
    SplitDescription::Tensor &t = withoutSizes.tensors[i];
    // Which nodes the split hands out is where ggml-alloc put them, which
    // the sizes move; dynamic-shape code hands out what its splits do.
    t.output = false;
    t.flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    if (t.weight || t.cache || written[i])
      continue;
    std::fill(std::begin(t.ne), std::end(t.ne), 0);
    std::fill(std::begin(t.nb), std::end(t.nb), 0);
    t.viewOffs = 0;
    // A view's offset is its parameters too.
    if (t.op == GGML_OP_VIEW)
      std::fill(std::begin(t.opParams), std::end(t.opParams), 0);
  }
  return signature(withoutSizes);
}

/// The caches the split writes: the leaves its SET_ROWS write views of.
std::vector<bool> caches(const SplitDescription &description) {
  std::vector<bool> isCache(description.tensors.size(), false);
  for (const SplitDescription::Tensor &t : description.tensors)
    if (t.op == GGML_OP_SET_ROWS && t.viewSrc >= 0) {
      int32_t root = t.viewSrc;
      while (description.tensors[static_cast<size_t>(root)].viewSrc >= 0)
        root = description.tensors[static_cast<size_t>(root)].viewSrc;
      isCache[static_cast<size_t>(root)] = true;
    }
  return isCache;
}

nlohmann::json toJson(const SymbolSpec &spec) {
  nlohmann::json symbols = nlohmann::json::array();
  for (const SymbolSpec::Symbol &s : spec.symbols) {
    nlohmann::json o{
        {"name", s.name}, {"minimum", s.minimum}, {"maximum", s.maximum}};
    if (!s.product.empty())
      o["product"] = s.product;
    nlohmann::json places = nlohmann::json::array();
    for (const SymbolSpec::Place &p : s.places)
      places.push_back({{"tensor", p.tensor},
                        {"dimension", p.dimension},
                        {"scale", p.scale}});
    if (!places.empty())
      o["places"] = std::move(places);
    symbols.push_back(std::move(o));
  }
  return nlohmann::json{{"symbols", std::move(symbols)}};
}

SymbolSpec symbolSpecFromJson(const nlohmann::json &document) {
  SymbolSpec spec;
  for (const nlohmann::json &o : document.at("symbols")) {
    SymbolSpec::Symbol s;
    s.name = o.at("name").get<std::string>();
    s.minimum = o.at("minimum").get<int64_t>();
    s.maximum = o.at("maximum").get<int64_t>();
    if (o.contains("product"))
      s.product = o.at("product").get<std::vector<std::string>>();
    if (o.contains("places"))
      for (const nlohmann::json &p : o.at("places"))
        s.places.push_back({p.at("tensor").get<int32_t>(),
                            p.at("dimension").get<int>(),
                            p.at("scale").get<int64_t>()});
    spec.symbols.push_back(std::move(s));
  }
  return spec;
}

SplitShapes::SplitShapes(const nlohmann::json &document) {
  for (const nlohmann::json &symbol : document.at("symbols")) {
    symbols_.push_back(symbol.at("name").get<std::string>());
    bounds_.emplace_back(
        symbol.value("minimum", int64_t{1}),
        symbol.value("maximum", std::numeric_limits<int64_t>::max()));
  }
  products_.assign(symbols_.size(), {-1, -1});
  const auto index = [&](const std::string &name) {
    const auto found = std::find(symbols_.begin(), symbols_.end(), name);
    if (found == symbols_.end())
      throw std::runtime_error("shapes name an undeclared symbol '" + name +
                               "'");
    return static_cast<int>(found - symbols_.begin());
  };
  for (size_t i = 0; i < symbols_.size(); ++i)
    if (const auto product = document.at("symbols")[i].find("product");
        product != document.at("symbols")[i].end() && product->size() == 2)
      products_[i] = {index((*product)[0]), index((*product)[1])};
  for (const nlohmann::json &tensor : document.at("tensors")) {
    Shape shape;
    if (!tensor.is_null()) {
      shape.present = true;
      for (int d = 0; d < 4; ++d) {
        shape.ne[d] = parse(tensor.at("ne")[d]);
        shape.nb[d] = parse(tensor.at("nb")[d]);
      }
      shape.viewOffs = parse(tensor.at("view_offs"));
    }
    tensors_.push_back(std::move(shape));
  }
}

SplitShapes::Expr SplitShapes::parse(const nlohmann::json &e) {
  Expr result;
  if (e.is_number()) {
    result.offset = e.get<int64_t>();
    return result;
  }
  result.offset = e.at("offset").get<int64_t>();
  for (const auto &[name, coefficient] : e.at("terms").items()) {
    const auto found = std::find(symbols_.begin(), symbols_.end(), name);
    if (found == symbols_.end())
      throw std::runtime_error("shapes name an undeclared symbol '" + name +
                               "'");
    result.terms.emplace_back(static_cast<int>(found - symbols_.begin()),
                              coefficient.get<int64_t>());
  }
  return result;
}

bool SplitShapes::fits(const SplitDescription &description,
                       std::string *error) const {
  std::vector<int64_t> values;
  return bind(description, values, error);
}

bool SplitShapes::bind(const SplitDescription &description,
                       std::vector<int64_t> &values, std::string *error) const {
  const char *const parts[] = {"extent", "stride", "view offset"};
  const auto refuse = [&](const std::string &reason) {
    if (error)
      *error = reason;
    return false;
  };
  const auto location = [&](size_t i, int part, int d) {
    return "tensor " + std::to_string(i) + " (" + description.tensors[i].name +
           ") " + parts[part] +
           (part == 2 ? std::string() : " " + std::to_string(d));
  };
  if (description.tensors.size() != tensors_.size())
    return refuse("other tensors");
  values.assign(symbols_.size(), -1);
  const auto actual = [&](size_t i, int part, int d) -> int64_t {
    const SplitDescription::Tensor &t = description.tensors[i];
    return part == 0   ? t.ne[d]
           : part == 1 ? static_cast<int64_t>(t.nb[d])
                       : static_cast<int64_t>(t.viewOffs);
  };
  const auto expected = [&](size_t i, int part, int d) -> const Expr & {
    const Shape &s = tensors_[i];
    return part == 0 ? s.ne[d] : part == 1 ? s.nb[d] : s.viewOffs;
  };
  // A dimension of one element steps nowhere: its stride says nothing (past
  // two symbolic extents the derivation keeps the one before).
  const auto unitStride = [&](size_t i, int part, int d) {
    return part == 1 && description.tensors[i].ne[d] == 1;
  };
  // Bind: a symbol alone (times a coefficient) is worth what is described.
  for (size_t i = 0; i < tensors_.size(); ++i) {
    if (!tensors_[i].present)
      continue;
    for (int part = 0; part < 3; ++part)
      for (int d = 0; d < (part == 2 ? 1 : 4); ++d) {
        const Expr &e = expected(i, part, d);
        if (e.terms.size() != 1 || products_[e.terms[0].first].first >= 0 ||
            unitStride(i, part, d))
          continue;
        const auto [symbol, coefficient] = e.terms[0];
        const int64_t rest = actual(i, part, d) - e.offset;
        if (coefficient == 0 || rest % coefficient != 0)
          return refuse(location(i, part, d) +
                        " is no multiple of its symbol's");
        if (values[symbol] < 0)
          values[symbol] = rest / coefficient;
        else if (values[symbol] != rest / coefficient)
          return refuse(location(i, part, d) + " makes " + symbols_[symbol] +
                        " " + std::to_string(rest / coefficient) +
                        ", elsewhere " + std::to_string(values[symbol]));
      }
  }
  for (size_t s = 0; s < symbols_.size(); ++s)
    if (products_[s].first >= 0 && values[products_[s].first] >= 0 &&
        values[products_[s].second] >= 0)
      values[s] = values[products_[s].first] * values[products_[s].second];
  for (size_t s = 0; s < symbols_.size(); ++s)
    if (values[s] >= 0 &&
        (values[s] < bounds_[s].first || values[s] > bounds_[s].second))
      return refuse(symbols_[s] + " " + std::to_string(values[s]) +
                    " is out of its bounds");
  // Check: every expression is what is described.
  for (size_t i = 0; i < tensors_.size(); ++i) {
    if (!tensors_[i].present)
      continue;
    for (int part = 0; part < 3; ++part)
      for (int d = 0; d < (part == 2 ? 1 : 4); ++d) {
        if (unitStride(i, part, d))
          continue;
        const Expr &e = expected(i, part, d);
        int64_t value = e.offset;
        for (const auto &[symbol, coefficient] : e.terms) {
          if (values[symbol] < 0)
            return refuse(symbols_[symbol] + " is bound nowhere");
          value += coefficient * values[symbol];
        }
        if (value != actual(i, part, d))
          return refuse(location(i, part, d) + " is " +
                        std::to_string(actual(i, part, d)) + ", the code's " +
                        std::to_string(value));
      }
  }
  return true;
}

} // namespace hip::ggml::capture
