/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SplitGraph.cpp - A piece of a ggml graph, as a graph of its own ----===//

#include "hip/Frontend/Ggml/SplitGraph.h"

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

class DescriptionBuilder {
public:
  DescriptionBuilder(const std::vector<const Tensor *> &nodes,
                     const SplitTensors &tensors)
      : nodes_(nodes.begin(), nodes.end()), tensors_(tensors) {}

  int32_t addSource(const Tensor *t) {
    if (t == nullptr)
      return -1;
    if (auto found = index_.find(t); found != index_.end())
      return found->second;
    if (nodes_.count(t) != 0)
      throw std::runtime_error("split node '" + t->name +
                               "' is read before it is made");
    // From outside: a view of an outside tensor stays a view of that
    // tensor's own leaf (a cache read through a window), anything else is a
    // leaf.
    if (t->viewSrc != nullptr && t->op != "NONE")
      return addTensor(t, /*node=*/false, false);
    SplitDescription::Tensor d = tensorMetadata(t);
    d.op = "NONE";
    d.flags = t->flags & kFlagInput;
    d.weight = tensors_.weights.count(t) != 0;
    d.cache = tensors_.caches.count(t) != 0;
    return push(t, std::move(d));
  }

  /// A view of storage the split does not compute (a cache's window another
  /// backend attends over): ggml points it at that storage, so the split has
  /// nothing to write for it.
  bool aliasesOutside(const Tensor *t) const {
    if (!isView(*t))
      return false;
    const Tensor *root = t;
    while (root->viewSrc != nullptr)
      root = root->viewSrc;
    return nodes_.count(root) == 0;
  }

  int32_t addTensor(const Tensor *t, bool node, bool output) {
    output = output && !aliasesOutside(t);
    int32_t src[kMaxSources];
    for (int i = 0; i < kMaxSources; ++i)
      src[i] = t->src[i] == t ? -2 : addSource(t->src[i]);
    const int32_t viewSrc = addSource(t->viewSrc);
    SplitDescription::Tensor d = tensorMetadata(t);
    d.op = t->op;
    std::memcpy(d.opParams, t->opParams, sizeof(d.opParams));
    d.flags = t->flags & kFlagInput;
    if (node && output)
      d.flags |= kFlagOutput;
    d.output = node && output;
    for (int i = 0; i < kMaxSources; ++i)
      d.src[i] = src[i];
    d.viewSrc = viewSrc;
    d.viewOffs = t->viewOffs;
    return push(t, std::move(d));
  }

  SplitDescription result;

private:
  SplitDescription::Tensor tensorMetadata(const Tensor *t) const {
    SplitDescription::Tensor d;
    d.name = t->name;
    d.type = t->type;
    for (int i = 0; i < 4; ++i) {
      d.ne[i] = t->ne[i];
      d.nb[i] = t->nb[i];
    }
    return d;
  }

  int32_t push(const Tensor *t, SplitDescription::Tensor d) {
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

  std::unordered_set<const Tensor *> nodes_;
  const SplitTensors &tensors_;
  std::unordered_map<const Tensor *, int32_t> index_;
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
    if (write.op != "SET_ROWS" || write.viewSrc < 0)
      continue;
    SplitDescription::Tensor &state =
        result.tensors[static_cast<size_t>(write.viewSrc)];
    const Tensor *original = result.originals[i];
    const Tensor *storage =
        result.originals[static_cast<size_t>(write.viewSrc)];
    if (state.op != "NONE" || write.viewOffs != 0 ||
        nbytes(*original) != nbytes(*storage) || !isContiguous(*original) ||
        !isContiguous(*storage))
      continue;
    std::memcpy(state.ne, write.ne, sizeof(state.ne));
    std::memcpy(state.nb, write.nb, sizeof(state.nb));
  }
  return result;
}

namespace {

template <typename T> llvm::json::Array arrayOf(const T *values, size_t n) {
  llvm::json::Array array;
  for (size_t i = 0; i < n; ++i)
    array.push_back(static_cast<int64_t>(values[i]));
  return array;
}

/// The integer `object` has at `key`; throws when it has none.
int64_t integerAt(const llvm::json::Object &object, llvm::StringRef key) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    throw std::runtime_error("no integer '" + key.str() + "'");
  return *value;
}

const llvm::json::Object &objectOf(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (object == nullptr)
    throw std::runtime_error("not an object");
  return *object;
}

const llvm::json::Array &arrayAt(const llvm::json::Object &object,
                                 llvm::StringRef key) {
  const llvm::json::Array *array = object.getArray(key);
  if (array == nullptr)
    throw std::runtime_error("no array '" + key.str() + "'");
  return *array;
}

std::string stringAt(const llvm::json::Object &object, llvm::StringRef key) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    throw std::runtime_error("no string '" + key.str() + "'");
  return value->str();
}

} // namespace

llvm::json::Value toJson(const SplitDescription &description) {
  llvm::json::Array tensors;
  for (const SplitDescription::Tensor &t : description.tensors) {
    llvm::json::Object o;
    o["name"] = t.name;
    o["type"] = t.type;
    o["ne"] = arrayOf(t.ne, 4);
    o["nb"] = arrayOf(t.nb, 4);
    o["op"] = t.op;
    if (t.op != "NONE") {
      o["op_params"] = arrayOf(t.opParams, kOpParams);
      o["src"] = arrayOf(t.src, kMaxSources);
    }
    o["flags"] = t.flags;
    if (t.viewSrc >= 0) {
      o["view_src"] = t.viewSrc;
      o["view_offs"] = static_cast<int64_t>(t.viewOffs);
    }
    if (t.output)
      o["output"] = true;
    if (t.weight)
      o["weight"] = true;
    if (t.cache)
      o["cache"] = true;
    tensors.push_back(std::move(o));
  }
  return llvm::json::Object{{"format", "hip-ggml split"},
                            {"version", 2},
                            {"tensors", std::move(tensors)}};
}

std::string signature(const SplitDescription &description) {
  // Two halves of different seeds: 128 bits.
  Fnv a, b;
  b.h ^= 0x9e3779b97f4a7c15ull;
  for (Fnv *f : {&a, &b}) {
    f->bytes("hip-ggml split 2", 16);
    for (const SplitDescription::Tensor &t : description.tensors) {
      f->value(t.type);
      f->bytes(t.ne, sizeof(t.ne));
      f->bytes(t.nb, sizeof(t.nb));
      f->value(t.op.size());
      f->bytes(t.op.data(), t.op.size());
      if (t.op != "NONE") {
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
    t.flags &= ~kFlagOutput;
    if (t.weight || t.cache || written[i])
      continue;
    std::fill(std::begin(t.ne), std::end(t.ne), 0);
    std::fill(std::begin(t.nb), std::end(t.nb), 0);
    t.viewOffs = 0;
    // A view's offset is its parameters too.
    if (t.op == "VIEW")
      std::fill(std::begin(t.opParams), std::end(t.opParams), 0);
  }
  return signature(withoutSizes);
}

/// The caches the split writes: the leaves its SET_ROWS write views of.
std::vector<bool> caches(const SplitDescription &description) {
  std::vector<bool> isCache(description.tensors.size(), false);
  for (const SplitDescription::Tensor &t : description.tensors)
    if (t.op == "SET_ROWS" && t.viewSrc >= 0) {
      int32_t root = t.viewSrc;
      while (description.tensors[static_cast<size_t>(root)].viewSrc >= 0)
        root = description.tensors[static_cast<size_t>(root)].viewSrc;
      isCache[static_cast<size_t>(root)] = true;
    }
  return isCache;
}

llvm::json::Value toJson(const SymbolSpec &spec) {
  llvm::json::Array symbols;
  for (const SymbolSpec::Symbol &s : spec.symbols) {
    llvm::json::Object o{
        {"name", s.name}, {"minimum", s.minimum}, {"maximum", s.maximum}};
    if (!s.product.empty()) {
      llvm::json::Array product;
      for (const std::string &factor : s.product)
        product.push_back(factor);
      o["product"] = std::move(product);
    }
    llvm::json::Array places;
    for (const SymbolSpec::Place &p : s.places)
      places.push_back(llvm::json::Object{{"tensor", p.tensor},
                                          {"dimension", p.dimension},
                                          {"scale", p.scale}});
    if (!places.empty())
      o["places"] = std::move(places);
    symbols.push_back(std::move(o));
  }
  return llvm::json::Object{{"symbols", std::move(symbols)}};
}

SymbolSpec symbolSpecFromJson(const llvm::json::Value &document) {
  SymbolSpec spec;
  for (const llvm::json::Value &value :
       arrayAt(objectOf(document), "symbols")) {
    const llvm::json::Object &o = objectOf(value);
    SymbolSpec::Symbol s;
    s.name = stringAt(o, "name");
    s.minimum = integerAt(o, "minimum");
    s.maximum = integerAt(o, "maximum");
    if (const llvm::json::Array *product = o.getArray("product"))
      for (const llvm::json::Value &factor : *product)
        s.product.push_back(factor.getAsString().value_or("").str());
    if (const llvm::json::Array *places = o.getArray("places"))
      for (const llvm::json::Value &place : *places) {
        const llvm::json::Object &p = objectOf(place);
        s.places.push_back({static_cast<int32_t>(integerAt(p, "tensor")),
                            static_cast<int>(integerAt(p, "dimension")),
                            integerAt(p, "scale")});
      }
    spec.symbols.push_back(std::move(s));
  }
  return spec;
}

SplitShapes::SplitShapes(const llvm::json::Value &document) {
  const llvm::json::Object &root = objectOf(document);
  const llvm::json::Array &symbols = arrayAt(root, "symbols");
  for (const llvm::json::Value &value : symbols) {
    const llvm::json::Object &symbol = objectOf(value);
    symbols_.push_back(stringAt(symbol, "name"));
    bounds_.emplace_back(symbol.getInteger("minimum").value_or(1),
                         symbol.getInteger("maximum").value_or(
                             std::numeric_limits<int64_t>::max()));
  }
  products_.assign(symbols_.size(), {-1, -1});
  const auto index = [&](llvm::StringRef name) {
    const auto found = std::find(symbols_.begin(), symbols_.end(), name);
    if (found == symbols_.end())
      throw std::runtime_error("shapes name an undeclared symbol '" +
                               name.str() + "'");
    return static_cast<int>(found - symbols_.begin());
  };
  for (size_t i = 0; i < symbols_.size(); ++i)
    if (const llvm::json::Array *product =
            objectOf(symbols[i]).getArray("product");
        product != nullptr && product->size() == 2)
      products_[i] = {index((*product)[0].getAsString().value_or("")),
                      index((*product)[1].getAsString().value_or(""))};
  for (const llvm::json::Value &tensor : arrayAt(root, "tensors")) {
    Shape shape;
    if (tensor.kind() != llvm::json::Value::Null) {
      const llvm::json::Object &t = objectOf(tensor);
      shape.present = true;
      const llvm::json::Array &ne = arrayAt(t, "ne"), &nb = arrayAt(t, "nb");
      if (ne.size() != 4 || nb.size() != 4)
        throw std::runtime_error("a tensor of other than four extents");
      for (int d = 0; d < 4; ++d) {
        shape.ne[d] = parse(ne[d]);
        shape.nb[d] = parse(nb[d]);
      }
      const llvm::json::Value *viewOffs = t.get("view_offs");
      if (viewOffs == nullptr)
        throw std::runtime_error("a tensor of no view offset");
      shape.viewOffs = parse(*viewOffs);
    }
    tensors_.push_back(std::move(shape));
  }
}

SplitShapes::Expr SplitShapes::parse(const llvm::json::Value &e) {
  Expr result;
  if (std::optional<int64_t> number = e.getAsInteger()) {
    result.offset = *number;
    return result;
  }
  const llvm::json::Object &o = objectOf(e);
  result.offset = integerAt(o, "offset");
  const llvm::json::Object *terms = o.getObject("terms");
  if (terms == nullptr)
    throw std::runtime_error("an expression of no terms");
  for (const auto &[name, coefficient] : *terms) {
    const auto found = std::find(symbols_.begin(), symbols_.end(), name.str());
    if (found == symbols_.end())
      throw std::runtime_error("shapes name an undeclared symbol '" +
                               name.str() + "'");
    std::optional<int64_t> value = coefficient.getAsInteger();
    if (!value)
      throw std::runtime_error("a coefficient that is no integer");
    result.terms.emplace_back(static_cast<int>(found - symbols_.begin()),
                              *value);
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
