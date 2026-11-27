/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- test_hipep_backend.cpp - libhipep_backend without a device --------===//
//
// A split as a ggml backend describes it (hipep_backend.h), built as ggml
// builds its graphs and described as ggml-hip describes it: what the
// library rebuilds of it
// (Graph.h), which extents the split's operations make one size and which the
// model fixes, what a dynamic signature keeps, whether compiled shapes fit a
// split, and which operations it takes.
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "Ops.h"

#include "hip/Frontend/Ggml/GgmlTensor.h"
#include "hip/Frontend/Ggml/SplitCompiler.h"
#include "hip/kquant_abi.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace capture = hip::ggml::capture;
using hip::ggml::Tensor;

namespace {

int failures = 0;

void check(bool passed, const std::string &what) {
  if (passed)
    return;
  std::printf("  FAILED: %s\n", what.c_str());
  ++failures;
}

//===----------------------------------------------------------------------===//
// Graphs built as ggml builds them (ggml.c's constructors), on no device
//===----------------------------------------------------------------------===//

/// ggml's type numbers the tests use.
constexpr int32_t kF32 = 0, kF16 = 1, kQ4K = 12, kI32 = 26, kI64 = 27,
                  kBF16 = 30;

size_t typeSize(int32_t type) {
  return static_cast<size_t>(hip::ggml::typeLayout(type)->typeSize);
}

struct Context {
  std::deque<Tensor> tensors;

  /// ggml_new_tensor_impl: a view's source is the root of a view chain.
  Tensor *tensor(int32_t type, std::initializer_list<int64_t> ne,
                 const size_t *nb = nullptr, const Tensor *viewSrc = nullptr,
                 size_t viewOffs = 0) {
    if (viewSrc != nullptr && viewSrc->viewSrc != nullptr) {
      viewOffs += viewSrc->viewOffs;
      viewSrc = viewSrc->viewSrc;
    }
    Tensor &t = tensors.emplace_back();
    t.type = type;
    int d = 0;
    for (int64_t n : ne)
      t.ne[d++] = n;
    if (nb != nullptr) {
      std::copy(nb, nb + 4, t.nb);
    } else {
      const auto layout = *hip::ggml::typeLayout(type);
      t.nb[0] = static_cast<size_t>(layout.typeSize);
      t.nb[1] = t.nb[0] * static_cast<size_t>(t.ne[0] / layout.blockElements);
      for (int i = 2; i < 4; ++i)
        t.nb[i] = t.nb[i - 1] * static_cast<size_t>(t.ne[i - 1]);
    }
    t.viewSrc = viewSrc;
    t.viewOffs = viewOffs;
    return &t;
  }
  Tensor *like(const Tensor *a) {
    return tensor(a->type, {a->ne[0], a->ne[1], a->ne[2], a->ne[3]});
  }
  Tensor *viewOf(const Tensor *a) {
    return tensor(a->type, {a->ne[0], a->ne[1], a->ne[2], a->ne[3]}, a->nb, a);
  }
  static Tensor *named(Tensor *t, const char *name) {
    t->name = name;
    return t;
  }

  Tensor *mulMat(const Tensor *a, const Tensor *b) {
    Tensor *r = tensor(kF32, {a->ne[1], b->ne[1], b->ne[2], b->ne[3]});
    r->op = "MUL_MAT";
    r->src[0] = a, r->src[1] = b;
    return r;
  }
  Tensor *add(const Tensor *a, const Tensor *b) {
    Tensor *r = like(a);
    r->op = "ADD";
    r->src[0] = a, r->src[1] = b;
    return r;
  }
  Tensor *cpy(const Tensor *a, const Tensor *b) {
    Tensor *r = viewOf(b);
    r->op = "CPY";
    r->src[0] = a, r->src[1] = b;
    return r;
  }
  Tensor *reshape3d(const Tensor *a, int64_t ne0, int64_t ne1, int64_t ne2) {
    Tensor *r = tensor(a->type, {ne0, ne1, ne2}, nullptr, a);
    r->op = "RESHAPE";
    r->src[0] = a;
    return r;
  }
  /// ggml_rope_ext without frequency factors.
  Tensor *rope(const Tensor *a, const Tensor *positions, int nDims, int mode,
               int nCtxOrig, float freqBase) {
    Tensor *r = like(a);
    const float floats[6] = {freqBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    r->opParams[1] = nDims;
    r->opParams[2] = mode;
    r->opParams[4] = nCtxOrig;
    std::memcpy(r->opParams + 5, floats, sizeof(floats));
    r->op = "ROPE";
    r->src[0] = a, r->src[1] = positions;
    return r;
  }
  Tensor *view(const Tensor *a, std::initializer_list<int64_t> ne,
               const size_t (&nb)[4], size_t offset) {
    Tensor *r = tensor(a->type, ne, nb, a, offset);
    std::memcpy(r->opParams, &offset, sizeof(offset));
    r->op = "VIEW";
    r->src[0] = a;
    return r;
  }
  Tensor *view2d(const Tensor *a, int64_t ne0, int64_t ne1, size_t nb1,
                 size_t offset) {
    const size_t nb[4] = {typeSize(a->type), nb1, nb1 * ne1, nb1 * ne1};
    return view(a, {ne0, ne1}, nb, offset);
  }
  Tensor *view3d(const Tensor *a, int64_t ne0, int64_t ne1, int64_t ne2,
                 size_t nb1, size_t nb2, size_t offset) {
    const size_t nb[4] = {typeSize(a->type), nb1, nb2, nb2 * ne2};
    return view(a, {ne0, ne1, ne2}, nb, offset);
  }
  Tensor *permute(const Tensor *a, int a0, int a1, int a2, int a3) {
    Tensor *r = viewOf(a);
    const int axes[4] = {a0, a1, a2, a3};
    for (int i = 0; i < 4; ++i) {
      r->ne[axes[i]] = a->ne[i];
      r->nb[axes[i]] = a->nb[i];
      r->opParams[i] = axes[i];
    }
    r->op = "PERMUTE";
    r->src[0] = a;
    return r;
  }
  Tensor *setRows(const Tensor *a, const Tensor *b, const Tensor *c) {
    Tensor *r = viewOf(a);
    r->op = "SET_ROWS";
    r->src[0] = b, r->src[1] = c, r->src[2] = a;
    return r;
  }
  Tensor *flashAttn(const Tensor *q, const Tensor *k, const Tensor *v,
                    const Tensor *mask, float scale) {
    Tensor *r = tensor(kF32, {v->ne[0], q->ne[2], q->ne[1], q->ne[3]});
    const float floats[3] = {scale, 0.0f, 0.0f};
    std::memcpy(r->opParams, floats, sizeof(floats));
    r->op = "FLASH_ATTN_EXT";
    r->src[0] = q, r->src[1] = k, r->src[2] = v, r->src[3] = mask;
    return r;
  }
  /// ggml_glu_split, or ggml_glu (`b` null: `a` halved).
  Tensor *glu(const Tensor *a, const Tensor *b, int32_t kind, bool swapped) {
    Tensor *r =
        b != nullptr
            ? like(a)
            : tensor(a->type, {a->ne[0] / 2, a->ne[1], a->ne[2], a->ne[3]});
    r->opParams[0] = kind;
    r->opParams[1] = swapped ? 1 : 0;
    r->op = "GLU";
    r->src[0] = a, r->src[1] = b;
    return r;
  }

  /// ggml_build_forward_expand's nodes: what `roots` compute, sources first.
  static std::vector<const Tensor *>
  forward(std::initializer_list<const Tensor *> roots) {
    std::vector<const Tensor *> nodes;
    std::unordered_set<const Tensor *> visited;
    std::function<void(const Tensor *)> visit = [&](const Tensor *t) {
      if (!visited.insert(t).second)
        return;
      for (const Tensor *source : t->src)
        if (source != nullptr)
          visit(source);
      if (t->op != "NONE")
        nodes.push_back(t);
    };
    for (const Tensor *root : roots)
      visit(root);
    return nodes;
  }
};

//===----------------------------------------------------------------------===//
// A split description, built as ggml-hip builds it
// (ggml-cuda/hipep/split-desc.cpp)
//===----------------------------------------------------------------------===//

struct SplitDescBuilder {
  std::vector<hipep_backend_tensor_desc> tensors;
  std::vector<const Tensor *> originals;
  std::unordered_map<const Tensor *, int32_t> index;
  /// Weights and caches by name: on the CPU no buffer says so.
  std::unordered_set<std::string> weights;
  std::unordered_set<std::string> caches;

  hipep_backend_split_desc desc() const {
    hipep_backend_split_desc d{};
    d.abi = HIPEP_BACKEND_ABI_VERSION;
    d.tensor_size = sizeof(hipep_backend_tensor_desc);
    d.ggml_version = 10000; // ggml 1.0.0
    d.tensors = tensors.data();
    d.n_tensors = tensors.size();
    return d;
  }

  hipep_backend_tensor_desc leafDesc(const Tensor *t) const {
    hipep_backend_tensor_desc d{};
    d.op = "NONE";
    d.name = t->name.c_str();
    d.type = t->type;
    for (int i = 0; i < 4; ++i) {
      d.ne[i] = t->ne[i];
      d.nb[i] = t->nb[i];
    }
    for (int32_t &s : d.src)
      s = -1;
    d.view_src = -1;
    d.flags = t->flags | 16; // GGML_TENSOR_FLAG_COMPUTE
    d.role = weights.count(t->name)  ? HIPEP_BACKEND_ROLE_WEIGHT
             : caches.count(t->name) ? HIPEP_BACKEND_ROLE_KV_CACHE
                                     : HIPEP_BACKEND_ROLE_NODE;
    return d;
  }
  void setOp(hipep_backend_tensor_desc &d, const Tensor *t, const int32_t *src,
             int32_t view) {
    d.op = t->op.c_str();
    std::memcpy(d.op_params, t->opParams, sizeof(d.op_params));
    for (int i = 0; i < hip::ggml::kMaxSources; ++i)
      d.src[i] = src[i];
    d.view_src = view;
    d.view_offs = t->viewOffs;
  }
  int32_t push(const Tensor *t, hipep_backend_tensor_desc d) {
    const int32_t i = static_cast<int32_t>(tensors.size());
    for (int32_t &s : d.src)
      if (s == -2)
        s = i;
    tensors.push_back(d);
    originals.push_back(t);
    index.emplace(t, i);
    return i;
  }
  int32_t addSource(const Tensor *t,
                    const std::unordered_set<const Tensor *> &nodes) {
    if (t == nullptr)
      return -1;
    if (auto found = index.find(t); found != index.end())
      return found->second;
    if (t->viewSrc != nullptr && t->op != "NONE" && nodes.count(t) == 0)
      return addTensor(t, nodes, false);
    return push(t, leafDesc(t));
  }
  int32_t addTensor(const Tensor *t,
                    const std::unordered_set<const Tensor *> &nodes,
                    bool node) {
    int32_t src[hip::ggml::kMaxSources];
    for (int i = 0; i < hip::ggml::kMaxSources; ++i)
      src[i] = t->src[i] == t ? -2 : addSource(t->src[i], nodes);
    const int32_t view = addSource(t->viewSrc, nodes);
    hipep_backend_tensor_desc d = leafDesc(t);
    setOp(d, t, src, view);
    d.in_split = node;
    d.exported = node;
    return push(t, d);
  }

  /// The split: `nodes`, in order.
  SplitDescBuilder(const std::vector<const Tensor *> &nodes,
                   std::unordered_set<std::string> weightNames = {},
                   std::unordered_set<std::string> cacheNames = {})
      : weights(std::move(weightNames)), caches(std::move(cacheNames)) {
    const std::unordered_set<const Tensor *> set(nodes.begin(), nodes.end());
    for (const Tensor *n : nodes)
      addTensor(n, set, true);
  }

  int32_t indexOf(const Tensor *t) const {
    auto found = index.find(t);
    return found == index.end() ? -1 : found->second;
  }
};

/// The symbol whose places include (tensor, dimension), or null.
const capture::SymbolSpec::Symbol *symbolAt(const capture::SymbolSpec &spec,
                                            int32_t tensor, int dimension) {
  for (const capture::SymbolSpec::Symbol &s : spec.symbols)
    for (const capture::SymbolSpec::Place &p : s.places)
      if (p.tensor == tensor && p.dimension == dimension)
        return &s;
  return nullptr;
}

capture::SymbolSpec deriveSymbols(const SplitDescBuilder &builder) {
  hipep_backend::Graph graph(builder.desc());
  return capture::captureSplit(graph.split, graph.tensors, nullptr)->symbols;
}

/// One attention layer as llama.cpp builds it, the way the backend is handed
/// it: its activation a copy the scheduler made of another device's tensor,
/// the cache's window as wide as the mask, the cache written at the step's
/// rows.
struct AttentionLayer : Context {
  std::vector<const Tensor *> nodes;
  AttentionLayer(int64_t tokens, int64_t slots) {
    Tensor *positions = named(tensor(kI32, {tokens}), "inp_pos");
    Tensor *x = named(tensor(kF32, {64, tokens}), "ROCm0#node_1#0");
    Tensor *w = named(tensor(kF32, {64, 64}), "blk.0.attn_q.weight");
    Tensor *q = reshape3d(mulMat(w, x), 16, 4, tokens);
    q = rope(q, positions, 16, 0, 4096, 10000.0f);
    Tensor *cache = named(tensor(kF16, {64, 1024}), "cache_k_l0");
    Tensor *rows = named(tensor(kI64, {tokens}), "self_k_idxs");
    Tensor *written = setRows(cache, x, rows);
    Tensor *k = view3d(cache, 16, 4, slots, 16 * 2, 64 * 2, 0);
    Tensor *mask = named(tensor(kF16, {slots, tokens}), "kq_mask");
    Tensor *out = flashAttn(permute(q, 0, 2, 1, 3), permute(k, 0, 2, 1, 3),
                            permute(k, 0, 2, 1, 3), mask, 0.25f);
    nodes = forward({written, out});
  }
  SplitDescBuilder splitDesc() const {
    return SplitDescBuilder(nodes, {"blk.0.attn_q.weight"}, {"cache_k_l0"});
  }
  int32_t leafIndex(const char *name) const {
    const SplitDescBuilder d = splitDesc();
    for (size_t i = 0; i < d.tensors.size(); ++i)
      if (std::strcmp(d.tensors[i].name, name) == 0 &&
          std::strcmp(d.tensors[i].op, "NONE") == 0)
        return static_cast<int32_t>(i);
    return -1;
  }
  /// The cache's window.
  int32_t cacheWindowIndex() const {
    const SplitDescBuilder d = splitDesc();
    for (size_t i = 0; i < d.tensors.size(); ++i)
      if (std::strcmp(d.tensors[i].op, "VIEW") == 0)
        return static_cast<int32_t>(i);
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// The checks
//===----------------------------------------------------------------------===//

void testAttentionLayerSymbols() {
  const AttentionLayer layer(6, 256);
  const capture::SymbolSpec spec = deriveSymbols(layer.splitDesc());
  const auto *tokens = symbolAt(spec, layer.leafIndex("inp_pos"), 0);
  const auto *slots = symbolAt(spec, layer.leafIndex("kq_mask"), 0);
  check(tokens && slots && tokens != slots,
        "inp_pos and kq_mask carry two symbols");
  if (!tokens || !slots)
    return;
  check(symbolAt(spec, layer.leafIndex("kq_mask"), 1) == tokens,
        "the mask's rows are the queries");
  check(symbolAt(spec, layer.leafIndex("self_k_idxs"), 0) == tokens,
        "a row per token is written");
  check(symbolAt(spec, layer.leafIndex("ROCm0#node_1#0"), 1) == tokens,
        "the copy's rows are the positions RoPE turns");
  check(symbolAt(spec, layer.cacheWindowIndex(), 2) == slots,
        "the window is as wide as the mask");
  check(symbolAt(spec, layer.leafIndex("ROCm0#node_1#0"), 0) == nullptr &&
            symbolAt(spec, layer.cacheWindowIndex(), 0) == nullptr,
        "a product's shared extent and a head are the weight's");
  for (int d = 0; d < 4; ++d) {
    check(symbolAt(spec, layer.leafIndex("blk.0.attn_q.weight"), d) == nullptr,
          "the weight is fixed");
    check(symbolAt(spec, layer.leafIndex("cache_k_l0"), d) == nullptr,
          "the cache is fixed");
  }
  check(slots->maximum == 1024,
        "the window is bounded by the cache it is a window of");
}

void testEqualExtentsAreSeparateSymbols() {
  const AttentionLayer layer(256, 256);
  const capture::SymbolSpec spec = deriveSymbols(layer.splitDesc());
  const auto *tokens = symbolAt(spec, layer.leafIndex("inp_pos"), 0);
  const auto *slots = symbolAt(spec, layer.leafIndex("kq_mask"), 0);
  check(tokens && slots && tokens != slots,
        "256 tokens and 256 slots are two symbols");
  check(symbolAt(spec, layer.cacheWindowIndex(), 2) == slots,
        "the window is the slots");
  check(symbolAt(spec, layer.leafIndex("ROCm0#node_1#0"), 1) == tokens,
        "the copy's rows are the tokens");
}

void testSingleTokenIsStatic() {
  const AttentionLayer layer(1, 256);
  const capture::SymbolSpec spec = deriveSymbols(layer.splitDesc());
  check(symbolAt(spec, layer.leafIndex("inp_pos"), 0) == nullptr,
        "one token is no symbol");
  const auto *slots = symbolAt(spec, layer.leafIndex("kq_mask"), 0);
  check(slots != nullptr &&
            symbolAt(spec, layer.cacheWindowIndex(), 2) == slots,
        "the slots still are");
}

/// A split reading a copy the scheduler made of a tensor computed elsewhere:
/// nothing outside the split says what it counts.
void testSchedulerCopyInput() {
  Context c;
  Tensor *copy = c.named(c.tensor(kF32, {64, 7}), "ROCm0#inp_embd_rows#0");
  Tensor *w = c.named(c.tensor(kF32, {64, 32}), "blk.0.ffn.weight");
  const SplitDescBuilder split({c.mulMat(w, copy)}, {"blk.0.ffn.weight"});
  const capture::SymbolSpec spec = deriveSymbols(split);
  const int32_t copyIndex = split.indexOf(copy);
  check(symbolAt(spec, copyIndex, 1) != nullptr,
        "the copy's rows are a symbol");
  check(symbolAt(spec, copyIndex, 0) == nullptr, "its width is the weight's");
}

/// A split that only reshapes what it reads: the reshape says the width is
/// its heads' (a number), the rotation that the positions are its rows.
void testReshapeFixesWidth() {
  Context c;
  Tensor *x = c.named(c.tensor(kF32, {64, 7}), "ROCm0#node_1#0");
  Tensor *positions = c.named(c.tensor(kI32, {7}), "ROCm0#inp_pos#0");
  Tensor *reshaped = c.reshape3d(x, 16, 4, 7);
  Tensor *rotated = c.rope(reshaped, positions, 16, 0, 4096, 10000.0f);
  const SplitDescBuilder split(Context::forward({rotated}), {});
  const capture::SymbolSpec spec = deriveSymbols(split);
  const auto *rows = symbolAt(spec, split.indexOf(x), 1);
  check(rows != nullptr && symbolAt(spec, split.indexOf(positions), 0) == rows,
        "the rows are the positions");
  check(symbolAt(spec, split.indexOf(x), 0) == nullptr,
        "the width is the reshape's heads");
}

/// A split that reads keys another backend rotated ([head width, heads,
/// tokens]) and writes them to the cache as rows ([width * heads, tokens]):
/// the view that merges the two says they multiply to the row's width.
void testMergedViewFixesHeads() {
  Context c;
  Tensor *keys = c.named(c.tensor(kF32, {16, 4, 7}), "Kcur-0");
  Tensor *rows = c.named(c.tensor(kI64, {7}), "ROCm0#attn_inp_k_idxs#0");
  Tensor *cache = c.named(c.tensor(kF16, {64, 256}), "cache_k_l0");
  Tensor *flat = c.view2d(keys, 64, 7, keys->nb[2], 0);
  const SplitDescBuilder split(Context::forward({c.setRows(cache, flat, rows)}),
                               {}, {"cache_k_l0"});
  const capture::SymbolSpec spec = deriveSymbols(split);
  const auto *tokens = symbolAt(spec, split.indexOf(rows), 0);
  check(tokens != nullptr && symbolAt(spec, split.indexOf(keys), 2) == tokens,
        "a row is written per token");
  check(symbolAt(spec, split.indexOf(keys), 0) == nullptr &&
            symbolAt(spec, split.indexOf(keys), 1) == nullptr,
        "the head width and the heads are the row's");
}

/// The split's tensors numbered sources first, as the library describes it;
/// any other numbering is refused.
void testDescriptionOrder() {
  const AttentionLayer layer(6, 256);
  SplitDescBuilder split = layer.splitDesc();
  bool read = true;
  try {
    hipep_backend::Graph graph(split.desc());
  } catch (const std::exception &) {
    read = false;
  }
  check(read, "a description numbered sources first is read");
  // The first two tensors swapped, every index with them.
  std::vector<hipep_backend_tensor_desc> swapped = split.tensors;
  std::swap(swapped[0], swapped[1]);
  const auto renumber = [](int32_t &i) { i = i == 0 ? 1 : i == 1 ? 0 : i; };
  for (hipep_backend_tensor_desc &t : swapped) {
    for (int32_t &s : t.src)
      renumber(s);
    renumber(t.view_src);
  }
  hipep_backend_split_desc desc = split.desc();
  desc.tensors = swapped.data();
  bool refused = false;
  try {
    hipep_backend::Graph graph(desc);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "a description numbered otherwise is refused");
}

capture::SplitDescription::Tensor leafTensor(std::initializer_list<int64_t> ne,
                                             bool weight = false) {
  capture::SplitDescription::Tensor t;
  t.op = "NONE";
  int d = 0;
  for (int64_t n : ne)
    t.ne[d++] = n;
  t.weight = weight;
  return t;
}

capture::SplitDescription::Tensor
nodeTensor(const char *op, std::initializer_list<int32_t> src,
           std::initializer_list<int64_t> ne) {
  capture::SplitDescription::Tensor t = leafTensor(ne);
  t.op = op;
  int i = 0;
  for (int32_t s : src)
    t.src[i++] = s;
  return t;
}

/// One attention layer's split, described: positions, activation, weight,
/// product, rotation, cache, mask, window, attention, rows, cache write; the
/// positions are the step's tokens.
capture::SplitDescription attentionSplit(int64_t tokens, int64_t slots,
                                         int64_t rows = 1024) {
  capture::SplitDescription d;
  d.tensors.push_back(leafTensor({tokens}));
  d.tensors.push_back(leafTensor({64, tokens}));
  d.tensors.push_back(leafTensor({64, 64}, /*weight=*/true));
  d.tensors.push_back(nodeTensor("MUL_MAT", {2, 1}, {64, tokens}));
  d.tensors.push_back(nodeTensor("RESHAPE", {3}, {16, 4, tokens}));
  d.tensors.push_back(nodeTensor("ROPE", {4, 0}, {16, 4, tokens}));
  d.tensors.push_back(leafTensor({64, rows}));
  d.tensors.push_back(leafTensor({slots, tokens}));
  capture::SplitDescription::Tensor view =
      nodeTensor("VIEW", {6}, {16, 4, slots});
  view.viewSrc = 6;
  d.tensors.push_back(view);
  d.tensors.push_back(
      nodeTensor("FLASH_ATTN_EXT", {5, 8, 8, 7}, {16, 4, tokens}));
  d.tensors.push_back(leafTensor({tokens}));
  capture::SplitDescription::Tensor write =
      nodeTensor("SET_ROWS", {6, 3, 10}, {64, rows});
  write.viewSrc = 6;
  d.tensors.push_back(write);
  return d;
}

void testDynamicSignature() {
  check(capture::dynamicSignature(attentionSplit(6, 256)) ==
            capture::dynamicSignature(attentionSplit(9, 512)),
        "other sizes are one dynamic signature");
  check(capture::signature(attentionSplit(6, 256)) !=
            capture::signature(attentionSplit(9, 512)),
        "other sizes are other signatures");
  check(capture::dynamicSignature(attentionSplit(6, 256)) !=
            capture::dynamicSignature(attentionSplit(6, 256, 2048)),
        "another cache size is another dynamic signature");
  check(capture::dynamicSignature(attentionSplit(1, 256)) ==
            capture::dynamicSignature(attentionSplit(6, 256)),
        "a step of one token is one too (code of many runs it)");
}

void testCompiledShapesFit() {
  // Compiled for the activation's rows as s0 (bounded 1..16) and their
  // bytes as 256 * s0.
  using llvm::json::Array, llvm::json::Object;
  const auto times = [](int64_t coefficient) {
    return Object{{"offset", 0}, {"terms", Object{{"s0", coefficient}}}};
  };
  const llvm::json::Value shapes = Object{
      {"symbols",
       Array{Object{{"name", "s0"}, {"minimum", 1}, {"maximum", 16}}}},
      {"tensors", Array{Object{{"ne", Array{64, times(1), 1, 1}},
                               {"nb", Array{4, 256, times(256), times(256)}},
                               {"view_offs", 0}}}}};
  const capture::SplitShapes fitted(shapes);
  const auto one = [](int64_t rows) {
    capture::SplitDescription d;
    capture::SplitDescription::Tensor t = leafTensor({64, rows});
    t.nb[0] = 4, t.nb[1] = 256,
    t.nb[2] = t.nb[3] = static_cast<size_t>(256 * rows);
    d.tensors.push_back(t);
    return d;
  };
  check(fitted.fits(one(6)) && fitted.fits(one(16)), "6 and 16 rows fit");
  check(!fitted.fits(one(17)), "17 rows do not");
  capture::SplitDescription strided = one(6);
  strided.tensors[0].nb[1] = 512;
  check(!fitted.fits(strided), "another stride does not");
  // A dimension of one element steps nowhere: its stride is anything.
  capture::SplitDescription unit = one(6);
  unit.tensors[0].nb[2] = 4096;
  check(fitted.fits(unit), "a stride of a dimension of one says nothing");
}

void testSupportedOps() {
  Context c;
  Tensor *a = c.tensor(kQ4K, {256, 8});
  Tensor *b = c.tensor(kF32, {256, 4});
  const auto supported = [](const Tensor *op) {
    SplitDescBuilder d({}, {});
    d.addTensor(op, {op}, true);
    return hipep_backend::supports(d.desc());
  };
  check(supported(c.mulMat(a, b)), "a product by a packed weight");
  check(!supported(c.mulMat(b, a)), "no product of a packed activation");
  Tensor *dense = c.tensor(kF16, {256, 8});
  check(!supported(c.mulMat(dense, b)),
        "no product by a dense weight (the native kernels read K-quants)");
  check(supported(c.add(b, b)), "a sum");
  Tensor *h = c.tensor(kBF16, {256, 4});
  check(!supported(c.add(h, h)), "no bf16 sum (the runtime has none)");
  Tensor *dst = c.tensor(kF32, {256, 4});
  check(!supported(c.cpy(b, dst)), "no copy into another tensor");
}

/// What the library takes of an operation is how convert-ggml-to-hip lowers
/// it configured (GgmlToHipLegality.h): a split of what it takes but the
/// lowering refuses would be refused whole.
void testSupportedConfigurations() {
  Context c;
  const auto supported = [](const Tensor *op) {
    SplitDescBuilder d({}, {});
    d.addTensor(op, {op}, true);
    return hipep_backend::supports(d.desc());
  };
  // ggml's enumerators: enum ggml_glu_op, GGML_ROPE_TYPE_*, enum ggml_prec.
  constexpr int32_t kGeglu = 1, kSwiglu = 2, kSwigluOai = 3;
  constexpr int32_t kNeox = 2, kMrope = 8;
  constexpr int32_t kPrecisionF32 = 10;

  Tensor *x = c.tensor(kF32, {64, 4});
  check(supported(c.glu(x, x, kSwiglu, false)), "SwiGLU of a gate and an up");
  check(!supported(c.glu(x, nullptr, kSwiglu, false)),
        "no SwiGLU of one operand halved");
  check(!supported(c.glu(x, x, kSwiglu, true)), "no swapped SwiGLU");
  check(!supported(c.glu(x, x, kGeglu, false)), "no GEGLU");
  check(!supported(c.glu(x, x, kSwigluOai, false)), "no SwiGLU-OAI");

  Tensor *q = c.tensor(kF32, {16, 4, 6});
  Tensor *positions = c.tensor(kI32, {6});
  check(supported(c.rope(q, positions, 16, 0, 4096, 10000.0f)),
        "a normal rotation");
  check(supported(c.rope(q, positions, 16, kNeox, 4096, 10000.0f)),
        "a NEOX rotation");
  check(!supported(c.rope(q, positions, 16, kMrope, 4096, 10000.0f)),
        "no multi-section rotation");
  Tensor *yarn = c.rope(q, positions, 16, kNeox, 4096, 10000.0f);
  const float extension = 1.0f;
  std::memcpy(yarn->opParams + 7, &extension, sizeof(extension));
  check(!supported(yarn), "no rotation with YaRN's extension");
  Tensor *unknown = c.rope(q, positions, 16, 1, 4096, 10000.0f);
  check(!supported(unknown), "no rotation of a mode the dialect does not have");

  Tensor *k = c.tensor(kF16, {16, 256, 4});
  Tensor *mask = c.tensor(kF16, {256, 6});
  const auto attention = [&](const Tensor *m, float softcap,
                             int32_t precision) {
    Tensor *r = c.flashAttn(q, k, k, m, 0.25f);
    std::memcpy(r->opParams + 2, &softcap, sizeof(softcap));
    r->opParams[3] = precision;
    return r;
  };
  check(supported(attention(mask, 0.0f, kPrecisionF32)),
        "attention under a mask, at f32");
  check(!supported(attention(nullptr, 0.0f, kPrecisionF32)),
        "no attention without a mask");
  check(!supported(attention(mask, 30.0f, kPrecisionF32)),
        "no attention under a logit softcap");
  check(!supported(attention(mask, 0.0f, 0)),
        "no attention at the default precision");
}

/// ggml's type numbers (GgmlTensor.h) name types the dialect has a block
/// layout of, and the K-quants' and Q8_1's layouts are the kernels'.
void testTypeTable() {
  int named = 0;
  for (int32_t t = 0; t < 64; ++t) {
    const std::optional<std::string> name = hip::ggml::typeName(t);
    if (!name)
      continue;
    ++named;
    check(hip::ggml::typeLayout(t).has_value(), *name + " has a block layout");
  }
  check(named == 36, "ggml has 36 types");
  check(hip::ggml::typeName(12) == std::string("q4_K") &&
            hip::ggml::typeName(26) == std::string("i32") &&
            !hip::ggml::typeName(4).has_value(),
        "q4_K is 12, i32 26, 4 a removed type");
  const auto layout = [](int32_t type) { return *hip::ggml::typeLayout(type); };
  check(layout(12).typeSize == HIP_KQUANT_Q4_K_BLOCK_SIZE &&
            layout(13).typeSize == HIP_KQUANT_Q5_K_BLOCK_SIZE &&
            layout(14).typeSize == HIP_KQUANT_Q6_K_BLOCK_SIZE &&
            layout(12).blockElements == HIP_KQUANT_BLOCK_ELEMENTS,
        "the K-quants' blocks are the kernels'");
  check(layout(9).typeSize == HIP_Q8_1_BLOCK_SIZE &&
            layout(9).blockElements == HIP_Q8_1_BLOCK_ELEMENTS,
        "Q8_1's blocks are the kernels'");
}

} // namespace

int main() {
  std::printf("hipep_backend:\n");
  testAttentionLayerSymbols();
  testEqualExtentsAreSeparateSymbols();
  testSingleTokenIsStatic();
  testSchedulerCopyInput();
  testReshapeFixesWidth();
  testMergedViewFixesHeads();
  testDescriptionOrder();
  testDynamicSignature();
  testCompiledShapesFit();
  testSupportedOps();
  testSupportedConfigurations();
  testTypeTable();
  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("  all checks passed\n");
  return 0;
}
