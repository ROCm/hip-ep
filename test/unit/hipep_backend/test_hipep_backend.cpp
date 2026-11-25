/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- test_hipep_backend.cpp - libhipep_backend without a device --------===//
//
// A split as a ggml backend describes it (hipep_backend.h), built on the CPU
// and described as ggml-hipep describes it: what the library rebuilds of it
// (Graph.h), which extents the split's operations make one size and which the
// model fixes, what a dynamic signature keeps, whether compiled shapes fit a
// split, and which operations it takes.
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "Ops.h"

#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Frontend/Ggml/SplitCompiler.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace capture = hip::ggml::capture;

namespace {

int failures = 0;

void check(bool passed, const std::string &what) {
  if (passed)
    return;
  std::printf("  FAILED: %s\n", what.c_str());
  ++failures;
}

//===----------------------------------------------------------------------===//
// A split description, built as ggml-hipep builds it (split-desc.cpp)
//===----------------------------------------------------------------------===//

struct SplitDescBuilder {
  std::vector<hipep_backend_tensor_desc> tensors;
  std::vector<const ggml_tensor *> originals;
  std::unordered_map<const ggml_tensor *, int32_t> index;
  /// Weights and caches by name: on the CPU no buffer says so.
  std::unordered_set<std::string> weights;
  std::unordered_set<std::string> caches;

  hipep_backend_split_desc desc() const {
    hipep_backend_split_desc d{};
    d.abi = HIPEP_BACKEND_ABI_VERSION;
    d.tensor_size = sizeof(hipep_backend_tensor_desc);
    unsigned major = 0, minor = 0, patch = 0;
    std::sscanf(ggml_version(), "%u.%u.%u", &major, &minor, &patch);
    d.ggml_version = major * 10000 + minor * 100 + patch;
    d.tensors = tensors.data();
    d.n_tensors = tensors.size();
    return d;
  }

  hipep_backend_tensor_desc leafDesc(const ggml_tensor *t) const {
    hipep_backend_tensor_desc d{};
    d.op = "NONE";
    d.name = t->name;
    d.type = t->type;
    for (int i = 0; i < 4; ++i) {
      d.ne[i] = t->ne[i];
      d.nb[i] = t->nb[i];
    }
    for (int32_t &s : d.src)
      s = -1;
    d.view_src = -1;
    d.flags = t->flags | GGML_TENSOR_FLAG_COMPUTE;
    d.role = weights.count(t->name)  ? HIPEP_BACKEND_ROLE_WEIGHT
             : caches.count(t->name) ? HIPEP_BACKEND_ROLE_KV_CACHE
                                     : HIPEP_BACKEND_ROLE_NODE;
    return d;
  }
  void setOp(hipep_backend_tensor_desc &d, const ggml_tensor *t,
             const int32_t *src, int32_t view) {
    d.op = ggml_op_name(t->op);
    std::memcpy(d.op_params, t->op_params, sizeof(d.op_params));
    for (int i = 0; i < GGML_MAX_SRC; ++i)
      d.src[i] = src[i];
    d.view_src = view;
    d.view_offs = t->view_offs;
  }
  int32_t push(const ggml_tensor *t, hipep_backend_tensor_desc d) {
    const int32_t i = static_cast<int32_t>(tensors.size());
    for (int32_t &s : d.src)
      if (s == -2)
        s = i;
    tensors.push_back(d);
    originals.push_back(t);
    index.emplace(t, i);
    return i;
  }
  int32_t addSource(const ggml_tensor *t,
                    const std::unordered_set<const ggml_tensor *> &nodes) {
    if (t == nullptr)
      return -1;
    if (auto found = index.find(t); found != index.end())
      return found->second;
    if (t->view_src != nullptr && t->op != GGML_OP_NONE && nodes.count(t) == 0)
      return addTensor(t, nodes, false);
    return push(t, leafDesc(t));
  }
  int32_t addTensor(const ggml_tensor *t,
                    const std::unordered_set<const ggml_tensor *> &nodes,
                    bool node) {
    int32_t src[GGML_MAX_SRC];
    for (int i = 0; i < GGML_MAX_SRC; ++i)
      src[i] = t->src[i] == t ? -2 : addSource(t->src[i], nodes);
    const int32_t view = addSource(t->view_src, nodes);
    hipep_backend_tensor_desc d = leafDesc(t);
    setOp(d, t, src, view);
    d.in_split = node;
    d.exported = node;
    return push(t, d);
  }

  /// The split: `nodes`, in order.
  SplitDescBuilder(const std::vector<const ggml_tensor *> &nodes,
                   std::unordered_set<std::string> weightNames = {},
                   std::unordered_set<std::string> cacheNames = {})
      : weights(std::move(weightNames)), caches(std::move(cacheNames)) {
    const std::unordered_set<const ggml_tensor *> set(nodes.begin(),
                                                      nodes.end());
    for (const ggml_tensor *n : nodes)
      addTensor(n, set, true);
  }

  int32_t indexOf(const ggml_tensor *t) const {
    auto found = index.find(t);
    return found == index.end() ? -1 : found->second;
  }
};

std::vector<const ggml_tensor *> nodesOf(const ggml_cgraph *graph) {
  std::vector<const ggml_tensor *> nodes;
  for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)); ++i)
    nodes.push_back(ggml_graph_node(const_cast<ggml_cgraph *>(graph), i));
  return nodes;
}

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

struct Context {
  ggml_context *context;
  Context() {
    ggml_init_params params{size_t(16) << 20, nullptr, /*no_alloc=*/true};
    context = ggml_init(params);
  }
  ~Context() { ggml_free(context); }
  ggml_tensor *named(ggml_tensor *t, const char *name) {
    return ggml_set_name(t, name);
  }
};

/// One attention layer as llama.cpp builds it, the way the backend is handed
/// it: its activation a copy the scheduler made of another device's tensor,
/// the cache's window as wide as the mask, the cache written at the step's
/// rows.
struct AttentionLayer : Context {
  ggml_cgraph *graph = nullptr;
  AttentionLayer(int64_t tokens, int64_t slots) {
    ggml_tensor *positions =
        named(ggml_new_tensor_1d(context, GGML_TYPE_I32, tokens), "inp_pos");
    ggml_tensor *x =
        named(ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, tokens),
              "HIPEP0#node_1#0");
    ggml_tensor *w = named(ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, 64),
                           "blk.0.attn_q.weight");
    ggml_tensor *q =
        ggml_reshape_3d(context, ggml_mul_mat(context, w, x), 16, 4, tokens);
    q = ggml_rope_ext(context, q, positions, nullptr, 16, 0, 4096, 10000.0f,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor *cache = named(
        ggml_new_tensor_2d(context, GGML_TYPE_F16, 64, 1024), "cache_k_l0");
    ggml_tensor *rows = named(
        ggml_new_tensor_1d(context, GGML_TYPE_I64, tokens), "self_k_idxs");
    ggml_tensor *written = ggml_set_rows(context, cache, x, rows);
    ggml_tensor *k =
        ggml_view_3d(context, cache, 16, 4, slots, 16 * 2, 64 * 2, 0);
    ggml_tensor *mask = named(
        ggml_new_tensor_2d(context, GGML_TYPE_F16, slots, tokens), "kq_mask");
    ggml_tensor *out = ggml_flash_attn_ext(
        context, ggml_permute(context, q, 0, 2, 1, 3),
        ggml_permute(context, k, 0, 2, 1, 3),
        ggml_permute(context, k, 0, 2, 1, 3), mask, 0.25f, 0.0f, 0.0f);
    graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, written);
    ggml_build_forward_expand(graph, out);
  }
  SplitDescBuilder splitDesc() const {
    return SplitDescBuilder(nodesOf(graph), {"blk.0.attn_q.weight"},
                            {"cache_k_l0"});
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
  check(symbolAt(spec, layer.leafIndex("HIPEP0#node_1#0"), 1) == tokens,
        "the copy's rows are the positions RoPE turns");
  check(symbolAt(spec, layer.cacheWindowIndex(), 2) == slots,
        "the window is as wide as the mask");
  check(symbolAt(spec, layer.leafIndex("HIPEP0#node_1#0"), 0) == nullptr &&
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
  check(symbolAt(spec, layer.leafIndex("HIPEP0#node_1#0"), 1) == tokens,
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
  ggml_tensor *copy =
      c.named(ggml_new_tensor_2d(c.context, GGML_TYPE_F32, 64, 7),
              "HIPEP0#inp_embd_rows#0");
  ggml_tensor *w = c.named(ggml_new_tensor_2d(c.context, GGML_TYPE_F32, 64, 32),
                           "blk.0.ffn.weight");
  const SplitDescBuilder split({ggml_mul_mat(c.context, w, copy)},
                               {"blk.0.ffn.weight"});
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
  ggml_tensor *x = c.named(ggml_new_tensor_2d(c.context, GGML_TYPE_F32, 64, 7),
                           "HIPEP0#node_1#0");
  ggml_tensor *positions = c.named(
      ggml_new_tensor_1d(c.context, GGML_TYPE_I32, 7), "HIPEP0#inp_pos#0");
  ggml_tensor *rotated = ggml_rope_ext(
      c.context, ggml_reshape_3d(c.context, x, 16, 4, 7), positions, nullptr,
      16, 0, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  const SplitDescBuilder split({rotated}, {});
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
  ggml_tensor *keys =
      c.named(ggml_new_tensor_3d(c.context, GGML_TYPE_F32, 16, 4, 7), "Kcur-0");
  ggml_tensor *rows = c.named(ggml_new_tensor_1d(c.context, GGML_TYPE_I64, 7),
                              "HIPEP0#attn_inp_k_idxs#0");
  ggml_tensor *cache = c.named(
      ggml_new_tensor_2d(c.context, GGML_TYPE_F16, 64, 256), "cache_k_l0");
  ggml_tensor *flat = ggml_view_2d(c.context, keys, 64, 7, keys->nb[2], 0);
  const SplitDescBuilder split({ggml_set_rows(c.context, cache, flat, rows)},
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
  t.op = GGML_OP_NONE;
  int d = 0;
  for (int64_t n : ne)
    t.ne[d++] = n;
  t.weight = weight;
  return t;
}

capture::SplitDescription::Tensor
nodeTensor(ggml_op op, std::initializer_list<int32_t> src,
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
  d.tensors.push_back(nodeTensor(GGML_OP_MUL_MAT, {2, 1}, {64, tokens}));
  d.tensors.push_back(nodeTensor(GGML_OP_RESHAPE, {3}, {16, 4, tokens}));
  d.tensors.push_back(nodeTensor(GGML_OP_ROPE, {4, 0}, {16, 4, tokens}));
  d.tensors.push_back(leafTensor({64, rows}));
  d.tensors.push_back(leafTensor({slots, tokens}));
  capture::SplitDescription::Tensor view =
      nodeTensor(GGML_OP_VIEW, {6}, {16, 4, slots});
  view.viewSrc = 6;
  d.tensors.push_back(view);
  d.tensors.push_back(
      nodeTensor(GGML_OP_FLASH_ATTN_EXT, {5, 8, 8, 7}, {16, 4, tokens}));
  d.tensors.push_back(leafTensor({tokens}));
  capture::SplitDescription::Tensor write =
      nodeTensor(GGML_OP_SET_ROWS, {6, 3, 10}, {64, rows});
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
  const nlohmann::json term = {{"offset", 0}, {"terms", {{"s0", 1}}}};
  const nlohmann::json bytes = {{"offset", 0}, {"terms", {{"s0", 256}}}};
  const nlohmann::json shapes = {
      {"symbols", {{{"name", "s0"}, {"minimum", 1}, {"maximum", 16}}}},
      {"tensors",
       {{{"ne", {64, term, 1, 1}},
         {"nb", {4, 256, bytes, bytes}},
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
  ggml_tensor *a = ggml_new_tensor_2d(c.context, GGML_TYPE_Q4_K, 256, 8);
  ggml_tensor *b = ggml_new_tensor_2d(c.context, GGML_TYPE_F32, 256, 4);
  const auto supported = [](const ggml_tensor *op) {
    SplitDescBuilder d({}, {});
    d.addTensor(op, {op}, true);
    return hipep_backend::supports(d.desc());
  };
  check(supported(ggml_mul_mat(c.context, a, b)),
        "a product by a packed weight");
  check(!supported(ggml_mul_mat(c.context, b, a)),
        "no product of a packed activation");
  check(supported(ggml_add(c.context, b, b)), "a sum");
  ggml_tensor *h = ggml_new_tensor_2d(c.context, GGML_TYPE_BF16, 256, 4);
  check(!supported(ggml_add(c.context, h, h)),
        "no bf16 sum (the runtime has none)");
  ggml_tensor *dst = ggml_new_tensor_2d(c.context, GGML_TYPE_F32, 256, 4);
  check(!supported(ggml_cpy(c.context, b, dst)), "no copy into another tensor");
}

/// The dialect's block layouts (GgmlLayout.h) against ggml's own tables, for
/// every type this ggml has.
void testBlockLayoutsMatchGgml() {
  for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
    const auto type = static_cast<ggml_type>(t);
    if (ggml_blck_size(type) == 0)
      continue; // A removed type.
    const std::string name = ggml_type_name(type);
    const std::optional<mlir::ggml::BlockLayout> layout =
        mlir::ggml::getBlockLayout(name);
    check(layout.has_value(), name + " has a block layout");
    if (layout)
      check(layout->typeSize == static_cast<int64_t>(ggml_type_size(type)) &&
                layout->blockElements == ggml_blck_size(type),
            name + "'s block layout is ggml's");
  }
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
  testBlockLayoutsMatchGgml();
  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("  all checks passed\n");
  return 0;
}
