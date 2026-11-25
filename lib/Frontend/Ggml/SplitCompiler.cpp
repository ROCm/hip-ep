/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SplitCompiler.cpp - Compiling a ggml backend's split ---------------===//
//
// A split is captured as libhipep_backend rebuilt it - its tensors walked
// from its computed nodes to what they read (CaptureGraph.h's observeSplit) -
// and each captured
// value is named by the index describeSplit gives its tensor ("t7"), which
// is how the bindings and the shapes the compile writes find the tensors
// again. Compiling is the whole frontend's: the module emitted with the
// caller's symbols and its extents derived, then lowered and compiled by
// hip-ep's compiler, in this process, to LLVM bitcode the backend JIT-loads
// as the execution provider does its models.
//
//===----------------------------------------------------------------------===//

#include "hip/Frontend/Ggml/SplitCompiler.h"

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"
#include "hip/Dialect/Ggml/IR/GgmlStatePairs.h"
#include "hip/Dialect/Ggml/Transforms/Passes.h"
#include "hip/Dialect/Ggml/Transforms/StateDonation.h"
#include "hip/Frontend/Ggml/CaptureGraph.h"
#include "hip/Frontend/Ggml/CapturedGraph.h"
#include "hip/Frontend/Ggml/Error.h"
#include "hip/Frontend/Ggml/GgmlEmitter.h"
#include "hip/Frontend/Ggml/SplitGraph.h"

#include "ggml.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "compilation_options_generated.h"
#include "hip/Compiler/CompilerDriver.h"
#include "hip/Support/DiskFileSystem.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace capture = hip::ggml::capture;
namespace fs = std::filesystem;
using hip::ggml::fail;

namespace {

/// The described tensor a captured value is: values are named t<index>.
int64_t describedIndex(const capture::Value &value) {
  if (value.name.size() < 2 || value.name[0] != 't')
    fail("captured value '", value.name, "' is not a described tensor");
  return std::stoll(value.name.substr(1));
}

/// Which described tensor each of the code's inputs and outputs is, in the
/// order the emitter makes them the entry point's arguments and results (the
/// capture's), and which outputs may be written into an input's memory: a
/// state's update that nothing reads the old bytes of afterwards (`donated`,
/// [output, input] pairs).
nlohmann::json bindings(const capture::CapturedGraph &graph,
                        mlir::ModuleOp module) {
  std::map<int64_t, const capture::Value *> values;
  for (const capture::Value &value : graph.values)
    values.emplace(value.id, &value);
  nlohmann::json inputs = nlohmann::json::array(),
                 outputs = nlohmann::json::array(),
                 donated = nlohmann::json::array();
  for (const capture::Input &input : graph.inputs)
    inputs.push_back(describedIndex(*values.at(input.id)));
  for (const capture::Output &output : graph.outputs)
    outputs.push_back(describedIndex(*values.at(output.id)));

  auto entry = module.lookupSymbol<mlir::func::FuncOp>("main_graph");
  std::string reason;
  mlir::FailureOr<llvm::SmallVector<mlir::ggml::StatePair>> pairs =
      mlir::ggml::readStatePairs(entry, reason);
  mlir::FailureOr<llvm::SmallVector<mlir::ggml::StateDonation>> donation =
      mlir::ggml::proveStateDonation(module);
  if (failed(pairs) || failed(donation) || pairs->size() != donation->size())
    fail("the split's states cannot be read: ", reason);
  for (auto [pair, decision] : llvm::zip(*pairs, *donation))
    if (decision.inPlace)
      donated.push_back({pair.outputIndex, pair.inputIndex});
  return nlohmann::json{{"inputs", std::move(inputs)},
                        {"outputs", std::move(outputs)},
                        {"donated", std::move(donated)}};
}

/// A shape expression as shapes.json spells it: a number, or
/// {"offset": c, "terms": {symbol: coefficient}}.
nlohmann::json expression(mlir::Attribute attribute) {
  std::optional<mlir::ggml::ShapeExprView> view =
      mlir::ggml::getShapeExprView(attribute);
  if (!view)
    fail("a split tensor's layout holds no shape expression");
  if (view->isConstant())
    return view->offset;
  nlohmann::json terms = nlohmann::json::object();
  for (const auto &[symbol, coefficient] : view->terms)
    terms[symbol.str()] = coefficient;
  return nlohmann::json{{"offset", view->offset}, {"terms", std::move(terms)}};
}

/// Each described tensor's extents, strides and view offset as the module
/// has them once its extents are derived (model.ggml.mlir), and
/// the symbols with their bounds: the guard a split is checked against
/// before this code runs it (SplitShapes). A tensor the module does not
/// hold is null: nothing compiled reads it.
nlohmann::json shapes(mlir::ModuleOp module,
                      const capture::SplitDescription &description) {
  auto entry = module.lookupSymbol<mlir::func::FuncOp>("main_graph");
  if (!entry)
    fail("the split's module has no main_graph");

  nlohmann::json tensors = nlohmann::json::array();
  for (size_t i = 0; i < description.tensors.size(); ++i)
    tensors.push_back(nullptr);
  const auto record = [&](mlir::Value value, mlir::StringAttr name) {
    mlir::ggml::LayoutAttr layout = mlir::ggml::getGgmlLayout(value);
    if (!name || !layout || name.getValue().size() < 2 ||
        name.getValue()[0] != 't')
      return;
    const size_t index = std::stoul(name.getValue().substr(1).str());
    if (index >= tensors.size())
      return;
    nlohmann::json ne = nlohmann::json::array(), nb = nlohmann::json::array();
    for (mlir::Attribute e : layout.getExtents())
      ne.push_back(expression(e));
    for (mlir::Attribute e : layout.getStrides())
      nb.push_back(expression(e));
    nlohmann::json shape{{"ne", std::move(ne)}, {"nb", std::move(nb)}};
    shape["view_offs"] =
        description.tensors[index].viewSrc >= 0
            ? expression(layout.getOffset())
            : nlohmann::json(description.tensors[index].viewOffs);
    tensors[index] = std::move(shape);
  };
  for (mlir::BlockArgument argument : entry.getArguments())
    record(argument, entry.getArgAttrOfType<mlir::StringAttr>(
                         argument.getArgNumber(), "ggml.source_name"));
  entry.walk([&](mlir::Operation *op) {
    if (op->getNumResults() == 1)
      record(op->getResult(0),
             op->getAttrOfType<mlir::StringAttr>("ggml.source_name"));
  });

  nlohmann::json symbols = nlohmann::json::array();
  if (auto declared = module->getAttrOfType<mlir::ArrayAttr>("ggml.symbols"))
    for (mlir::Attribute attribute : declared) {
      auto d = llvm::dyn_cast<mlir::DictionaryAttr>(attribute);
      auto name = d ? d.getAs<mlir::StringAttr>("name") : nullptr;
      if (!name)
        continue;
      nlohmann::json symbol{{"name", name.getValue().str()}};
      if (auto minimum = d.getAs<mlir::IntegerAttr>("minimum"))
        symbol["minimum"] = minimum.getInt();
      if (auto maximum = d.getAs<mlir::IntegerAttr>("maximum"))
        symbol["maximum"] = maximum.getInt();
      if (auto product = d.getAs<mlir::ArrayAttr>("product"))
        for (mlir::Attribute factor : product)
          symbol["product"].push_back(
              llvm::cast<mlir::StringAttr>(factor).getValue().str());
      symbols.push_back(std::move(symbol));
    }
  return nlohmann::json{{"symbols", std::move(symbols)},
                        {"tensors", std::move(tensors)}};
}

void write(const fs::path &path, const nlohmann::json &document,
           int indent = -1) {
  std::ofstream out(path);
  out << document.dump(indent) << "\n";
  if (!out)
    fail("cannot write ", path.string());
}

/// Each captured value named by the described tensor it is ("t7"); a leaf
/// read in the shape the description gives it (a cache written whole is read
/// in the shape it is written in).
void nameValues(capture::CapturedGraph &graph,
                const capture::SplitDescription &description,
                const std::unordered_map<const ggml_tensor *, int64_t> &ids) {
  std::unordered_map<int64_t, int64_t> indexOfId;
  for (size_t i = 0; i < description.originals.size(); ++i) {
    auto found = ids.find(description.originals[i]);
    if (found != ids.end())
      indexOfId.emplace(found->second, static_cast<int64_t>(i));
  }
  for (capture::Value &value : graph.values) {
    auto found = indexOfId.find(value.id);
    if (found == indexOfId.end())
      fail("captured value '", value.name, "' is no described tensor");
    const capture::SplitDescription::Tensor &t =
        description.tensors[found->second];
    value.name = "t" + std::to_string(found->second);
    if (t.op == GGML_OP_NONE)
      for (unsigned d = 0; d < 4; ++d) {
        value.ne[d] = t.ne[d];
        value.nb[d] = static_cast<int64_t>(t.nb[d]);
      }
  }
  std::unordered_map<int64_t, std::string> names;
  for (const capture::Value &value : graph.values)
    names.emplace(value.id, value.name);
  for (capture::Input &input : graph.inputs)
    input.name = names.at(input.id);
  for (capture::Output &output : graph.outputs)
    output.name = names.at(output.id);
  for (capture::Node &node : graph.nodes)
    node.name = names.at(node.id);
}

//===----------------------------------------------------------------------===//
// Which extents are symbols
//===----------------------------------------------------------------------===//
//
// The split alone says which of its extents are one size: each extent of what
// it reads (a leaf) and each length of a view the host chose (a run along an
// extent of its source) is a candidate, a guess of its own. ggml-symbolize-
// shapes derives the split from the guesses in its order and learns as it
// goes: where an operation requires two extents equal (a product's shared
// dimension, the keys as many as the mask is wide, a position per token, a
// reshape's elements), two guesses are one, or a guess is the number the
// model's side leaves it (a weight's, a cache's, one an operation makes). A
// candidate of one is one (a broadcast; a step of one token is code for one
// row). Each class left that an input carries is a symbol, read where it
// enters.

/// Where a symbol is put before the symbolizer derives the rest.
struct Seed {
  int32_t tensor = 0;
  int dimension = 0;
  std::string symbol;
};

/// The largest value a symbol is compiled for when nothing bounds it.
constexpr int64_t kUnbounded = std::numeric_limits<int32_t>::max();

std::vector<hip::ggml::emit::CallerSymbol>
callerSymbols(const std::vector<Seed> &seeds,
              const std::map<std::string, int64_t> &bounds, bool products) {
  std::vector<hip::ggml::emit::CallerSymbol> symbols;
  for (const auto &[name, maximum] : bounds) {
    hip::ggml::emit::CallerSymbol symbol;
    symbol.name = name;
    symbol.maximum = maximum;
    for (const Seed &seed : seeds)
      if (seed.symbol == name)
        symbol.places.push_back({"t" + std::to_string(seed.tensor),
                                 static_cast<unsigned>(seed.dimension), 1});
    symbols.push_back(std::move(symbol));
  }
  // A tensor of two symbolic extents has their product in its strides.
  for (auto a = bounds.begin(); products && a != bounds.end(); ++a)
    for (auto b = std::next(a); b != bounds.end(); ++b) {
      hip::ggml::emit::CallerSymbol product;
      product.name = a->first + "_" + b->first;
      product.maximum = a->second * b->second;
      product.product = {a->first, b->first};
      symbols.push_back(std::move(product));
    }
  return symbols;
}

/// What ggml-symbolize-shapes inferred of guesses: each guess's class, none
/// for one it fixed at a number.
using InferredClasses = std::map<std::string, std::optional<std::string>>;

/// ggml-symbolize-shapes on the split with `seeds`: whether every extent
/// follows from them (fails as the pass does), or, inferring, which of them
/// are one size.
InferredClasses symbolize(const capture::CapturedGraph &graph,
                          const std::vector<Seed> &seeds,
                          const std::map<std::string, int64_t> &bounds,
                          bool infer) {
  hip::ggml::emit::Options options;
  options.callerSymbols = callerSymbols(seeds, bounds, /*products=*/!infer);
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::ggml::GgmlDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    if (d.getSeverity() == mlir::DiagnosticSeverity::Error &&
        diagnostics.empty())
      diagnostics = d.str();
    return mlir::success();
  });
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          hip::ggml::emit::serializeModule(graph, options), &context);
  if (!module)
    fail("cannot parse the emitted module");
  if (infer)
    (*module)->setAttr("ggml.infer_symbols", mlir::UnitAttr::get(&context));
  mlir::PassManager manager(&context);
  manager.addPass(mlir::ggml::createGgmlSymbolizeShapesPass());
  if (mlir::failed(manager.run(*module)))
    fail("the extents do not follow: ", diagnostics);
  InferredClasses inferred;
  if (auto inferredAttr =
          (*module)->getAttrOfType<mlir::DictionaryAttr>("ggml.inferred"))
    for (mlir::NamedAttribute entry : inferredAttr) {
      auto root = llvm::dyn_cast<mlir::StringAttr>(entry.getValue());
      inferred[entry.getName().str()] =
          root ? std::optional<std::string>(root.str()) : std::nullopt;
    }
  return inferred;
}

/// The model's memory: weights, and the caches the split reads or writes.
std::vector<bool> modelMemory(const capture::SplitDescription &description) {
  std::vector<bool> isModelMemory = capture::caches(description);
  for (size_t i = 0; i < description.tensors.size(); ++i)
    isModelMemory[i] = isModelMemory[i] || description.tensors[i].weight ||
                       description.tensors[i].cache;
  return isModelMemory;
}

/// A candidate: an extent of a leaf that is not the model's, or a length of
/// a view of a leaf that runs along an extent of it (as stepped as it): a
/// window into a cache's cells at any length, into anything else short of
/// all of it (all of it is that extent). A view of a weight, or of what the
/// split computes, is as long as its structure says.
bool isCandidate(const capture::SplitDescription &description,
                 const std::vector<bool> &isModelMemory, int32_t t, int d) {
  const capture::SplitDescription::Tensor &x =
      description.tensors[static_cast<size_t>(t)];
  if (x.op == GGML_OP_NONE)
    return !isModelMemory[static_cast<size_t>(t)];
  if (x.op != GGML_OP_VIEW || x.viewSrc < 0)
    return false;
  const capture::SplitDescription::Tensor &from =
      description.tensors[static_cast<size_t>(x.viewSrc)];
  if (from.op != GGML_OP_NONE || from.weight)
    return false;
  const bool cache = isModelMemory[static_cast<size_t>(x.viewSrc)];
  bool along = false;
  // A cache's window runs along its cells; what a row holds (a head's
  // width) the model fixes.
  for (int e = cache ? 1 : 0; e < 4; ++e)
    along = along || (from.nb[e] == x.nb[d] &&
                      (cache ? from.ne[e] >= x.ne[d] : from.ne[e] > x.ne[d]));
  return along;
}

/// The longest a view may be along extent `d` and stay inside
/// its source, the other extents as they are.
int64_t viewBound(const capture::SplitDescription &description, int32_t t,
                  int d) {
  const capture::SplitDescription::Tensor &x =
      description.tensors[static_cast<size_t>(t)];
  int32_t root = x.viewSrc;
  while (root >= 0 &&
         description.tensors[static_cast<size_t>(root)].viewSrc >= 0)
    root = description.tensors[static_cast<size_t>(root)].viewSrc;
  if (root < 0)
    return kUnbounded;
  const capture::SplitDescription::Tensor &storage =
      description.tensors[static_cast<size_t>(root)];
  int64_t room =
      static_cast<int64_t>(storage.nb[3]) * storage.ne[3] -
      static_cast<int64_t>(x.viewOffs) -
      static_cast<int64_t>(ggml_type_size(static_cast<ggml_type>(x.type)));
  for (int e = 0; e < 4; ++e)
    if (e != d)
      room -= (x.ne[e] - 1) * static_cast<int64_t>(x.nb[e]);
  if (x.nb[d] == 0 || room < 0)
    return kUnbounded;
  return std::min(kUnbounded, room / static_cast<int64_t>(x.nb[d]) + 1);
}

/// The symbols of a split (SplitCompiler.h). Empty when no extent is free:
/// compiled for its sizes.
capture::SymbolSpec deriveSymbols(const capture::SplitDescription &description,
                                  const capture::CapturedGraph &graph) {
  const std::vector<bool> isModelMemory = modelMemory(description);
  const auto extent = [&](int32_t t, int d) {
    return description.tensors[static_cast<size_t>(t)]
        .ne[static_cast<size_t>(d)];
  };
  // Only what the capture holds: a tensor it left out is no value to place
  // a guess on.
  std::set<int32_t> captured;
  for (const capture::Value &value : graph.values)
    captured.insert(static_cast<int32_t>(describedIndex(value)));
  std::vector<Seed> guesses;
  std::map<std::string, int64_t> guessBounds;
  for (int32_t t : captured)
    for (int d = 0; d < 4; ++d)
      if (isCandidate(description, isModelMemory, t, d) && extent(t, d) > 1) {
        guesses.push_back({t, d, "g" + std::to_string(guesses.size())});
        guessBounds[guesses.back().symbol] = kUnbounded;
      }
  if (guesses.empty())
    return {};
  const InferredClasses inferred =
      symbolize(graph, guesses, guessBounds, /*infer=*/true);
  const auto classOf = [&](const Seed &guess) -> std::optional<std::string> {
    auto found = inferred.find(guess.symbol);
    return found == inferred.end() ? std::nullopt : found->second;
  };

  // A size the code reads from an input of its: one only views have is no
  // value the code could know.
  std::set<std::string> inputClasses;
  for (const Seed &guess : guesses)
    if (std::optional<std::string> c = classOf(guess);
        c && description.tensors[static_cast<size_t>(guess.tensor)].op ==
                 GGML_OP_NONE)
      inputClasses.insert(*c);

  // Each class left a symbol, s0, s1, ... in the order they are read.
  std::map<std::string, std::string> names;
  std::vector<Seed> seeds;
  std::map<std::string, int64_t> bounds;
  for (const Seed &guess : guesses) {
    std::optional<std::string> c = classOf(guess);
    if (!c || !inputClasses.count(*c))
      continue;
    auto [name, inserted] = names.emplace(*c, "");
    if (inserted)
      name->second = "s" + std::to_string(names.size() - 1);
    seeds.push_back({guess.tensor, guess.dimension, name->second});
    int64_t &maximum = bounds.emplace(name->second, kUnbounded).first->second;
    if (description.tensors[static_cast<size_t>(guess.tensor)].op ==
        GGML_OP_VIEW)
      maximum = std::min(maximum,
                         viewBound(description, guess.tensor, guess.dimension));
  }
  if (seeds.empty())
    return {};
  symbolize(graph, seeds, bounds, /*infer=*/false);

  std::map<std::string, capture::SymbolSpec::Symbol> symbols;
  for (const Seed &seed : seeds) {
    capture::SymbolSpec::Symbol &symbol = symbols[seed.symbol];
    symbol.name = seed.symbol;
    symbol.maximum = bounds.at(seed.symbol);
    symbol.places.push_back({seed.tensor, seed.dimension, 1});
  }
  capture::SymbolSpec spec;
  for (auto &[name, symbol] : symbols)
    spec.symbols.push_back(std::move(symbol));
  const size_t count = spec.symbols.size();
  for (size_t a = 0; a < count; ++a)
    for (size_t b = a + 1; b < count; ++b) {
      capture::SymbolSpec::Symbol product;
      product.name = spec.symbols[a].name + "_" + spec.symbols[b].name;
      product.maximum = spec.symbols[a].maximum * spec.symbols[b].maximum;
      product.product = {spec.symbols[a].name, spec.symbols[b].name};
      spec.symbols.push_back(std::move(product));
    }
  return spec;
}

} // namespace

namespace hip::ggml::capture {

std::unique_ptr<SplitCapture> captureSplit(const SplitNodes &split,
                                           const SplitTensors &tensors,
                                           const SymbolSpec *symbols) {
  auto result = std::make_unique<SplitCapture>();
  if (split.nodes.empty())
    fail("the split computes nothing");
  result->description = describeSplit(split, tensors);
  // What the code computes: the nodes it hands out (as described: not a view
  // of storage outside the split) and what they read. A view nothing of it
  // reads is not captured - its extents (a cache window's length) would be
  // numbers the code is compiled for, to no use.
  std::unordered_map<const ggml_tensor *, size_t> indexOf;
  for (size_t i = 0; i < result->description.originals.size(); ++i)
    indexOf.emplace(result->description.originals[i], i);
  std::vector<const ggml_tensor *> nodes;
  std::vector<bool> exported;
  std::unordered_set<const ggml_tensor *> read;
  for (size_t i = split.nodes.size(); i-- > 0;) {
    const ggml_tensor *node = split.nodes[i];
    const bool output = result->description.tensors[indexOf.at(node)].output;
    const bool view = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                      node->op == GGML_OP_PERMUTE ||
                      node->op == GGML_OP_TRANSPOSE;
    if (view && !output && read.count(node) == 0)
      continue;
    nodes.push_back(node);
    exported.push_back(output);
    for (const ggml_tensor *source : node->src)
      if (source != nullptr)
        read.insert(source);
    if (node->view_src != nullptr)
      read.insert(node->view_src);
  }
  std::reverse(nodes.begin(), nodes.end());
  std::reverse(exported.begin(), exported.end());
  std::unordered_map<const ggml_tensor *, int64_t> ids;
  CapturedGraph graph = observeSplit(nodes, exported, "split", &ids);
  if (!graph.untypedNodes.empty())
    fail("the split holds an operation the capture has no semantics for: ",
         graph.untypedNodes.front().operation);
  nameValues(graph, result->description, ids);
  if (symbols == nullptr) {
    try {
      result->symbols = deriveSymbols(result->description, graph);
    } catch (const std::exception &error) {
      // Not derivable at these sizes: compiled for them.
      result->symbols = {};
      result->symbolsError = error.what();
    }
  } else {
    result->symbols = *symbols;
  }
  result->description.originals.clear();
  result->graph = std::move(graph);
  return result;
}

void compileSplit(const SplitCapture &captured,
                  const std::string &directoryName) {
  const fs::path directory(directoryName);
  fs::create_directories(directory);

  hip::ggml::emit::Options emitOptions;
  for (const SymbolSpec::Symbol &s : captured.symbols.symbols) {
    hip::ggml::emit::CallerSymbol symbol;
    symbol.name = s.name;
    symbol.minimum = s.minimum;
    symbol.maximum = s.maximum;
    symbol.product = s.product;
    for (const SymbolSpec::Place &p : s.places)
      symbol.places.push_back({"t" + std::to_string(p.tensor),
                               static_cast<unsigned>(p.dimension), p.scale});
    emitOptions.callerSymbols.push_back(std::move(symbol));
  }

  // The module, its extents derived: what the compiler takes, and what the
  // bindings and shapes are read from.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::ggml::GgmlDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          hip::ggml::emit::serializeModule(captured.graph, emitOptions),
          &context);
  if (!module)
    fail("cannot parse the emitted module");
  mlir::PassManager manager(&context);
  manager.addPass(mlir::ggml::createGgmlSymbolizeShapesPass());
  if (mlir::failed(manager.run(*module)))
    fail("the compiler cannot derive the split's extents");
  std::string text;
  {
    llvm::raw_string_ostream stream(text);
    module->print(stream);
  }
  {
    std::ofstream out(directory / "model.ggml.mlir");
    out << text << "\n";
  }

  // hip-ep's compiler, in this process: LLVM bitcode, its constants in
  // constants.bin beside it.
  mlir::hip::CompilationOptionsT options;
  mlir::hip::DiskFileSystem files(directory.string().c_str());
  hip::compiler::CompilerDriver driver;
  driver.setFileSystem(&files);
  std::string compileError;
  if (!driver.compile(text, (directory / "model.bc").string(), options,
                      compileError))
    fail("the HIP compiler refused the module",
         compileError.empty() ? std::string() : ": " + compileError);

  write(directory / "bindings.json", bindings(captured.graph, *module), 2);
  write(directory / "shapes.json", shapes(*module, captured.description));
  write(directory / "split.json", toJson(captured.description));
  write(directory / "symbols.json", toJson(captured.symbols), 2);
}

} // namespace hip::ggml::capture
