/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Engine.h"

#include "Graph.h"

#include "LlvmIrJit.h"
#include "custom_op_mlir.hpp"
#include "hip/Frontend/Ggml/SplitCompiler.h"
#include "hip/Support/DiskFileSystem.h"
#include "hip/artifact_abi.h"

#include <hip/hip_runtime_api.h>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <iterator>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>

namespace hipep_backend {
namespace {

namespace fs = std::filesystem;
using capture::SplitDescription;
using mlir_compilation::output_allocator_t;
using mlir_compilation::span_t;
using mlir_compilation::tensor_t;
using mlir_compilation::customop::LlvmIrJit;
namespace abi = hipdnn::abi;

std::string environment(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return value && *value ? value : fallback;
}

bool logging() {
  static const bool log = environment("HIPEP_BACKEND_LOG", "0") != "0";
  return log;
}

/// This library's file.
const std::string &libraryPath() {
  static const std::string path = [] {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&libraryPath), &info) == 0 ||
        !info.dli_fname)
      return std::string();
    return fs::absolute(info.dli_fname).string();
  }();
  return path;
}

/// Which compiler a refusal is of: a library built since is asked again.
const std::string &compilerIdentity() {
  static const std::string identity = [] {
    std::error_code ec;
    const fs::path path(libraryPath());
    const auto size = fs::file_size(path, ec);
    const auto time = fs::last_write_time(path, ec).time_since_epoch().count();
    return path.string() + ":" + std::to_string(size) + ":" +
           std::to_string(time);
  }();
  return identity;
}

/// The document at `path`; null when there is none or it is no JSON.
llvm::json::Value readJson(const fs::path &path) {
  std::ifstream in(path);
  if (!in)
    return nullptr;
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  llvm::Expected<llvm::json::Value> document = llvm::json::parse(text);
  if (!document) {
    llvm::consumeError(document.takeError());
    return nullptr;
  }
  return std::move(*document);
}

/// The document on one line, its objects' keys sorted.
std::string printJson(const llvm::json::Value &document) {
  return llvm::formatv("{0}", document).str();
}

void writeJson(const fs::path &path, const llvm::json::Value &document) {
  const fs::path temporary = path.string() + ".tmp." + std::to_string(getpid());
  {
    std::ofstream out(temporary);
    out << printJson(document) << "\n";
  }
  std::error_code ignored;
  fs::rename(temporary, path, ignored);
}

/// The bytes a tensor of these sizes spans (ggml_nbytes).
size_t bytesOf(const SplitDescription::Tensor &t) {
  const hip::ggml::TypeLayout layout =
      hip::ggml::typeLayout(t.type).value_or(hip::ggml::TypeLayout{0, 1});
  const size_t block = static_cast<size_t>(layout.blockElements);
  size_t bytes = block == 1 ? static_cast<size_t>(layout.typeSize)
                            : static_cast<size_t>(t.ne[0]) * t.nb[0] /
                                  std::max<size_t>(block, 1);
  for (int d = block == 1 ? 0 : 1; d < 4; ++d)
    bytes += static_cast<size_t>(t.ne[d] - 1) * t.nb[d];
  for (int d = 0; d < 4; ++d)
    if (t.ne[d] == 0)
      return 0;
  return bytes;
}

} // namespace

/// One split's code: the JIT-loaded artifact, its state, and which described
/// tensor each of its inputs and outputs is.
struct Compiled {
  std::unique_ptr<LlvmIrJit> jit;
  /// What it reads its constants through (constants.bin beside the code).
  std::unique_ptr<mlir::hip::DiskFileSystem> files;
  void *state = nullptr;
  int (*compute)(void *, span_t *) = nullptr;
  int (*cleanup)(void *) = nullptr;
  void (*beginCompute)(void *) = nullptr;
  void (*setOutputAllocator)(void *, const output_allocator_t *) = nullptr;
  /// The inputs' shapes (-1 where the code is symbolic) and element sizes,
  /// from the artifact's metadata.
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<int64_t> inputElementSizes;
  /// Per input, per output: the described tensor it is.
  std::vector<int32_t> inputs;
  std::vector<int32_t> outputs;
  /// `outputs`, sorted.
  std::vector<int32_t> sortedOutputs;
  /// Per output: the input of the state it updates in place (the code writes
  /// the rows it changes into the input's memory and copies nothing when
  /// the output is handed that memory), or -1.
  std::vector<int> donatedFrom;
  /// Which descriptions it runs.
  std::shared_ptr<const capture::SplitShapes> shapes;
  /// One run at a time: an artifact's state is its own.
  mutable std::mutex runMutex;
  /// Where an output is written first when ggml put it over one of the
  /// split's inputs (ggml-alloc reuses an input's memory after its last
  /// reader; the code may write the output before it has read the input).
  struct Staging {
    void *data = nullptr;
    size_t bytes = 0;
  };
  mutable std::vector<Staging> staging;
  /// Zeros: the rows of a padded input (Graph::padded).
  mutable Staging zeros;

  Compiled() = default;
  Compiled(const Compiled &) = delete;
  Compiled &operator=(const Compiled &) = delete;
  ~Compiled() {
    if (state && cleanup)
      cleanup(state);
    for (Staging &buffer : staging)
      if (buffer.data)
        (void)hipFree(buffer.data);
    if (zeros.data)
      (void)hipFree(zeros.data);
  }
};

/// Whether `code` runs `description`: its sizes, and every node it hands out
/// (dynamic-shape code may hand out more: what any split it was compiled for
/// did).
static bool canRun(const Compiled &code, const SplitDescription &description) {
  if (!code.shapes || !code.shapes->fits(description))
    return false;
  for (size_t i = 0; i < description.tensors.size(); ++i)
    if (description.tensors[i].output &&
        !std::binary_search(code.sortedOutputs.begin(),
                            code.sortedOutputs.end(), static_cast<int32_t>(i)))
      return false;
  return true;
}

struct Engine::SignatureEntry {
  enum class State { Unknown, Compiling, Ready, Failed } state = State::Unknown;
  std::shared_ptr<Compiled> code;
  std::string error;
};

struct Engine::DynamicCode {
  /// Its codes, one for each assignment of symbols its splits were compiled
  /// with (a prompt and a step of several sequences may be one graph whose
  /// extents follow from different symbols), each its latest generation's.
  struct Variant {
    std::shared_ptr<Compiled> code;
    std::string symbols;
    std::string directory;
  };
  std::vector<Variant> variants;
  /// The latest generation in the cache.
  long generation = 0;
  /// The symbols its code could not be compiled with: not tried again.
  std::set<std::string> refused;
  /// Why a split of it could be compiled neither with dynamic shapes nor
  /// with its own static shapes, "" otherwise: its splits of other sizes are
  /// not compiled either (what failed is the structure, which they share).
  std::string failure;

  /// The code that runs `description`, or null.
  std::shared_ptr<Compiled> codeFor(const SplitDescription &description) const {
    for (const Variant &v : variants)
      if (v.code && canRun(*v.code, description))
        return v.code;
    return nullptr;
  }
  Variant *variantFor(const std::string &symbols) {
    for (Variant &v : variants)
      if (v.symbols == symbols)
        return &v;
    return nullptr;
  }
};

Engine::Engine(int device, const char *cacheDir) : device_(device) {
  const char *home = std::getenv("HOME");
  cacheDir_ =
      cacheDir && *cacheDir
          ? cacheDir
          : environment("HIPEP_BACKEND_CACHE",
                        std::string(home ? home : "/tmp") + "/.cache/hipep");
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || device < 0 ||
      device >= devices)
    throw std::runtime_error("no HIP device " + std::to_string(device));
  std::error_code ec;
  fs::create_directories(cacheDir_, ec);
}

Engine::~Engine() {
  stop();
  if (logging() || compiles_ > 0)
    std::fprintf(stderr,
                 "hipep_backend: %lld compiles (%lld refused) in %.1f s\n",
                 static_cast<long long>(compiles_),
                 static_cast<long long>(failures_), compileSeconds_);
}

void Engine::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  if (thread_.joinable())
    thread_.join();
}

std::shared_ptr<Engine::SignatureEntry>
Engine::signatureEntryOf(const std::string &signature) {
  std::shared_ptr<SignatureEntry> entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_ptr<SignatureEntry> &slot = signatureEntries_[signature];
    if (slot)
      return slot;
    slot = std::make_shared<SignatureEntry>();
    entry = slot;
  }
  // Compiled in an earlier run: from the cache, at once.
  const fs::path directory = fs::path(cacheDir_) / signature;
  std::string error;
  std::shared_ptr<Compiled> code;
  if (fs::exists(directory / "model.bc")) {
    code = load(directory.string(), error);
    // Compiled by another build (its kernel library is not this one's):
    // compiled again.
    if (!code) {
      if (logging())
        std::fprintf(stderr,
                     "hipep_backend: split %s in the cache is stale: %s\n",
                     signature.substr(0, 12).c_str(), error.c_str());
      std::error_code ignored;
      fs::remove_all(directory, ignored);
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (code) {
    entry->code = code;
    entry->state = SignatureEntry::State::Ready;
  } else if (std::ifstream failed(fs::path(cacheDir_) /
                                  (signature + ".failed"));
             failed) {
    std::string compiler;
    std::getline(failed, compiler);
    if (compiler == compilerIdentity()) {
      entry->state = SignatureEntry::State::Failed;
      entry->error = "refused in an earlier run";
    }
  }
  return entry;
}

std::shared_ptr<Engine::DynamicCode>
Engine::dynamicCodeOf(const std::string &name) {
  std::shared_ptr<DynamicCode> dynamic;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_ptr<DynamicCode> &slot = dynamicCodes_[name];
    if (slot)
      return slot;
    slot = std::make_shared<DynamicCode>();
    dynamic = slot;
  }
  const fs::path cache(cacheDir_);
  // Its codes: d<dynamic signature>-<generation>, the latest of each
  // symbols'.
  std::map<long, fs::path> generations;
  const std::string prefix = "d" + name + "-";
  std::error_code ec;
  for (const fs::directory_entry &e : fs::directory_iterator(cache, ec)) {
    const std::string file = e.path().filename().string();
    if (file.compare(0, prefix.size(), prefix) != 0 ||
        file.find_first_not_of("0123456789", prefix.size()) !=
            std::string::npos ||
        !fs::exists(e.path() / "model.bc"))
      continue;
    generations.emplace(std::stol(file.substr(prefix.size())), e.path());
  }
  std::vector<DynamicCode::Variant> variants;
  for (auto it = generations.rbegin(); it != generations.rend(); ++it) {
    const llvm::json::Value symbols = readJson(it->second / "symbols.json");
    const std::string symbolsKey =
        symbols.getAsObject() != nullptr ? printJson(symbols) : "";
    std::error_code ignored;
    if (std::any_of(variants.begin(), variants.end(),
                    [&](const DynamicCode::Variant &v) {
                      return v.symbols == symbolsKey;
                    })) {
      // Covered by a later generation of these symbols.
      fs::remove_all(it->second, ignored);
      continue;
    }
    std::string error;
    std::shared_ptr<Compiled> code = load(it->second.string(), error);
    if (!code) {
      if (logging())
        std::fprintf(
            stderr,
            "hipep_backend: dynamic-shape code %s in the cache is stale: %s\n",
            name.substr(0, 12).c_str(), error.c_str());
      fs::remove_all(it->second, ignored);
      continue;
    }
    variants.push_back({code, symbolsKey, it->second.filename().string()});
  }
  const llvm::json::Value refused =
      readJson(cache / ("d" + name + ".refused.json"));
  std::string failure;
  if (std::ifstream failed(cache / ("d" + name + ".failed")); failed) {
    std::string compiler;
    std::getline(failed, compiler);
    if (compiler == compilerIdentity())
      std::getline(failed, failure);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  dynamic->failure = failure;
  dynamic->variants = std::move(variants);
  dynamic->generation = generations.empty() ? 0L : generations.rbegin()->first;
  if (const llvm::json::Object *o = refused.getAsObject();
      o != nullptr && o->getString("compiler") == compilerIdentity())
    if (const llvm::json::Array *list = o->getArray("refused"))
      for (const llvm::json::Value &r : *list)
        if (std::optional<llvm::StringRef> symbols = r.getAsString())
          dynamic->refused.insert(symbols->str());
  return dynamic;
}

hipep_backend_code *Engine::readyLocked(SignatureEntry &entry,
                                        DynamicCode &dynamic,
                                        const std::string &signature,
                                        const Graph &graph) {
  const SplitDescription &description = graph.description;
  if (entry.state != SignatureEntry::State::Ready)
    if (std::shared_ptr<Compiled> code = dynamic.codeFor(description)) {
      entry.code = code;
      entry.state = SignatureEntry::State::Ready;
    }
  if (entry.state != SignatureEntry::State::Ready)
    return nullptr;
  std::unique_ptr<hipep_backend_code> &handle =
      codes_[signature + graph.variant];
  if (!handle || handle->compiled != entry.code) {
    auto newHandle = std::make_unique<hipep_backend_code>();
    newHandle->engine = this;
    newHandle->compiled = entry.code;
    newHandle->description = description;
    newHandle->description.originals.clear();
    newHandle->signature = signature;
    newHandle->padded = graph.padded;
    // A handle given out stays valid: the old one is kept beside.
    if (handle)
      codes_.emplace(signature + graph.variant + "@" +
                         std::to_string(codes_.size()),
                     std::move(handle));
    handle = std::move(newHandle);
  }
  return handle.get();
}

hipep_backend_code *Engine::find(const hipep_backend_split_desc &desc,
                                 std::string &error) {
  std::unique_ptr<Graph> graph;
  try {
    graph = std::make_unique<Graph>(desc);
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
  const std::string signature = capture::signature(graph->description);
  std::shared_ptr<SignatureEntry> entry = signatureEntryOf(signature);
  std::shared_ptr<DynamicCode> dynamic =
      dynamicCodeOf(capture::dynamicSignature(graph->description));
  std::lock_guard<std::mutex> lock(mutex_);
  hipep_backend_code *code = readyLocked(*entry, *dynamic, signature, *graph);
  if (!code)
    error = entry->state == SignatureEntry::State::Failed ? entry->error
                                                          : "not compiled";
  return code;
}

hipep_backend_code *Engine::compile(const hipep_backend_split_desc &desc,
                                    bool wait, std::string &error) {
  std::unique_ptr<Graph> graph;
  try {
    graph = std::make_unique<Graph>(desc);
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
  const std::string signature = capture::signature(graph->description);
  const std::string dynamicSignature =
      capture::dynamicSignature(graph->description);
  std::shared_ptr<SignatureEntry> entry = signatureEntryOf(signature);
  std::shared_ptr<DynamicCode> dynamic = dynamicCodeOf(dynamicSignature);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (hipep_backend_code *code =
            readyLocked(*entry, *dynamic, signature, *graph))
      return code;
    if (entry->state == SignatureEntry::State::Failed) {
      error = entry->error;
      return nullptr;
    }
    if (entry->state == SignatureEntry::State::Compiling) {
      error = "being compiled";
      return nullptr;
    }
    if (!dynamic->failure.empty()) {
      entry->state = SignatureEntry::State::Failed;
      entry->error =
          "a split of its structure was refused: " + dynamic->failure;
      error = entry->error;
      return nullptr;
    }
    if (environment("HIPEP_BACKEND_COMPILE", "1") == "0") {
      error = "compiling is off (HIPEP_BACKEND_COMPILE=0)";
      return nullptr;
    }
    // Compiled with the symbols the compiler derives from it - dynamic-shape
    // code, or static-shape code of its own -, unless dynamic-shape code
    // compiled meanwhile fits it.
    entry->state = SignatureEntry::State::Compiling;
  }
  Job job;
  job.key = "d" + dynamicSignature;
  job.dynamicSignature = dynamicSignature;
  job.dynamic = true;
  job.desc = std::make_shared<DescriptionCopy>(desc);
  job.requestedSignature = signature;
  if (!wait) {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = false;
    queue_.push_back(std::move(job));
    if (!thread_.joinable())
      thread_ = std::thread([this] { worker(); });
    wake_.notify_one();
    error = "being compiled";
    return nullptr;
  }
  process(std::move(job));
  std::lock_guard<std::mutex> lock(mutex_);
  hipep_backend_code *code = readyLocked(*entry, *dynamic, signature, *graph);
  if (!code)
    error = entry->error.empty() ? "not compiled" : entry->error;
  return code;
}

void Engine::process(Job job) {
  std::string error;
  std::unique_ptr<capture::SplitCapture> captured;
  const capture::SymbolSpec staticSymbols;
  std::unique_ptr<Graph> graph;
  try {
    graph = std::make_unique<Graph>(job.desc->get());
  } catch (const std::exception &e) {
    error = e.what();
  }
  if (graph) {
    // Dynamic-shape code compiled since it was asked for may run it.
    std::shared_ptr<SignatureEntry> entry =
        signatureEntryOf(job.requestedSignature);
    std::shared_ptr<DynamicCode> dynamic = dynamicCodeOf(job.key.substr(1));
    std::lock_guard<std::mutex> lock(mutex_);
    entry->state = SignatureEntry::State::Unknown;
    if (readyLocked(*entry, *dynamic, job.requestedSignature, *graph))
      return;
    entry->state = SignatureEntry::State::Compiling;
  }
  try {
    if (!graph) {
    } else if (job.dynamic) {
      // The symbols derived from this split: dynamic-shape code of them,
      // unless there are none, they were refused, or the dynamic-shape code
      // of them does not fit it - then static-shape code of its own.
      std::shared_ptr<DynamicCode> dynamic = dynamicCodeOf(job.key.substr(1));
      const std::vector<bool> originalExported = graph->split.exported;
      captured = capture::captureSplit(graph->split, graph->tensors, nullptr);
      job.symbols = captured->symbols.empty()
                        ? ""
                        : printJson(capture::toJson(captured->symbols));
      bool staticShapes = false;
      std::string staticReason;
      std::shared_ptr<Compiled> previous;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const DynamicCode::Variant *existing = dynamic->variantFor(job.symbols);
        previous = existing ? existing->code : nullptr;
        std::string fitError;
        staticReason =
            job.symbols.empty()
                ? "no symbols derived" + (captured->symbolsError.empty()
                                              ? std::string()
                                              : ": " + captured->symbolsError)
            : dynamic->refused.count(job.symbols) != 0
                ? "its dynamic-shape code refused these symbols"
            : previous && previous->shapes &&
                    !previous->shapes->fits(graph->description, &fitError)
                ? "its dynamic-shape code has these symbols and does not "
                  "fit it: " +
                      fitError
                : "";
        staticShapes = !staticReason.empty();
        if (!staticShapes)
          job.directory = job.key + "-" + std::to_string(++dynamic->generation);
      }
      if (!staticShapes && previous) {
        // The nodes its code of these symbols handed out are handed out too:
        // the new code runs the splits the old one ran.
        std::unordered_map<const hip::ggml::Tensor *, size_t> nodeIndex;
        for (size_t j = 0; j < graph->split.nodes.size(); ++j)
          nodeIndex.emplace(graph->split.nodes[j], j);
        for (int32_t i : previous->sortedOutputs)
          if (static_cast<size_t>(i) < graph->description.originals.size())
            if (auto found = nodeIndex.find(
                    graph->description.originals[static_cast<size_t>(i)]);
                found != nodeIndex.end())
              graph->split.exported[found->second] = true;
        const capture::SymbolSpec spec = captured->symbols;
        captured = capture::captureSplit(graph->split, graph->tensors, &spec);
      }
      if (staticShapes) {
        if (logging())
          std::fprintf(
              stderr,
              "hipep_backend: split %s compiled with static shapes: %s\n",
              job.requestedSignature.substr(0, 12).c_str(),
              staticReason.c_str());
        graph->split.exported = originalExported;
        captured =
            capture::captureSplit(graph->split, graph->tensors, &staticSymbols);
        job.key = job.requestedSignature;
        job.dynamic = false;
        job.symbols.clear();
        job.directory = job.requestedSignature;
      }
    } else {
      captured =
          capture::captureSplit(graph->split, graph->tensors, &staticSymbols);
    }
  } catch (const std::exception &e) {
    error = std::string("capture failed: ") + e.what();
  }
  if (!captured) {
    job.key = job.requestedSignature;
    job.dynamic = false;
    finish(job, nullptr, error);
    return;
  }
  std::shared_ptr<Compiled> code = build(job, *captured, error);
  finish(job, code, error);
  if (!job.dynamic)
    return;
  // The new dynamic-shape code is derived from this split; should it be
  // refused, or its guard still not fit the split, the split is compiled with
  // static shapes.
  if (code && canRun(*code, graph->description))
    return;
  if (logging()) {
    std::string fitError;
    if (code && code->shapes)
      code->shapes->fits(graph->description, &fitError);
    std::fprintf(stderr,
                 "hipep_backend: split %s compiled with static shapes: its "
                 "dynamic-shape code %s\n",
                 job.requestedSignature.substr(0, 12).c_str(),
                 code ? ("does not fit it: " + fitError).c_str()
                      : "was refused");
  }
  job.key = job.directory = job.requestedSignature;
  job.dynamic = false;
  job.symbols.clear();
  try {
    captured = nullptr;
    graph = std::make_unique<Graph>(job.desc->get());
    captured =
        capture::captureSplit(graph->split, graph->tensors, &staticSymbols);
    code = build(job, *captured, error);
  } catch (const std::exception &e) {
    code = nullptr;
    error = std::string("capture failed: ") + e.what();
  }
  finish(job, code, error);
}

void Engine::finish(const Job &job, std::shared_ptr<Compiled> code,
                    const std::string &error) {
  const fs::path cache(cacheDir_);
  if (job.dynamic) {
    const std::string name = job.key.substr(1);
    std::shared_ptr<DynamicCode> dynamic = dynamicCodeOf(name);
    llvm::json::Array refusedSymbols;
    std::string replaced;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (code) {
        // The code of these symbols: in place of the earlier generation's,
        // which it covers, beside the others'.
        if (DynamicCode::Variant *existing = dynamic->variantFor(job.symbols)) {
          replaced = existing->directory;
          *existing = {code, job.symbols, job.directory};
        } else {
          dynamic->variants.push_back({code, job.symbols, job.directory});
        }
      }
      // The split that asked: run by this code if it fits, else asked again.
      if (auto requested = signatureEntries_.find(job.requestedSignature);
          requested != signatureEntries_.end() &&
          requested->second->state == SignatureEntry::State::Compiling)
        requested->second->state = SignatureEntry::State::Unknown;
    }
    if (code) {
      // Code of the replaced generation still running here, or in another
      // process, keeps its mapped files.
      std::error_code ec;
      if (!replaced.empty())
        fs::remove_all(cache / replaced, ec);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      dynamic->refused.insert(job.symbols);
      for (const std::string &r : dynamic->refused)
        refusedSymbols.push_back(r);
    }
    writeJson(cache / (job.key + ".refused.json"),
              llvm::json::Object{{"compiler", compilerIdentity()},
                                 {"refused", std::move(refusedSymbols)}});
    if (logging())
      std::fprintf(stderr, "hipep_backend: dynamic-shape code %s refused: %s\n",
                   name.substr(0, 12).c_str(), error.c_str());
    return;
  }
  std::shared_ptr<SignatureEntry> entry = signatureEntryOf(job.key);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    entry->code = code;
    entry->state =
        code ? SignatureEntry::State::Ready : SignatureEntry::State::Failed;
    entry->error = error;
  }
  if (!code) {
    std::ofstream failed(cache / (job.key + ".failed"));
    failed << compilerIdentity() << "\n" << error << "\n";
    // Its dynamic-shape code failed it first: the structure is refused.
    if (!job.dynamicSignature.empty()) {
      std::shared_ptr<DynamicCode> dynamic =
          dynamicCodeOf(job.dynamicSignature);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        dynamic->failure = error;
      }
      std::ofstream(cache / ("d" + job.dynamicSignature + ".failed"))
          << compilerIdentity() << "\n"
          << error << "\n";
    }
    if (logging())
      std::fprintf(stderr, "hipep_backend: split %s: %s\n",
                   job.key.substr(0, 12).c_str(), error.c_str());
  }
}

void Engine::worker() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_) {
        queue_.clear();
        return;
      }
      job = std::move(queue_.front());
      queue_.pop_front();
    }
    process(std::move(job));
  }
}

std::shared_ptr<Compiled> Engine::build(const Job &job,
                                        const capture::SplitCapture &captured,
                                        std::string &error) {
  const fs::path cache(cacheDir_);
  const fs::path directory = cache / job.directory;
  const fs::path work =
      cache / (job.directory + ".work." + std::to_string(getpid()));
  std::error_code ec;
  fs::remove_all(work, ec);
  fs::create_directories(work, ec);
  if (ec) {
    error = "cannot make " + work.string() + ": " + ec.message();
    return nullptr;
  }
  const auto start = std::chrono::steady_clock::now();
  // The split's code, LLVM bitcode, and which compiler made it: code of
  // another is compiled again.
  bool ok = true;
  try {
    capture::compileSplit(captured, work.string());
    std::ofstream(work / "compiler.txt") << compilerIdentity() << "\n";
  } catch (const std::exception &e) {
    ok = false;
    error = std::string("compile failed: ") + e.what();
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++compiles_;
    compileSeconds_ += seconds;
    if (!ok)
      ++failures_;
  }
  if (!ok) {
    // Why is the caller's to keep (`<signature>.failed`).
    fs::remove_all(work, ec);
    return nullptr;
  }
  fs::rename(work, directory, ec);
  if (ec) {
    // Another process compiled it first: take its.
    fs::remove_all(work, ec);
  }
  if (logging())
    std::fprintf(stderr, "hipep_backend: %s %s compiled in %.1f s\n",
                 job.dynamic ? "dynamic-shape code" : "split",
                 job.key.substr(job.dynamic ? 1 : 0, 12).c_str(), seconds);
  return load(directory.string(), error);
}

std::shared_ptr<Compiled> Engine::load(const std::string &directory,
                                       std::string &error) {
  const fs::path dir(directory);
  {
    std::ifstream stamp(dir / "compiler.txt");
    std::string compiler;
    std::getline(stamp, compiler);
    if (compiler != compilerIdentity()) {
      error = "compiled by another build";
      return nullptr;
    }
  }
  std::vector<uint8_t> bitcode;
  {
    std::ifstream in(dir / "model.bc", std::ios::binary);
    bitcode.assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
  }
  if (bitcode.empty()) {
    error = "cannot read model.bc";
    return nullptr;
  }
  if (hipSetDevice(device_) != hipSuccess) {
    error = "cannot select HIP device " + std::to_string(device_);
    return nullptr;
  }
  auto code = std::make_shared<Compiled>();
  // In-process, as the execution provider loads its models: the runtime's
  // bitcode is linked in, the kernel libraries found beside this library.
  code->jit = LlvmIrJit::create(bitcode, "split");
  if (!code->jit) {
    error = "the code cannot be JIT-loaded";
    return nullptr;
  }
  auto init = code->jit->get_method<int, void **, void *, const void *>(
      abi::kInferenceInit);
  auto metadataJson =
      code->jit->get_method<const char *>(abi::kInferenceGetMetadataJson);
  code->compute =
      code->jit->get_method<int, void *, span_t *>(abi::kInferenceCompute);
  code->cleanup = code->jit->get_method<int, void *>(abi::kInferenceCleanup);
  code->beginCompute =
      code->jit->get_method<void, void *>(abi::kRuntimeBeginCompute);
  code->setOutputAllocator =
      code->jit->get_method<void, void *, const output_allocator_t *>(
          abi::kSetOutputAllocator);
  if (!init || !metadataJson || !code->compute || !code->cleanup ||
      !code->setOutputAllocator) {
    error = "the code lacks an artifact entry point";
    return nullptr;
  }
  code->files =
      std::make_unique<mlir::hip::DiskFileSystem>(dir.string().c_str());
  if (int status = init(&code->state, code->files.get(), nullptr);
      status != 0) {
    code->state = nullptr;
    error = "inference_init failed: " + std::to_string(status);
    return nullptr;
  }

  llvm::Expected<llvm::json::Value> metadataDocument =
      llvm::json::parse(metadataJson());
  if (!metadataDocument) {
    error = "the artifact's metadata: " +
            llvm::toString(metadataDocument.takeError());
    return nullptr;
  }
  const llvm::json::Object *metadata = metadataDocument->getAsObject();
  const llvm::json::Value bindingsDocument = readJson(dir / "bindings.json");
  const llvm::json::Object *bindings = bindingsDocument.getAsObject();
  if (metadata == nullptr || bindings == nullptr) {
    error = "no metadata or no bindings.json";
    return nullptr;
  }
  const llvm::json::Array empty;
  const auto arrayOr = [&](const llvm::json::Object &o, llvm::StringRef key) {
    const llvm::json::Array *array = o.getArray(key);
    return array != nullptr ? array : &empty;
  };
  for (const llvm::json::Value &value : *arrayOr(*metadata, "inputs")) {
    std::vector<int64_t> shape;
    int64_t elementSize = 4;
    if (const llvm::json::Object *t = value.getAsObject()) {
      for (const llvm::json::Value &e : *arrayOr(*t, "shape"))
        shape.push_back(e.getAsInteger().value_or(0));
      elementSize = t->getInteger("element_size").value_or(4);
    }
    code->inputShapes.push_back(std::move(shape));
    code->inputElementSizes.push_back(elementSize);
  }
  for (const llvm::json::Value &i : *arrayOr(*bindings, "inputs"))
    code->inputs.push_back(static_cast<int32_t>(i.getAsInteger().value_or(-1)));
  for (const llvm::json::Value &i : *arrayOr(*bindings, "outputs"))
    code->outputs.push_back(
        static_cast<int32_t>(i.getAsInteger().value_or(-1)));
  code->sortedOutputs = code->outputs;
  std::sort(code->sortedOutputs.begin(), code->sortedOutputs.end());
  const size_t outputs = arrayOr(*metadata, "outputs")->size();
  if (code->inputs.size() != code->inputShapes.size() ||
      code->outputs.size() != outputs) {
    error = "the code's inputs and outputs are not the split's tensors";
    return nullptr;
  }
  code->donatedFrom.assign(code->outputs.size(), -1);
  for (const llvm::json::Value &value : *arrayOr(*bindings, "donated")) {
    const llvm::json::Array *pair = value.getAsArray();
    if (pair == nullptr || pair->size() != 2)
      continue;
    const int64_t output = (*pair)[0].getAsInteger().value_or(-1);
    if (output >= 0 && static_cast<size_t>(output) < code->donatedFrom.size())
      code->donatedFrom[output] =
          static_cast<int>((*pair)[1].getAsInteger().value_or(-1));
  }
  if (fs::exists(dir / "shapes.json")) {
    const llvm::json::Value shapes = readJson(dir / "shapes.json");
    try {
      code->shapes = std::make_shared<capture::SplitShapes>(shapes);
    } catch (const std::exception &e) {
      error = std::string("shapes.json: ") + e.what();
      return nullptr;
    }
  }
  return code;
}

namespace {

/// Hands each output the memory the caller gave its tensor (the artifact's
/// output allocator).
struct TensorOutputs {
  const std::vector<size_t> &bytes;
  const std::vector<void *> &targets;
  std::string mismatch;

  static void *allocate(void *self, int64_t outputIndex, const int64_t *shape,
                        int64_t rank, int64_t elementSize) {
    auto &outputs = *static_cast<TensorOutputs *>(self);
    const auto index = static_cast<size_t>(outputIndex);
    if (outputIndex < 0 || index >= outputs.bytes.size())
      return nullptr;
    size_t bytes = static_cast<size_t>(elementSize);
    for (int64_t i = 0; i < rank; ++i)
      bytes *= static_cast<size_t>(shape[i]);
    if (bytes != outputs.bytes[index] || outputs.targets[index] == nullptr) {
      outputs.mismatch = "output " + std::to_string(index) + ": " +
                         std::to_string(bytes) + " bytes newHandle, " +
                         std::to_string(outputs.bytes[index]) + " allocated";
      return nullptr;
    }
    return outputs.targets[index];
  }
};

/// `data`, of at least `bytes` (and 256), all zero.
bool ensureZeroed(void *&data, size_t &capacity, size_t bytes) {
  if (capacity >= bytes)
    return true;
  if (data)
    (void)hipFree(data);
  data = nullptr;
  capacity = 0;
  bytes = std::max<size_t>(bytes, 256);
  if (hipMalloc(&data, bytes) != hipSuccess) {
    data = nullptr;
    return false;
  }
  capacity = bytes;
  return hipMemset(data, 0, bytes) == hipSuccess;
}

bool overlaps(const void *a, size_t aBytes, const void *b, size_t bBytes) {
  const auto *x = static_cast<const char *>(a);
  const auto *y = static_cast<const char *>(b);
  return x < y + bBytes && y < x + aBytes;
}

} // namespace

bool Engine::run(const hipep_backend_code &handle, void *const *data,
                 std::string &error) {
  const Compiled &code = *handle.compiled;
  const SplitDescription &description = handle.description;
  const auto tensor = [&](int32_t index) -> const SplitDescription::Tensor & {
    return description.tensors.at(static_cast<size_t>(index));
  };
  const auto isPadded = [&](int32_t index) {
    return std::binary_search(handle.padded.begin(), handle.padded.end(),
                              index);
  };
  std::lock_guard<std::mutex> lock(code.runMutex);
  if (hipSetDevice(device_) != hipSuccess) {
    error = "cannot select HIP device " + std::to_string(device_);
    return false;
  }
  std::vector<std::vector<int64_t>> shapes(code.inputs.size());
  std::vector<tensor_t> inputs(code.inputs.size());
  // Padded inputs read one buffer of zeros, as large as the largest of
  // them.
  size_t zeroBytes = 0;
  std::vector<size_t> zeroInputs;
  for (size_t i = 0; i < code.inputs.size(); ++i) {
    shapes[i] = code.inputShapes[i];
    const size_t rank = shapes[i].size();
    inputs[i].element_size = static_cast<size_t>(code.inputElementSizes[i]);
    inputs[i].memory_type = mlir_compilation::TENSOR_MEMORY_GPU;
    const SplitDescription::Tensor &described = tensor(code.inputs[i]);
    // A symbolic code's dynamic extents are the tensor's.
    for (size_t d = 0; d < rank; ++d)
      if (shapes[i][d] < 0)
        shapes[i][d] = rank - 1 - d < 4 ? described.ne[rank - 1 - d] : 1;
    inputs[i].data = data[code.inputs[i]];
    if (isPadded(code.inputs[i])) {
      // A row of zeros (index 0 of a gathering).
      zeroBytes = std::max(zeroBytes, bytesOf(described));
      zeroInputs.push_back(i);
    }
    inputs[i].shape = shapes[i].data();
    inputs[i].rank = shapes[i].size();
  }
  if (!zeroInputs.empty()) {
    if (!ensureZeroed(code.zeros.data, code.zeros.bytes, zeroBytes)) {
      error = "cannot allocate " + std::to_string(zeroBytes) + " bytes";
      return false;
    }
    for (size_t i : zeroInputs)
      inputs[i].data = code.zeros.data;
  }
  std::vector<size_t> bytes;
  std::vector<void *> targets;
  std::vector<size_t> staged;
  code.staging.resize(code.outputs.size());
  for (size_t o = 0; o < code.outputs.size(); ++o) {
    void *target = data[code.outputs[o]];
    const size_t size = bytesOf(tensor(code.outputs[o]));
    bytes.push_back(size);
    Compiled::Staging &buffer = code.staging[o];
    const auto stage = [&]() {
      if (buffer.bytes >= size)
        return true;
      if (buffer.data)
        (void)hipFree(buffer.data);
      buffer = {};
      if (hipMalloc(&buffer.data, size) != hipSuccess) {
        buffer.data = nullptr;
        error = "cannot allocate " + std::to_string(size) + " bytes";
        return false;
      }
      buffer.bytes = size;
      return true;
    };
    // A node dynamic-shape code hands out but this split does not, or one of
    // no elements: ggml may have put another tensor over it. Written aside.
    if (!tensor(code.outputs[o]).output || isPadded(code.outputs[o])) {
      if (!stage())
        return false;
      targets.push_back(buffer.data);
      continue;
    }
    if (const int from = code.donatedFrom[o]; from >= 0) {
      // A cache's rows written through a view of all of it.
      if (target == data[code.inputs[from]] &&
          size == bytesOf(tensor(code.inputs[from]))) {
        targets.push_back(target);
        continue;
      }
    }
    bool aliased = false;
    for (size_t i = 0; i < inputs.size() && !aliased; ++i)
      aliased =
          code.inputs[i] >= 0 && overlaps(target, size, inputs[i].data,
                                          bytesOf(tensor(code.inputs[i])));
    if (!aliased) {
      targets.push_back(target);
      continue;
    }
    if (!stage())
      return false;
    targets.push_back(buffer.data);
    staged.push_back(o);
  }
  TensorOutputs tensorOutputs{bytes, targets, {}};
  const output_allocator_t allocator{&tensorOutputs, &TensorOutputs::allocate};
  code.setOutputAllocator(code.state, &allocator);
  if (code.beginCompute)
    code.beginCompute(code.state);
  span_t span{inputs.data(), inputs.size()};
  const int status = code.compute(code.state, &span);
  code.setOutputAllocator(code.state, nullptr);
  if (status != 0) {
    error =
        "inference_compute failed: " + std::to_string(status) +
        (tensorOutputs.mismatch.empty() ? ""
                                        : " (" + tensorOutputs.mismatch + ")");
    return false;
  }
  // inference_compute returns with its stream synchronized.
  for (size_t o : staged)
    if (hipMemcpy(data[code.outputs[o]], targets[o], bytes[o],
                  hipMemcpyDeviceToDevice) != hipSuccess) {
      error = "cannot copy output " + std::to_string(o);
      return false;
    }
  // A device-to-device hipMemcpy may return before it is done, and the
  // caller's next work may be on a stream that does not wait for the null
  // stream: the run is done when the null stream is.
  if (hipStreamSynchronize(nullptr) != hipSuccess) {
    error = "the run's copies did not finish";
    return false;
  }
  return true;
}

} // namespace hipep_backend
