/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Engine - a context of hipep_backend.h: the code of one HIP device's
// splits, compiled in this process and kept in a cache on disk.
//
// Sizes. A split is compiled with dynamic shapes: splits of one structure at
// other sizes have one dynamic signature (SplitGraph.h), and its dynamic-shape
// code is compiled with the symbols the compiler derives from the split
// alone (SplitCompiler.h) and runs every split its guard (SplitShapes) fits.
// A split it
// does not fit has its symbols derived again: other symbols compile the
// dynamic-shape code's next generation; the same ones, none, or refused ones
// compile the split with static shapes, as it is.
//
// Failures. A split the compiler refuses (a hip::ggml::Error: a statement
// about the split, Error.h) it would refuse again: the cache keeps the
// refusal, for every process of this library. A split whose code could not be
// made or loaded otherwise - the cache's disk, memory, the device - has no
// code in this process alone: the next one tries again.

#ifndef HIPEP_BACKEND_ENGINE_H
#define HIPEP_BACKEND_ENGINE_H

#include "hipep_backend.h"

#include "Graph.h"
#include "hip/Frontend/Ggml/SplitGraph.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace hip::ggml::capture {
struct SplitCapture;
}

namespace hipep_backend {

class DescriptionCopy;
class Engine;
struct Compiled;

} // namespace hipep_backend

/// Code that runs splits of one signature: its static-shape or dynamic-shape
/// code, and the split as described (its tensors' sizes).
struct hipep_backend_code {
  hipep_backend::Engine *engine = nullptr;
  std::shared_ptr<hipep_backend::Compiled> compiled;
  hip::ggml::capture::SplitDescription description;
  std::string signature;
  /// The description's tensors of no elements made one row (Graph::padded).
  std::vector<int32_t> padded;
};

namespace hipep_backend {

class Engine {
public:
  /// Throws when there is no such device.
  Engine(int device, const char *cacheDir);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  hipep_backend_code *find(const hipep_backend_split_desc &desc,
                           hipep_backend_status &status, std::string &error);
  hipep_backend_code *compile(const hipep_backend_split_desc &desc, bool wait,
                              hipep_backend_status &status, std::string &error);
  bool run(const hipep_backend_code &code, void *const *data,
           std::string &error);

private:
  /// By signature: the code that runs splits of it (its static-shape code, or
  /// dynamic-shape code that fits), or why there is none.
  struct SignatureEntry;
  /// Dynamic-shape code, by its dynamic signature, and what it was compiled
  /// from.
  struct DynamicCode;
  /// A compile: the split's description, where its code goes.
  struct Job {
    std::string key;
    bool dynamic = false;
    std::shared_ptr<const DescriptionCopy> desc;
    /// The symbols it is compiled with (SymbolSpec JSON), "" for none.
    std::string symbols;
    /// In the cache: dynamic-shape code by its generation (code of an earlier
    /// one may still be running).
    std::string directory;
    /// The signature of the split it was asked for, and its dynamic
    /// signature.
    std::string requestedSignature;
    std::string dynamicSignature;
  };

  std::shared_ptr<SignatureEntry>
  signatureEntryOf(const std::string &signature);
  std::shared_ptr<DynamicCode>
  dynamicCodeOf(const std::string &dynamicSignature);
  /// The code of `entry` if it has some - its static-shape code, or the
  /// dynamic-shape code where it fits -, as a handle; null otherwise. Under
  /// mutex_.
  hipep_backend_code *readyLocked(SignatureEntry &entry, DynamicCode &dynamic,
                                  const std::string &signature,
                                  const Graph &graph);
  /// What find and compile say of `entry` when it has no code: refused,
  /// being compiled or not compiled, and why (`error`). Under mutex_.
  hipep_backend_status statusLocked(SignatureEntry &entry,
                                    const DynamicCode &dynamic,
                                    std::string &error);
  void process(Job job);
  /// `refused`: no `code` because the compiler refused the split, which the
  /// cache then keeps.
  void finish(const Job &job, std::shared_ptr<Compiled> code,
              const std::string &error, bool refused);
  void worker();
  std::shared_ptr<Compiled>
  build(const Job &job, const hip::ggml::capture::SplitCapture &capture,
        std::string &error, bool &refused);
  std::shared_ptr<Compiled> load(const std::string &directory,
                                 std::string &error);
  /// Waits for the compile under way and stops compiling.
  void stop();

  int device_;
  std::string cacheDir_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::map<std::string, std::shared_ptr<SignatureEntry>> signatureEntries_;
  std::map<std::string, std::shared_ptr<DynamicCode>> dynamicCodes_;
  std::map<std::string, std::unique_ptr<hipep_backend_code>> codes_;
  std::deque<Job> queue_;
  bool stopping_ = false;
  std::thread thread_;
  int64_t compiles_ = 0;
  int64_t refusals_ = 0;
  int64_t failures_ = 0;
  double compileSeconds_ = 0;
};

} // namespace hipep_backend

#endif // HIPEP_BACKEND_ENGINE_H
