//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_DIALECT
#define HIPSR_DIALECT

include "mlir/IR/DialectBase.td"

def Hipsr_Dialect : Dialect {
  let name = "hipsr";
  let summary = "Declarative AMD GPU compute dialect with first-class shape regions";
  let description = [{
    An AMD GPU compute dialect that represents tensor computations
    declaratively and models dynamic shapes as first-class shape regions.
  }];
  let cppNamespace = "::mlir::hipsr";

  // Lets the dialect parse/print `#hipsr.mem<...>`.
  let useDefaultAttributePrinterParser = 1;

  // Lets the dialect parse/print `!hipsr.context`.
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Returns a cached memory-mapped view of the file at `path`, loading it on
    /// first use. The buffer lives for the dialect's lifetime (no eviction --
    /// constants are read only during compilation). Returns nullptr on failure.
    /// Thread-safe: the dialect is a shared singleton and MLIR runs passes
    /// multi-threaded, so the cache is guarded by a mutex.
    ::llvm::MemoryBuffer *getOrLoadFileMap(::llvm::StringRef path);

    /// Sink for the constants sidecar written by -hipsr-externalize-constants.
    /// Non-owning; the compile driver injects the same FileSystem it hands the
    /// hip.* pipeline. Null when nobody injected one (e.g. hip-mlir-opt), in
    /// which case the pass writes no sidecar (pure IR transform).
    void setFileSystem(::morphizen::FileSystem *fs) { fileSystem = fs; }
    ::morphizen::FileSystem *getFileSystem() { return fileSystem; }

  private:
    ::llvm::StringMap<std::unique_ptr<::llvm::MemoryBuffer>> fileMaps;
    std::mutex fileMapsMutex;
    ::morphizen::FileSystem *fileSystem = nullptr;
  }];
}

#endif // HIPSR_DIALECT
