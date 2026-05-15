/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// hip-inspect-bc - Print metadata embedded in a compiled MLIR bitcode artifact
//===----------------------------------------------------------------------===//
// Loads a `.bc` produced by hip-compiler, parses the IR with
// `llvm::parseBitcodeFile`, reads the `@__metadata_json` global directly,
// and prints a human-readable summary of the model's inputs, outputs, and
// constants.
//
// Unlike hip-test-bc this tool does *not* JIT-compile or link the bitcode.
// That is a deliberate choice: inspection must work on plain dev boxes that
// have no ROCm / GPU runtime installed (no MIOpen.dll, no libamdhip64.dll),
// where module-level global constructors emitted by hipcc would otherwise
// fail JIT-link the moment we tried to materialize them. Parsing the IR
// and walking globals avoids all of that.
//
// Usage:
//   hip-inspect-bc <model.bc> [--json]
//
//   --json   Dump the raw JSON instead of the formatted summary.
//===----------------------------------------------------------------------===//

#include "CrashHandler.h"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // Conversion warnings in LLVM JSON.h
#endif
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// Name of the global emitted by the GenerateInterface pass (see
// lib/Dialect/Transforms/GenerateInterface.cpp::generateMetadataGlobal).
// Internal, constant `[N x i8]` array holding the null-terminated JSON.
static constexpr const char *kMetadataGlobalName = "__metadata_json";

// Return a human-readable element type string from element_size bytes.
static std::string elementTypeName(int64_t element_size) {
  switch (element_size) {
  case 1:
    return "int8";
  case 2:
    return "float16";
  case 4:
    return "float32";
  case 8:
    return "float64/int64";
  default:
    return "unknown(" + std::to_string(element_size) + "B)";
  }
}

// Format a shape array as "[d0, d1, ...]".
static std::string formatShape(const llvm::json::Array *shapeArr) {
  if (!shapeArr || shapeArr->empty())
    return "[]";
  std::string s = "[";
  for (size_t i = 0; i < shapeArr->size(); ++i) {
    if (i)
      s += ", ";
    if (auto v = (*shapeArr)[i].getAsInteger())
      s += std::to_string(*v);
    else
      s += "?";
  }
  s += "]";
  return s;
}

// Print formatted summary of parsed metadata JSON.
static bool printSummary(const char *json_str) {
  auto parsed = llvm::json::parse(llvm::StringRef(json_str));
  if (!parsed) {
    std::cerr << "Failed to parse metadata JSON: ";
    llvm::handleAllErrors(parsed.takeError(), [](const llvm::ErrorInfoBase &e) {
      std::cerr << e.message();
    });
    std::cerr << "\n";
    return false;
  }

  auto *root = parsed->getAsObject();
  if (!root) {
    std::cerr << "Metadata JSON root is not an object.\n";
    return false;
  }

  if (auto v = root->getInteger("version"))
    std::cout << "Version         : " << *v << "\n";

  if (auto cf = root->getString("constants_filename"))
    std::cout << "Constants file  : " << cf->str() << "\n";

  if (auto *consts = root->getArray("constants")) {
    std::cout << "Constants       : " << consts->size() << "\n";
    int64_t total = 0;
    for (auto &c : *consts) {
      if (auto *obj = c.getAsObject()) {
        if (auto sz = obj->getInteger("size"))
          total += *sz;
      }
    }
    std::cout << "Constants total : " << total << " bytes\n";
  }

  std::cout << "\n";

  auto printTensors = [&](const char *key, const char *label) {
    auto *arr = root->getArray(key);
    int64_t count = arr ? static_cast<int64_t>(arr->size()) : 0;
    std::string countKey =
        std::string(key) == "inputs" ? "input_count" : "output_count";
    if (auto c = root->getInteger(countKey))
      count = *c;

    std::cout << label << " (" << count << "):\n";
    if (!arr) {
      std::cout << "  (none)\n";
      return;
    }
    for (size_t i = 0; i < arr->size(); ++i) {
      auto *obj = (*arr)[i].getAsObject();
      if (!obj)
        continue;
      std::string dtype = "float32";
      if (auto es = obj->getInteger("element_size"))
        dtype = elementTypeName(*es);
      std::string shape = formatShape(obj->getArray("shape"));
      std::cout << "  [" << i << "] shape=" << shape << "  dtype=" << dtype
                << "\n";
    }
  };

  printTensors("inputs", "Inputs");
  std::cout << "\n";
  printTensors("outputs", "Outputs");

  return true;
}

// Read `@__metadata_json` out of the bitcode module. The global is emitted
// as an internal constant `[N x i8]` whose initializer is a
// `ConstantDataArray` -- `getAsCString` strips the trailing NUL and gives
// us a `StringRef` we can copy into a std::string for printing.
static bool extractMetadataJson(llvm::Module &module, std::string &out) {
  llvm::GlobalVariable *gv = module.getGlobalVariable(
      kMetadataGlobalName, /*AllowInternal=*/true);
  if (!gv) {
    std::cerr << "Global '" << kMetadataGlobalName
              << "' not found in bitcode -- not a hip-compiler artifact?\n";
    return false;
  }
  if (!gv->hasInitializer()) {
    std::cerr << "Global '" << kMetadataGlobalName << "' has no initializer\n";
    return false;
  }
  auto *cda =
      llvm::dyn_cast<llvm::ConstantDataArray>(gv->getInitializer());
  if (!cda || !cda->isString()) {
    std::cerr << "Global '" << kMetadataGlobalName
              << "' is not a C string constant\n";
    return false;
  }
  out = cda->getAsCString().str();
  return true;
}

static void printHelp(const char *argv0) {
  std::cout
      << "Print metadata embedded in a compiled MLIR bitcode artifact.\n\n"
      << "Usage: " << argv0 << " <model.bc> [--json]\n\n"
      << "Options:\n"
      << "  --json   Dump raw metadata JSON instead of formatted summary\n"
      << "  -h, --help   Show this help\n";
}

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-inspect-bc");
  std::string bcPath;
  bool dumpJson = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json") {
      dumpJson = true;
    } else if (arg == "-h" || arg == "--help") {
      printHelp(argv[0]);
      return 0;
    } else if (arg[0] != '-') {
      bcPath = arg;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printHelp(argv[0]);
      return 1;
    }
  }

  if (bcPath.empty()) {
    printHelp(argv[0]);
    return 1;
  }

  // mmap the file -- `getFile` gives us a `MemoryBuffer` that the
  // BitcodeReader can consume without an extra copy.
  auto bufOrErr = llvm::MemoryBuffer::getFile(bcPath);
  if (!bufOrErr) {
    std::cerr << "Failed to open '" << bcPath
              << "': " << bufOrErr.getError().message() << "\n";
    return 1;
  }

  llvm::LLVMContext ctx;
  auto moduleOrErr =
      llvm::parseBitcodeFile(bufOrErr.get()->getMemBufferRef(), ctx);
  if (!moduleOrErr) {
    std::cerr << "Failed to parse bitcode '" << bcPath << "': ";
    llvm::handleAllErrors(moduleOrErr.takeError(),
                          [](const llvm::ErrorInfoBase &e) {
                            std::cerr << e.message();
                          });
    std::cerr << "\n";
    return 1;
  }
  std::unique_ptr<llvm::Module> module = std::move(*moduleOrErr);

  std::string json;
  if (!extractMetadataJson(*module, json))
    return 1;

  if (dumpJson) {
    std::cout << json << "\n";
    return 0;
  }

  return printSummary(json.c_str()) ? 0 : 1;
}
