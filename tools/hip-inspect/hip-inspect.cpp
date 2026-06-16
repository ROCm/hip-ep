/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// hip-inspect - Print metadata embedded in a compiled model artifact
//===----------------------------------------------------------------------===//
// Prints a human-readable summary of the model's inputs, outputs, and
// constants. Two artifact formats, dispatched by magic bytes:
//   * LLVM bitcode (.bc): parse the IR and read the `@__metadata_json`
//     global -- lightweight, no ROCm/JIT needed.
//   * native .dll/.so: load via morphizen::Plugin and call
//     inference_get_metadata_json (only when built with BUILD_EP).
//
// Usage:
//   hip-inspect <model.{bc,dll,so}> [--json]
//
//   --json   Dump the raw JSON instead of the formatted summary.
//===----------------------------------------------------------------------===//

#include "CrashHandler.h"
#include "artifact_format.h"  // hasNativeMagic
#include "hip/artifact_abi.h" // hipdnn::abi symbol names
#ifdef HIPDNN_INSPECT_NATIVE
// morphizen.hpp must precede plugin.hpp (morphizen/_sanity_check.hpp enforces
// this include order).
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <filesystem>
#endif
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

static constexpr const char *kMetadataGlobalName =
    hipdnn::abi::kMetadataJsonGlobal;

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

  // Schema version
  if (auto v = root->getInteger("version"))
    std::cout << "Version         : " << *v << "\n";

  // Constants
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

  // Inputs
  auto printTensors = [&](const char *key, const char *label) {
    auto *arr = root->getArray(key);
    int64_t count = arr ? static_cast<int64_t>(arr->size()) : 0;
    // Prefer the explicit count field when present.
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

static bool extractMetadataJson(llvm::Module &module, std::string &out) {
  llvm::GlobalVariable *gv =
      module.getGlobalVariable(kMetadataGlobalName, /*AllowInternal=*/true);
  if (!gv) {
    std::cerr << "Global '" << kMetadataGlobalName
              << "' not found in bitcode -- not a hip-compiler artifact?\n";
    return false;
  }
  if (!gv->hasInitializer()) {
    std::cerr << "Global '" << kMetadataGlobalName << "' has no initializer\n";
    return false;
  }
  auto *cda = llvm::dyn_cast<llvm::ConstantDataArray>(gv->getInitializer());
  if (!cda || !cda->isString()) {
    std::cerr << "Global '" << kMetadataGlobalName
              << "' is not a C string constant\n";
    return false;
  }
  out = cda->getAsCString().str();
  return true;
}

#ifdef HIPDNN_INSPECT_NATIVE
// Load a native .dll/.so via morphizen::Plugin and read its metadata by
// calling the exported inference_get_metadata_json (the same symbol
// hip-test uses). Loading resolves the artifact's imports, so its runtime
// dependencies must be present.
static bool extractMetadataNative(const std::string &path, std::string &out) {
  // Plugin::create takes a base path (no extension) and adds the platform
  // suffix itself.
  std::string base = std::filesystem::path(path).replace_extension("").string();
  auto plugin = morphizen::Plugin::create(base.c_str());
  if (!plugin) {
    std::cerr << "Failed to load native artifact: " << path
              << " (check that its dependencies are resolvable)\n";
    return false;
  }
  auto get_meta =
      plugin->get_method<const char *>(hipdnn::abi::kInferenceGetMetadataJson);
  if (!get_meta) {
    std::cerr << "Symbol 'inference_get_metadata_json' not found in: " << path
              << "\n";
    return false;
  }
  const char *json = get_meta();
  if (!json) {
    std::cerr << "inference_get_metadata_json returned null.\n";
    return false;
  }
  out = json;
  return true;
}
#endif // HIPDNN_INSPECT_NATIVE

static void printHelp(const char *argv0) {
  std::cout
      << "Print metadata embedded in a compiled model artifact.\n\n"
      << "Usage: " << argv0 << " <model.{bc,dll,so}> [--json]\n\n"
      << "Formats: LLVM bitcode (.bc) always; native .dll/.so when built with "
         "BUILD_EP.\n"
      << "Options:\n"
      << "  --json   Dump raw metadata JSON instead of formatted summary\n"
      << "  -h, --help   Show this help\n";
}

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-inspect");
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

  auto bufOrErr = llvm::MemoryBuffer::getFile(bcPath);
  if (!bufOrErr) {
    std::cerr << "Failed to open '" << bcPath
              << "': " << bufOrErr.getError().message() << "\n";
    return 1;
  }
  llvm::StringRef buf = bufOrErr.get()->getBuffer();

  std::string json;
  const bool is_native = mlir_compilation::customop::hasNativeMagic(
      reinterpret_cast<const uint8_t *>(buf.data()), buf.size());

  if (is_native) {
#ifdef HIPDNN_INSPECT_NATIVE
    if (!extractMetadataNative(bcPath, json))
      return 1;
#else
    std::cerr << "'" << bcPath
              << "' is a native artifact (PE/ELF), but this hip-inspect was "
                 "built without native support. Reconfigure with BUILD_EP=ON "
                 "to inspect native artifacts.\n";
    return 1;
#endif
  } else {
    llvm::LLVMContext ctx;
    auto moduleOrErr =
        llvm::parseBitcodeFile(bufOrErr.get()->getMemBufferRef(), ctx);
    if (!moduleOrErr) {
      std::cerr << "Failed to parse bitcode '" << bcPath << "': ";
      llvm::handleAllErrors(
          moduleOrErr.takeError(),
          [](const llvm::ErrorInfoBase &e) { std::cerr << e.message(); });
      std::cerr << "\n";
      return 1;
    }
    std::unique_ptr<llvm::Module> module = std::move(*moduleOrErr);
    if (!extractMetadataJson(*module, json))
      return 1;
  }

  if (dumpJson) {
    std::cout << json << "\n";
    return 0;
  }

  return printSummary(json.c_str()) ? 0 : 1;
}
