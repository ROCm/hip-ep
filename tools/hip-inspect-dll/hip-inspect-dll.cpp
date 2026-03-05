/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// hip-inspect-dll - Print metadata embedded in a compiled model DLL
//===----------------------------------------------------------------------===//
// Loads a DLL produced by hip-compiler, calls inference_get_metadata_json(),
// and prints a human-readable summary of the model's inputs, outputs, and
// constants.
//
// Usage:
//   hip-inspect-dll <model.dll> [--json]
//
//   --json   Dump the raw JSON instead of the formatted summary.
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

typedef const char* (*InferenceGetMetadataJsonFunc)(void);

// Cross-platform DLL loader (same pattern as test-model-dll).
class DllLoader {
public:
  explicit DllLoader(const std::string& path) {
    std::string errMsg;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(),
                                                          &errMsg)) {
      std::cerr << "Failed to load DLL: " << path << " - " << errMsg << "\n";
    } else {
      lib_ =
          llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &errMsg);
      valid_ = lib_.isValid();
    }
  }

  void* getSymbol(const char* name) {
    if (!valid_)
      return nullptr;
    return lib_.getAddressOfSymbol(name);
  }

  bool isValid() const { return valid_; }

private:
  llvm::sys::DynamicLibrary lib_;
  bool valid_ = false;
};

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
static std::string formatShape(const llvm::json::Array* shapeArr) {
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
static bool printSummary(const char* json_str) {
  auto parsed = llvm::json::parse(llvm::StringRef(json_str));
  if (!parsed) {
    std::cerr << "Failed to parse metadata JSON: ";
    llvm::handleAllErrors(parsed.takeError(),
                          [](const llvm::ErrorInfoBase& e) {
                            std::cerr << e.message();
                          });
    std::cerr << "\n";
    return false;
  }

  auto* root = parsed->getAsObject();
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

  if (auto* consts = root->getArray("constants")) {
    std::cout << "Constants       : " << consts->size() << "\n";
    int64_t total = 0;
    for (auto& c : *consts) {
      if (auto* obj = c.getAsObject()) {
        if (auto sz = obj->getInteger("size"))
          total += *sz;
      }
    }
    std::cout << "Constants total : " << total << " bytes\n";
  }

  std::cout << "\n";

  // Inputs
  auto printTensors = [&](const char* key, const char* label) {
    auto* arr = root->getArray(key);
    int64_t count = arr ? static_cast<int64_t>(arr->size()) : 0;
    // Prefer the explicit count field when present.
    std::string countKey = std::string(key) == "inputs" ? "input_count"
                                                        : "output_count";
    if (auto c = root->getInteger(countKey))
      count = *c;

    std::cout << label << " (" << count << "):\n";
    if (!arr) {
      std::cout << "  (none)\n";
      return;
    }
    for (size_t i = 0; i < arr->size(); ++i) {
      auto* obj = (*arr)[i].getAsObject();
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

static void printHelp(const char* argv0) {
  std::cout << "Print metadata embedded in a compiled model DLL.\n\n"
            << "Usage: " << argv0 << " <model.dll> [--json]\n\n"
            << "Options:\n"
            << "  --json   Dump raw metadata JSON instead of formatted summary\n"
            << "  -h, --help   Show this help\n";
}

int main(int argc, char** argv) {
  std::string dllPath;
  bool dumpJson = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json") {
      dumpJson = true;
    } else if (arg == "-h" || arg == "--help") {
      printHelp(argv[0]);
      return 0;
    } else if (arg[0] != '-') {
      dllPath = arg;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printHelp(argv[0]);
      return 1;
    }
  }

  if (dllPath.empty()) {
    printHelp(argv[0]);
    return 1;
  }

  DllLoader loader(dllPath);
  if (!loader.isValid())
    return 1;

  auto* getMetadata = reinterpret_cast<InferenceGetMetadataJsonFunc>(
      loader.getSymbol("inference_get_metadata_json"));
  if (!getMetadata) {
    std::cerr
        << "Symbol 'inference_get_metadata_json' not found in: " << dllPath
        << "\n";
    return 1;
  }

  const char* json = getMetadata();
  if (!json) {
    std::cerr << "inference_get_metadata_json returned null.\n";
    return 1;
  }

  if (dumpJson) {
    std::cout << json << "\n";
    return 0;
  }

  return printSummary(json) ? 0 : 1;
}
