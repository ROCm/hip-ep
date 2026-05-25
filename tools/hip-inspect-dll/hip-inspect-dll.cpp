/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
//   hip-inspect-dll <model.dll> [--json] [--dim-specs]
//
//   --json        Dump the raw JSON instead of the formatted summary.
//   --dim-specs   Extend the default summary with one line per output dim
//                 spelling its DimSpec tree, matching the textual form
//                 used by the EP-side resolver tracer
//                 (HIPDNN_EP_DEBUG_SHAPES).
//===----------------------------------------------------------------------===//

#include "../common/DllLoader.h"
#include "CrashHandler.h"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // Conversion warnings in LLVM JSON.h
#endif
#include "llvm/Support/JSON.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

typedef const char *(*InferenceGetMetadataJsonFunc)(void);

// Mirror of mlir::hip::DimSpecKind in include/hip/Dialect/IR/HipShapeInterface.h
// and schemas/model_metadata.fbs. Kept in sync by inspection — there is no
// shared header because hip-inspect-dll deliberately does not link the
// HipDialect (keeps the inspector small and standalone).
enum class DimSpecKindLocal : int {
  Static = 0,
  InputDim = 1,
  InputValueI64 = 2,
  RuntimeSlot = 3,
  Add = 16,
  Sub = 17,
  Mul = 18,
  FloorDiv = 19,
  CeilDiv = 20,
  Min = 21,
  Max = 22,
};

static const char *binaryNameLocal(DimSpecKindLocal k) {
  switch (k) {
  case DimSpecKindLocal::Add:
    return "add";
  case DimSpecKindLocal::Sub:
    return "sub";
  case DimSpecKindLocal::Mul:
    return "mul";
  case DimSpecKindLocal::FloorDiv:
    return "floordiv";
  case DimSpecKindLocal::CeilDiv:
    return "ceildiv";
  case DimSpecKindLocal::Min:
    return "min";
  case DimSpecKindLocal::Max:
    return "max";
  default:
    return "<?>";
  }
}

// Decode a DimSpecNode's `kind` field. FlatBuffers' `toJson` serializer
// emits enums by name ("RuntimeSlot", "Add", ...) by default, but a
// hand-rolled metadata producer might emit the integer form. We accept
// both, defaulting to Static when the field is absent (which is the FB
// convention for "use the schema default" — `kind: DimSpecKind = Static`
// at the FB level resolves to no JSON field).
static DimSpecKindLocal decodeKind(const llvm::json::Object &n) {
  if (auto s = n.getString("kind")) {
    llvm::StringRef sv = *s;
    if (sv == "Static")
      return DimSpecKindLocal::Static;
    if (sv == "InputDim")
      return DimSpecKindLocal::InputDim;
    if (sv == "InputValueI64")
      return DimSpecKindLocal::InputValueI64;
    if (sv == "RuntimeSlot")
      return DimSpecKindLocal::RuntimeSlot;
    if (sv == "Add")
      return DimSpecKindLocal::Add;
    if (sv == "Sub")
      return DimSpecKindLocal::Sub;
    if (sv == "Mul")
      return DimSpecKindLocal::Mul;
    if (sv == "FloorDiv")
      return DimSpecKindLocal::FloorDiv;
    if (sv == "CeilDiv")
      return DimSpecKindLocal::CeilDiv;
    if (sv == "Min")
      return DimSpecKindLocal::Min;
    if (sv == "Max")
      return DimSpecKindLocal::Max;
    return DimSpecKindLocal::Static; // unknown name → treat as Static
  }
  if (auto i = n.getInteger("kind"))
    return static_cast<DimSpecKindLocal>(*i);
  return DimSpecKindLocal::Static;
}

// Render the DimSpec JSON tree rooted at `nodes[idx]` to the same textual
// form as DimSpec::toString() in the compiler:
//     mul(arg[0].shape[0], slot[3])
// On any structural fault returns the literal token "<bad>" rather than
// throwing — the inspector is a diagnostic tool, not a validator.
static void renderDimSpecNode(const llvm::json::Array &nodes, int idx,
                              std::string &out) {
  if (idx < 0 || (size_t)idx >= nodes.size()) {
    out += "<bad>";
    return;
  }
  const auto *n = nodes[idx].getAsObject();
  if (!n) {
    out += "<bad>";
    return;
  }
  DimSpecKindLocal k = decodeKind(*n);
  switch (k) {
  case DimSpecKindLocal::Static: {
    out += std::to_string(n->getInteger("value").value_or(0));
    break;
  }
  case DimSpecKindLocal::InputDim: {
    out += "arg[";
    out += std::to_string(n->getInteger("input_index").value_or(0));
    out += "].shape[";
    out += std::to_string(n->getInteger("dim_index").value_or(0));
    out += "]";
    break;
  }
  case DimSpecKindLocal::InputValueI64: {
    out += "arg[";
    out += std::to_string(n->getInteger("input_index").value_or(0));
    out += "].i64[";
    out += std::to_string(n->getInteger("flat_offset").value_or(0));
    out += "]";
    break;
  }
  case DimSpecKindLocal::RuntimeSlot: {
    out += "slot[";
    out += std::to_string(n->getInteger("slot_id").value_or(-1));
    out += "]";
    break;
  }
  default: {
    out += binaryNameLocal(k);
    out += "(";
    const auto *children = n->getArray("children");
    int lhs = -1, rhs = -1;
    if (children && children->size() >= 2) {
      lhs = static_cast<int>((*children)[0].getAsInteger().value_or(-1));
      rhs = static_cast<int>((*children)[1].getAsInteger().value_or(-1));
    }
    renderDimSpecNode(nodes, lhs, out);
    out += ", ";
    renderDimSpecNode(nodes, rhs, out);
    out += ")";
    break;
  }
  }
}

// Render a whole DimSpec (root at nodes[0]) to its compact toString form.
// Returns "<empty>" for an empty / missing nodes array.
static std::string renderDimSpec(const llvm::json::Object *specObj) {
  if (!specObj)
    return "<empty>";
  const auto *nodes = specObj->getArray("nodes");
  if (!nodes || nodes->empty())
    return "<empty>";
  std::string out;
  renderDimSpecNode(*nodes, /*idx=*/0, out);
  return out;
}

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

// Print formatted summary of parsed metadata JSON. When `dumpDimSpecs`
// is true, also emits one line per (output, dim) spelling the DimSpec
// tree (and likewise for inputs that carry dim_specs entries).
static bool printSummary(const char *json_str, bool dumpDimSpecs) {
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
      if (dumpDimSpecs) {
        const auto *specs = obj->getArray("dim_specs");
        if (!specs || specs->empty()) {
          std::cout << "      dim_specs: <none — all dims static or "
                       "legacy model>\n";
        } else {
          for (size_t d = 0; d < specs->size(); ++d) {
            std::cout << "      dim[" << d << "] = "
                      << renderDimSpec((*specs)[d].getAsObject()) << "\n";
          }
        }
      }
    }
  };

  printTensors("inputs", "Inputs");
  std::cout << "\n";
  printTensors("outputs", "Outputs");

  if (dumpDimSpecs) {
    // dyn_dim_slots_count is useful context for the RuntimeSlot[N]
    // references that may appear above.
    std::cout << "\n";
    if (auto n = root->getInteger("dyn_dim_slots_count")) {
      std::cout << "dyn_dim_slots_count : " << *n << "\n";
    } else {
      std::cout << "dyn_dim_slots_count : <not set>\n";
    }
  }

  return true;
}

static void printHelp(const char *argv0) {
  std::cout
      << "Print metadata embedded in a compiled model DLL.\n\n"
      << "Usage: " << argv0 << " <model.dll> [--json] [--dim-specs]\n\n"
      << "Options:\n"
      << "  --json        Dump raw metadata JSON instead of formatted "
         "summary\n"
      << "  --dim-specs   Extend the summary with one line per output dim "
         "spelling its DimSpec tree\n"
      << "  -h, --help    Show this help\n";
}

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-inspect-dll");
  std::string dllPath;
  bool dumpJson = false;
  bool dumpDimSpecs = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json") {
      dumpJson = true;
    } else if (arg == "--dim-specs") {
      dumpDimSpecs = true;
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

  auto *getMetadata = reinterpret_cast<InferenceGetMetadataJsonFunc>(
      loader.getSymbol("inference_get_metadata_json"));
  if (!getMetadata) {
    std::cerr << "Symbol 'inference_get_metadata_json' not found in: "
              << dllPath << "\n";
    return 1;
  }

  const char *json = getMetadata();
  if (!json) {
    std::cerr << "inference_get_metadata_json returned null.\n";
    return 1;
  }

  if (dumpJson) {
    std::cout << json << "\n";
    return 0;
  }

  return printSummary(json, dumpDimSpecs) ? 0 : 1;
}
