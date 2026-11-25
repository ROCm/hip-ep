/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ggml-tblgen - Generate the GGML parameter traversals ------------===//
//
// An operation's configured values are declared once, in `GgmlOps.td`: the
// argument list says what each is called and what attribute it becomes, and
// `ggmlSources` says where GGML keeps it. Everything that walks those values
// is generated from that declaration, so none of them can be added to one
// traversal and forgotten in another.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Main.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"

#include "llvm/TableGen/Record.h"

#include <cctype>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace {

enum Action {
  Parameters,
  Decode,
  Attributes,
  EnumNames,
  Aliasing,
  OperationNames,
  OperandArguments
};

cl::opt<Action> action(
    cl::desc("what to generate"),
    cl::values(clEnumValN(Parameters, "gen-ggml-parameters",
                          "the structs the capture records configuration in"),
               clEnumValN(Decode, "gen-ggml-decode",
                          "reading the configuration out of op_params"),
               clEnumValN(Attributes, "gen-ggml-attributes",
                          "putting the configuration on the dialect operation"),
               clEnumValN(EnumNames, "gen-ggml-enum-names",
                          "naming a GGML enumerator the way the dialect does"),
               clEnumValN(Aliasing, "gen-ggml-aliasing",
                          "which operand an operation's result reads through"),
               clEnumValN(OperationNames, "gen-ggml-operation-names",
                          "the dialect operation a GGML one becomes"),
               clEnumValN(OperandArguments, "gen-ggml-operand-arguments",
                          "the arguments that say which optional operands were "
                          "given")));

/// The capture's name for one configured value.
std::string memberOf(const Record &source) {
  StringRef declared = source.getValueAsString("member");
  if (!declared.empty())
    return declared.str();
  std::string name;
  bool upper = false;
  for (char character : source.getValueAsString("argument")) {
    if (character == '_') {
      upper = true;
      continue;
    }
    name.push_back(upper ? static_cast<char>(std::toupper(character))
                         : character);
    upper = false;
  }
  return name;
}

/// The C++ type a value takes in the capture's struct, which follows from
/// where GGML keeps it.
std::string typeOf(const Record &source) {
  if (source.isSubClassOf("FromI32") || source.isSubClassOf("FromI32Flag") ||
      source.isSubClassOf("FromI32Bit"))
    return source.isSubClassOf("FromI32") ? "int64_t" : "bool";
  if (source.isSubClassOf("FromF32"))
    return "float";
  if (source.isSubClassOf("FromI32Run"))
    return "std::array<int64_t, " +
           std::to_string(source.getValueAsInt("count")) + ">";
  if (source.isSubClassOf("FromI32Enum") ||
      source.isSubClassOf("FromI32MaskedEnum") ||
      source.isSubClassOf("FromTensorEnum"))
    return "std::string";
  if (source.isSubClassOf("FromTensor"))
    return source.getValueAsString("type").str();
  PrintFatalError(&source, "no C++ type for this parameter source");
}

/// Every operation the capture has typed semantics for, in declaration order.
std::vector<const Record *> operations(const RecordKeeper &records) {
  std::vector<const Record *> result;
  for (const Record *op : records.getAllDerivedDefinitions("Ggml_Op")) {
    if (!op->getValueAsString("ggmlOperation").empty())
      result.push_back(op);
  }
  sort(result, [](const Record *lhs, const Record *rhs) {
    return lhs->getValueAsString("ggmlOperation") <
           rhs->getValueAsString("ggmlOperation");
  });
  return result;
}

/// Operations that carry a struct, one entry per distinct struct. Operations
/// configured the same way share one, and the first to declare it wins.
std::vector<const Record *> structures(const RecordKeeper &records) {
  std::vector<const Record *> result;
  std::set<std::string> seen;
  for (const Record *op : operations(records)) {
    StringRef name = op->getValueAsString("ggmlParameters");
    if (name.empty() || !seen.insert(name.str()).second)
      continue;
    result.push_back(op);
  }
  return result;
}

void emitParameters(const RecordKeeper &records, raw_ostream &os) {
  for (const Record *op : structures(records)) {
    std::vector<const Record *> sources =
        op->getValueAsListOfDefs("ggmlSources");
    if (sources.empty())
      continue; // Declared by hand, because the operation is not regular.
    StringRef name = op->getValueAsString("ggmlParameters");
    os << "struct " << name << " {\n";
    for (const Record *source : sources)
      os << "  " << typeOf(*source) << " " << memberOf(*source) << "{};\n";
    // Two operations are configured the same way exactly when every value
    // agrees, which is what sharing one table of them means.
    os << "\n  bool operator==(const " << name << " &other) const {\n"
       << "    return ";
    for (auto [index, source] : enumerate(sources)) {
      if (index)
        os << " &&\n           ";
      os << memberOf(*source) << " == other." << memberOf(*source);
    }
    if (sources.empty())
      os << "true";
    os << ";\n  }\n};\n\n";
  }

  os << "/// What an operation was configured with. Which alternative applies\n"
        "/// follows from the operation, so the operation is not named again\n"
        "/// here, and one that takes no configuration carries `monostate`.\n";
  os << "using NodeParameters =\n    std::variant<std::monostate";
  for (const Record *op : structures(records))
    os << ",\n                 " << op->getValueAsString("ggmlParameters");
  os << ",\n                 UnsupportedParameters>;\n";
}

void emitDecode(const RecordKeeper &records, raw_ostream &os) {
  // Operations GGML configures with nothing. What they do to their operands is
  // the whole of their semantics.
  for (const Record *op : operations(records)) {
    if (op->getValueAsString("ggmlParameters").empty())
      os << "case GGML_OP_" << op->getValueAsString("ggmlOperation") << ":\n";
  }
  os << "  return std::monostate{};\n";

  const auto decodeCase = [&](StringRef operation, StringRef structure,
                              ArrayRef<const Record *> sources) {
    os << "case GGML_OP_" << operation << ": {\n";
    os << "  capture::" << structure << " parameters;\n";
    for (const Record *source : sources) {
      os << "  parameters." << memberOf(*source) << " = ";
      if (source->isSubClassOf("FromI32"))
        os << "i32(" << source->getValueAsInt("index") << ")";
      else if (source->isSubClassOf("FromI32Flag"))
        os << "i32(" << source->getValueAsInt("index") << ") != 0";
      else if (source->isSubClassOf("FromI32Bit"))
        os << "(i32(" << source->getValueAsInt("index") << ") & "
           << source->getValueAsInt("mask") << ") != 0";
      else if (source->isSubClassOf("FromF32"))
        os << "f32(" << source->getValueAsInt("index") << ")";
      else if (source->isSubClassOf("FromI32Enum"))
        os << source->getValueAsString("spelling") << "(static_cast<"
           << source->getValueAsString("cast") << ">(i32("
           << source->getValueAsInt("index") << ")))";
      else if (source->isSubClassOf("FromI32MaskedEnum"))
        os << source->getValueAsString("spelling") << "(static_cast<"
           << source->getValueAsString("cast") << ">(i32("
           << source->getValueAsInt("index") << ") & "
           << source->getValueAsInt("mask") << "))";
      else if (source->isSubClassOf("FromI32Run")) {
        os << "{";
        for (int64_t index = 0; index < source->getValueAsInt("count"); ++index)
          os << (index ? ", " : "") << "i32("
             << source->getValueAsInt("index") + index << ")";
        os << "}";
      } else if (source->isSubClassOf("FromTensorEnum"))
        os << source->getValueAsString("spelling") << "("
           << source->getValueAsString("expression") << ")";
      else if (source->isSubClassOf("FromTensor"))
        os << "tensor." << source->getValueAsString("expression");
      else if (source->isSubClassOf("Fixed"))
        os << source->getValueAsString("expression");
      else
        PrintFatalError(source, "no decoding for this parameter source");
      os << ";\n";
    }
    os << "  return parameters;\n}\n";
  };

  for (const Record *op : operations(records)) {
    std::vector<const Record *> sources =
        op->getValueAsListOfDefs("ggmlSources");
    if (sources.empty())
      continue;
    decodeCase(op->getValueAsString("ggmlOperation"),
               op->getValueAsString("ggmlParameters"), sources);
    // A second spelling fills the same struct; what it does not read out of
    // `op_params`, it fixes.
    for (const Record *spelling : op->getValueAsListOfDefs("ggmlAlsoSpelled")) {
      decodeCase(spelling->getValueAsString("operation"),
                 op->getValueAsString("ggmlParameters"),
                 spelling->getValueAsListOfDefs("sources"));
    }
  }
}

void emitAttributes(const RecordKeeper &records, raw_ostream &os) {
  for (const Record *op : structures(records)) {
    std::vector<const Record *> sources =
        op->getValueAsListOfDefs("ggmlSources");
    if (sources.empty())
      continue;
    os << "if (const auto *configured = std::get_if<capture::"
       << op->getValueAsString("ggmlParameters") << ">(&parameters)) {\n";
    for (const Record *source : sources) {
      if (!source->getValueAsBit("attribute"))
        continue; // The result type already states it.
      const bool symbolic = source->isSubClassOf("FromI32Enum") ||
                            source->isSubClassOf("FromI32MaskedEnum") ||
                            source->isSubClassOf("FromTensorEnum");
      os << (symbolic ? "  symbolized(\"" : "  attribute(\"")
         << source->getValueAsString("argument") << "\", configured->"
         << memberOf(*source) << ");\n";
    }
    os << "  return;\n}\n";
  }
}

/// The function that names one GGML enumerator the way the dialect spells it.
///
/// A dialect enumeration and the GGML one it comes from agree on the names
/// they use, not on the numbers: `GGML_PREC_F32` is ten where the dialect's
/// `f32` is one. So this switches on the enumerator, and llama.cpp reordering
/// its enumerations cannot silently remap anything.
void emitEnumNames(const RecordKeeper &records, raw_ostream &os) {
  for (const Record *enumeration :
       records.getAllDerivedDefinitions("Ggml_GgmlEnum")) {
    StringRef type = enumeration->getValueAsString("ggmlType");
    const bool numeric = type == "int";
    os << "const char *" << enumeration->getValueAsString("ggmlSpelling") << "("
       << type << " value) {\n  switch (value) {\n";
    for (const Record *value :
         enumeration->getValueAsListOfDefs("enumerants")) {
      os << "  case " << value->getValueAsString("ggmlEnumerator")
         << ":\n    return \"" << value->getValueAsString("str") << "\";\n";
    }
    StringRef exhausted = enumeration->getValueAsString("ggmlExhausted");
    if (!exhausted.empty())
      os << "  case " << exhausted << ":\n    break;\n";
    if (numeric)
      os << "  default:\n    break;\n";
    os << "  }\n  fail(\"unsupported pinned llama.cpp " << type << " value \", "
       << (numeric ? "value" : "static_cast<int>(value)") << ");\n}\n\n";
  }
}

/// Which operand an operation's result reads through, when it reads through
/// one. A view hands back its source's bytes and a write hands back its
/// destination's; either way the storage a value occupies is the one that
/// operand occupies.
void emitAliasing(const RecordKeeper &records, raw_ostream &os) {
  for (const Record *op : operations(records)) {
    const int64_t slot = op->getValueAsInt("ggmlAliasedOperand");
    if (slot < 0)
      continue;
    os << "if (node.operation == \"" << op->getValueAsString("ggmlOperation")
       << "\") {\n";
    os << "  return " << slot << ";\n}\n";
  }
}

/// The dialect operation a GGML one becomes.
///
/// Mostly the dialect spells an operation the way GGML does, in lower case,
/// but not always: one name reads badly, and one GGML operation is a fixed
/// instance of another and the dialect keeps only the general one. Rather than
/// lower-case the name and carry the exceptions beside the rule, every
/// correspondence is read off the declaration that already states it.
void emitOperationNames(const RecordKeeper &records, raw_ostream &os) {
  const auto entry = [&](StringRef operation, const Record *op) {
    os << "if (operation == \"" << operation << "\")\n  return \"ggml."
       << op->getValueAsString("opName") << "\";\n";
  };
  for (const Record *op : operations(records)) {
    entry(op->getValueAsString("ggmlOperation"), op);
    for (const Record *spelling : op->getValueAsListOfDefs("ggmlAlsoSpelled"))
      entry(spelling->getValueAsString("operation"), op);
  }
}

/// The arguments whose value is which optional operands GGML was given.
///
/// GGML fills a fixed-width operand array and leaves a slot empty; the dialect
/// has a variadic operand list, so the emitter has to say how many operands
/// each argument took. That is `operandSegmentSizes`, and which arguments are
/// optional is already in the operation's own argument list. An operation may
/// also name the emptiness itself, when it has an argument for it.
void emitOperandArguments(const RecordKeeper &records, raw_ostream &os) {
  for (const Record *op : operations(records)) {
    std::vector<const Record *> presences =
        op->getValueAsListOfDefs("ggmlOperandArguments");
    // `traits` holds trait lists as well as traits, so the declaration is
    // recognized by name rather than by a field only some of them have.
    const bool segmented =
        any_of(op->getValueAsListOfDefs("traits"), [](const Record *trait) {
          return trait->getName() == "AttrSizedOperandSegments";
        });
    if (!segmented && presences.empty())
      continue;

    os << "if (operation == \"" << op->getValueAsString("ggmlOperation")
       << "\") {\n";
    if (segmented) {
      const DagInit *arguments = op->getValueAsDag("arguments");
      os << "  segments({";
      int64_t slot = 0;
      bool first = true;
      for (unsigned index = 0; index < arguments->getNumArgs(); ++index) {
        const Record *kind = cast<DefInit>(arguments->getArg(index))->getDef();
        const bool optional = kind->isSubClassOf("Optional");
        if (!optional && !kind->isSubClassOf("TypeConstraint"))
          continue; // An attribute takes no operand slot.
        os << (first ? "" : ", ") << (optional ? "present(" : "")
           << (optional ? std::to_string(slot) + ")" : "1");
        first = false;
        ++slot;
      }
      os << "});\n";
    }
    for (const Record *presence : presences) {
      os << "  flag(\"" << presence->getValueAsString("argument")
         << "\", present(" << presence->getValueAsInt("slot") << ") != 0);\n";
    }
    os << "  return;\n}\n";
  }
}

bool generate(raw_ostream &os, const RecordKeeper &records) {
  os << "// Generated by ggml-tblgen from GgmlOps.td. Do not edit.\n\n";
  switch (action) {
  case Parameters:
    emitParameters(records, os);
    break;
  case Decode:
    emitDecode(records, os);
    break;
  case Attributes:
    emitAttributes(records, os);
    break;
  case EnumNames:
    emitEnumNames(records, os);
    break;
  case Aliasing:
    emitAliasing(records, os);
    break;
  case OperationNames:
    emitOperationNames(records, os);
    break;
  case OperandArguments:
    emitOperandArguments(records, os);
    break;
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &generate);
}
