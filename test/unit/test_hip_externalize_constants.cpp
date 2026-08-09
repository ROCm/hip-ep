/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "morphizen-foundation/file_io.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, llvm::StringRef message) {
  if (condition)
    llvm::outs() << "[ OK ] " << message << "\n";
  else {
    llvm::errs() << "[FAIL] " << message << "\n";
    ++failures;
  }
}

class CapturingFileSystem : public morphizen::FileSystem {
public:
  class Writer : public morphizen::FileWriter {
  public:
    explicit Writer(std::vector<char> &data) : data(data) {}

    std::size_t fwrite(const void *source, std::size_t size) const override {
      const char *bytes = static_cast<const char *>(source);
      data.insert(data.end(), bytes, bytes + size);
      return size;
    }

  private:
    std::vector<char> &data;
  };

  morphizen::FileReader *create_reader(const char *) override {
    return nullptr;
  }
  morphizen::FileWriter *create_writer(const char *path) override {
    if (failNames.count(path))
      return nullptr;
    return new Writer(files[path]);
  }
  void destroy_reader(morphizen::FileReader *reader) override { delete reader; }
  void destroy_writer(morphizen::FileWriter *writer) override { delete writer; }

  std::set<std::string> failNames;
  std::map<std::string, std::vector<char>> files;
};

struct Harness {
  mlir::MLIRContext context;

  Harness() {
    context.loadDialect<mlir::hip::HipDialect, mlir::arith::ArithDialect,
                        mlir::bufferization::BufferizationDialect,
                        mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef ir) {
    return mlir::parseSourceString<mlir::ModuleOp>(ir, &context);
  }

  bool run(mlir::ModuleOp module, morphizen::FileSystem &fs, int64_t threshold,
           bool skipData) {
    mlir::PassManager manager(&context);
    manager.addPass(
        mlir::hip::createExternalizeConstantsPass(&fs, threshold, skipData));
    return mlir::succeeded(manager.run(module));
  }
};

template <typename T>
std::vector<T> arrayValues(mlir::ModuleOp module, llvm::StringRef name) {
  if constexpr (std::is_same_v<T, int64_t>) {
    if (auto attr = module->getAttrOfType<mlir::DenseI64ArrayAttr>(name))
      return std::vector<T>(attr.asArrayRef().begin(), attr.asArrayRef().end());
  } else if constexpr (std::is_same_v<T, int32_t>) {
    if (auto attr = module->getAttrOfType<mlir::DenseI32ArrayAttr>(name))
      return std::vector<T>(attr.asArrayRef().begin(), attr.asArrayRef().end());
  }
  return {};
}

std::vector<std::string> stringArrayValues(mlir::ModuleOp module,
                                           llvm::StringRef name) {
  std::vector<std::string> result;
  auto attr = module->getAttrOfType<mlir::ArrayAttr>(name);
  if (!attr)
    return result;
  for (mlir::Attribute value : attr)
    result.push_back(mlir::cast<mlir::StringAttr>(value).getValue().str());
  return result;
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

std::string printType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

std::filesystem::path writeSourceFile() {
  auto path =
      std::filesystem::temp_directory_path() / "hip_constant_source.bin";
  std::ofstream output(path, std::ios::binary);
  const char bytes[] = {40, 41, 42, 43, 44, 45, 46, 47};
  output.write(bytes, sizeof(bytes));
  return path;
}

std::filesystem::path writeTypedSourceFile() {
  auto path =
      std::filesystem::temp_directory_path() / "hip_typed_constant_source.bin";
  std::ofstream output(path, std::ios::binary);
  const uint8_t bytes[] = {
      0xfe, 0xff, // si16 -2
      0x00, 0x80, // si16 -32768
      0x00, 0x80, // ui16 32768
      0xff, 0xff, // ui16 65535
  };
  output.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
  return path;
}

struct GlobalExpectation {
  int64_t index;
  int64_t offset;
  int64_t size;
  std::string type;
  std::string symbol;
};

void verifyGlobals(mlir::ModuleOp module,
                   llvm::ArrayRef<GlobalExpectation> expected,
                   llvm::StringRef label) {
  std::map<int64_t, mlir::memref::GlobalOp> byIndex;
  module.walk([&](mlir::memref::GlobalOp global) {
    auto data =
        global->getAttrOfType<mlir::DictionaryAttr>("hip.external_data");
    if (!data)
      return;
    auto index = data.getAs<mlir::IntegerAttr>("index");
    if (index)
      byIndex[index.getInt()] = global;
  });
  check(byIndex.size() == expected.size(),
        (label + ": exact external global count").str());
  for (const GlobalExpectation &item : expected) {
    auto found = byIndex.find(item.index);
    bool ok = found != byIndex.end();
    if (ok) {
      mlir::memref::GlobalOp global = found->second;
      auto data =
          global->getAttrOfType<mlir::DictionaryAttr>("hip.external_data");
      ok = global.getSymName() == item.symbol &&
           printType(global.getType()) == item.type &&
           global->getAttrOfType<mlir::StringAttr>("sym_visibility")
                   .getValue() == "private" &&
           global->getAttrOfType<mlir::IntegerAttr>("alignment").getInt() ==
               64 &&
           data.getAs<mlir::IntegerAttr>("offset").getInt() == item.offset &&
           data.getAs<mlir::IntegerAttr>("size").getInt() == item.size;
    }
    check(ok, (label + ": global index " + std::to_string(item.index) +
               " attrs/type/symbol")
                  .str());
  }
}

std::vector<char> expectedBlob(
    int64_t size,
    std::initializer_list<std::pair<int64_t, std::vector<uint8_t>>> entries) {
  std::vector<char> blob(static_cast<size_t>(size), 0);
  for (const auto &[offset, bytes] : entries)
    for (size_t i = 0; i < bytes.size(); ++i)
      blob[static_cast<size_t>(offset) + i] = static_cast<char>(bytes[i]);
  return blob;
}

bool checkJson(const std::vector<char> &bytes,
               llvm::ArrayRef<GlobalExpectation> expected,
               llvm::ArrayRef<std::string> elementTypes,
               llvm::ArrayRef<std::vector<int64_t>> shapes,
               llvm::StringRef binaryFile, int64_t totalBytes) {
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(llvm::StringRef(bytes.data(), bytes.size()));
  if (!parsed)
    return false;
  llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->size() != 5 || root->getInteger("version") != 1 ||
      root->getString("binary_file") != binaryFile ||
      root->getInteger("num_constants") !=
          static_cast<int64_t>(expected.size()) ||
      root->getInteger("total_bytes") != totalBytes)
    return false;
  llvm::json::Array *constants = root->getArray("constants");
  if (!constants || constants->size() != expected.size())
    return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    llvm::json::Object *item = (*constants)[i].getAsObject();
    if (!item || item->size() != 6 ||
        item->getString("name") != expected[i].symbol ||
        item->getString("element_type") != elementTypes[i] ||
        item->getInteger("offset") != expected[i].offset ||
        item->getInteger("size") != expected[i].size ||
        item->getInteger("alignment") != 64)
      return false;
    llvm::json::Array *shape = item->getArray("shape");
    if (!shape || shape->size() != shapes[i].size())
      return false;
    for (size_t dim = 0; dim < shapes[i].size(); ++dim)
      if ((*shape)[dim].getAsInteger() != shapes[i][dim])
        return false;
  }
  return true;
}

void testFullModeExactArtifacts() {
  std::filesystem::path source = writeSourceFile();
  std::vector<uint8_t> memory = {70, 71};
  uintptr_t address = reinterpret_cast<uintptr_t>(memory.data());
  std::string ir =
      "module @fixture {\n"
      "  func.func @f() -> (tensor<3xi8>, tensor<4xui8>, tensor<4xi8>, "
      "tensor<2xi8>, tensor<si16>) {\n"
      "    %0 = hip.constant {onnx_node_name = \"plain\", value = "
      "dense<[1, 2, 3]> : tensor<3xi8>} : tensor<3xi8>\n"
      "    %1 = hip.constant {onnx_node_name = \"unsigned\", value = "
      "dense<9> : tensor<4xui8>} : tensor<4xui8>\n"
      "    %2 = hip.constant {onnx_node_name = \"file\", location = \"" +
      source.generic_string() +
      "\", offset = 2 : i64, size = 4 : i64} : tensor<4xi8>\n"
      "    %3 = hip.constant {onnx_node_name = \"memory\", location = "
      "\"*/_ORT_MEM_ADDR_/*\", offset = " +
      std::to_string(address) +
      " : i64, size = 2 : i64} : tensor<2xi8>\n"
      "    %4 = hip.constant {onnx_node_name = \"signed_scalar\", value = "
      "dense<-2> : tensor<si16>} : tensor<si16>\n"
      "    return %0, %1, %2, %3, %4 : tensor<3xi8>, tensor<4xui8>, "
      "tensor<4xi8>, tensor<2xi8>, tensor<si16>\n"
      "  }\n"
      "}\n";

  Harness harness;
  CapturingFileSystem fs;
  auto module = harness.parse(ir);
  bool ok = module && harness.run(*module, fs, 1, false);
  check(ok, "full: pass succeeds");
  if (!ok)
    return;

  std::vector<int64_t> sizes = {3, 4, 4, 2, 2};
  std::vector<int64_t> offsets = {0, 64, 128, 192, 256};
  check(arrayValues<int64_t>(*module, "hipdnn.constant_sizes") == sizes,
        "full: exact sizes array");
  check(arrayValues<int64_t>(*module, "hipdnn.constant_offsets") == offsets,
        "full: exact offsets array");
  check(!(*module)->hasAttr("hipdnn.constant_source_kinds") &&
            !(*module)->hasAttr("hipdnn.constant_splat_elem_values") &&
            !(*module)->hasAttr("hipdnn.constant_splat_elem_sizes") &&
            !(*module)->hasAttr("hipdnn.constant_file_paths") &&
            !(*module)->hasAttr("hipdnn.constant_file_offsets") &&
            !(*module)->hasAttr("hipdnn.constant_mem_offsets"),
        "full: streaming metadata arrays absent");
  check((*module)
                ->getAttrOfType<mlir::StringAttr>("hip.constants_file")
                .getValue() == "fixture.constants.bin",
        "full: exact constants filename");

  std::vector<char> expected = expectedBlob(258, {{0, {1, 2, 3}},
                                                  {64, {9, 9, 9, 9}},
                                                  {128, {42, 43, 44, 45}},
                                                  {192, {70, 71}},
                                                  {256, {0xfe, 0xff}}});
  check(fs.files["fixture.constants.bin"] == expected,
        "full: complete bytes, gaps, and trailing behavior");

  std::vector<GlobalExpectation> globals = {
      {0, 0, 3, "memref<3xi8>", "hip_ext_constant_plain_0"},
      {1, 64, 4, "memref<4xui8>", "hip_ext_constant_unsigned_1"},
      {2, 128, 4, "memref<4xi8>", "hip_ext_constant_file_2"},
      {3, 192, 2, "memref<2xi8>", "hip_ext_constant_memory_3"},
      {4, 256, 2, "memref<si16>", "hip_ext_constant_signed_scalar_4"},
  };
  verifyGlobals(*module, globals, "full");
  std::vector<std::string> elementTypes = {"i8", "i8", "i8", "i8", "i16"};
  std::vector<std::vector<int64_t>> shapes = {{3}, {4}, {4}, {2}, {}};
  check(checkJson(fs.files["fixture.constants.json"], globals, elementTypes,
                  shapes, "fixture.constants.bin", 258),
        "full: complete JSON fields and baseline integer type strings");
  std::filesystem::remove(source);
}

void testPureStreamingExactMetadata() {
  const std::string missing = "/definitely/missing/streaming_weights.bin";
  std::string ir = "func.func @f() -> (tensor<4xui8>, tensor<4xi8>) {\n"
                   "  %0 = hip.constant {value = dense<5> : tensor<4xui8>} : "
                   "tensor<4xui8>\n"
                   "  %1 = hip.constant {location = \"" +
                   missing +
                   "\", offset = 123 : i64, size = 4 : i64} : tensor<4xi8>\n"
                   "  return %0, %1 : tensor<4xui8>, tensor<4xi8>\n"
                   "}\n";
  Harness harness;
  CapturingFileSystem fs;
  auto module = harness.parse(ir);
  bool ok = module && harness.run(*module, fs, 1, true);
  check(ok, "streaming: missing source is not opened");
  if (!ok)
    return;

  check(fs.files.empty(), "streaming: no binary or JSON artifact");
  check((*module)
                ->getAttrOfType<mlir::StringAttr>("hip.constants_file")
                .getValue() == "model.constants.bin",
        "streaming: exact constants filename");
  check(arrayValues<int64_t>(*module, "hipdnn.constant_sizes") ==
                std::vector<int64_t>({4, 4}) &&
            arrayValues<int64_t>(*module, "hipdnn.constant_offsets") ==
                std::vector<int64_t>({0, 64}),
        "streaming: exact size and canonical offset arrays");
  check(
      arrayValues<int32_t>(*module, "hipdnn.constant_source_kinds") ==
              std::vector<int32_t>({1, 2}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_splat_elem_values") ==
              std::vector<int64_t>({5, 0}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_splat_elem_sizes") ==
              std::vector<int64_t>({1, 0}) &&
          stringArrayValues(*module, "hipdnn.constant_file_paths") ==
              std::vector<std::string>({"", missing}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_file_offsets") ==
              std::vector<int64_t>({0, 123}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_mem_offsets") ==
              std::vector<int64_t>({0, 0}),
      "streaming: every source metadata array is exact");
  verifyGlobals(*module,
                {{0, 0, 4, "memref<4xui8>", "hip_ext_constant_0"},
                 {1, 64, 4, "memref<4xi8>", "hip_ext_constant_1"}},
                "streaming");
}

void testHybridExactArtifacts() {
  const std::string missing = "/definitely/missing/hybrid_weights.bin";
  std::vector<uint8_t> memory = {20, 21};
  uintptr_t address = reinterpret_cast<uintptr_t>(memory.data());
  std::string ir =
      "func.func @f() -> (tensor<3xi8>, tensor<2xi8>, tensor<si16>, "
      "tensor<4xi8>) {\n"
      "  %0 = hip.constant {value = dense<[1, 2, 3]> : tensor<3xi8>} : "
      "tensor<3xi8>\n"
      "  %1 = hip.constant {location = \"*/_ORT_MEM_ADDR_/*\", offset = " +
      std::to_string(address) +
      " : i64, size = 2 : i64} : tensor<2xi8>\n"
      "  %2 = hip.constant {value = dense<-2> : tensor<si16>} : tensor<si16>\n"
      "  %3 = hip.constant {location = \"" +
      missing +
      "\", offset = 77 : i64, size = 4 : i64} : tensor<4xi8>\n"
      "  return %0, %1, %2, %3 : tensor<3xi8>, tensor<2xi8>, tensor<si16>, "
      "tensor<4xi8>\n"
      "}\n";
  Harness harness;
  CapturingFileSystem fs;
  auto module = harness.parse(ir);
  bool ok = module && harness.run(*module, fs, 1, true);
  check(ok, "hybrid: pass succeeds without opening file-ref source");
  if (!ok)
    return;

  check((*module)
                ->getAttrOfType<mlir::StringAttr>("hip.constants_file")
                .getValue() == "model.constants.bin",
        "hybrid: exact constants filename");
  check(fs.files["model.constants.bin"] ==
            expectedBlob(66, {{0, {1, 2, 3}}, {64, {20, 21}}}),
        "hybrid: complete partial binary bytes and gaps");
  check(fs.files.count("model.constants.json") == 0, "hybrid: JSON absent");
  check(
      arrayValues<int64_t>(*module, "hipdnn.constant_sizes") ==
              std::vector<int64_t>({3, 2, 2, 4}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_offsets") ==
              std::vector<int64_t>({0, 64, 128, 192}) &&
          arrayValues<int32_t>(*module, "hipdnn.constant_source_kinds") ==
              std::vector<int32_t>({3, 3, 1, 2}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_splat_elem_values") ==
              std::vector<int64_t>({0, 0, 65534, 0}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_splat_elem_sizes") ==
              std::vector<int64_t>({0, 0, 2, 0}) &&
          stringArrayValues(*module, "hipdnn.constant_file_paths") ==
              std::vector<std::string>({"", "", "", missing}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_file_offsets") ==
              std::vector<int64_t>({0, 0, 0, 77}) &&
          arrayValues<int64_t>(*module, "hipdnn.constant_mem_offsets") ==
              std::vector<int64_t>({0, 64, 0, 0}),
      "hybrid: every metadata array is exact");
  verifyGlobals(*module,
                {{0, 0, 3, "memref<3xi8>", "hip_ext_constant_0"},
                 {1, 64, 2, "memref<2xi8>", "hip_ext_constant_1"},
                 {2, 128, 2, "memref<si16>", "hip_ext_constant_2"},
                 {3, 192, 4, "memref<4xi8>", "hip_ext_constant_3"}},
                "hybrid");
}

void testThresholdAndInlineParity() {
  std::filesystem::path source = writeSourceFile();
  std::filesystem::path typedSource = writeTypedSourceFile();
  std::string ir =
      "func.func @f() -> (tensor<3xi8>, tensor<4xi8>, tensor<5xi8>) {\n"
      "  %0 = hip.constant {value = dense<[1, 2, 3]> : tensor<3xi8>} : "
      "tensor<3xi8>\n"
      "  %1 = hip.constant {value = dense<[4, 5, 6, 7]> : tensor<4xi8>} : "
      "tensor<4xi8>\n"
      "  %2 = hip.constant {value = dense<[8, 9, 10, 11, 12]> : "
      "tensor<5xi8>} : tensor<5xi8>\n"
      "  return %0, %1, %2 : tensor<3xi8>, tensor<4xi8>, tensor<5xi8>\n"
      "}\n";
  Harness harness;
  CapturingFileSystem fs;
  auto module = harness.parse(ir);
  bool ok = module && harness.run(*module, fs, 4, false);
  check(ok, "threshold: equality and larger externalize");
  if (ok) {
    check(arrayValues<int64_t>(*module, "hipdnn.constant_sizes") ==
                  std::vector<int64_t>({4, 5}) &&
              arrayValues<int64_t>(*module, "hipdnn.constant_offsets") ==
                  std::vector<int64_t>({0, 64}),
          "threshold: smaller inline, equality and larger indexed");
    size_t inlineConstants = 0;
    module->walk([&](mlir::arith::ConstantOp) { ++inlineConstants; });
    check(inlineConstants == 1, "threshold: exactly one smaller inline value");
  }

  std::string externalInline =
      "func.func @f() -> (tensor<2xsi16>, tensor<2xui16>) {\n"
      "  %s = hip.constant {location = \"" +
      typedSource.generic_string() +
      "\", offset = 0 : i64, size = 4 : i64} : tensor<2xsi16>\n"
      "  %u = hip.constant {location = \"" +
      typedSource.generic_string() +
      "\", offset = 4 : i64, size = 4 : i64} : tensor<2xui16>\n"
      "  return %s, %u : tensor<2xsi16>, tensor<2xui16>\n"
      "}\n";
  CapturingFileSystem inlineFs;
  auto inlineModule = harness.parse(externalInline);
  ok = inlineModule && harness.run(*inlineModule, inlineFs, 0, false);
  check(ok, "threshold zero: external reference materializes inline");
  if (ok) {
    bool signedValueOk = false;
    bool unsignedValueOk = false;
    inlineModule->walk([&](mlir::arith::ConstantOp op) {
      auto value = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
      if (!value)
        return;
      auto valueRange = value.getValues<llvm::APInt>();
      std::vector<llvm::APInt> elements(valueRange.begin(), valueRange.end());
      llvm::ArrayRef<char> raw = value.getRawData();
      if (printType(op.getResult().getType()) == "tensor<2xsi16>")
        signedValueOk =
            elements.size() == 2 && elements[0].getSExtValue() == -2 &&
            elements[1].getSExtValue() == -32768 && raw.size() == 4 &&
            static_cast<uint8_t>(raw[0]) == 0xfe &&
            static_cast<uint8_t>(raw[1]) == 0xff &&
            static_cast<uint8_t>(raw[2]) == 0x00 &&
            static_cast<uint8_t>(raw[3]) == 0x80;
      if (printType(op.getResult().getType()) == "tensor<2xui16>")
        unsignedValueOk =
            elements.size() == 2 && elements[0].getZExtValue() == 32768 &&
            elements[1].getZExtValue() == 65535 && raw.size() == 4 &&
            static_cast<uint8_t>(raw[0]) == 0x00 &&
            static_cast<uint8_t>(raw[1]) == 0x80 &&
            static_cast<uint8_t>(raw[2]) == 0xff &&
            static_cast<uint8_t>(raw[3]) == 0xff;
    });
    mlir::func::FuncOp function;
    inlineModule->walk([&](mlir::func::FuncOp op) { function = op; });
    bool exactFunctionTypes =
        function && function.getNumResults() == 2 &&
        printType(function.getResultTypes()[0]) == "tensor<2xsi16>" &&
        printType(function.getResultTypes()[1]) == "tensor<2xui16>";
    mlir::func::ReturnOp returnOp;
    if (function)
      returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
          function.front().getTerminator());
    bool exactReturnTypes =
        returnOp && returnOp.getNumOperands() == 2 &&
        printType(returnOp.getOperand(0).getType()) == "tensor<2xsi16>" &&
        printType(returnOp.getOperand(1).getType()) == "tensor<2xui16>";
    check(signedValueOk && unsignedValueOk && exactFunctionTypes &&
              exactReturnTypes && inlineFs.files.empty() &&
              !(*inlineModule)->hasAttr("hip.constants_file"),
          "threshold zero: multi-byte signed/unsigned raw values and types");
  }

  CapturingFileSystem scalarFs;
  auto scalarModule = harness.parse(R"mlir(
    func.func @scalars() {
      %0 = hip.constant {value = dense<-1> : tensor<si8>} : tensor<si8>
      %1 = hip.constant {value = dense<255> : tensor<ui8>} : tensor<ui8>
      %2 = hip.constant {value = dense<7> : tensor<i32>} : tensor<i32>
      return
    }
  )mlir");
  ok = scalarModule && harness.run(*scalarModule, scalarFs, 0, false);
  std::set<std::string> scalarTypes;
  if (ok)
    scalarModule->walk([&](mlir::arith::ConstantOp op) {
      scalarTypes.insert(printType(op.getResult().getType()));
    });
  check(ok && scalarTypes == std::set<std::string>(
                                 {"tensor<i32>", "tensor<si8>", "tensor<ui8>"}),
        "threshold zero: inline signed/unsigned/signless scalars preserved");

  CapturingFileSystem oneFs;
  auto oneModule = harness.parse(externalInline);
  ok = oneModule && harness.run(*oneModule, oneFs, 1, false);
  check(ok &&
            arrayValues<int64_t>(*oneModule, "hipdnn.constant_sizes") ==
                std::vector<int64_t>({4, 4}) &&
            arrayValues<int64_t>(*oneModule, "hipdnn.constant_offsets") ==
                std::vector<int64_t>({0, 64}),
        "threshold one: external reference remains external");
  std::filesystem::remove(source);
  std::filesystem::remove(typedSource);
}

void testMultiFunctionAbsoluteOrder() {
  std::string ir = R"mlir(
    func.func @f0() -> (tensor<2xi8>, tensor<1xi8>, tensor<1xi8>) {
      %plugin = hip.constant {onnx_node_name = "plugin_f0", value = dense<[90, 91]> : tensor<2xi8>} : tensor<2xi8>
      %synth = hip.constant {hip.constant_order = 1 : i64, hip.constant_origin = "onnx-synthesized", onnx_node_name = "f0_synthesized", value = dense<30> : tensor<1xi8>} : tensor<1xi8>
      %import = hip.constant {hip.constant_order = 0 : i64, hip.constant_origin = "onnx-imported", onnx_node_name = "f0_imported", value = dense<20> : tensor<1xi8>} : tensor<1xi8>
      return %plugin, %synth, %import : tensor<2xi8>, tensor<1xi8>, tensor<1xi8>
    }
    func.func @f1() -> (tensor<1xi8>, tensor<1xi8>, tensor<2xi8>) {
      %plugin = hip.constant {onnx_node_name = "plugin_f1", value = dense<92> : tensor<1xi8>} : tensor<1xi8>
      %synth = hip.constant {hip.constant_order = 3 : i64, hip.constant_origin = "onnx-synthesized", onnx_node_name = "f1_synthesized", value = dense<31> : tensor<1xi8>} : tensor<1xi8>
      %import = hip.constant {hip.constant_order = 2 : i64, hip.constant_origin = "onnx-imported", onnx_node_name = "f1_imported", value = dense<[21, 22]> : tensor<2xi8>} : tensor<2xi8>
      return %plugin, %synth, %import : tensor<1xi8>, tensor<1xi8>, tensor<2xi8>
    }
  )mlir";
  Harness harness;
  CapturingFileSystem fs;
  auto module = harness.parse(ir);
  bool ok = module && harness.run(*module, fs, 1, false);
  check(ok, "order: pass succeeds");
  if (!ok)
    return;

  check(arrayValues<int64_t>(*module, "hipdnn.constant_sizes") ==
                std::vector<int64_t>({1, 1, 2, 1, 2, 1}) &&
            arrayValues<int64_t>(*module, "hipdnn.constant_offsets") ==
                std::vector<int64_t>({0, 64, 128, 192, 256, 320}),
        "order: f0 imported/synth, f1 imported/synth, then plugins");
  check(fs.files["model.constants.bin"] == expectedBlob(321, {{0, {20}},
                                                              {64, {30}},
                                                              {128, {21, 22}},
                                                              {192, {31}},
                                                              {256, {90, 91}},
                                                              {320, {92}}}),
        "order: exact multi-function bytes, indices, offsets, and gaps");
  std::vector<GlobalExpectation> globals = {
      {0, 0, 1, "memref<1xi8>", "hip_ext_constant_f0_imported_0"},
      {1, 64, 1, "memref<1xi8>", "hip_ext_constant_f0_synthesized_1"},
      {2, 128, 2, "memref<2xi8>", "hip_ext_constant_f1_imported_2"},
      {3, 192, 1, "memref<1xi8>", "hip_ext_constant_f1_synthesized_3"},
      {4, 256, 2, "memref<2xi8>", "hip_ext_constant_plugin_f0_4"},
      {5, 320, 1, "memref<1xi8>", "hip_ext_constant_plugin_f1_5"},
  };
  verifyGlobals(*module, globals, "order");
  std::vector<std::string> elementTypes(6, "i8");
  std::vector<std::vector<int64_t>> shapes = {{1}, {1}, {2}, {1}, {2}, {1}};
  check(checkJson(fs.files["model.constants.json"], globals, elementTypes,
                  shapes, "model.constants.bin", 321),
        "order: exact multi-function JSON order and contents");
}

void expectFailureUnchanged(llvm::StringRef label, llvm::StringRef ir,
                            int64_t threshold, bool skipData,
                            CapturingFileSystem &fs) {
  Harness harness;
  mlir::ScopedDiagnosticHandler handler(
      &harness.context, [](mlir::Diagnostic &) { return mlir::success(); });
  auto module = harness.parse(ir);
  if (!module) {
    check(false, (label + ": fixture parses").str());
    return;
  }
  std::string before = printModule(*module);
  bool ok = harness.run(*module, fs, threshold, skipData);
  check(!ok, (label + ": pass fails").str());
  check(printModule(*module) == before, (label + ": IR unchanged").str());
}

void expectRejected(llvm::StringRef label, llvm::StringRef ir,
                    int64_t threshold, bool skipData, CapturingFileSystem &fs) {
  Harness harness;
  mlir::ScopedDiagnosticHandler handler(
      &harness.context, [](mlir::Diagnostic &) { return mlir::success(); });
  auto module = harness.parse(ir);
  bool rejected = !module || !harness.run(*module, fs, threshold, skipData);
  check(rejected, (label + ": invalid carrier rejected").str());
}

void testFailuresAndStructuralValidation() {
  const char *zeroFile = R"mlir(
    func.func @f() -> tensor<0xi8> {
      %0 = hip.constant {location = "/missing", offset = 0 : i64, size = 0 : i64} : tensor<0xi8>
      return %0 : tensor<0xi8>
    }
  )mlir";
  CapturingFileSystem fullZero;
  expectRejected("zero full", zeroFile, 1, false, fullZero);
  CapturingFileSystem streamingZero;
  expectRejected("zero streaming", zeroFile, 1, true, streamingZero);
  CapturingFileSystem hybridZero;
  std::vector<uint8_t> live = {1};
  std::string hybridZeroIr =
      "func.func @f() -> (tensor<1xi8>, tensor<0xi8>) {\n"
      "  %0 = hip.constant {location = \"*/_ORT_MEM_ADDR_/*\", offset = " +
      std::to_string(reinterpret_cast<uintptr_t>(live.data())) +
      " : i64, size = 1 : i64} : tensor<1xi8>\n"
      "  %1 = hip.constant {location = \"*/_ORT_MEM_ADDR_/*\", offset = 1 : "
      "i64, size = 0 : i64} : tensor<0xi8>\n"
      "  return %0, %1 : tensor<1xi8>, tensor<0xi8>\n"
      "}\n";
  expectRejected("zero hybrid", hybridZeroIr, 1, true, hybridZero);
  check(fullZero.files.empty() && streamingZero.files.empty() &&
            hybridZero.files.empty(),
        "zero sources: no artifacts created");

  CapturingFileSystem missingFull;
  expectFailureUnchanged("missing full source",
                         R"mlir(func.func @f() -> tensor<2xi8> {
        %0 = hip.constant {location = "/definitely/missing/full.bin", offset = 0 : i64, size = 2 : i64} : tensor<2xi8>
        return %0 : tensor<2xi8>
      })mlir",
                         1, false, missingFull);

  CapturingFileSystem sinkFailure;
  sinkFailure.failNames.insert("model.constants.bin");
  expectFailureUnchanged("binary writer failure",
                         R"mlir(func.func @f() -> tensor<2xi8> {
        %0 = hip.constant {value = dense<[1, 2]> : tensor<2xi8>} : tensor<2xi8>
        return %0 : tensor<2xi8>
      })mlir",
                         1, false, sinkFailure);

  CapturingFileSystem jsonFailure;
  jsonFailure.failNames.insert("model.constants.json");
  expectFailureUnchanged("JSON writer failure",
                         R"mlir(func.func @f() -> tensor<2xi8> {
        %0 = hip.constant {value = dense<[1, 2]> : tensor<2xi8>} : tensor<2xi8>
        return %0 : tensor<2xi8>
      })mlir",
                         1, false, jsonFailure);
  check(jsonFailure.files.count("model.constants.bin") == 1,
        "JSON failure: non-transactional binary side effect documented");

  CapturingFileSystem stale;
  expectFailureUnchanged(
      "stale metadata",
      R"mlir(module attributes {hipdnn.constant_offsets = array<i64: 0>} {
        func.func @f() -> tensor<1xi8> {
          %0 = hip.constant {value = dense<1> : tensor<1xi8>} : tensor<1xi8>
          return %0 : tensor<1xi8>
        }
      })mlir",
      1, false, stale);

  {
    Harness duplicateHarness;
    mlir::ScopedDiagnosticHandler handler(
        &duplicateHarness.context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    CapturingFileSystem duplicateFs;
    auto module = duplicateHarness.parse(R"mlir(
      func.func @f() -> tensor<1xi8> {
        %0 = hip.constant {value = dense<1> : tensor<1xi8>} : tensor<1xi8>
        return %0 : tensor<1xi8>
      }
    )mlir");
    bool first = module && duplicateHarness.run(*module, duplicateFs, 1, false);
    std::string beforeSecond = module ? printModule(*module) : "";
    bool second =
        module && duplicateHarness.run(*module, duplicateFs, 1, false);
    check(first && !second,
          "duplicate invocation: first succeeds, second fails");
    check(module && printModule(*module) == beforeSecond,
          "duplicate invocation: second run leaves IR unchanged");
  }

  CapturingFileSystem duplicate;
  expectFailureUnchanged("duplicate order",
                         R"mlir(func.func @f() -> (tensor<1xi8>, tensor<1xi8>) {
        %0 = hip.constant {hip.constant_origin = "onnx-imported", hip.constant_order = 0 : i64, value = dense<1> : tensor<1xi8>} : tensor<1xi8>
        %1 = hip.constant {hip.constant_origin = "onnx-synthesized", hip.constant_order = 0 : i64, value = dense<2> : tensor<1xi8>} : tensor<1xi8>
        return %0, %1 : tensor<1xi8>, tensor<1xi8>
      })mlir",
                         1, false, duplicate);

  CapturingFileSystem unknownOrigin;
  expectFailureUnchanged("unknown order origin",
                         R"mlir(func.func @f() -> tensor<1xi8> {
        %0 = hip.constant {hip.constant_origin = "plugin-forged", hip.constant_order = 1 : i64, value = dense<1> : tensor<1xi8>} : tensor<1xi8>
        return %0 : tensor<1xi8>
      })mlir",
                         1, false, unknownOrigin);

  CapturingFileSystem partial;
  expectFailureUnchanged("partial order metadata",
                         R"mlir(func.func @f() -> tensor<1xi8> {
        %0 = hip.constant {hip.constant_order = 0 : i64, value = dense<1> : tensor<1xi8>} : tensor<1xi8>
        return %0 : tensor<1xi8>
      })mlir",
                         1, false, partial);

  CapturingFileSystem symbol;
  expectFailureUnchanged("symbol collision",
                         R"mlir(module {
        func.func private @hip_ext_constant_0()
        func.func @f() -> tensor<1xi8> {
          %0 = hip.constant {value = dense<1> : tensor<1xi8>} : tensor<1xi8>
          return %0 : tensor<1xi8>
        }
      })mlir",
                         1, false, symbol);
}

} // namespace

int main() {
  testFullModeExactArtifacts();
  testPureStreamingExactMetadata();
  testHybridExactArtifacts();
  testThresholdAndInlineParity();
  testMultiFunctionAbsoluteOrder();
  testFailuresAndStructuralValidation();
  if (failures == 0) {
    llvm::outs() << "All hip-externalize-constants unit tests passed.\n";
    return 0;
  }
  llvm::errs() << failures << " check(s) failed.\n";
  return 1;
}
