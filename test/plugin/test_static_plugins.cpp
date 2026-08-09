/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Static compiler-plugin unit test.
//
// Runs the static registrar (dispatchPluginRegistrationsOnce,
// StaticPlugins.cpp) and checks the recorded registry state. CMake sets
// HIP_EP_EXPECT_SAMPLE:
//   * 1 (sample selected): the registry records the sample's slot request,
//     library + search path, bitcode buffer (empty => degraded no-clang build,
//     skipped), and dialect registration (loaded here via loadAllDialects).
//   * 0 (no plugins): dispatch is a clean no-op; the registry stays empty.
// Either way dispatch must be safe + idempotent. Plain main() (no GTest).

#include "hip/Compiler/PluginRegistry.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/InitAllPasses.h"

#include "morphizen-foundation/file_io.hpp"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#ifndef HIP_EP_EXPECT_SAMPLE
#define HIP_EP_EXPECT_SAMPLE 0
#endif

extern "C" void
hipEpRegisterPlugin_sample(hip::compiler::HipEpPluginRegistry &registry);

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond) {
    llvm::outs() << "[ OK ] " << what << "\n";
  } else {
    llvm::errs() << "[FAIL] " << what << "\n";
    ++g_failures;
  }
}

template <typename Range>
bool contains(const Range &range, llvm::StringRef needle) {
  for (const auto &e : range)
    if (llvm::StringRef(e) == needle)
      return true;
  return false;
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
    return new Writer(files[path]);
  }
  void destroy_reader(morphizen::FileReader *reader) override { delete reader; }
  void destroy_writer(morphizen::FileWriter *writer) override { delete writer; }

  std::map<std::string, std::vector<char>> files;
};

mlir::OwningOpRef<mlir::ModuleOp>
runOnnxToHipPipeline(mlir::MLIRContext &context, llvm::StringRef text,
                     CapturingFileSystem &fs, bool &ok) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module) {
    ok = false;
    return module;
  }
  mlir::PassManager manager(&context);
  mlir::hip::OnnxToHipPipelineOptions options;
  options.externalizeMinNumElements = 1;
  mlir::hip::buildOnnxToHipPipeline(manager, options, &fs);
  ok = mlir::succeeded(manager.run(*module));
  return module;
}

} // namespace

int main() {
  using namespace hip::compiler;

  // Dispatch must be safe to call and idempotent (call twice).
  dispatchPluginRegistrationsOnce();
  dispatchPluginRegistrationsOnce();
  check(true, "dispatchPluginRegistrationsOnce() ran without crashing");

  auto slotPasses = pluginPassesForSlot(PipelineSlot::AfterConvertOnnxToHip);
  auto lateSlotPasses = pluginPassesForSlot(PipelineSlot::BeforeBufferization);
  auto libs = pluginLibraries();
  auto libPaths = pluginLibraryPaths();
  auto bitcode = pluginBitcodeBuffers();
  auto dialectRegs = pluginDialectRegistrations();

#if HIP_EP_EXPECT_SAMPLE
  llvm::outs() << "Mode: sample plugin EXPECTED (statically linked)\n";

  check(contains(slotPasses, "func.func(hip-ep-sample-print-functions)"),
        "AfterConvertOnnxToHip slot records the sample pass request");
  check(contains(lateSlotPasses, "func.func(hip-ep-sample-emit-late-constant)"),
        "BeforeBufferization slot records the late-carrier fixture");
  check(contains(libs, "hip_ep_sample_lib"),
        "pluginLibraries() records 'hip_ep_sample_lib'");
  check(!libPaths.empty(), "pluginLibraryPaths() records a search path");

  // Bitcode is present only when the build had clang to compile it; an empty
  // set is the documented degraded-build case, not a failure.
  if (bitcode.empty()) {
    llvm::outs() << "[SKIP] no plugin bitcode (degraded build without clang)\n";
  } else {
    check(bitcode.size() == 1, "exactly one plugin bitcode buffer recorded");
    const auto &buf = bitcode.front();
    const auto *bytes = static_cast<const unsigned char *>(buf.data);
    bool magicOk = buf.sizeBytes >= 4 && bytes[0] == 'B' && bytes[1] == 'C' &&
                   bytes[2] == 0xC0 && bytes[3] == 0xDE;
    check(magicOk, "plugin bitcode carries the LLVM bitcode magic");
  }

  // Dialect-registration path. The sample plugin contributes a minimal vendor
  // dialect (hip_ep_sample) with a ConvertToLLVMPatternInterface. This is the
  // only in-tree coverage of loadAllDialects()'s plugin loop and the
  // convert-hip-to-llvm hasPromisedInterface guard, so exercise both here.
  check(!dialectRegs.empty(),
        "pluginDialectRegistrations() records the sample dialect callback");

  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);
  mlir::Dialect *sampleDialect = context.getLoadedDialect("hip_ep_sample");
  check(sampleDialect != nullptr,
        "loadAllDialects() loaded the plugin's hip_ep_sample dialect");
  if (sampleDialect) {
    // Mirror the convert-hip-to-llvm guard: a genuinely-registered interface
    // (attached via DialectExtension, not just promised) has
    // hasPromisedInterface() false AND dyn_casts, so its patterns are used.
    bool onlyPromised = sampleDialect->hasPromisedInterface(
        sampleDialect->getTypeID(),
        mlir::ConvertToLLVMPatternInterface::getInterfaceID());
    check(!onlyPromised,
          "hip_ep_sample's ConvertToLLVM interface is registered, not just "
          "promised (hasPromisedInterface == false)");
    check(mlir::dyn_cast<mlir::ConvertToLLVMPatternInterface>(sampleDialect) !=
              nullptr,
          "hip_ep_sample implements ConvertToLLVMPatternInterface "
          "(convert-hip-to-llvm guard admits it)");
  }
#else
  llvm::outs() << "Mode: NO plugins selected (dispatch must be a no-op)\n";

  check(slotPasses.empty(), "no plugin slot requests recorded");
  check(lateSlotPasses.empty(), "no late plugin slot requests recorded");
  check(libs.empty(), "no plugin libraries recorded");
  check(libPaths.empty(), "no plugin library paths recorded");
  check(bitcode.empty(), "no plugin bitcode recorded");
  check(dialectRegs.empty(), "no plugin dialect registrations recorded");

  // The authoritative default build deliberately selects no production
  // plugins, but still links the real sample archive into this focused test.
  // Register it explicitly and exercise the actual pipeline slots so carrier
  // production/consumption and the late-carrier diagnostic run in default CI.
  hipEpRegisterPlugin_sample(getProcessPluginRegistry());
  check(contains(pluginPassesForSlot(PipelineSlot::AfterConvertOnnxToHip),
                 "func.func(hip-ep-sample-print-functions)"),
        "focused sample registration records AfterConvertOnnxToHip");
  check(contains(pluginPassesForSlot(PipelineSlot::BeforeBufferization),
                 "func.func(hip-ep-sample-emit-late-constant)"),
        "focused sample registration records BeforeBufferization");

  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  hip::compiler::loadAllDialects(context);

  CapturingFileSystem carrierFs;
  bool carrierOk = false;
  auto carrierModule = runOnnxToHipPipeline(context,
                                            R"mlir(
        module {
          func.func @main_graph(
              %input: tensor<2xf32> {onnx.name = "input"})
              -> tensor<2xf32> attributes {hip_ep_sample.emit_constant} {
            "onnx.Return"(%input) : (tensor<2xf32>) -> ()
          }
        }
      )mlir",
                                            carrierFs, carrierOk);
  check(carrierOk && carrierModule,
        "AfterConvert sample carrier pipeline succeeds");
  if (carrierOk && carrierModule) {
    auto sizes =
        (*carrierModule)
            ->getAttrOfType<mlir::DenseI64ArrayAttr>("hipdnn.constant_sizes");
    auto offsets =
        (*carrierModule)
            ->getAttrOfType<mlir::DenseI64ArrayAttr>("hipdnn.constant_offsets");
    size_t carriers = 0;
    carrierModule->walk([&](mlir::hip::ConstantOp) { ++carriers; });
    check(sizes && sizes.asArrayRef().size() == 1 &&
              sizes.asArrayRef()[0] == 2 && offsets &&
              offsets.asArrayRef().size() == 1 &&
              offsets.asArrayRef()[0] == 0 &&
              carrierFs.files["model.constants.bin"] ==
                  std::vector<char>({101, 102}) &&
              carriers == 0,
          "AfterConvert sample carrier is externalized with exact artifact");
  }

  bool sawLateCarrierDiagnostic = false;
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        diagnostic.print(stream);
        sawLateCarrierDiagnostic |= llvm::StringRef(text).contains(
            "hip.constant survived past hip-externalize-constants");
        return mlir::success();
      });
  CapturingFileSystem lateFs;
  bool lateOk = true;
  runOnnxToHipPipeline(context,
                       R"mlir(
        module {
          func.func @main_graph(
              %input: tensor<2xf32> {onnx.name = "input"})
              -> tensor<2xf32>
              attributes {hip_ep_sample.emit_late_constant} {
            "onnx.Return"(%input) : (tensor<2xf32>) -> ()
          }
        }
      )mlir",
                       lateFs, lateOk);
  check(!lateOk && sawLateCarrierDiagnostic,
        "BeforeBufferization sample carrier is diagnosed");
#endif

  if (g_failures == 0) {
    llvm::outs() << "\nAll checks passed.\n";
    return 0;
  }
  llvm::errs() << "\n" << g_failures << " check(s) failed.\n";
  return 1;
}
