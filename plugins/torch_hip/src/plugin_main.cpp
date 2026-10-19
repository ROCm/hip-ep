/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- plugin_main.cpp - torch_hip plugin entry point ---------------------===//
//
// The ONLY plugin-specific source file. Everything else is plain hip-ep-style
// code, so it can move into lib/Conversion/TorchToHip/ unchanged later; at that
// point this file is deleted and the registration moves into hip-ep itself.
//
// Selected with -DHIPDNN_EP_COMPILER_PLUGINS=torch_hip; the generated static
// registrar calls hipEpRegisterPlugin_torch_hip() once per process.
//
//===----------------------------------------------------------------------===//

#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"

#include "torch_hip/Passes.h"

HIP_EP_DEFINE_PLUGIN(torch_hip) {
  // Same effect as R.registerPass<T>() (mlir::PassRegistration<T>), but keeps
  // the pass class private to its translation unit.
  mlir::hip::registerConvertTorchToHipPass();

  // After hip-add-context-arg (so !hip.context exists) and before
  // convert-onnx-to-hip (which builds module metadata from @main_graph).
  // Module pass, so no func.func(...) nesting.
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterOnnxLoopOutline,
                        mlir::hip::kConvertTorchToHipPassName);
}
