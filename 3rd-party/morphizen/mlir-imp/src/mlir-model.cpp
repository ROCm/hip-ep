/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir-model.hpp"
#include "mlir-constants.hpp"
#include "mlir-context-manager.hpp"
#include "mlir-graph.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "morphizen-foundation/env_config.hpp"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_MODEL, "0")
DEF_ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_GENERIC, "0")
DEF_ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_DEBUG_INFO, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_MODEL) >= n)
namespace morphizen {
namespace mlir_impl {

MLIRModel::MLIRModel(PrivateTag, mlir::OwningOpRef<mlir::ModuleOp> module)
    : module_(std::move(module)) {
  // get the function with sym_name = "main_graph" from the module.
  mlir::func::FuncOp mainFunc = nullptr;
  auto ops = module_->getOps<mlir::func::FuncOp>();
  for (auto func : ops) {
    if (func.getSymName() == "main_graph") {
      mainFunc = func;
      break;
    }
  }
  if (mainFunc) {
    MY_LOG(1) << "Found main_graph FuncOp: " << mainFunc.getName().str();
    // Initialize main_graph_ with the main_graph FuncOp found
    main_graph_ =
        std::make_unique<MLIRGraph>(const_cast<MLIRModel &>(*this), mainFunc);
  } else {
    LOG(ERROR) << "No FuncOp with sym_name 'main_graph' found in the module";
    throw std::runtime_error(
        "No FuncOp with sym_name 'main_graph' found in the module");
  }
}

std::unique_ptr<MLIRModel>
MLIRModel::create(mlir::OwningOpRef<mlir::ModuleOp> module) {
  // Create an OwningOpRef from the module
  return std::make_unique<MLIRModel>(PrivateTag{}, std::move(module));
}

std::unique_ptr<MLIRModel> MLIRModel::create_empty(
    const std::filesystem::path &path,
    const std::vector<std::pair<std::string, int64_t>> &opset) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  // Create an empty MLIR module
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create the main graph function first
  auto mainFunc = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), "main_graph",
      /*type=*/builder.getFunctionType({}, {}), /*attrs=*/{});
  module.push_back(mainFunc);

  // Set the onnx.graph.name attribute
  mainFunc->setAttr(attr_names::ONNX_GRAPH_NAME,
                    builder.getStringAttr("main_graph"));

  // Create and set insertion point to entry block.
  mainFunc.getBody().push_back(new mlir::Block);

  auto model = MLIRModel::create(mlir::OwningOpRef<mlir::ModuleOp>(module));

  // Set model path as metadata if provided
  if (!path.empty()) {
    model->set_metadata_prop("model_path", path.u8string());
    // model->main_graph().set_model_path(path);
  }

  // Set opset information as metadata
  if (!opset.empty()) {
    for (const auto &[domain, version] : opset) {
      std::string key = "opset." + domain;
      model->set_metadata_prop(key, std::to_string(version));
    }
  }

  // Debug: Save empty module if debug level > 5
  model->maybe_dump_mlir(path);
  return model;
}

std::unique_ptr<MLIRModel> MLIRModel::load(const std::string &filename) {
  auto &context = MLIRContextManager::getInstance().getContext();

  // Check if this is an ONNX file
  std::filesystem::path path(filename);
  // Try to parse as MLIR source file
  auto ret = std::unique_ptr<MLIRModel>();
  {
    auto moduleRef = mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
    if (!moduleRef) {
      LOG(ERROR) << "Failed to load MLIR module from: " << filename;
      return nullptr;
    }
    ret = MLIRModel::create(std::move(moduleRef));
    ret->set_metadata_prop("model_path", path.u8string());
  }
  MY_LOG(1) << "ret is create: " << ret.get();
  return ret;
}

std::unique_ptr<MLIRModel>
MLIRModel::clone(int64_t /*external_data_threshold*/) const {
  auto *cloned_op = module_.get()->clone();
  auto cloned_module = mlir::cast<mlir::ModuleOp>(cloned_op);
  auto clonedModel =
      MLIRModel::create(mlir::OwningOpRef<mlir::ModuleOp>(cloned_module));
  clonedModel->metadata_ = this->metadata_;
  return clonedModel;
}

MLIRGraph &MLIRModel::main_graph() const { return *main_graph_; }

void MLIRModel::set_metadata_prop(const std::string &key,
                                  const std::string &value) {
  metadata_[key] = value;
}

std::string MLIRModel::get_metadata_prop(const std::string &key) const {
  auto it = metadata_.find(key);
  return (it != metadata_.end()) ? it->second : "";
}

bool MLIRModel::has_metadata_prop(const std::string &key) const {
  return metadata_.find(key) != metadata_.end();
}

mlir::ModuleOp MLIRModel::getModule() const { return *module_; }

void MLIRModel::maybe_dump_mlir(const std::filesystem::path &path) const {
  if (ENV_PARAM(MORPHIZEN_DEBUG_MLIR_MODEL) > 5) {
    std::string debug_path = "debug_empty_module.mlir";
    if (!path.empty()) {
      std::filesystem::path debug_file_path =
          path.parent_path() / "debug_empty_module.mlir";
      debug_path = debug_file_path.string();
    }

    try {
      std::ofstream output(debug_path);
      if (output.is_open()) {
        std::string mlir_str;
        llvm::raw_string_ostream stream(mlir_str);
        getModule().print(stream);
        output << mlir_str;
        output.close();
        LOG(INFO) << "Debug: Saved empty MLIR module to: " << debug_path;
      } else {
        LOG(WARNING) << "Debug: Failed to open file for writing: "
                     << debug_path;
      }
    } catch (const std::exception &e) {
      LOG(WARNING) << "Debug: Failed to save empty MLIR module: " << e.what();
    }
  }
}

std::unique_ptr<MLIRGraph>
MLIRModel::create_main_graph(MLIRModel &model, mlir::OpBuilder &builder,
                             mlir::ModuleOp &module,
                             const std::string &graph_name) {
  const std::string &name = "main_graph";
  auto mainFunc = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), name,
      /*type=*/builder.getFunctionType({}, {}), /*attrs=*/{});
  module.push_back(mainFunc);

  // Set the onnx.graph.name attribute
  mainFunc->setAttr(attr_names::ONNX_GRAPH_NAME,
                    builder.getStringAttr(graph_name));

  // retrieve the attribute back and print it.
  if (auto attr = mainFunc->getAttr(attr_names::ONNX_GRAPH_NAME)) {
    if (auto string_attr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
      MY_LOG(1) << "Set attribute " << attr_names::ONNX_GRAPH_NAME << " = "
                << string_attr.getValue().str();
    }
  }

  // Create and set insertion point to entry block.
  mainFunc.getBody().push_back(new mlir::Block);
  builder.setInsertionPointToStart(&mainFunc.getBody().back());

  // Note: The following code requires ONNX-MLIR specific operations
  // which are not available in this context. This is a placeholder
  // for the actual ONNX graph import functionality.
  /*

  */

  // Create and return MLIRGraph object
  return std::make_unique<MLIRGraph>(model, mainFunc);
}

std::string MLIRModel::serialize_as_string() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  mlir::OpPrintingFlags flags;
  if (ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_GENERIC)) {
    flags.printGenericOpForm();
  }
  if (ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_DEBUG_INFO)) {
    flags.enableDebugInfo();
    flags.printValueUsers();
  }

  getModule().print(stream, flags);
  return result;
}

} // namespace mlir_impl
} // namespace morphizen
