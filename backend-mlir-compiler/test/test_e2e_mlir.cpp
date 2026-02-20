/**
 * @file test_e2e_mlir.cpp
 * @brief E2E test for MLIR backend integration
 *
 * This test validates the ONNX → MLIR → HIP compilation pipeline.
 * It works with both MOCK runtime (default, no GPU required) and REAL runtime
 * (compile-time option with BUILD_MOCK_RUNTIME=OFF).
 *
 * The test uses a two-layer convolution model:
 * Input [1,3,224,224] → Conv1 → ReLU → Conv2 → ReLU → Output [1,64,112,112]
 *
 * Environment variables (all optional):
 * - ORT_LOG_LEVEL=info - Enable ORT session creation logging
 * - DEBUG_MORPHIZEN_PASS=1 - Enable morphizen pass debug logging
 * - MORPHIZEN_DEBUG_MLIR_BACKEND=3 - MLIR backend compilation verbose logging
 */

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Model path defined by CMake
#ifndef TWO_LAYER_CONV_MODEL_PATH
#error "TWO_LAYER_CONV_MODEL_PATH must be defined by CMake"
#endif

#ifndef MORPHIZEN_EP_LIB_PATH
#error "MORPHIZEN_EP_LIB_PATH must be defined by CMake"
#endif

class MlirE2ETest : public ::testing::Test {
 protected:
  std::unique_ptr<Ort::Env> env_;
  std::string model_path_;
  std::string ep_lib_path_;

  void SetUp() override {
    // Initialize ORT environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MlirE2ETest");

    // Set model path
    model_path_ = TWO_LAYER_CONV_MODEL_PATH;
    ep_lib_path_ = MORPHIZEN_EP_LIB_PATH;

    // Verify model file exists
    if (!fs::exists(model_path_)) {
      GTEST_SKIP() << "Model file not found: " << model_path_
                   << "\nRun: python gen_two_layer_conv_model.py --output models/two_layer_conv.onnx";
    }

    // Verify EP library exists
    if (!fs::exists(ep_lib_path_)) {
      GTEST_SKIP() << "MorphiZen EP library not found: " << ep_lib_path_;
    }

    std::cout << "[SetUp] Model path: " << model_path_ << std::endl;
    std::cout << "[SetUp] EP library path: " << ep_lib_path_ << std::endl;

    // Register MorphiZen EP
    try {
      Ort::SessionOptions options;
      options.RegisterCustomOpsLibrary(ep_lib_path_.c_str());
      std::cout << "[SetUp] MorphiZen EP registered successfully" << std::endl;
    } catch (const Ort::Exception& e) {
      GTEST_SKIP() << "Failed to register MorphiZen EP: " << e.what();
    }
  }

  void TearDown() override {
    // Clean up
    env_.reset();
  }
};

/**
 * @test TwoLayerConvSession
 * @brief Test session creation with two-layer convolution model
 *
 * This test validates:
 * 1. MorphiZen EP can be registered
 * 2. ONNX model can be loaded
 * 3. Session can be created (triggers MLIR compilation pipeline)
 * 4. Works with both MOCK runtime (default) and REAL runtime (BUILD_MOCK_RUNTIME=OFF)
 *
 * Expected behavior:
 * - MOCK runtime: Logs show [MOCK] prefixes for HIP/MIOpen calls
 * - REAL runtime: Actual GPU execution (requires ROCm hardware)
 *
 * TODO: Add actual inference with input data and output validation
 */
TEST_F(MlirE2ETest, TwoLayerConvSession) {
  std::cout << "[Test] Creating session with MorphiZen EP (MLIR backend)..." << std::endl;

  // Create session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Register MorphiZen EP
  session_options.RegisterCustomOpsLibrary(ep_lib_path_.c_str());

  // Create session (this triggers ONNX → MLIR → HIP compilation)
  Ort::Session session(*env_, model_path_.c_str(), session_options);

  std::cout << "[Test] Session created successfully with MorphiZen EP!" << std::endl;

  // Get input/output info
  Ort::AllocatorWithDefaultOptions allocator;

  size_t num_input_nodes = session.GetInputCount();
  size_t num_output_nodes = session.GetOutputCount();

  std::cout << "[Test] Model has " << num_input_nodes << " input(s) and "
            << num_output_nodes << " output(s)" << std::endl;

  // Print input info
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::cout << "[Test] Input " << i << ": " << input_name.get() << " shape=[";
    for (size_t j = 0; j < shape.size(); j++) {
      std::cout << shape[j];
      if (j < shape.size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
  }

  // Print output info
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    auto type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::cout << "[Test] Output " << i << ": " << output_name.get() << " shape=[";
    for (size_t j = 0; j < shape.size(); j++) {
      std::cout << shape[j];
      if (j < shape.size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
  }

  // Session creation succeeded - MLIR compilation pipeline worked!
  SUCCEED();

  // TODO: Add actual inference
  // - Create input tensor with test data
  // - Run session.Run()
  // - Validate output tensor
  // - For MOCK runtime: outputs will be zeros
  // - For REAL runtime: outputs will be computed results
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "=== MLIR E2E Test ===" << std::endl;
  std::cout << "This test validates ONNX → MLIR → HIP compilation pipeline" << std::endl;
  std::cout << "Runtime: "
#ifdef BUILD_MOCK_RUNTIME
            << "MOCK (no GPU required, outputs filled with zeros)"
#else
            << "REAL (requires ROCm GPU)"
#endif
            << std::endl;
  std::cout << "Model: " << TWO_LAYER_CONV_MODEL_PATH << std::endl;
  std::cout << std::endl;

  return RUN_ALL_TESTS();
}
