/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip-test: load a per-model artifact (LLVM .bc via LlvmIrJit, or a native
// .dll/.so via morphizen::Plugin -- selected by file extension), parse
// metadata, generate test inputs, run inference, validate outputs. Exercises
// both loaders the EP uses at runtime.

#include "CrashHandler.h"
#include "LoadedArtifact.h" // LoadedArtifact, ArtifactKind
#include "hip/Support/DiskFileSystem.h"
#include "hip/artifact_abi.h" // hipdnn::abi symbol names
#include <memory>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // Conversion warnings in LLVM JSON.h
#endif
#include "llvm/Support/JSON.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Local copy of the tensor_t wire-protocol ABI; sibling copies live in
// custom_op_mlir.hpp (compiler) and hipdnn_ep_runtime.h (runtime). The
// static_asserts below catch layout drift between the three copies.
enum {
  TENSOR_MEMORY_CPU = 0,  // == OrtMemoryInfoDeviceType_CPU
  TENSOR_MEMORY_GPU = 1,  // == OrtMemoryInfoDeviceType_GPU
  TENSOR_MEMORY_FPGA = 2, // == OrtMemoryInfoDeviceType_FPGA
  TENSOR_MEMORY_NPU = 3,  // == OrtMemoryInfoDeviceType_NPU
};

typedef struct {
  void *data;          // Data pointer (host or GPU-accessible per memory_type)
  int64_t *shape;      // Array of dimension sizes
  size_t rank;         // Number of dimensions
  size_t element_size; // Bytes per element (e.g. 4=float32, 2=float16, 8=int64)
  int memory_type;     // TENSOR_MEMORY_CPU / _GPU / _FPGA / _NPU
} tensor_t;

static_assert(offsetof(tensor_t, data) == 0,
              "tensor_t.data must remain the first field");
static_assert(offsetof(tensor_t, shape) == sizeof(void *),
              "tensor_t.shape moved -- update all three tensor_t copies");
static_assert(offsetof(tensor_t, memory_type) ==
                  offsetof(tensor_t, element_size) + sizeof(size_t),
              "tensor_t.memory_type moved -- update all three tensor_t "
              "copies");

typedef struct {
  tensor_t *data; // Array of tensors
  size_t count;   // Number of tensors
} span_t;

// Parse --input-shape arguments: "0=8,3,224,224;1=8,512"
std::map<int, std::vector<int64_t>>
parseShapeOverrides(const std::string &arg) {
  std::map<int, std::vector<int64_t>> result;
  std::istringstream iss(arg);
  std::string token;

  while (std::getline(iss, token, ';')) {
    size_t eq_pos = token.find('=');
    if (eq_pos == std::string::npos)
      continue;

    int index = std::stoi(token.substr(0, eq_pos));
    std::string dims_str = token.substr(eq_pos + 1);
    std::vector<int64_t> shape;

    std::istringstream dims_iss(dims_str);
    std::string dim;
    while (std::getline(dims_iss, dim, ',')) {
      shape.push_back(std::stoll(dim));
    }

    result[index] = shape;
  }

  return result;
}

// Resolve dynamic dimensions to concrete values. Standard ONNX uses -1 for
// unknown dims; Range emits INT64_MIN because its output size is computed
// at runtime from scalar inputs and has no static value in the metadata.
// Clamp any negative dim to default_batch (1) so the test harness can
// allocate a valid non-null buffer regardless of the dynamic shape.
void resolveShape(std::vector<int64_t> &shape, int64_t default_batch = 1) {
  for (auto &dim : shape) {
    if (dim < 0) {
      dim = default_batch;
    }
  }
}

// Calculate total element count
size_t elementCount(const std::vector<int64_t> &shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  return count;
}

// Convert float to IEEE 754 half-precision (binary16)
static uint16_t floatToHalf(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof(f));

  uint32_t sign = (f >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xFF) - 127;
  uint32_t mant = f & 0x7FFFFF;

  uint16_t h;
  if (exp > 15) {
    h = static_cast<uint16_t>((sign << 15) | 0x7C00); // clamp to Inf
  } else if (exp < -14) {
    h = static_cast<uint16_t>(sign << 15); // flush to zero
  } else {
    h = static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
  }
  return h;
}

// Generate test data with valid values for the element type
void generateTestData(void *data, size_t sizeBytes, size_t elemSize) {
  if (elemSize == 4) {
    auto *fdata = static_cast<float *>(data);
    size_t count = sizeBytes / 4;
    for (size_t i = 0; i < count; i++)
      fdata[i] = static_cast<float>(i % 100 + 1) * 0.0001f;
  } else if (elemSize == 2) {
    auto *hdata = static_cast<uint16_t *>(data);
    size_t count = sizeBytes / 2;
    for (size_t i = 0; i < count; i++)
      hdata[i] = floatToHalf(static_cast<float>(i % 100 + 1) * 0.0001f);
  } else {
    auto *bytes = static_cast<uint8_t *>(data);
    for (size_t i = 0; i < sizeBytes; i++)
      bytes[i] = static_cast<uint8_t>(i % 256);
  }
}

// Convert IEEE 754 half-precision (binary16) to float for display/validation
static float halfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;

  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31; // +/- zero
    } else {
      // Subnormal: normalize
      float val = std::ldexp(static_cast<float>(mant), -24);
      return sign ? -val : val;
    }
  } else if (exp == 0x1F) {
    // Inf or NaN
    f = (sign << 31) | 0x7F800000 | (mant << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }

  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

// Validate output buffer (check for NaN/Inf which may indicate failure)
bool validateOutput(const void *data, size_t sizeBytes, size_t elemSize,
                    bool verbose) {
  size_t elemCount = sizeBytes / elemSize;

  if (verbose && sizeBytes > 0) {
    std::cout << "  First 10 values: ";
    for (size_t i = 0; i < std::min(elemCount, size_t(10)); i++) {
      if (elemSize == 4) {
        std::cout << static_cast<const float *>(data)[i] << " ";
      } else if (elemSize == 2) {
        const auto *p = static_cast<const uint16_t *>(data);
        std::cout << halfToFloat(p[i]) << " ";
      } else if (elemSize == 8) {
        const auto *p = static_cast<const int64_t *>(data);
        std::cout << p[i] << " ";
      } else {
        const auto *p = static_cast<const uint8_t *>(data);
        std::cout << static_cast<int>(p[i]) << " ";
      }
    }
    std::cout << "\n";
  }

  // NaN/Inf validation for floating-point types
  size_t nan_count = 0, inf_count = 0;

  if (elemSize == 4) {
    const auto *fdata = static_cast<const float *>(data);
    for (size_t i = 0; i < elemCount; i++) {
      if (std::isnan(fdata[i]))
        nan_count++;
      else if (std::isinf(fdata[i]))
        inf_count++;
    }
  } else if (elemSize == 2) {
    const auto *hdata = static_cast<const uint16_t *>(data);
    for (size_t i = 0; i < elemCount; i++) {
      float val = halfToFloat(hdata[i]);
      if (std::isnan(val))
        nan_count++;
      else if (std::isinf(val))
        inf_count++;
    }
  }

  if (nan_count > 0 || inf_count > 0) {
    std::cerr << "VALIDATION FAILED: " << nan_count << " NaN, " << inf_count
              << " Inf values\n";
    return false;
  }

  return true;
}

// Parsed metadata for a single tensor
struct TensorMeta {
  std::vector<int64_t> shape;
  size_t element_size = sizeof(float);
};

// Parse the metadata JSON string returned by inference_get_metadata_json().
// Schema defined in proto/model_metadata.fbs.
static bool parseMetadata(const char *json_str, std::vector<TensorMeta> &inputs,
                          std::vector<TensorMeta> &outputs,
                          std::string &error) {
  auto parsed = llvm::json::parse(llvm::StringRef(json_str));
  if (!parsed) {
    error = "JSON parse error: ";
    llvm::handleAllErrors(
        parsed.takeError(),
        [&](const llvm::ErrorInfoBase &e) { error += e.message(); });
    return false;
  }

  auto *root = parsed->getAsObject();
  if (!root) {
    error = "metadata JSON root is not an object";
    return false;
  }

  auto parseTensors = [&](llvm::StringRef key,
                          std::vector<TensorMeta> &out) -> bool {
    auto *arr = root->getArray(key);
    if (!arr)
      return true; // empty is fine
    for (auto &elem : *arr) {
      auto *obj = elem.getAsObject();
      if (!obj) {
        error = "tensor entry is not an object";
        return false;
      }
      TensorMeta tm;
      if (auto *shapeArr = obj->getArray("shape")) {
        for (auto &d : *shapeArr) {
          if (auto v = d.getAsInteger())
            tm.shape.push_back(*v);
        }
      }
      if (auto es = obj->getInteger("element_size")) {
        tm.element_size = (*es > 0) ? static_cast<size_t>(*es) : sizeof(float);
      }
      out.push_back(std::move(tm));
    }
    return true;
  };

  return parseTensors("inputs", inputs) && parseTensors("outputs", outputs);
}
int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-test");
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <model.{bc,dll,so}> [--input-shape INDEX=DIMS;...] "
                 "[--iterations N] [--verbose] [--validate]\n";
    std::cerr << "Example: " << argv[0]
              << " model.bc --input-shape 0=8,3,224,224 --iterations 10 "
                 "--verbose --validate\n";
    std::cerr << "The loader is chosen from the artifact's file extension "
                 "(.bc -> LLVM IR, .dll/.so -> native).\n";
    return 1;
  }

  std::string bc_path = argv[1];
  std::map<int, std::vector<int64_t>> shape_overrides;
  int iterations = 1;
  bool verbose = false;
  bool validate = false;

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input-shape" && i + 1 < argc) {
      shape_overrides = parseShapeOverrides(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--validate") {
      validate = true;
    }
  }

  if (verbose)
    std::cout << "Loading artifact: " << bc_path << "\n";
  // LoadedArtifact picks the format from the file extension and selects the
  // loader (LlvmIrJit for LLVM IR, morphizen::Plugin for native) -- the same
  // loader the EP uses, so the tool and the library never drift.
  std::string load_err;
  auto artifact = mlir_compilation::customop::LoadedArtifact::createFromFile(
      bc_path, &load_err);
  if (!artifact) {
    std::cerr << "ERROR: " << load_err << "\n";
    return 1;
  }
  if (verbose &&
      artifact->kind() == mlir_compilation::customop::ArtifactKind::NATIVE)
    std::cout << "Detected native artifact (loaded via morphizen::Plugin)\n";

  auto init_func =
      artifact->get_method<int, void **, void *>(hipdnn::abi::kInferenceInit);
  auto compute_func = artifact->get_method<int, void *, void *, void *>(
      hipdnn::abi::kInferenceCompute);
  auto cleanup_func =
      artifact->get_method<int, void *>(hipdnn::abi::kInferenceCleanup);
  auto get_metadata_func = artifact->get_method<const char *>(
      hipdnn::abi::kInferenceGetMetadataJson);

  if (!init_func || !compute_func || !cleanup_func || !get_metadata_func) {
    std::cerr << "ERROR: Missing required interface functions\n";
    std::cerr << "  inference_init: " << (init_func ? "OK" : "MISSING") << "\n";
    std::cerr << "  inference_compute: " << (compute_func ? "OK" : "MISSING")
              << "\n";
    std::cerr << "  inference_cleanup: " << (cleanup_func ? "OK" : "MISSING")
              << "\n";
    std::cerr << "  inference_get_metadata_json: "
              << (get_metadata_func ? "OK" : "MISSING") << "\n";
    return 1;
  }

  // Get and parse metadata JSON
  const char *json_str = get_metadata_func();
  if (verbose)
    std::cout << "Metadata JSON: " << json_str << "\n";

  std::vector<TensorMeta> input_metas, output_metas;
  std::string parseErr;
  if (!parseMetadata(json_str, input_metas, output_metas, parseErr)) {
    std::cerr << "ERROR: Failed to parse metadata: " << parseErr << "\n";
    return 1;
  }

  size_t num_inputs = input_metas.size();
  size_t num_outputs = output_metas.size();

  if (verbose) {
    std::cout << "Model metadata:\n";
    std::cout << "  Inputs: " << num_inputs << "\n";
    std::cout << "  Outputs: " << num_outputs << "\n";
  }

  // Prepare input shapes (apply overrides and resolve dynamics)
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t i = 0; i < num_inputs; i++) {
    std::vector<int64_t> shape = input_metas[i].shape;

    if (shape_overrides.count(static_cast<int>(i)))
      shape = shape_overrides[static_cast<int>(i)];

    resolveShape(shape);
    input_shapes.push_back(shape);

    if (verbose) {
      std::cout << "  Input " << i << " shape: [";
      for (size_t j = 0; j < shape.size(); j++) {
        std::cout << shape[j];
        if (j + 1 < shape.size())
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }

  // Prepare output shapes
  std::vector<std::vector<int64_t>> output_shapes;
  for (size_t i = 0; i < num_outputs; i++) {
    std::vector<int64_t> shape = output_metas[i].shape;
    resolveShape(shape);
    output_shapes.push_back(shape);

    if (verbose) {
      std::cout << "  Output " << i << " shape: [";
      for (size_t j = 0; j < shape.size(); j++) {
        std::cout << shape[j];
        if (j + 1 < shape.size())
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }

  // Element sizes from metadata
  std::vector<size_t> input_elem_sizes;
  for (size_t i = 0; i < num_inputs; i++)
    input_elem_sizes.push_back(input_metas[i].element_size);
  std::vector<size_t> output_elem_sizes;
  for (size_t i = 0; i < num_outputs; i++)
    output_elem_sizes.push_back(output_metas[i].element_size);

  // Allocate input buffers and generate test data
  std::vector<std::vector<uint8_t>> input_buffers;
  std::vector<std::vector<int64_t>> input_shape_storage;
  std::vector<tensor_t> input_tensors;
  for (size_t i = 0; i < input_shapes.size(); i++) {
    size_t count = elementCount(input_shapes[i]);
    size_t sizeBytes = count * input_elem_sizes[i];
    std::vector<uint8_t> buffer(sizeBytes);
    generateTestData(buffer.data(), sizeBytes, input_elem_sizes[i]);
    input_buffers.push_back(std::move(buffer));
    input_shape_storage.push_back(input_shapes[i]);

    tensor_t tensor;
    tensor.data = input_buffers.back().data();
    tensor.shape = input_shape_storage.back().data();
    tensor.rank = input_shape_storage.back().size();
    tensor.element_size = input_elem_sizes[i];
    tensor.memory_type = TENSOR_MEMORY_CPU; // host buffers, runtime does H2D
    input_tensors.push_back(tensor);
  }

  span_t inputs_span;
  inputs_span.data = input_tensors.data();
  inputs_span.count = input_tensors.size();

  // Allocate output buffers
  std::vector<std::vector<uint8_t>> output_buffers;
  std::vector<std::vector<int64_t>> output_shape_storage;
  std::vector<tensor_t> output_tensors;
  for (size_t i = 0; i < output_shapes.size(); i++) {
    size_t count = elementCount(output_shapes[i]);
    size_t sizeBytes = count * output_elem_sizes[i];
    std::vector<uint8_t> buffer(sizeBytes);
    output_buffers.push_back(std::move(buffer));
    output_shape_storage.push_back(output_shapes[i]);

    tensor_t tensor;
    tensor.data = output_buffers.back().data();
    tensor.shape = output_shape_storage.back().data();
    tensor.rank = output_shape_storage.back().size();
    tensor.element_size = output_elem_sizes[i];
    tensor.memory_type = TENSOR_MEMORY_CPU; // host buffers, runtime does D2H
    output_tensors.push_back(tensor);
  }

  span_t outputs_span;
  outputs_span.data = output_tensors.data();
  outputs_span.count = output_tensors.size();

  // Initialize inference context
  if (verbose)
    std::cout << "\nInitializing inference context...\n";
  // DiskFileSystem resolves "constants.bin" relative to the current working
  // directory, which matches the WORKING_DIRECTORY set in the e2e CMakeLists.
  mlir::hip::DiskFileSystem fs(".");
  void *state = nullptr;
  int ret = init_func(&state, &fs);
  if (ret != 0) {
    std::cerr << "ERROR: inference_init failed with code " << ret << "\n";
    return 1;
  }

  // Run inference
  if (verbose)
    std::cout << "Running inference (" << iterations << " iteration(s))...\n";

  for (int iter = 0; iter < iterations; iter++) {
    ret = compute_func(state, &inputs_span, &outputs_span);
    if (ret != 0) {
      std::cerr << "ERROR: inference_compute failed with code " << ret
                << " at iteration " << iter << "\n";
      cleanup_func(state);
      return 1;
    }

    if (verbose && iterations > 1)
      std::cout << "  Iteration " << (iter + 1) << "/" << iterations << "\n";
  }

  // Validate outputs
  if (validate) {
    if (verbose)
      std::cout << "\nValidating outputs...\n";
    bool all_valid = true;
    for (size_t i = 0; i < output_buffers.size(); i++) {
      if (verbose)
        std::cout << "Output " << i << ":\n";
      if (!validateOutput(output_buffers[i].data(), output_buffers[i].size(),
                          output_elem_sizes[i], verbose)) {
        all_valid = false;
      }
    }

    if (!all_valid) {
      cleanup_func(state);
      return 1;
    }

    if (verbose)
      std::cout << "All outputs valid!\n";
  }

  // Cleanup
  if (verbose)
    std::cout << "\nCleaning up...\n";
  ret = cleanup_func(state);
  if (ret != 0) {
    std::cerr << "WARNING: inference_cleanup failed with code " << ret << "\n";
  }

  std::cout << "SUCCESS: Model executed successfully\n";
  return 0;
}
