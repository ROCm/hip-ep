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
#include "hip/timing.h"
#include <memory>
// On real (GPU) builds we link hip::host so the output-allocator callback can
// give the model a real device buffer and copy it back to host for validation
// (same as the EP's callback). Mock builds use CPU stubs that write host memory
// directly, so the callback just returns a host buffer -- no HIP needed.
#ifdef HIPDNN_EP_LINK_HIP_HOST
#include <hip/hip_runtime.h>
#endif
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

// Local mirror of `hipdnn_output_allocator_t` (lib/Runtime/hipdnn_ep_runtime.h)
// / `output_allocator_t` (custom_op_mlir.hpp). Layout MUST match: the runtime
// setter does a plain struct copy across the model.dll <-> caller boundary. See
// those headers for the ABI contract.
typedef struct {
  void *self; // opaque caller context (borrowed; runtime never owns/frees)
  void *(*allocate)(void *self, int64_t out_idx, const int64_t *shape,
                    int64_t rank, int64_t elem_size);
} output_allocator_t;

static_assert(offsetof(output_allocator_t, self) == 0,
              "output_allocator_t.self must remain the first field");

// One graph output produced in-graph via hip.alloc_output -> the callback
// below. Owns the buffer the model.dll writes into (device memory in real
// builds, host memory in mock builds) plus a host-side copy used for
// validation.
struct OutputSlot {
  std::vector<int64_t> shape;
  size_t elem_size = 0;
  size_t nbytes = 0;
  std::vector<uint8_t> host; // validation copy (== the write buffer in mock)
#ifdef HIPDNN_EP_LINK_HIP_HOST
  void *gpu = nullptr; // device buffer the model.dll writes into
#endif
};

// Context threaded through output_allocator_t.self. In allocator mode the DLL
// calls back once per graph output (indexed by out_idx) as each output's shape
// becomes known in-graph; we allocate a write buffer here and return it.
struct OutputAllocatorCtx {
  std::vector<OutputSlot> outs; // indexed by compiler-order out_idx
};

// output_allocator_t.allocate: allocate the buffer for graph output `out_idx`
// at the DLL's in-graph shape and return the pointer the DLL writes into. On a
// real build that must be device memory (the GPU kernels target it); on a mock
// build a host buffer suffices. Never returns null (the DLL builds a memref
// from the pointer -- a null would segfault with no diagnostic).
static void *hip_test_allocate(void *self, int64_t out_idx,
                               const int64_t *shape, int64_t rank,
                               int64_t elem_size) {
  auto *ctx = static_cast<OutputAllocatorCtx *>(self);
  if (out_idx < 0) {
    std::cerr << "ERROR: output allocator got negative out_idx " << out_idx
              << "\n";
    std::abort();
  }
  if (static_cast<size_t>(out_idx) >= ctx->outs.size())
    ctx->outs.resize(static_cast<size_t>(out_idx) + 1);
  OutputSlot &slot = ctx->outs[static_cast<size_t>(out_idx)];
  const std::vector<int64_t> newShape(shape, shape + rank);
  size_t nelem = 1;
  for (int64_t i = 0; i < rank; ++i)
    nelem *= static_cast<size_t>(shape[i]);
  const size_t newBytes = nelem * static_cast<size_t>(elem_size);
  const bool reuse = slot.shape == newShape &&
                     slot.elem_size == static_cast<size_t>(elem_size) &&
                     slot.nbytes == newBytes;
  slot.shape = std::move(newShape);
  slot.elem_size = static_cast<size_t>(elem_size);
  slot.nbytes = newBytes;
  if (!reuse)
    slot.host.assign(slot.nbytes, 0);
#ifdef HIPDNN_EP_LINK_HIP_HOST
  if (slot.gpu && reuse)
    return slot.gpu;
  if (slot.gpu) {
    (void)hipFree(slot.gpu);
    slot.gpu = nullptr;
  }
  hipError_t e = hipMalloc(&slot.gpu, slot.nbytes ? slot.nbytes : 1);
  if (e != hipSuccess || !slot.gpu) {
    std::cerr << "ERROR: hipMalloc(" << slot.nbytes << ") for output "
              << out_idx << " failed: " << hipGetErrorString(e) << "\n";
    std::abort();
  }
  return slot.gpu;
#else
  return slot.host.data();
#endif
}

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

// Parse --inputs arguments: "0=pixels.bin;1=ids.bin". Each file is raw tensor
// bytes in element order and must match that input's byte size. Paths may
// contain ':' (Windows drives); ';' separates entries.
std::map<int, std::string> parseInputFiles(const std::string &arg) {
  std::map<int, std::string> result;
  std::istringstream iss(arg);
  std::string token;
  while (std::getline(iss, token, ';')) {
    if (token.empty())
      continue;
    size_t eq_pos = token.find('=');
    if (eq_pos == std::string::npos || eq_pos == 0)
      continue;
    int index = std::stoi(token.substr(0, eq_pos));
    result[index] = token.substr(eq_pos + 1);
  }
  return result;
}

// Replace a generated input buffer with raw bytes from `path`.
static bool loadRawInput(const std::string &path, std::vector<uint8_t> &buffer,
                         std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  if (n < 0 || static_cast<size_t>(n) != buffer.size()) {
    error = path + " is " + std::to_string(n < 0 ? 0 : static_cast<long long>(n)) +
            " bytes, expected " + std::to_string(buffer.size());
    return false;
  }
  in.seekg(0, std::ios::beg);
  if (buffer.empty())
    return true;
  in.read(reinterpret_cast<char *>(buffer.data()),
          static_cast<std::streamsize>(buffer.size()));
  if (!in) {
    error = "short read from " + path;
    return false;
  }
  return true;
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
  } else if (elemSize == 8) {
    // HIP_TEST_I64=0,1,2,3,4,5,6,7 uses those ids instead of raw bytes.
    // Raw bytes are out of range for an embedding table.
    if (const char *spec = std::getenv("HIP_TEST_I64")) {
      auto *idata = static_cast<int64_t *>(data);
      size_t count = sizeBytes / 8;
      size_t n = 0;
      const char *p = spec;
      while (*p && n < count) {
        char *end = nullptr;
        long long v = std::strtoll(p, &end, 10);
        if (end == p)
          break;
        idata[n++] = static_cast<int64_t>(v);
        p = (*end == ',') ? end + 1 : end;
      }
      while (n < count)
        idata[n++] = 0;
      return;
    }
    auto *bytes = static_cast<uint8_t *>(data);
    for (size_t i = 0; i < sizeBytes; i++)
      bytes[i] = static_cast<uint8_t>(i % 256);
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
    if (elemSize == 4 && elemCount > 0) {
      const auto *fdata = static_cast<const float *>(data);
      if (const char *dump = std::getenv("HIP_TEST_DUMP")) {
        std::ofstream out(dump, std::ios::binary);
        out.write(reinterpret_cast<const char *>(fdata),
                  static_cast<std::streamsize>(elemCount * sizeof(float)));
        std::cout << "  Dumped " << elemCount << " f32 values to " << dump
                  << "\n";
      }
    }
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

/// Parse one stdin record of comma-separated int64 token ids.
static bool parseTokenIds(const std::string &line, size_t expected,
                          std::vector<int64_t> &ids, std::string &error) {
  ids.clear();
  std::istringstream stream(line);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      error = "empty token id";
      return false;
    }
    try {
      ids.push_back(std::stoll(token));
    } catch (const std::exception &) {
      error = "invalid token id '" + token + "'";
      return false;
    }
  }
  if (ids.size() != expected) {
    error = "expected " + std::to_string(expected) + " token ids, got " +
            std::to_string(ids.size());
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-test");
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <model.{bc,dll,so}> [--input-shape INDEX=DIMS;...] "
                 "[--inputs INDEX=FILE;...] [--iterations N] [--verbose] "
                 "[--validate] [--token-ids-stdin POSITION|dynamic]\n";
    std::cerr << "Example: " << argv[0]
              << " model.bc --input-shape 0=8,3,224,224 --iterations 10 "
                 "--verbose --validate\n";
    std::cerr << "The loader is chosen from the artifact's file extension "
                 "(.bc -> LLVM IR, .dll/.so -> native).\n";
    return 1;
  }

  std::string bc_path = argv[1];
  std::map<int, std::vector<int64_t>> shape_overrides;
  std::map<int, std::string> input_files;
  int iterations = 1;
  bool verbose = false;
  bool validate = false;
  int64_t tokenIdsPosition = -1;
  bool tokenIdsStdin = false;
  bool dynamicTokenIdsPosition = false;

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input-shape" && i + 1 < argc) {
      shape_overrides = parseShapeOverrides(argv[++i]);
    } else if (arg == "--inputs" && i + 1 < argc) {
      input_files = parseInputFiles(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--validate") {
      validate = true;
    } else if (arg == "--token-ids-stdin" && i + 1 < argc) {
      tokenIdsStdin = true;
      const std::string positionArg = argv[++i];
      if (positionArg == "dynamic")
        dynamicTokenIdsPosition = true;
      else
        tokenIdsPosition = std::stoll(positionArg);
    }
  }

  if (verbose)
    std::cout << "Loading artifact: " << bc_path << "\n";
  // Same split as hip-onnx-runner: session is load plus runtime init, and
  // inference is only the compute call. Input fill sits between the two.
  const auto sessionStart = std::chrono::steady_clock::now();
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

  auto init_func = artifact->get_method<int, void **, void *, const void *>(
      hipdnn::abi::kInferenceInit);
  // Output-allocator ABI: 2-arg inference_compute(state, inputs). Graph outputs
  // are allocated in-graph via the callback installed through
  // hipdnn_ep_set_output_allocator -- there is no outputs span.
  auto compute_func =
      artifact->get_method<int, void *, void *>(hipdnn::abi::kInferenceCompute);
  auto set_alloc_func =
      artifact->get_method<void, void *, const output_allocator_t *>(
          hipdnn::abi::kSetOutputAllocator);
  auto cleanup_func =
      artifact->get_method<int, void *>(hipdnn::abi::kInferenceCleanup);
  auto get_metadata_func = artifact->get_method<const char *>(
      hipdnn::abi::kInferenceGetMetadataJson);

  if (!init_func || !compute_func || !set_alloc_func || !cleanup_func ||
      !get_metadata_func) {
    std::cerr << "ERROR: Missing required interface functions\n";
    std::cerr << "  inference_init: " << (init_func ? "OK" : "MISSING") << "\n";
    std::cerr << "  inference_compute: " << (compute_func ? "OK" : "MISSING")
              << "\n";
    std::cerr << "  hipdnn_ep_set_output_allocator: "
              << (set_alloc_func ? "OK" : "MISSING") << "\n";
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

  // Load and metadata are part of session setup. Input fill is not.
  double sessionSeconds = elapsed_since(sessionStart);

  // Output shapes are NOT pre-computed here: in output-allocator mode the
  // model.dll sizes each graph output in-graph and requests its buffer via the
  // callback below (hip_test_allocate), passing the concrete shape at that
  // point. `output_metas` is still parsed above for the output count / logging.

  // Element sizes from metadata (inputs only; output element sizes arrive via
  // the allocator callback's elem_size argument).
  std::vector<size_t> input_elem_sizes;
  for (size_t i = 0; i < num_inputs; i++)
    input_elem_sizes.push_back(input_metas[i].element_size);

  // Allocate input buffers and generate test data
  std::vector<std::vector<uint8_t>> input_buffers;
  std::vector<std::vector<int64_t>> input_shape_storage;
  std::vector<tensor_t> input_tensors;
  for (size_t i = 0; i < input_shapes.size(); i++) {
    size_t count = elementCount(input_shapes[i]);
    size_t sizeBytes = count * input_elem_sizes[i];
    std::vector<uint8_t> buffer(sizeBytes);
    auto file = input_files.find(static_cast<int>(i));
    if (file != input_files.end()) {
      std::string loadErr;
      if (!loadRawInput(file->second, buffer, loadErr)) {
        std::cerr << "ERROR: input " << i << ": " << loadErr << "\n";
        return 1;
      }
      if (verbose)
        std::cout << "  Loaded input " << i << " from " << file->second << " ("
                  << sizeBytes << " bytes)\n";
    } else {
      generateTestData(buffer.data(), sizeBytes, input_elem_sizes[i]);
    }
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

  // Output-allocator context: the model.dll's in-graph hip.alloc_output calls
  // back through this to obtain each graph-output buffer once its shape is
  // known. Reserve num_outputs slots up front so a graph that produces them out
  // of order still lands each at its compiler-order index.
  OutputAllocatorCtx octx;
  octx.outs.resize(num_outputs);
  output_allocator_t alloc;
  alloc.self = &octx;
  alloc.allocate = &hip_test_allocate;

  // Initialize inference context
  if (verbose)
    std::cout << "\nInitializing inference context...\n";
  // DiskFileSystem resolves "constants.bin" relative to the current working
  // directory, which matches the WORKING_DIRECTORY set in the e2e CMakeLists.
  mlir::hip::DiskFileSystem fs(".");
  void *state = nullptr;
  const auto initStart = std::chrono::steady_clock::now();
  int ret = init_func(&state, &fs, /*config=*/nullptr);
  if (ret != 0) {
    std::cerr << "ERROR: inference_init failed with code " << ret << "\n";
    return 1;
  }

  // Install the output allocator so the model.dll can request output buffers
  // in-graph. Cleared after the run so a stale `self` can never be used.
  set_alloc_func(state, &alloc);
  sessionSeconds += elapsed_since(initStart);
  std::cout << "Session created in "
            << static_cast<int>(sessionSeconds * 1000) << " ms\n";

  // Persistent decode mode used by small autoregressive clients. The model,
  // constants, HIP state, input allocation, and output allocation stay alive.
  // Each stdin line updates the one int64 input tensor and prints the argmax
  // of one output position. This avoids reopening multi-GB constants and
  // copying an entire logits tensor for every generated token.
  if (tokenIdsStdin) {
    if (num_inputs != 1 || input_elem_sizes[0] != sizeof(int64_t) ||
        input_shapes[0].size() != 2 || input_shapes[0][0] != 1) {
      std::cerr << "ERROR: --token-ids-stdin requires one [1,N] int64 input\n";
      set_alloc_func(state, nullptr);
      cleanup_func(state);
      return 1;
    }
    const size_t tokenCount = elementCount(input_shapes[0]);
    std::vector<int64_t> tokenIds;
    std::string line, error;
    std::cout << "READY\n" << std::flush;
    while (std::getline(std::cin, line)) {
      if (line == "QUIT")
        break;
      int64_t position = tokenIdsPosition;
      if (dynamicTokenIdsPosition) {
        const size_t separator = line.find(';');
        if (separator == std::string::npos) {
          std::cerr << "ERROR: expected POSITION;TOKEN_ID,... record\n";
          continue;
        }
        try {
          position = std::stoll(line.substr(0, separator));
        } catch (const std::exception &) {
          std::cerr << "ERROR: invalid logits position\n";
          continue;
        }
        line.erase(0, separator + 1);
      }
      if (!parseTokenIds(line, tokenCount, tokenIds, error)) {
        std::cerr << "ERROR: " << error << "\n";
        continue;
      }
      std::memcpy(input_buffers[0].data(), tokenIds.data(),
                  tokenCount * sizeof(int64_t));
      ret = compute_func(state, &inputs_span);
      if (ret != 0) {
        std::cerr << "ERROR: inference_compute failed with code " << ret
                  << "\n";
        break;
      }
      if (octx.outs.empty() || octx.outs[0].elem_size != sizeof(float) ||
          octx.outs[0].shape.size() != 3 || octx.outs[0].shape[0] != 1 ||
          position < 0 || position >= octx.outs[0].shape[1]) {
        std::cerr << "ERROR: expected f32 logits with shape [1,N,V]\n";
        break;
      }
      const OutputSlot &slot = octx.outs[0];
      const size_t vocab = static_cast<size_t>(slot.shape[2]);
      const size_t offset = static_cast<size_t>(position) * vocab;
      std::vector<float> logits(vocab);
#ifdef HIPDNN_EP_LINK_HIP_HOST
      hipError_t e = hipMemcpy(logits.data(),
                               static_cast<const uint8_t *>(slot.gpu) +
                                   offset * sizeof(float),
                               vocab * sizeof(float), hipMemcpyDeviceToHost);
      if (e != hipSuccess) {
        std::cerr << "ERROR: D2H logits copy failed: " << hipGetErrorString(e)
                  << "\n";
        break;
      }
#else
      std::memcpy(logits.data(), slot.host.data() + offset * sizeof(float),
                  vocab * sizeof(float));
#endif
      const auto it = std::max_element(logits.begin(), logits.end());
      std::cout << std::distance(logits.begin(), it) << "\n" << std::flush;
    }
    set_alloc_func(state, nullptr);
    (void)cleanup_func(state);
#ifdef HIPDNN_EP_LINK_HIP_HOST
    for (auto &slot : octx.outs)
      if (slot.gpu)
        (void)hipFree(slot.gpu);
#endif
    return 0;
  }

  // Run inference. The timer covers inference_compute only, matching
  // hip-onnx-runner's session->Run() measurement. Validation copies are later.
  if (verbose)
    std::cout << "Running inference (" << iterations << " iteration(s))...\n";

  const auto inferStart = std::chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {
    ret = compute_func(state, &inputs_span);
    if (ret != 0) {
      std::cerr << "ERROR: inference_compute failed with code " << ret
                << " at iteration " << iter << "\n";
      set_alloc_func(state, nullptr);
      cleanup_func(state);
      return 1;
    }

    if (verbose && iterations > 1)
      std::cout << "  Iteration " << (iter + 1) << "/" << iterations << "\n";
  }
  std::cout << "Inference: "
            << static_cast<int64_t>(elapsed_since(inferStart) * 1e6)
            << " us\n";

  set_alloc_func(state, nullptr);

  if (verbose) {
    for (size_t i = 0; i < octx.outs.size(); i++) {
      std::cout << "  Output " << i << " shape: [";
      for (size_t j = 0; j < octx.outs[i].shape.size(); j++) {
        std::cout << octx.outs[i].shape[j];
        if (j + 1 < octx.outs[i].shape.size())
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }

#ifdef HIPDNN_EP_LINK_HIP_HOST
  // Copy each device output back into its host validation buffer. The 2-arg
  // inference_compute already stream-synced, so a blocking hipMemcpy is safe.
  for (auto &slot : octx.outs) {
    if (!slot.gpu || slot.nbytes == 0)
      continue;
    hipError_t e = hipMemcpy(slot.host.data(), slot.gpu, slot.nbytes,
                             hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
      std::cerr << "ERROR: D2H copy of output failed: " << hipGetErrorString(e)
                << "\n";
      cleanup_func(state);
      return 1;
    }
  }
#endif

  // Validate outputs
  if (validate) {
    if (verbose)
      std::cout << "\nValidating outputs...\n";
    bool all_valid = true;
    for (size_t i = 0; i < octx.outs.size(); i++) {
      if (verbose)
        std::cout << "Output " << i << ":\n";
      const OutputSlot &slot = octx.outs[i];
      if (slot.elem_size == 0) {
        std::cerr << "VALIDATION FAILED: output " << i
                  << " was never produced by the model\n";
        all_valid = false;
        continue;
      }
      if (!validateOutput(slot.host.data(), slot.host.size(), slot.elem_size,
                          verbose)) {
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

#ifdef HIPDNN_EP_LINK_HIP_HOST
  for (auto &slot : octx.outs)
    if (slot.gpu)
      (void)hipFree(slot.gpu);
#endif

  std::cout << "SUCCESS: Model executed successfully\n";
  return 0;
}
