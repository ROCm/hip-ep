/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "morphizen/onnxruntime_api.hpp"

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace mlir_compilation::customop::cpugate {

/// Process-wide CPUGate (Quark-style): one inner CPU `Session` over a shell
/// `CpuGate` custom op borrows `OrtKernelContext*`; cached `Ort::Op` handles
/// invoke standard ONNX kernels on that context. Debug-only CPU fallback.
class Manager {
public:
  static Manager &instance();

  /// Idempotent; blocks until `kernel_ctx_` is ready or returns false on error.
  bool ensure_initialized();

  void invoke(Ort::Op &op, const OrtValue **inputs, size_t input_count,
              OrtValue **outputs, size_t output_count);

  /// Lazy `Ort::Op::Create` for ONNX Gather (opset 13). Returns null on failure.
  Ort::Op *get_or_create_gather_op(int64_t axis,
                                   ONNXTensorElementDataType data_type,
                                   ONNXTensorElementDataType indices_type);

  // CpuGate custom-op hooks (called from cpu_fallback_cpugate.cpp only).
  void on_kernel_constructed(const OrtKernelInfo *info);
  void on_kernel_compute_called(OrtKernelContext *context);

private:
  Manager() = default;
  ~Manager();

  Manager(const Manager &) = delete;
  Manager &operator=(const Manager &) = delete;

  struct GatherOpKey {
    int64_t axis = 0;
    ONNXTensorElementDataType data_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType indices_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    bool operator==(const GatherOpKey &o) const {
      return axis == o.axis && data_type == o.data_type &&
             indices_type == o.indices_type;
    }
  };

  struct GatherOpKeyHash {
    size_t operator()(const GatherOpKey &k) const noexcept {
      return static_cast<size_t>(k.axis) ^
             (static_cast<size_t>(k.data_type) * 1315423911u) ^
             (static_cast<size_t>(k.indices_type) * 2654435761u);
    }
  };

  bool initialize_locked();
  Ort::Op create_gather_op_locked(const GatherOpKey &key);
  void dispatch_invoke_on_gate_thread();

  struct PendingInvoke {
    Ort::Op *op = nullptr;
    const OrtValue **inputs = nullptr;
    OrtValue **outputs = nullptr;
    size_t input_count = 0;
    size_t output_count = 0;
    bool done = false;
    std::exception_ptr error;
  };

  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MorphiZenCpuGate"};
  std::shared_ptr<Ort::Session> session_;
  const OrtKernelInfo *gate_kernel_info_ = nullptr;
  OrtKernelContext *kernel_ctx_ = nullptr;

  std::mutex guard_;
  std::condition_variable cv_;
  bool stopping_{false};
  bool init_failed_{false};
  bool has_pending_invoke_{false};
  PendingInvoke pending_invoke_;
  std::thread gate_thread_;

  std::unordered_map<GatherOpKey, Ort::Op, GatherOpKeyHash> gather_ops_;
};

} // namespace mlir_compilation::customop::cpugate
