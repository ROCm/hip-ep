/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "morphizen/onnxruntime_api.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mlir_compilation::customop::cpugate {

/// Process-wide CPUGate (Quark-style): one inner CPU `Session` over a shell
/// `CpuGate` custom op borrows `OrtKernelContext*`; cached `Ort::Op` handles
/// invoke standard ONNX kernels on that context. Debug-only CPU fallback.
class Manager {
public:
  static Manager &instance();

  /// Blocks until inner shell Session is loaded (`gate_kernel_info_` ready).
  bool ensure_session_loaded();

  /// Blocks until `kernel_ctx_` is ready (starts inner `Run` on first call).
  bool ensure_kernel_context();

  void invoke(Ort::Op &op, const OrtValue **inputs, size_t input_count,
              OrtValue **outputs, size_t output_count);

  /// Generic ONNX op cache keyed by name, types, and attributes.
  Ort::Op *get_or_create_onnx_op(
      const char *op_name, const char *domain, int64_t opset,
      const ONNXTensorElementDataType *input_types, size_t input_type_count,
      size_t onnx_input_count, size_t onnx_output_count,
      const Ort::OpAttr *attrs, size_t attr_count);

  // CpuGate custom-op hooks (called from cpu_fallback_cpugate.cpp only).
  void on_kernel_constructed(const OrtKernelInfo *info);
  void on_kernel_compute_called(OrtKernelContext *context);

private:
  Manager() = default;
  ~Manager();

  Manager(const Manager &) = delete;
  Manager &operator=(const Manager &) = delete;

  struct PendingInvoke {
    Ort::Op *op = nullptr;
    const OrtValue **inputs = nullptr;
    OrtValue **outputs = nullptr;
    size_t input_count = 0;
    size_t output_count = 0;
    bool done = false;
    std::exception_ptr error;
  };

  struct OpCreateSpec {
    std::string key;
    std::string op_name;
    std::string domain;
    int64_t opset = 13;
    std::vector<ONNXTensorElementDataType> input_types;
    size_t onnx_input_count = 0;
    size_t onnx_output_count = 0;
    const Ort::OpAttr *attrs = nullptr;
    size_t attr_count = 0;
  };

  void gate_thread_main();
  bool create_session_under_lock(std::unique_lock<std::mutex> &lock);
  void run_gate_session_outside_lock();
  void queue_op_create_init(const OpCreateSpec &spec);
  std::string make_generic_op_cache_key(
      const char *op_name, const char *domain, int64_t opset,
      const ONNXTensorElementDataType *input_types, size_t input_type_count,
      size_t onnx_input_count, size_t onnx_output_count,
      const Ort::OpAttr *attrs, size_t attr_count) const;
  static Ort::Op create_onnx_op_from_kernel_info(const OrtKernelInfo *info,
                                                 const OpCreateSpec &spec);
  void dispatch_invoke_on_gate_thread();

  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MorphiZenCpuGate"};
  std::shared_ptr<Ort::Session> session_;
  const OrtKernelInfo *gate_kernel_info_ = nullptr;
  OrtKernelContext *kernel_ctx_ = nullptr;

  std::mutex guard_;
  std::condition_variable cv_;
  bool stopping_{false};
  bool init_failed_{false};
  bool session_loaded_{false};
  bool run_requested_{false};
  bool run_started_{false};
  bool has_pending_invoke_{false};
  PendingInvoke pending_invoke_;
  std::thread gate_thread_;

  std::vector<std::function<void(const OrtKernelInfo *)>> pending_inits_;
  std::unordered_map<std::string, Ort::Op> generic_ops_;
};

} // namespace mlir_compilation::customop::cpugate
