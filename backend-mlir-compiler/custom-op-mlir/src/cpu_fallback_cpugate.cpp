/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Quark-style CPUGate for debug CPU fallback: one inner CPU Session over a shell
// `CpuGate` custom op holds `OrtKernelContext*`; standard ops run via
// `Ort::Op::Create` + `Ort::Op::Invoke(kernel_ctx_, ...)`.
//
// Before:
//   onnx.Constant
//     %gate_blob = ...
// After (conceptual — shell graph is embedded protobuf, not MLIR):
//   com.amd.morphizen.cpu.CpuGate(%i) -> %o   // blocks; borrows kernel_ctx_
//===----------------------------------------------------------------------===//

#include "cpu_fallback_cpugate.h"

#include <glog/logging.h>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// Embedded shell ONNX (com.amd.morphizen.cpu.CpuGate); see onnx/README.md.
#include "cpugate_shell_onnx_data.inc"

namespace mlir_compilation::customop::cpugate {
namespace {

struct CpuGateKernel {
  CpuGateKernel(const OrtApi &, const OrtKernelInfo *info, Manager *manager)
      : manager_(manager) {
    manager_->on_kernel_constructed(info);
  }

  void Compute(OrtKernelContext *context) {
    manager_->on_kernel_compute_called(context);
  }

  Manager *manager_;
};

struct CpuGateCustomOp
    : Ort::CustomOpBase<CpuGateCustomOp, CpuGateKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    const auto str = last_session_options.GetConfigEntry("manager");
    if (str.empty()) {
      throw std::runtime_error("CpuGate: missing session config entry 'manager'");
    }
    auto *mgr = reinterpret_cast<Manager *>(std::strtoull(str.c_str(), nullptr, 10));
    if (!mgr) {
      throw std::runtime_error("CpuGate: invalid manager config entry");
    }
    return new CpuGateKernel(api, info, mgr);
  }

  const char *GetName() const noexcept { return "CpuGate"; }

  const char *GetExecutionProviderType() const noexcept {
    return "CPUExecutionProvider";
  }

  size_t GetInputTypeCount() const noexcept { return 1; }
  size_t GetOutputTypeCount() const noexcept { return 1; }

  ONNXTensorElementDataType GetInputType(size_t) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
      size_t) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(
      size_t) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  bool GetVariadicInputHomogeneity() const noexcept { return false; }
  bool GetVariadicOutputHomogeneity() const noexcept { return false; }

  mutable Ort::ConstSessionOptions last_session_options{nullptr};
};

OrtStatus *register_cpu_gate_custom_ops(OrtSessionOptions *options) {
  static Ort::CustomOpDomain domain{"com.amd.morphizen.cpu"};
  static CpuGateCustomOp cpu_gate_op;
  cpu_gate_op.last_session_options = Ort::ConstSessionOptions{options};
  try {
    Ort::UnownedSessionOptions session_options(options);
    domain.Add(&cpu_gate_op);
    session_options.Add(domain);
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }
  return nullptr;
}

} // namespace

Manager &Manager::instance() {
  static Manager mgr;
  return mgr;
}

Manager::~Manager() {
  {
    std::lock_guard<std::mutex> lock(guard_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (gate_thread_.joinable()) {
    gate_thread_.join();
  }
}

bool Manager::ensure_initialized() {
  std::unique_lock<std::mutex> lock(guard_);
  if (init_failed_) {
    return false;
  }
  if (kernel_ctx_ != nullptr) {
    return true;
  }
  if (!initialize_locked()) {
    init_failed_ = true;
    return false;
  }
  if (!cv_.wait_for(lock, std::chrono::seconds(30),
                    [this] { return kernel_ctx_ != nullptr || stopping_; })) {
    LOG(ERROR) << "cpu_fallback CPUGate: timed out waiting for kernel context";
    init_failed_ = true;
    return false;
  }
  return kernel_ctx_ != nullptr;
}

bool Manager::initialize_locked() {
  if (session_ || init_failed_) {
    return session_ != nullptr;
  }

  try {
    Ort::SessionOptions so;
    so.AddConfigEntry(
        "manager",
        std::to_string(reinterpret_cast<std::uintptr_t>(this)).c_str());
    so.DisableCpuMemArena();
    so.DisableMemPattern();
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    so.AddConfigEntry("session.inter_op.allow_spinning", "0");
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");

    if (OrtStatus *st = register_cpu_gate_custom_ops(so)) {
      const char *msg = Ort::GetApi().GetErrorMessage(st);
      LOG(ERROR) << "cpu_fallback CPUGate: RegisterCustomOps failed: "
                 << (msg ? msg : "(null)");
      Ort::GetApi().ReleaseStatus(st);
      return false;
    }

    session_ = std::make_shared<Ort::Session>(
        env_, reinterpret_cast<const char *>(kCpugateShellOnnx),
        kCpugateShellOnnx_len, so);

    gate_thread_ = std::thread{[this, session = session_] {
      try {
        // Shell graph input is UINT8[1]; must supply one byte (not nullptr).
        uint8_t gate_input_byte = 0;
        constexpr int64_t gate_input_shape = 1;
        constexpr const char *input_names[] = {"i"};
        constexpr const char *output_names[] = {"o"};
        Ort::Value tensors[1] = {Ort::Value::CreateTensor(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            &gate_input_byte, 1, &gate_input_shape, 1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)};
        session->Run(Ort::RunOptions{nullptr}, input_names, tensors, 1,
                     output_names, tensors, 1);
      } catch (const std::exception &e) {
        LOG(ERROR) << "cpu_fallback CPUGate: gate thread Run failed: "
                   << e.what();
        std::lock_guard<std::mutex> lock(guard_);
        stopping_ = true;
        cv_.notify_all();
      }
    }};
    return true;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate: session create failed: " << e.what();
    return false;
  } catch (const std::exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate: " << e.what();
    return false;
  }
}

void Manager::on_kernel_constructed(const OrtKernelInfo *info) {
  // Gate kernel ctor runs during `Session` construction while
  // `ensure_initialized` holds `guard_` (Quark CPUGate contract).
  if (guard_.try_lock()) {
    guard_.unlock();
    throw std::runtime_error(
        "CpuGate: manager mutex must be held during inner Session load");
  }
  gate_kernel_info_ = info;
}

void Manager::on_kernel_compute_called(OrtKernelContext *context) {
  std::unique_lock<std::mutex> lock(guard_);
  kernel_ctx_ = context;
  cv_.notify_all();
  while (!stopping_) {
    cv_.wait(lock, [this] { return stopping_ || has_pending_invoke_; });
    if (stopping_) {
      break;
    }
    if (has_pending_invoke_ && pending_invoke_.op) {
      dispatch_invoke_on_gate_thread();
      pending_invoke_.done = true;
      has_pending_invoke_ = false;
      cv_.notify_all();
    }
  }
  kernel_ctx_ = nullptr;
}

void Manager::dispatch_invoke_on_gate_thread() {
  // Runs on the gate thread (inside CpuGate::Compute) — the only thread that
  // may call Ort::Op::Invoke on this kernel_ctx_. The inference thread must
  // not call Invoke while nested inside the outer MorphiZen ORT Session::Run.
  try {
    pending_invoke_.op->Invoke(
        kernel_ctx_, pending_invoke_.inputs, pending_invoke_.input_count,
        pending_invoke_.outputs, pending_invoke_.output_count);
  } catch (...) {
    pending_invoke_.error = std::current_exception();
  }
}

void Manager::invoke(Ort::Op &op, const OrtValue **inputs, size_t input_count,
                     OrtValue **outputs, size_t output_count) {
  std::unique_lock<std::mutex> lock(guard_);
  if (!kernel_ctx_) {
    throw std::runtime_error("CPUGate: kernel context not ready");
  }
  if (has_pending_invoke_) {
    throw std::runtime_error("CPUGate: concurrent invoke not supported");
  }
  pending_invoke_ = PendingInvoke{};
  pending_invoke_.op = &op;
  pending_invoke_.inputs = inputs;
  pending_invoke_.outputs = outputs;
  pending_invoke_.input_count = input_count;
  pending_invoke_.output_count = output_count;
  pending_invoke_.done = false;
  pending_invoke_.error = nullptr;
  has_pending_invoke_ = true;
  cv_.notify_all();
  if (!cv_.wait_for(lock, std::chrono::seconds(300),
                    [this] { return pending_invoke_.done || stopping_; })) {
    has_pending_invoke_ = false;
    throw std::runtime_error("CPUGate: invoke timed out");
  }
  if (stopping_) {
    throw std::runtime_error("CPUGate: shutting down");
  }
  if (pending_invoke_.error) {
    std::rethrow_exception(pending_invoke_.error);
  }
}

Ort::Op Manager::create_gather_op_locked(const GatherOpKey &key) {
  if (!gate_kernel_info_) {
    throw std::runtime_error("CPUGate: gate OrtKernelInfo not available");
  }
  if (key.indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("CPUGate Gather: indices must be int64 for ORT");
  }

  const char *type_names[] = {"T", "Tind"};
  const ONNXTensorElementDataType type_values[] = {key.data_type,
                                                   key.indices_type};
  int64_t axis = key.axis;
  Ort::OpAttr axis_attr("axis", &axis, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  return Ort::Op::Create(gate_kernel_info_, "Gather", "", 13, type_names,
                         type_values, 2, &axis_attr, 1, 2, 1);
}

Ort::Op *Manager::get_or_create_gather_op(int64_t axis,
                                        ONNXTensorElementDataType data_type,
                                        ONNXTensorElementDataType indices_type) {
  if (!ensure_initialized()) {
    return nullptr;
  }

  const GatherOpKey key{axis, data_type, indices_type};
  std::lock_guard<std::mutex> lock(guard_);
  auto it = gather_ops_.find(key);
  if (it != gather_ops_.end()) {
    return &it->second;
  }

  try {
    auto inserted =
        gather_ops_.emplace(key, create_gather_op_locked(key));
    return &inserted.first->second;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate: Ort::Op::Create(Gather) failed: "
               << e.what();
    return nullptr;
  } catch (const std::exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate: " << e.what();
    return nullptr;
  }
}

} // namespace mlir_compilation::customop::cpugate
