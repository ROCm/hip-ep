/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Quark-style CPUGate for debug CPU fallback: one inner CPU Session over a shell
// `CpuGate` custom op holds `OrtKernelContext*`; standard ops run via
// `Ort::Op::Create` + `Ort::Op::Invoke(kernel_ctx_, ...)`.
//
// Threading contract (must not violate):
//   - Inner Session load on gate thread while `guard_` is held.
//   - Ort::Op::Create: queued on `pending_inits_`, runs inside
//     on_kernel_constructed during Session construction (Quark CPUGate).
//   - Ort::Op::Invoke: gate thread, inside CpuGate::Compute only.
//   - EP callback thread must NOT create inner Session.
//===----------------------------------------------------------------------===//

#include "cpu_fallback_cpugate.h"

#include <glog/logging.h>

#include <chrono>
#include <cstring>
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
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }

  // Fixed IO matching cpugate_shell.onnx (UINT8[1] -> UINT8[1]).

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

void fill_type_constraints(
    const char *op_name, const ONNXTensorElementDataType *input_types,
    size_t input_type_count, std::vector<const char *> &names,
    std::vector<ONNXTensorElementDataType> &values) {
  names.clear();
  values.clear();
  if (!op_name || input_type_count == 0)
    return;
  if (std::strcmp(op_name, "Gather") == 0 && input_type_count >= 2) {
    names = {"T", "Tind"};
    values = {input_types[0], input_types[1]};
    return;
  }
  if (std::strcmp(op_name, "Cast") == 0) {
    names = {"T"};
    values = {input_types[0]};
    return;
  }
  if (std::strcmp(op_name, "Where") == 0 && input_type_count >= 2) {
    names = {"T"};
    values = {input_types[1]};
    return;
  }
  names.push_back("T");
  values.push_back(input_types[0]);
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

bool Manager::ensure_session_loaded() {
  std::unique_lock<std::mutex> lock(guard_);
  if (init_failed_) {
    return false;
  }
  if (session_loaded_ && gate_kernel_info_ != nullptr) {
    return true;
  }
  if (!gate_thread_.joinable()) {
    gate_thread_ = std::thread{[this] { gate_thread_main(); }};
  }
  if (!cv_.wait_for(lock, std::chrono::seconds(60),
                    [this] {
                      return (session_loaded_ && gate_kernel_info_ != nullptr) ||
                             stopping_ || init_failed_;
                    })) {
    LOG(ERROR) << "cpu_fallback CPUGate: timed out waiting for shell session";
    init_failed_ = true;
    return false;
  }
  return session_loaded_ && gate_kernel_info_ != nullptr && !init_failed_;
}

bool Manager::ensure_kernel_context() {
  if (!ensure_session_loaded()) {
    return false;
  }
  std::unique_lock<std::mutex> lock(guard_);
  if (kernel_ctx_ != nullptr) {
    return true;
  }
  if (init_failed_) {
    return false;
  }
  run_requested_ = true;
  cv_.notify_all();
  if (!cv_.wait_for(lock, std::chrono::seconds(60),
                    [this] { return kernel_ctx_ != nullptr || stopping_; })) {
    LOG(ERROR) << "cpu_fallback CPUGate: timed out waiting for kernel context";
    init_failed_ = true;
    return false;
  }
  return kernel_ctx_ != nullptr;
}

bool Manager::create_session_under_lock(std::unique_lock<std::mutex> &lock) {
  if (session_) {
    return true;
  }
  (void)lock;
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

    LOG(INFO) << "cpu_fallback CPUGate: loading shell session on gate thread";
    session_ = std::make_shared<Ort::Session>(
        env_, reinterpret_cast<const char *>(kCpugateShellOnnx),
        kCpugateShellOnnx_len, so);
    return true;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate: session create failed: " << e.what();
    return false;
  } catch (const std::exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate: " << e.what();
    return false;
  }
}

void Manager::run_gate_session_outside_lock() {
  if (!session_) {
    throw std::runtime_error("CPUGate: shell session missing");
  }
  uint8_t gate_input_byte = 0;
  constexpr int64_t gate_input_shape = 1;
  constexpr const char *input_names[] = {"i"};
  constexpr const char *output_names[] = {"o"};
  Ort::Value tensors[1] = {Ort::Value::CreateTensor(
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
      &gate_input_byte, 1, &gate_input_shape, 1,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)};
  LOG(INFO) << "cpu_fallback CPUGate: starting shell Session::Run on gate thread";
  session_->Run(Ort::RunOptions{nullptr}, input_names, tensors, 1, output_names,
                tensors, 1);
}

void Manager::gate_thread_main() {
  try {
    std::unique_lock<std::mutex> lock(guard_);
    if (!create_session_under_lock(lock)) {
      init_failed_ = true;
      cv_.notify_all();
      return;
    }
    session_loaded_ = true;
    cv_.notify_all();

    while (!stopping_) {
      if (run_requested_ && !run_started_) {
        run_started_ = true;
        lock.unlock();
        try {
          run_gate_session_outside_lock();
        } catch (const std::exception &e) {
          LOG(ERROR) << "cpu_fallback CPUGate: gate thread Run failed: "
                     << e.what();
          lock.lock();
          stopping_ = true;
          cv_.notify_all();
          return;
        }
        lock.lock();
        run_started_ = false;
        continue;
      }

      cv_.wait(lock, [this] {
        return stopping_ || (run_requested_ && !run_started_);
      });
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "cpu_fallback CPUGate gate thread: " << e.what();
    std::lock_guard<std::mutex> lock(guard_);
    init_failed_ = true;
    stopping_ = true;
    cv_.notify_all();
  } catch (...) {
    LOG(ERROR) << "cpu_fallback CPUGate gate thread: unknown failure";
    std::lock_guard<std::mutex> lock(guard_);
    init_failed_ = true;
    stopping_ = true;
    cv_.notify_all();
  }
}

void Manager::on_kernel_constructed(const OrtKernelInfo *info) {
  // Gate kernel ctor runs during shell Session construction while the gate
  // thread holds `guard_` (Quark CPUGate contract).
  if (guard_.try_lock()) {
    guard_.unlock();
    throw std::runtime_error(
        "CpuGate: manager mutex must be held during inner Session load");
  }
  gate_kernel_info_ = info;
  for (const auto &init : pending_inits_) {
    init(info);
  }
  pending_inits_.clear();
}

void Manager::queue_op_create_init(const OpCreateSpec &spec) {
  pending_inits_.push_back([this, spec](const OrtKernelInfo *info) {
    if (generic_ops_.find(spec.key) != generic_ops_.end()) {
      return;
    }
    LOG(INFO) << "cpu_fallback CPUGate: Ort::Op::Create(" << spec.op_name
              << ") in OnKernelConstructed (Quark CPUGate)";
    generic_ops_.emplace(spec.key, create_onnx_op_from_kernel_info(info, spec));
  });
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

Ort::Op Manager::create_onnx_op_from_kernel_info(const OrtKernelInfo *info,
                                                 const OpCreateSpec &spec) {
  if (!info) {
    throw std::runtime_error("CPUGate: gate OrtKernelInfo not available");
  }
  std::vector<const char *> type_names;
  std::vector<ONNXTensorElementDataType> type_values;
  fill_type_constraints(spec.op_name.c_str(), spec.input_types.data(),
                        spec.input_types.size(), type_names, type_values);
  if (type_names.empty()) {
    throw std::runtime_error("CPUGate: could not infer type constraints");
  }
  return Ort::Op::Create(
      info, spec.op_name.c_str(), spec.domain.c_str(),
      static_cast<int>(spec.opset), type_names.data(), type_values.data(),
      type_names.size(), spec.attrs, spec.attr_count, spec.onnx_input_count,
      spec.onnx_output_count);
}

void Manager::dispatch_invoke_on_gate_thread() {
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
  if (!ensure_kernel_context()) {
    throw std::runtime_error("CPUGate: kernel context not ready");
  }
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

std::string Manager::make_generic_op_cache_key(
    const char *op_name, const char *domain, int64_t opset,
    const ONNXTensorElementDataType *input_types, size_t input_type_count,
    size_t onnx_input_count, size_t onnx_output_count,
    const Ort::OpAttr *attrs, size_t attr_count) const {
  std::string key = std::string(op_name ? op_name : "") + "|" +
                    std::string(domain ? domain : "") + "|" +
                    std::to_string(opset) + "|" +
                    std::to_string(onnx_input_count) + "|" +
                    std::to_string(onnx_output_count) + "|";
  for (size_t i = 0; i < input_type_count; ++i) {
    key += std::to_string(static_cast<int>(input_types[i]));
    key += ',';
  }
  key += "|";
  for (size_t i = 0; i < attr_count; ++i) {
    key += attrs[i].GetName();
    key += ';';
  }
  return key;
}

Ort::Op *Manager::get_or_create_onnx_op(
    const char *op_name, const char *domain, int64_t opset,
    const ONNXTensorElementDataType *input_types, size_t input_type_count,
    size_t onnx_input_count, size_t onnx_output_count,
    const Ort::OpAttr *attrs, size_t attr_count) {
  const std::string key = make_generic_op_cache_key(
      op_name, domain, opset, input_types, input_type_count, onnx_input_count,
      onnx_output_count, attrs, attr_count);

  std::unique_lock<std::mutex> lock(guard_);
  if (auto it = generic_ops_.find(key); it != generic_ops_.end()) {
    return &it->second;
  }
  if (init_failed_) {
    return nullptr;
  }

  OpCreateSpec spec{};
  spec.key = key;
  spec.op_name = op_name ? op_name : "";
  spec.domain = domain ? domain : "";
  spec.opset = opset;
  spec.input_types.assign(input_types, input_types + input_type_count);
  spec.onnx_input_count = onnx_input_count;
  spec.onnx_output_count = onnx_output_count;
  spec.attrs = attrs;
  spec.attr_count = attr_count;

  if (!session_loaded_) {
    queue_op_create_init(spec);
    if (!gate_thread_.joinable()) {
      gate_thread_ = std::thread{[this] { gate_thread_main(); }};
    }
    if (!cv_.wait_for(lock, std::chrono::seconds(60),
                      [this] {
                        return session_loaded_ || init_failed_ || stopping_;
                      })) {
      LOG(ERROR) << "cpu_fallback CPUGate: timed out waiting for shell session";
      init_failed_ = true;
      return nullptr;
    }
  } else if (gate_kernel_info_ != nullptr) {
    lock.unlock();
    Ort::Op new_op;
    try {
      new_op = create_onnx_op_from_kernel_info(gate_kernel_info_, spec);
    } catch (const Ort::Exception &e) {
      LOG(ERROR) << "cpu_fallback CPUGate: Ort::Op::Create("
                 << (op_name ? op_name : "?") << ") failed: " << e.what();
      return nullptr;
    } catch (const std::exception &e) {
      LOG(ERROR) << "cpu_fallback CPUGate: " << e.what();
      return nullptr;
    }
    lock.lock();
    auto inserted = generic_ops_.emplace(key, std::move(new_op));
    return &inserted.first->second;
  }

  if (init_failed_ || stopping_) {
    return nullptr;
  }
  auto it = generic_ops_.find(key);
  if (it == generic_ops_.end()) {
    LOG(ERROR) << "cpu_fallback CPUGate: Ort::Op::Create("
               << (op_name ? op_name : "?") << ") missing after shell load";
    return nullptr;
  }
  return &it->second;
}

} // namespace mlir_compilation::customop::cpugate
