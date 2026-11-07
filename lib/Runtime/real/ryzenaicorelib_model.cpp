/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ryzenaicorelib_model.cpp - Whole models through RyzenAI CoreLib ----===//
//
// The native side of corelib_model.h: creates and runs CoreLib model
// sessions, through the shared runtime loader (corelib_loader.h). See
// docs/design/ryzenai-corelib-model.md.
//
//===----------------------------------------------------------------------===//

#include "corelib_model.h"

#include "corelib_loader.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace hipdnn_ep::corelib {

class ModelSession {
public:
  ~ModelSession() {
    if (report_)
      fprintf(stderr,
              "[corelib] model report: %s, %llu forwards on the NPU (%llu "
              "prefills, %llu decodes, %llu rows)\n",
              arch_.c_str(), (unsigned long long)(prefills_ + decodes_),
              (unsigned long long)prefills_, (unsigned long long)decodes_,
              (unsigned long long)rows_);
    if (session_)
      api_->ryzenai_corelib_object_release(session_);
    if (model_)
      api_->ryzenai_corelib_object_release(model_);
    remove_user(*api_);
  }

  const CorelibApi *api_ = nullptr;
  ryzenai_corelib_model_ptr model_ = nullptr;
  ryzenai_corelib_model_session_ptr session_ = nullptr;
  std::string arch_;
  int64_t vocab_ = 0;
  bool report_ = false;
  uint64_t prefills_ = 0, decodes_ = 0, rows_ = 0;
};

std::shared_ptr<ModelSession> create_model_session(const std::string &arch,
                                                   const std::string &path,
                                                   int64_t max_sequence_length,
                                                   std::string *why) {
  const CorelibApi *api = load_api();
  if (!api) {
    *why = "corelib is not loadable (see the [corelib] line above)";
    return nullptr;
  }
  if (!api->ryzenai_corelib_npu_usable()) {
    *why = "no usable AIE4 NPU (corelib's models run on the NPU only on AIE4, "
           "with XRT 2.26.0 or later)";
    return nullptr;
  }
  ryzenai_corelib_model_ptr model = nullptr;
  if (api->ryzenai_corelib_create_model_from_path(arch.c_str(), path.c_str(),
                                                  &model) !=
      ryzenai_corelib_status_success) {
    *why = "corelib's " + arch + " refused " + path + ": " +
           api->ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  // Counted from here: the destructor releases the model and may clean up.
  add_user();
  auto s = std::make_shared<ModelSession>();
  s->api_ = api;
  s->model_ = model;
  s->arch_ = arch;
  s->report_ = read_env("RYZENAICORELIB_REPORT") == "1";

  ryzenai_corelib_model_session_options options{};
  options.size = sizeof(options);
  options.max_sequence_length = max_sequence_length;
  // hip-ep owns the GPU: CoreLib's GPU tier would load its own HIP and
  // register its buffers alongside the EP's. The CPU stays, for the token
  // pick and gathers CoreLib runs there.
  options.enabled_backends =
      ryzenai_corelib_backend_npu | ryzenai_corelib_backend_cpu;
  if (api->ryzenai_corelib_model_create_session(
          model, &options, &s->session_) != ryzenai_corelib_status_success) {
    *why = "creating a " + arch +
           " session failed: " + api->ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  ryzenai_corelib_model_metadata metadata{};
  metadata.size = sizeof(metadata);
  if (api->ryzenai_corelib_model_get_metadata(s->session_, &metadata) !=
          ryzenai_corelib_status_success ||
      metadata.vocab_size <= 0) {
    *why = "reading the " + arch + " model's metadata failed: " +
           api->ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  s->vocab_ = metadata.vocab_size;
  fprintf(stderr,
          "[corelib] model: %s from %s on the NPU (%lld layers, hidden %lld, "
          "vocab %lld, context %lld)\n",
          arch.c_str(), path.c_str(), (long long)metadata.layer_count,
          (long long)metadata.hidden_size, (long long)metadata.vocab_size,
          (long long)metadata.context_length);
  return s;
}

int64_t vocab_size(const ModelSession &session) { return session.vocab_; }

bool forward(ModelSession &s, const int32_t *tokens, size_t rows,
             int64_t position, float *logits, std::string *why) {
  ryzenai_corelib_model_value inputs[2] = {};
  inputs[0].size = sizeof(ryzenai_corelib_model_value);
  inputs[0].name = "input_ids";
  inputs[0].data_type = ryzenai_corelib_data_type_int32;
  inputs[0].rank = 1;
  inputs[0].dims[0] = static_cast<int64_t>(rows);
  inputs[0].data = const_cast<int32_t *>(tokens);
  inputs[1].size = sizeof(ryzenai_corelib_model_value);
  inputs[1].name = "position";
  inputs[1].data_type = ryzenai_corelib_data_type_int64;
  inputs[1].rank = 0;
  inputs[1].data = &position;

  ryzenai_corelib_model_value output{};
  output.size = sizeof(output);
  output.name = "logits";
  output.data_type = ryzenai_corelib_data_type_fp32;
  output.data = logits;
  output.capacity = size_t(s.vocab_) * sizeof(float);

  if (s.api_->ryzenai_corelib_model_run(s.session_, "forward", inputs, 2,
                                        &output, 1,
                                        0) != ryzenai_corelib_status_success) {
    *why = s.api_->ryzenai_corelib_get_last_error_message();
    return false;
  }
  (rows == 1 ? s.decodes_ : s.prefills_)++;
  s.rows_ += rows;
  return true;
}

} // namespace hipdnn_ep::corelib
