/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "CorelibModelCustomOp.h"

#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include <glog/logging.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace mlir_compilation {

namespace {

int find(const google::protobuf::RepeatedPtrField<std::string> &names,
         const std::string &name) {
  for (int i = 0; i < names.size(); ++i)
    if (names[i] == name)
      return i;
  return -1;
}

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error("[corelib] model: " + msg);
}

// Reads an int32 or int64 tensor's i-th element.
int64_t int_at(const Ort::ConstValue &v, size_t i) {
  if (v.GetTensorTypeAndShapeInfo().GetElementType() ==
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    return v.GetTensorData<int32_t>()[i];
  return v.GetTensorData<int64_t>()[i];
}

} // namespace

CorelibModelCustomOp::CorelibModelCustomOp(
    std::shared_ptr<const morphizen::PassContext> context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
    onnxruntime::Model *model)
    : CustomOpImp(context, meta_def, model) {
  ids_ = find(meta_def->inputs(), "input_ids");
  mask_ = find(meta_def->inputs(), "attention_mask");
  logits_ = find(meta_def->outputs(), "logits");
  if (ids_ < 0 || mask_ < 0 || logits_ < 0)
    fail("the fused graph lacks input_ids, attention_mask or logits");
  for (int o = 0; o < meta_def->outputs_size(); ++o) {
    const std::string &name = meta_def->outputs(o);
    if (name.rfind("present.", 0) != 0)
      continue;
    const int past =
        find(meta_def->inputs(), "past_key_values." + name.substr(8));
    if (past < 0)
      fail("no past_key_values input for " + name);
    presents_.emplace_back(o, past);
  }

  // The pass leaves the session it created; a model loaded another way (an
  // EPContext reload) creates its own from the fused node's parameter.
  session_ = hipdnn_ep::corelib::take_session(meta_def->id());
  if (!session_) {
    google::protobuf::Struct param;
    if (!google::protobuf::util::JsonStringToMessage(get_meta_def_param(),
                                                     &param)
             .ok())
      fail("the fused node's parameter is not JSON");
    const auto &fields = param.fields();
    auto text = [&](const char *key) {
      auto it = fields.find(key);
      return it == fields.end() ? std::string() : it->second.string_value();
    };
    auto len = fields.find("max_sequence_length");
    const int64_t max_len =
        len == fields.end() ? 0 : int64_t(len->second.number_value());
    std::string why;
    session_ = hipdnn_ep::corelib::create_model_session(
        text("arch"), text("path"), max_len, &why);
    if (!session_)
      fail(why);
  }
  logits_buf_.resize(size_t(hipdnn_ep::corelib::vocab_size(*session_)));
}

void CorelibModelCustomOp::Compute(const OrtApi * /*api*/,
                                   OrtKernelContext *context) const {
  Ort::KernelContext ctx(context);
  Ort::ConstValue ids = ctxGetInput(ctx, ids_);
  Ort::ConstValue mask = ctxGetInput(ctx, mask_);
  const std::vector<int64_t> ids_shape =
      ids.GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> mask_shape =
      mask.GetTensorTypeAndShapeInfo().GetShape();
  if (ids_shape.size() != 2 || ids_shape[0] != 1 || mask_shape.size() != 2)
    fail("only batch size 1 is supported");
  const int64_t rows = ids_shape[1];

  // attention_mask spans the positions so far plus these rows. A mask holding
  // fewer (onnxruntime_perf_test's generated zeros) starts a new sequence.
  if (rows <= 0)
    fail("input_ids is empty");
  int64_t total = 0;
  for (int64_t i = 0; i < mask_shape[1]; ++i)
    total += int_at(mask, size_t(i));
  if (total < rows) {
    if (!warned_mask_) {
      warned_mask_ = true;
      fprintf(stderr,
              "[corelib] model: attention_mask holds %lld positions "
              "for %lld rows; running them from position 0\n",
              (long long)total, (long long)rows);
    }
    total = rows;
  }
  const int64_t position = total - rows;

  tokens_.resize(size_t(rows));
  for (int64_t i = 0; i < rows; ++i)
    tokens_[size_t(i)] = static_cast<int32_t>(int_at(ids, size_t(i)));
  std::string why;
  if (!hipdnn_ep::corelib::forward(*session_, tokens_.data(), size_t(rows),
                                   position, logits_buf_.data(), &why))
    fail(why);

  // CoreLib computes the last row only, the one generation reads; the others
  // are zeros.
  const int64_t vocab = int64_t(logits_buf_.size());
  const int64_t shape[3] = {1, rows, vocab};
  Ort::UnownedValue out = ctxGetOutput(ctx, logits_, shape, 3);
  const size_t last = size_t((rows - 1) * vocab);
  if (out.GetTensorTypeAndShapeInfo().GetElementType() ==
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    auto *dst = out.GetTensorMutableData<Ort::Float16_t>();
    std::memset(dst, 0, last * sizeof(Ort::Float16_t));
    for (int64_t v = 0; v < vocab; ++v)
      dst[last + size_t(v)] = Ort::Float16_t(logits_buf_[size_t(v)]);
  } else {
    auto *dst = out.GetTensorMutableData<float>();
    std::memset(dst, 0, last * sizeof(float));
    std::memcpy(dst + last, logits_buf_.data(), size_t(vocab) * sizeof(float));
  }

  // CoreLib keeps the KV cache, so present.* is left unwritten. With a shared
  // past/present buffer it is that buffer and takes its shape; otherwise it
  // spans the positions so far.
  for (const auto &[out_index, past_index] : presents_) {
    std::vector<int64_t> shape =
        ctxGetInput(ctx, past_index).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 4 && shape[2] < total)
      shape[2] = total;
    ctxGetOutput(ctx, out_index, shape.data(), shape.size());
  }
}

} // namespace mlir_compilation
