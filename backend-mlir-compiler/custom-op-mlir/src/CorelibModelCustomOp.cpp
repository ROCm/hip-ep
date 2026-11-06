/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "CorelibModelCustomOp.h"

#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include <glog/logging.h>

#ifdef HIPDNN_EP_LINK_HIP_HOST
#include <hip/hip_runtime_api.h>
#endif

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

// The message is printed first: an exception unwinding through ONNX Runtime's
// C API can end the process before anyone reads it.
[[noreturn]] void fail(const std::string &msg) {
  fprintf(stderr, "[corelib] model: %s\n", msg.c_str());
  fflush(stderr);
  throw std::runtime_error("[corelib] model: " + msg);
}

// The EP's default device memory is device-only VRAM (hipMalloc), which the
// host cannot read: such tensors move through hipMemcpy.
bool on_gpu(const Ort::ConstMemoryInfo &info) {
  return info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU;
}

void copy(void *dst, const void *src, size_t bytes, bool gpu_src,
          bool gpu_dst) {
  if (!gpu_src && !gpu_dst) {
    std::memcpy(dst, src, bytes);
    return;
  }
#ifdef HIPDNN_EP_LINK_HIP_HOST
  // ORT may still be filling an input on a stream of its own.
  if (gpu_src && hipDeviceSynchronize() != hipSuccess)
    fail("hipDeviceSynchronize failed");
  if (hipMemcpy(dst, src, bytes, hipMemcpyDefault) != hipSuccess)
    fail("hipMemcpy of " + std::to_string(bytes) + " bytes failed");
#else
  fail("a tensor in GPU memory, in a build without HIP");
#endif
}

// An int32 or int64 tensor's elements, as int64, wherever it lives.
std::vector<int64_t> read_ints(const Ort::ConstValue &v, size_t n) {
  const bool i32 = v.GetTensorTypeAndShapeInfo().GetElementType() ==
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  const size_t elem = i32 ? sizeof(int32_t) : sizeof(int64_t);
  std::vector<uint8_t> raw(n * elem);
  copy(raw.data(), v.GetTensorRawData(), raw.size(),
       on_gpu(v.GetTensorMemoryInfo()), false);
  std::vector<int64_t> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = i32 ? reinterpret_cast<const int32_t *>(raw.data())[i]
                 : reinterpret_cast<const int64_t *>(raw.data())[i];
  return out;
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
  for (int64_t m : read_ints(mask, size_t(mask_shape[1])))
    total += m;
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

  const std::vector<int64_t> ids_host = read_ints(ids, size_t(rows));
  tokens_.resize(size_t(rows));
  for (int64_t i = 0; i < rows; ++i)
    tokens_[size_t(i)] = static_cast<int32_t>(ids_host[size_t(i)]);
  std::string why;
  if (!hipdnn_ep::corelib::forward(*session_, tokens_.data(), size_t(rows),
                                   position, logits_buf_.data(), &why))
    fail(why);

  // CoreLib computes the last row only, the one generation reads; the others
  // are zeros.
  const int64_t vocab = int64_t(logits_buf_.size());
  const int64_t shape[3] = {1, rows, vocab};
  Ort::UnownedValue out = ctxGetOutput(ctx, logits_, shape, 3);
  const bool fp16 = out.GetTensorTypeAndShapeInfo().GetElementType() ==
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  const size_t elem = fp16 ? sizeof(Ort::Float16_t) : sizeof(float);
  const size_t row_bytes = size_t(vocab) * elem;
  const size_t lead_bytes = size_t(rows - 1) * row_bytes;
  const void *last_row = logits_buf_.data();
  if (fp16) {
    logits_fp16_.resize(size_t(vocab));
    for (int64_t v = 0; v < vocab; ++v)
      logits_fp16_[size_t(v)] = Ort::Float16_t(logits_buf_[size_t(v)]);
    last_row = logits_fp16_.data();
  }
  auto *dst = static_cast<uint8_t *>(out.GetTensorMutableRawData());
  const bool gpu_out = on_gpu(out.GetTensorMemoryInfo());
  if (lead_bytes != 0) {
    if (gpu_out) {
#ifdef HIPDNN_EP_LINK_HIP_HOST
      if (hipMemset(dst, 0, lead_bytes) != hipSuccess)
        fail("hipMemset of the logits failed");
#endif
    } else {
      std::memset(dst, 0, lead_bytes);
    }
  }
  copy(dst + lead_bytes, last_row, row_bytes, false, gpu_out);

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
