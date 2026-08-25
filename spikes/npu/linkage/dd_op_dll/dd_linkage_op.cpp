/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// T0.2 spike -- built against a prebuilt DynamicDispatch with its own
// toolchain settings (/MD, DD's own protobuf), independent of hip-ep's
// (/MT). See ../README.md and docs/design/hybrid-npu-gpu-design.md's
// "Packaging and the shim boundary".

#define DD_LINKAGE_OP_BUILD_DLL
#include "dd_linkage_op.h"

#include <any>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include <ops/llm_ops/elwmul/elwmul.hpp>
#include <ops/op_interface.hpp>

namespace {

void SetError(char *err_msg, size_t cap, const std::string &what) {
  if (err_msg == nullptr || cap == 0) {
    return;
  }
  size_t n = what.size() < cap - 1 ? what.size() : cap - 1;
  std::memcpy(err_msg, what.data(), n);
  err_msg[n] = '\0';
}

} // namespace

int dd_linkage_run_trivial_op(void *output_buffer, size_t output_buffer_bytes,
                              char *err_msg, size_t err_msg_cap) {
  if (err_msg != nullptr && err_msg_cap > 0) {
    err_msg[0] = '\0';
  }
  if (output_buffer == nullptr ||
      output_buffer_bytes < DD_LINKAGE_OUTPUT_BYTES) {
    SetError(err_msg, err_msg_cap, "output_buffer too small");
    return DD_LINKAGE_ERR_BAD_ARGS;
  }

  try {
    const std::vector<size_t> shape = {DD_LINKAGE_ROWS, DD_LINKAGE_COLS};
    const size_t elems = static_cast<size_t>(DD_LINKAGE_ROWS) *
                         static_cast<size_t>(DD_LINKAGE_COLS);

    std::vector<uint16_t> a(elems, 1);
    std::vector<uint16_t> b(elems, 2);

    // ryzenai::elw_mul<uint16_t,uint16_t,uint16_t> was picked (over the
    // originally-tried ryzenai::elw_add<uint16_t,...>) because a `dumpbin
    // /exports` check against this machine's prebuilt dyn_dispatch_core.dll
    // found elw_add's explicit template instantiation is NOT exported by
    // this particular build (present in DynamicDispatch's source tree, but
    // this binary predates or excludes it), while elw_mul's is. See
    // ../README.md. "bfloat16" is DD's dtype tag for this instantiation even
    // though the C++ storage type is uint16_t -- the constructor DD_THROWs on
    // any other string for this template.
    //
    // load_xrt=true: unlike elw_add (which defers device/xclbin opening to
    // set_params()), elw_mul opens the XRT context inside its own
    // constructor when load_xrt is true -- so construction alone is expected
    // to reach (and, on a machine with no NPU, fail at) the hardware
    // boundary this spike exists to find.
    std::map<std::string, std::any> attr;
    ryzenai::elw_mul<uint16_t, uint16_t, uint16_t> op("bfloat16",
                                                      /*load_xrt=*/true, attr);

    std::vector<Tensor> inputs = {
        {a.data(), shape, "bfloat16"},
        {b.data(), shape, "bfloat16"},
    };
    std::vector<Tensor> outputs = {
        {output_buffer, shape, "bfloat16"},
    };
    op.execute(inputs, outputs);
  } catch (const std::exception &e) {
    SetError(err_msg, err_msg_cap, e.what());
    return DD_LINKAGE_ERR_EXCEPTION;
  } catch (...) {
    SetError(err_msg, err_msg_cap,
             "unknown (non-std::exception) exception in "
             "dd_linkage_run_trivial_op");
    return DD_LINKAGE_ERR_UNKNOWN;
  }

  return DD_LINKAGE_OK;
}
