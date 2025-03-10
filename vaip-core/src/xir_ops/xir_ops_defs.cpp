/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "./xir_ops_defs.hpp"
#include "./ort_custom_ops.hpp"
#include "./xir_ops_genereted_names.inc"
#include "core/session/onnxruntime_c_api.h"

#include <map>
#include <memory>

namespace vaip_core {
static std::vector<XilinxCustomOp> xir_custom_ops;
Ort::CustomOpDomain get_xir_domain() {
  Ort::CustomOpDomain domain("com.xilinx");
  if (xir_custom_ops.empty()) {
    XIR_OP_NAMES.push_back("super_layer");
    XIR_OP_NAMES.push_back("unknown");
    XIR_OP_NAMES.push_back("conv2d_nchw");
    XIR_OP_NAMES.push_back("depthwise_conv2d_nchw");
    XIR_OP_NAMES.push_back("conv1d_ncd");
    XIR_OP_NAMES.push_back("depthwise_conv1d_ncd");
    XIR_OP_NAMES.push_back("depthwise_conv1d");
    XIR_OP_NAMES.push_back("transposed_conv2d_nchw");
    XIR_OP_NAMES.push_back("maxpool2d_nchw");
    XIR_OP_NAMES.push_back("avgpool2d_nchw");
    XIR_OP_NAMES.push_back("resize_nchw");
    XIR_OP_NAMES.push_back("ConvTranspose_nhwc");
    XIR_OP_NAMES.push_back("GlobalAveragePool_nhwc");
    XIR_OP_NAMES.push_back("depthwise_conv2d_ihwo");
    XIR_OP_NAMES.push_back("pixel_shuffle_nchw");
    XIR_OP_NAMES.push_back("gstiling_nchw");
    XIR_OP_NAMES.push_back("quantize_linear");
    XIR_OP_NAMES.push_back("dequantize_linear");
    XIR_OP_NAMES.push_back("layernorm");
    XIR_OP_NAMES.push_back("abs");
    XIR_OP_NAMES.push_back("clamp");
    XIR_OP_NAMES.push_back("instancenorm_nchw");
    XIR_OP_NAMES.push_back("instancenorm_ncd");
    XIR_OP_NAMES.push_back("groupnorm_nchw");
    XIR_OP_NAMES.push_back("groupnorm_ncd");
    XIR_OP_NAMES.push_back("space_to_depth"); // #1448
    XIR_OP_NAMES.push_back("space_to_depth_nchw");
    XIR_OP_NAMES.push_back("sqrt");
    XIR_OP_NAMES.push_back("pow");
    XIR_OP_NAMES.push_back("gather");
    XIR_OP_NAMES.push_back("expand");
    XIR_OP_NAMES.push_back("reduction_min");
    XIR_OP_NAMES.push_back("broadcast_tile");
    XIR_OP_NAMES.push_back("QMHAGRPB");
    XIR_OP_NAMES.push_back("DQAdd");
    XIR_OP_NAMES.push_back("QActConstAdd");
    XIR_OP_NAMES.push_back("QExpand");
    XIR_OP_NAMES.push_back("QMatMul");
    XIR_OP_NAMES.push_back("QBatchMatMul");
    XIR_OP_NAMES.push_back("QMatMulAdd");
    XIR_OP_NAMES.push_back("QMatMulAddGelu");
    XIR_OP_NAMES.push_back("QGemmvGelu");
    XIR_OP_NAMES.push_back("mzdk5MHA");
    XIR_OP_NAMES.push_back("QMatMulDynamic");
    XIR_OP_NAMES.push_back("QMatMulDynamicSoftmax");
    XIR_OP_NAMES.push_back("QMulSoftmax");
    XIR_OP_NAMES.push_back("QLayerNorm");
    XIR_OP_NAMES.push_back("QEltWiseAdd");
    XIR_OP_NAMES.push_back("QEltWiseDiv");
    XIR_OP_NAMES.push_back("QGlobalAvgPool");
    XIR_OP_NAMES.push_back("QMHA");
    XIR_OP_NAMES.push_back("QMHACHANNEL");
    XIR_OP_NAMES.push_back("QMHAWINDOW");
    XIR_OP_NAMES.push_back("IConv");
    XIR_OP_NAMES.push_back("QReshapeTranspose");
    XIR_OP_NAMES.push_back("QConv");
    XIR_OP_NAMES.push_back("QReduceSum");
    XIR_OP_NAMES.push_back("QConcateOPs");
    XIR_OP_NAMES.push_back("QLstm");
    XIR_OP_NAMES.push_back("QL2norm");
    XIR_OP_NAMES.push_back("QGroupNorm");
    XIR_OP_NAMES.push_back("QConv2MatMul");
    XIR_OP_NAMES.push_back("QELWEMUL_qdq");
    XIR_OP_NAMES.push_back("QELWEMUL_mxgan");
    XIR_OP_NAMES.push_back("QSlice");
    XIR_OP_NAMES.push_back("DPS");
    XIR_OP_NAMES.push_back("QConcat");
    XIR_OP_NAMES.push_back("QSilu");
    XIR_OP_NAMES.push_back("SILU");
    XIR_OP_NAMES.push_back("AttentionMaskPrePro");
    XIR_OP_NAMES.push_back("QResize");
    XIR_OP_NAMES.push_back("QuantOP");
    XIR_OP_NAMES.push_back("DeQuantOP");
    XIR_OP_NAMES.push_back("QGelu");
    XIR_OP_NAMES.push_back("QBroadcastAdd");
    XIR_OP_NAMES.push_back("Mladfelwmul");
    XIR_OP_NAMES.push_back("ELWMUL");
    XIR_OP_NAMES.push_back("MLADFMATMULA16A16");
    XIR_OP_NAMES.push_back("MladfMatMul");
    XIR_OP_NAMES.push_back("FlatMLP");
    XIR_OP_NAMES.push_back("Qtanh_lpnorm");
    XIR_OP_NAMES.push_back("MLADFRMSNORM");
    XIR_OP_NAMES.push_back("Mladfsoftmax");
    XIR_OP_NAMES.push_back("QSigmoid");
    XIR_OP_NAMES.push_back("MLADFADD");
    XIR_OP_NAMES.push_back("Qbias_add");
    XIR_OP_NAMES.push_back("round");
    XIR_OP_NAMES.push_back("equal");
    XIR_OP_NAMES.push_back("reciprocal");
    XIR_OP_NAMES.push_back("QDeMHA");

    for (auto& name : XIR_OP_NAMES) {
      xir_custom_ops.emplace_back(XilinxCustomOp(name));
    }

    // multi-output ops
    // (op_name , num_outputs)
    xir_custom_ops.emplace_back("sample_multi_outputs_op", 3);
  }
  for (auto& op : xir_custom_ops) {
    domain.Add(&op);
  }
  return domain;
}

OrtCustomOpDomain* register_xir_ops() {
  static OrtCustomOpDomain* xir_domain = nullptr;
  if (xir_domain == nullptr)
    xir_domain = get_xir_domain().release();
  // Memory leak, passing data across dlls
  return xir_domain;
}

} // namespace vaip_core
