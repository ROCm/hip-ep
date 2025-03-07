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

namespace vaip_core {
static std::vector<std::string> XIR_OP_NAMES =
    std::vector<std::string>{"cost_volume_fix",
                             "cost_volume",
                             "correlation1d_elemwise",
                             "correlation2d_elemwise_fix",
                             "correlation2d_elemwise",
                             "scale",
                             "exp",
                             "tile_fix",
                             "matmul",
                             "stack",
                             "strided_slice",
                             "instancenorm",
                             "cast",
                             "hard_softmax",
                             "softmax",
                             "batchnorm",
                             "pixel_shuffle_fix",
                             "reorg",
                             "concat_fix",
                             "concat",
                             "downsample_fix",
                             "resize",
                             "pad_fix",
                             "flatten",
                             "eltwise_fix",
                             "transpose",
                             "strided_slice_fix",
                             "squeeze",
                             "reshape_fix",
                             "shape",
                             "upload",
                             "conv2d_fix",
                             "transposed_depthwise_conv2d_fix",
                             "transposed_conv2d_fix",
                             "relu",
                             "reduction_product",
                             "correlation1d_elemwise_fix",
                             "conv3d_fix",
                             "prelu",
                             "const",
                             "avgpool2d",
                             "transposed_depthwise_conv3d_fix",
                             "hard_sigmoid",
                             "transposed_depthwise_conv3d",
                             "identity",
                             "mish",
                             "download",
                             "transposed_depthwise_conv2d",
                             "const_fix",
                             "random_standard_normal",
                             "data_fix",
                             "fix2float",
                             "groupnorm",
                             "depthwise_conv3d_fix",
                             "transposed_conv3d_fix",
                             "conv2d",
                             "reshape",
                             "selu",
                             "depthwise_conv2d_fix",
                             "add",
                             "maxpool2d",
                             "elu",
                             "sub",
                             "conv3d",
                             "depthwise_conv3d",
                             "celu",
                             "pool_fix",
                             "reorg_fix",
                             "maxpool1d",
                             "placeholder",
                             "data",
                             "transposed_conv3d",
                             "gelu",
                             "div",
                             "min",
                             "fix",
                             "float2fix",
                             "max",
                             "argmax_fix",
                             "relu6",
                             "neg",
                             "depthwise_fix",
                             "priorbox",
                             "inner_product",
                             "sigmoid",
                             "pad",
                             "upsample_fix",
                             "swish",
                             "tanh",
                             "conv1d",
                             "hard_sigmoid_fix",
                             "depthwise_conv2d",
                             "reduction_max_fix",
                             "hard_tanh",
                             "transposed_conv2d",
                             "argmax",
                             "threshold",
                             "reduction_mean",
                             "gstiling",
                             "mul",
                             "hard_swish",
                             "reduction_sum",
                             "leaky_relu",
                             "reduction_max",
                             "pixel_shuffle",
                             "l2_normalize",
                             "quantize_linear_int8",
                             "quantize_linear_uint8",
                             "dequantize_linear_int8",
                             "dequantize_linear_uint8"};
}
