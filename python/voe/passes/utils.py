##
## Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np
from collections import OrderedDict
import onnx_tool
from onnx_tool import Graph
from onnx_tool.fusion import *
import onnx
import copy
import json
import os
from .cal_coeff_utils import *
import csv
from colorama import init, Fore

init(autoreset=True)
VALID_LOG_LEVELS = {"error", "info", "debug", "warning", "fatal"}
## Node Names for pattern Extraction
mzdk5_MHA = [
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1_output_0_DequantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2_output_0_DequantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_output_0_DequantizeLinear",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_2.0",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_1.0",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_2.0/MatMul_output_0_QuantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_QuantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_DequantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Constant_10_output_0_DequantizeLinear__15",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_QuantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_DequantizeLinear",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.softmax_1.0",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_QuantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_DequantizeLinear",
]
mzdk5_Gelu_nodes = [
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1_output_0_DequantizeLinear",
    "Gelu_fused_Erf_0",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/gelu_1/Mul_1_output_0_QuantizeLinear",
]


mzdk5_GEMM_nodes = [
    "/down_blocks.0/resnets.1/nonlinearity_2/Mul_output_0_DequantizeLinear",
    "down_blocks.0.resnets.1.time_emb_proj.weight_DequantizeLinear",
    "down_blocks.0.resnets.1.time_emb_proj.bias_DequantizeLinear",
    "down_blocks.0.resnets.1.time_emb_proj",
    "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_QuantizeLinear",
    "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_DequantizeLinear",
    "/down_blocks.0/resnets.1/Unsqueeze_1",
    "/down_blocks.0/resnets.1/Unsqueeze_1_output_0_QuantizeLinear",
]

mzdk5_dict = {
    "mzdk5MHA": [mzdk5_MHA],
    "QGemmv": [mzdk5_GEMM_nodes],
    "QGelu": [mzdk5_Gelu_nodes],
}

mzdk5_ELEMUL_nodes = [
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_output_0_DequantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/gelu_1/Mul_1_output_0_DequantizeLinear",
    "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.mult_1",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/mult_1/Mul_output_0_QuantizeLinear",
]

mzdk5_ELEMUL_dict = {"ELEMUL": [mzdk5_ELEMUL_nodes]}

mzdk5_SLICE_1 = [
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/proj/Add_output_0_DequantizeLinear",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_output_0_QuantizeLinear",
]

mzdk5_SLICE_2 = [
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/proj/Add_output_0_DequantizeLinear__1",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/proj/Add_output_0_DequantizeLinear__2",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Shape",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Gather",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Add",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Div",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Mul_1",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1",
    "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1_output_0_QuantizeLinear",
]
mzdk5_SLICE_dict = {"SLICE": [mzdk5_SLICE_1, mzdk5_SLICE_2]}

mzdk5_RESIZE = [
    "/up_blocks.0/resnets.2/add_2/Add_output_0_DequantizeLinear",
    "up_blocks.0.upsamplers.0.interpolate_1",
    "/up_blocks.0/upsamplers.0/interpolate_1/Resize_output_0_QuantizeLinear",
]
dict_mzdk5_resize = {"QResize": [mzdk5_RESIZE]}
PSS_PST_SOFTMAX_nodes = [
    "/Cast_output_0_DequantizeLinear",
    "encoder.mid_block.attentions.0.module_softmax",
    "/module_softmax/Softmax_output_0_QuantizeLinear",
]
# Only enabled mul in MHA, shape is (1x4096x4096) vs (1).
PSS_PST_MUL_nodes = [
    "encoder.mid_block.attentions.0.module_mul_22",
    "/decoder/mid_block/attentions.0/Mul",
    # "decoder.mid_block.resnets.0.norm1#3",
    # "decoder.mid_block.resnets.0.norm2#3",
    # "decoder.mid_block.resnets.1.norm1#3",
    # "decoder.mid_block.resnets.1.norm2#3",
    # "decoder.up_blocks.0.resnets.0.norm1#3",
    # "decoder.up_blocks.0.resnets.0.norm2#3",
    # "decoder.up_blocks.0.resnets.1.norm1#3",
    # "decoder.up_blocks.0.resnets.1.norm2#3",
    # "decoder.up_blocks.0.resnets.2.norm1#3",
    # "decoder.up_blocks.0.resnets.2.norm2#3",
    # "encoder.down_blocks.3.resnets.0.norm1#3",
    # "encoder.down_blocks.3.resnets.0.norm2#3",
    # "encoder.down_blocks.3.resnets.1.norm1#3",
    # "encoder.down_blocks.3.resnets.1.norm2#3",
    # "encoder.mid_block.resnets.0.norm1#3",
    # "encoder.mid_block.resnets.0.norm2#3",
    # "encoder.mid_block.resnets.1.norm1#3",
    # "encoder.conv_norm_out#3",
    # "decoder.mid_block.attentions.0.group_norm#3",
    # "encoder.mid_block.attentions.0.group_norm#3",
    # "/decoder/mid_block/attentions.0/Mul",
    # "decoder.up_blocks.1.resnets.0.norm1#3",
    # "decoder.up_blocks.1.resnets.0.norm2#3",
    # "decoder.up_blocks.1.resnets.1.norm1#3",
    # "decoder.up_blocks.1.resnets.1.norm2#3",
    # "decoder.up_blocks.1.resnets.2.norm1#3",
    # "decoder.up_blocks.1.resnets.2.norm2#3",
    # "encoder.down_blocks.2.resnets.0.norm2#3",
    # "encoder.down_blocks.2.resnets.1.norm1#3",
    # "encoder.down_blocks.2.resnets.1.norm2#3",
    # "encoder.mid_block.resnets.1.norm1#3",
    # "decoder.up_blocks.2.resnets.0.norm1#3",
    # "decoder.up_blocks.2.resnets.0.norm2#3",
    # "decoder.up_blocks.2.resnets.1.norm1#3",
    # "decoder.up_blocks.2.resnets.1.norm2#3",
    # "decoder.up_blocks.2.resnets.2.norm1#3",
    # "decoder.up_blocks.2.resnets.2.norm2#3",
    # "decoder.up_blocks.3.resnets.0.norm1#3",
    # "decoder.up_blocks.3.resnets.0.norm2#3",
    # "decoder.up_blocks.3.resnets.1.norm1#3",
    # "decoder.up_blocks.3.resnets.1.norm2#3",
    # "decoder.up_blocks.3.resnets.2.norm1#3",
    # "decoder.up_blocks.3.resnets.2.norm2#3",
    # "decoder.conv_norm_out#3",
    # "encoder.down_blocks.0.resnets.0.norm1#3",
    # "encoder.down_blocks.0.resnets.0.norm2#3",
    # "encoder.down_blocks.0.resnets.1.norm1#3",
    # "encoder.down_blocks.0.resnets.1.norm2#3",
    # "encoder.down_blocks.1.resnets.0.norm1#3",
    # "decoder.up_blocks.3.resnets.0.norm2#3",
    # "encoder.down_blocks.1.resnets.0.norm2#3",
    # "encoder.down_blocks.1.resnets.1.norm1#3",
    # "encoder.down_blocks.1.resnets.1.norm2#3",
    # "encoder.down_blocks.2.resnets.0.norm1#3",
]

PSS_PST_SOFTMAX_dict = {"SOFTMAX": [PSS_PST_SOFTMAX_nodes]}

m7h4xjg_MHA_nodes = [
    "/text_model/encoder/layers.0/self_attn/k_proj/Add_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/q_proj/Add_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/v_proj/Add_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape",
    "/text_model/encoder/layers.0/self_attn/Mul",
    "/text_model/encoder/layers.0/self_attn/Reshape_1",
    "/text_model/encoder/layers.0/self_attn/Reshape_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Constant_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Mul_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Mul_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose",
    "/text_model/encoder/layers.0/self_attn/Reshape_2",
    "/text_model/encoder/layers.0/self_attn/Transpose_1",
    "/text_model/encoder/layers.0/self_attn/Transpose_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_4",
    "/text_model/encoder/layers.0/self_attn/Transpose_2",
    "/text_model/encoder/layers.0/self_attn/Reshape_5",
    "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_3",
    "/text_model/encoder/layers.0/self_attn/Reshape_3",
    "/text_model/encoder/layers.0/self_attn/Transpose_4",
    "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_DequantizeLinear",
    "text_model.encoder.layers.0.self_attn.bmm_1",
    "text_model.encoder.layers.0.self_attn.bmm_2",
    "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_6",
    "/text_model/encoder/layers.0/self_attn/Transpose_6",
    "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_DequantizeLinear",
    "onnx::Add_4070_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Add",
    "/text_model/encoder/layers.0/self_attn/Reshape_8",
    "/text_model/encoder/layers.0/self_attn/Add_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Add_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_7",
    "/text_model/encoder/layers.0/self_attn/Transpose_7",
    "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_DequantizeLinear",
    "text_model.encoder.layers.0.self_attn.softmax",
    "/text_model/encoder/layers.0/self_attn/Reshape_9",
    "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Reshape_9_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_DequantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_5",
    "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_QuantizeLinear",
    "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_DequantizeLinear",
]

m7h4xjg_dict = {"QMHA": [m7h4xjg_MHA_nodes]}
PSS_Conv_nodes = [
    "/Mul_output_0_DequantizeLinear",
    "post_quant_conv.weight_DequantizeLinear",
    "post_quant_conv.bias_DequantizeLinear",
    "post_quant_conv",
    "/post_quant_conv/Conv_output_0_QuantizeLinear",
]

PSS_dict = {"QConv": [PSS_Conv_nodes]}

mtea0a_add = [
    "/text_model/embeddings/token_embedding/Gather_output_0_DequantizeLinear",
    "/text_model/embeddings/position_embedding/Gather_output_0_DequantizeLinear",
    "text_model.embeddings.embedding_add",
]

mswbjvw_LSTM = [
    "im2seq_reshape_DequantizeLinear",
    "lstm_0/W_DequantizeLinear",
    "lstm_0/R_DequantizeLinear",
    "lstm_0/B_DequantizeLinear",
    "lstm_0",
    "lstm_0_QuantizeLinear",
    "lstm_0_DequantizeLinear",
    "lstm_0_transpose",
    "lstm_0_transpose_QuantizeLinear",
    "lstm_0_transpose_DequantizeLinear",
    "lstm_0_reshape",
    "lstm_0_reshape_QuantizeLinear",
    "lstm_0_reshape_DequantizeLinear",
    "lstm_1/W_DequantizeLinear",
    "lstm_1/R_DequantizeLinear",
    "lstm_1/B_DequantizeLinear",
    "lstm_1",
    "lstm_1_QuantizeLinear",
    "lstm_1_DequantizeLinear",
    "lstm_1_transpose",
    "lstm_1_transpose_QuantizeLinear",
    "lstm_1_transpose_DequantizeLinear",
    "lstm_1_reshape",
    "lstm_1_reshape_QuantizeLinear",
]

mswbjvw_CONV = [
    "data_DequantizeLinear",
    "conv1/weight_DequantizeLinear",
    "conv1/bias_DequantizeLinear",
    "conv1",
    "conv1_relu_QuantizeLinear",
    "conv1_relu_DequantizeLinear",
    "pool1",
    "pool1_QuantizeLinear",
    "pool1_DequantizeLinear",
    "conv2/weight_DequantizeLinear",
    "conv2/bias_DequantizeLinear",
    "conv2",
    "conv2_relu_QuantizeLinear",
    "conv2_relu_DequantizeLinear",
    "pool2",
    "pool2_QuantizeLinear",
    "pool2_DequantizeLinear",
    "conv3_1s/weight_DequantizeLinear",
    "conv3_1s/bias_DequantizeLinear",
    "conv3_1s",
    "conv3_1s_QuantizeLinear",
    "conv3_1s_DequantizeLinear",
    "conv3_1d/weight_DequantizeLinear",
    "conv3_1d",
    "conv3_1d_QuantizeLinear",
    "conv3_1d_DequantizeLinear",
    "conv3_1/weight_DequantizeLinear",
    "conv3_1/bias_DequantizeLinear",
    "conv3_1",
    "conv3_1_relu_QuantizeLinear",
    "conv3_1_relu_DequantizeLinear",
    "conv3_2s/weight_DequantizeLinear",
    "conv3_2s/bias_DequantizeLinear",
    "conv3_2s",
    "conv3_2s_QuantizeLinear",
    "conv3_2s_DequantizeLinear",
    "conv3_2d/weight_DequantizeLinear",
    "conv3_2d",
    "conv3_2d_QuantizeLinear",
    "conv3_2d_DequantizeLinear",
    "conv3_2/weight_DequantizeLinear",
    "conv3_2/bias_DequantizeLinear",
    "conv3_2",
    "conv3_2_relu_QuantizeLinear",
    "conv3_2_relu_DequantizeLinear",
    "pool3",
    "pool3_QuantizeLinear",
    "pool3_DequantizeLinear",
    "conv4_1s/weight_DequantizeLinear",
    "conv4_1s/bias_DequantizeLinear",
    "conv4_1s",
    "conv4_1s_QuantizeLinear",
    "conv4_1s_DequantizeLinear",
    "conv4_1d/weight_DequantizeLinear",
    "conv4_1d",
    "conv4_1d_QuantizeLinear",
    "conv4_1d_DequantizeLinear",
    "conv4_1/weight_DequantizeLinear",
    "conv4_1/bias_DequantizeLinear",
    "conv4_1",
    "conv4_1_relu_QuantizeLinear",
    "conv4_1_relu_DequantizeLinear",
    "conv4_2s/weight_DequantizeLinear",
    "conv4_2s/bias_DequantizeLinear",
    "conv4_2s",
    "conv4_2s_QuantizeLinear",
    "conv4_2s_DequantizeLinear",
    "conv4_2d/weight_DequantizeLinear",
    "conv4_2d",
    "conv4_2d_QuantizeLinear",
    "conv4_2d_DequantizeLinear",
    "conv4_2/weight_DequantizeLinear",
    "conv4_2/bias_DequantizeLinear",
    "conv4_2",
    "conv4_2_relu_QuantizeLinear",
    "conv4_2_relu_DequantizeLinear",
    "pool4",
    "pool4_QuantizeLinear",
    "pool4_DequantizeLinear",
    "conv5_1s/weight_DequantizeLinear",
    "conv5_1s/bias_DequantizeLinear",
    "conv5_1s",
    "conv5_1s_QuantizeLinear",
    "conv5_1s_DequantizeLinear",
    "conv5_1d/weight_DequantizeLinear",
    "conv5_1d",
    "conv5_1d_QuantizeLinear",
    "conv5_1d_DequantizeLinear",
    "conv5_1/weight_DequantizeLinear",
    "conv5_1/bias_DequantizeLinear",
    "conv5_1",
    "conv5_1_relu_QuantizeLinear",
    "conv5_1_relu_DequantizeLinear",
    "conv5_2s/weight_DequantizeLinear",
    "conv5_2s/bias_DequantizeLinear",
    "conv5_2s",
    "conv5_2s_QuantizeLinear",
    "conv5_2s_DequantizeLinear",
    "conv5_2d/weight_DequantizeLinear",
    "conv5_2d",
    "conv5_2d_QuantizeLinear",
    "conv5_2d_DequantizeLinear",
    "conv5_2/weight_DequantizeLinear",
    "conv5_2/bias_DequantizeLinear",
    "conv5_2",
    "conv5_2_relu_QuantizeLinear",
    "conv5_2_relu_DequantizeLinear",
    "convfeat/weight_DequantizeLinear",
    "convfeat/bias_DequantizeLinear",
    "convfeat",
    "convfeat_QuantizeLinear",
]

m3uec_conv_nodes = [
    "input_image_DequantizeLinear",
    "image_encoder.convs.0.proj.weight_DequantizeLinear",
    "image_encoder.convs.0.proj.bias_DequantizeLinear",
    "/convs.0/proj/Conv",
    "/convs.0/proj/Conv_output_0_QuantizeLinear",
    "/convs.0/proj/Conv_output_0_DequantizeLinear",
    "/convs.0/Transpose",
    "/convs.0/Transpose_output_0_QuantizeLinear",
    "/convs.0/Transpose_output_0_DequantizeLinear",
    "/convs.0/Reshape_1",
    "/convs.0/Reshape_1_output_0_QuantizeLinear",
]

m3uec_conv_nodes_1 = [
    "/convs.0/norm/Add_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_DequantizeLinear",
    "image_encoder.blocks.0.0.spatial_block.conv1.fn.dw.weight_DequantizeLinear",
    "image_encoder.blocks.0.0.spatial_block.conv1.fn.dw.bias_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1",
    "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1_output_0_QuantizeLinear",
]

m3uec_conv_nodes_2 = [
    "/convs.1/norm/Add_1_output_0_DequantizeLinear",
    "/convs.1/Reshape",
    "/convs.1/Reshape_output_0_QuantizeLinear",
    "/convs.1/Reshape_output_0_DequantizeLinear",
    "/convs.1/Transpose",
    "/convs.1/Transpose_output_0_QuantizeLinear",
    "/convs.1/Transpose_output_0_DequantizeLinear",
    "image_encoder.convs.1.proj.weight_DequantizeLinear",
    "image_encoder.convs.1.proj.bias_DequantizeLinear",
    "/convs.1/proj/Conv",
    "/convs.1/proj/Conv_output_0_QuantizeLinear",
    "/convs.1/proj/Conv_output_0_DequantizeLinear",
    "/convs.1/Transpose_1",
    "/convs.1/Transpose_1_output_0_QuantizeLinear",
    "/convs.1/Transpose_1_output_0_DequantizeLinear",
    "/convs.1/Reshape_3",
    "/convs.1/Reshape_3_output_0_QuantizeLinear",
]

m3uec_reshape_transpose_reshape = [
    "/blocks.0/blocks.0.0/spatial_block/window_attn/norm/Add_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5_output_0_QuantizeLinear",
]


m3uec_global_average_pool = [
    "/blocks.3/blocks.3.0/channel_block/ffn/Add_output_0_DequantizeLinear",
    "/Transpose",
    "/Transpose_output_0_QuantizeLinear",
    "/Transpose_output_0_DequantizeLinear",
    "Reshape_5129",
    "/Transpose_output_0_reshape_QuantizeLinear",
    "/Transpose_output_0_reshape_DequantizeLinear",
    "/avgpool/GlobalAveragePool",
    "/avgpool/GlobalAveragePool_output_0_QuantizeLinear",
    "/avgpool/GlobalAveragePool_output_0_DequantizeLinear",
    "/Flatten",
    "/Flatten_output_0_QuantizeLinear",
]

m3uec_spatial_MHA_nodes = [
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Constant_33_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__1",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__2",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12",
    "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12_output_0_QuantizeLinear",
]

# ['/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_DequantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_QuantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_DequantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_QuantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_DequantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_QuantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_DequantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_QuantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_DequantizeLinear','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2','/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear']

m3uec_channel_MHA_nodes = [
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Pow_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__1",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__2",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_QuantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_DequantizeLinear",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6",
    "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6_output_0_QuantizeLinear",
]

matmul_add_nodes = [
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
    "onnx::MatMul_2195_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/Add",
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
]

matmul_add_gelu_nodes = [
    "/tulrv6/encoder/layer.0/attention/output/LayerNorm/Add_1_output_0_DequantizeLinear",
    "onnx::MatMul_2209_DequantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/MatMul",
    "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/Add",
    "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
    "Gelu_363",
    "/tulrv6/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0_QuantizeLinear",
]

mha_pattern_in_PSF = [
    "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/Mul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
    "onnx::MatMul_2204_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
    "/tulrv6/Constant_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Sub",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
    "/tulrv6/GatherElements_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
]

PSF_a8w8_MHAGRPB = [
    "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_DequantizeLinear",
    "/tulrv6/Mul_output_0_convert_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
    "onnx::MatMul_2204_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
    "/tulrv6/Constant_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Sub",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
    "/tulrv6/GatherElements_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
]

mxgan_MHAGRPB = [
    "279_DequantizeLinear",
    "Reshape_198",
    "334_QuantizeLinear",
    "334_DequantizeLinear",
    "Transpose_199",
    "335_QuantizeLinear",
    "335_DequantizeLinear",
    "276_DequantizeLinear",
    "Reshape_186",
    "316_QuantizeLinear",
    "316_DequantizeLinear",
    "Transpose_200",
    "336_QuantizeLinear",
    "336_DequantizeLinear",
    "274_DequantizeLinear",
    "Reshape_173",
    "297_QuantizeLinear",
    "297_DequantizeLinear",
    "Transpose_174",
    "298_QuantizeLinear",
    "298_DequantizeLinear",
    "MatMul_201",
    "337_QuantizeLinear",
    "337_DequantizeLinear",
    "1062_DequantizeLinear",
    "Div_203",
    "339_QuantizeLinear",
    "339_DequantizeLinear",
    "110_DequantizeLinear",
    "Add_204",
    "340_QuantizeLinear",
    "340_DequantizeLinear",
    "298_DequantizeLinear__1",
    "1077_DequantizeLinear",
    "MatMul_214",
    "351_QuantizeLinear",
    "351_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "Add_215",
    "352_QuantizeLinear",
    "352_DequantizeLinear",
    "Reshape_223",
    "366_QuantizeLinear",
    "366_DequantizeLinear",
    "ReduceSum_225",
    "368_QuantizeLinear",
    "368_DequantizeLinear",
    "Sigmoid_226",
    "369_QuantizeLinear",
    "369_DequantizeLinear",
    "Slice_237",
    "380_QuantizeLinear",
    "380_DequantizeLinear",
    "369_DequantizeLinear__1",
    "Slice_240",
    "383_QuantizeLinear",
    "383_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "Mul_241",
    "384_QuantizeLinear",
    "384_DequantizeLinear",
    "107_DequantizeLinear__1",
    "Sub_243",
    "386_QuantizeLinear",
    "386_DequantizeLinear",
    "Mul_244",
    "387_QuantizeLinear",
    "387_DequantizeLinear",
    "130_DequantizeLinear",
    "Add_246",
    "389_QuantizeLinear",
    "389_DequantizeLinear",
    "271_DequantizeLinear",
    "Mul_247",
    "390_QuantizeLinear",
    "390_DequantizeLinear",
    "Add_248",
    "391_QuantizeLinear",
    "391_DequantizeLinear",
    "Softmax_249",
    "392_QuantizeLinear",
    "392_DequantizeLinear",
    "MatMul_250",
    "393_QuantizeLinear",
    "393_DequantizeLinear",
    "Transpose_251",
    "394_QuantizeLinear",
    "394_DequantizeLinear",
    "Reshape_263",
    "409_QuantizeLinear",
]

mxgan_QMatMulADD = [
    "138_DequantizeLinear",
    "1068_DequantizeLinear",
    "MatMul_157",
    "273_QuantizeLinear",
    "273_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
    "Add_158",
    "274_QuantizeLinear",
]

mxgan_QMatMulADDGELU = [
    "424_DequantizeLinear",
    "1082_DequantizeLinear",
    "MatMul_278",
    "426_QuantizeLinear",
    "426_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
    "Add_279",
    "427_QuantizeLinear",
    "427_DequantizeLinear",
    "Gelu_229",
    "435_QuantizeLinear",
]

mxgan_QMatMul = [
    "138_DequantizeLinear__1",
    "1069_DequantizeLinear",
    "MatMul_159",
    "276_QuantizeLinear",
]

mxgan_SkipAdd = [
    "412_DequantizeLinear",
    "138_DequantizeLinear__3",
    "Add_265",
    "412_QuantizeLinear",
]

QDQ = [
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
]


mxgan_a8w8_MHAGRPB = [
    "274_DequantizeLinear",
    "Reshape_173",
    "297_QuantizeLinear",
    "297_DequantizeLinear",
    "Transpose_174",
    "298_QuantizeLinear",
    "298_DequantizeLinear",
    "276_DequantizeLinear",
    "Reshape_186",
    "316_QuantizeLinear",
    "316_DequantizeLinear",
    "Transpose_200",
    "336_QuantizeLinear",
    "336_DequantizeLinear",
    "MatMul_201",
    "337_QuantizeLinear",
    "337_DequantizeLinear",
    "1062_DequantizeLinear",
    "Div_203",
    "339_QuantizeLinear",
    "339_DequantizeLinear",
    "339_convert_QuantizeLinear",
    "339_convert_DequantizeLinear",
    "110_convert_DequantizeLinear",
    "Add_204",
    "340_QuantizeLinear",
    "340_DequantizeLinear",
    "298_DequantizeLinear__1",
    "1077_DequantizeLinear",
    "MatMul_214",
    "351_QuantizeLinear",
    "351_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "Add_215",
    "352_QuantizeLinear",
    "352_DequantizeLinear",
    "Reshape_223",
    "366_QuantizeLinear",
    "366_DequantizeLinear",
    "ReduceSum_225",
    "368_QuantizeLinear",
    "368_DequantizeLinear",
    "Sigmoid_226",
    "369_QuantizeLinear",
    "369_DequantizeLinear",
    "Slice_237",
    "380_QuantizeLinear",
    "380_DequantizeLinear",
    "369_DequantizeLinear__1",
    "Slice_240",
    "383_QuantizeLinear",
    "383_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "Mul_241",
    "384_QuantizeLinear",
    "384_DequantizeLinear",
    "107_DequantizeLinear__1",
    "Sub_243",
    "386_QuantizeLinear",
    "386_DequantizeLinear",
    "Mul_244",
    "387_QuantizeLinear",
    "387_DequantizeLinear",
    "130_DequantizeLinear",
    "Add_246",
    "389_QuantizeLinear",
    "389_DequantizeLinear",
    "271_DequantizeLinear",
    "Mul_247",
    "390_QuantizeLinear",
    "390_DequantizeLinear",
    "390_convert_QuantizeLinear",
    "390_convert_DequantizeLinear",
    "Add_248",
    "391_QuantizeLinear",
    "391_DequantizeLinear",
    "Softmax_249",
    "392_QuantizeLinear",
    "392_DequantizeLinear",
    "392_convert_QuantizeLinear",
    "392_convert_DequantizeLinear",
    "279_DequantizeLinear",
    "Reshape_198",
    "334_QuantizeLinear",
    "334_DequantizeLinear",
    "Transpose_199",
    "335_QuantizeLinear",
    "335_DequantizeLinear",
    "MatMul_250",
    "393_QuantizeLinear",
    "393_DequantizeLinear",
    "Transpose_251",
    "394_QuantizeLinear",
    "394_DequantizeLinear",
    "Reshape_263",
    "409_QuantizeLinear",
]

mxgan_a8w8_MatMulAdd_Gelu = [
    "424_DequantizeLinear",
    "1082_DequantizeLinear",
    "MatMul_278",
    "426_QuantizeLinear",
    "426_DequantizeLinear",
    "426_convert_QuantizeLinear",
    "426_convert_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
    "Add_279",
    "427_QuantizeLinear",
    "427_DequantizeLinear",
    "Gelu_fused_Erf_0",
    "435_QuantizeLinear",
]

### pattern for unit test not present in PSF
Qmatmul = ["DeQuantizeLinear_1", "matmul_1", "QuantizeLinear_2"]

QDQMatmul = [
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__1",
    "onnx::MatMul_2196_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_QuantizeLinear",
]

QKTMATMUL = [
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
]

LAYERNORM = [
    "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear",
    "LayerNormalization_242",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear",
]

PSF_a8w8_LayerNorm = [
    "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear",
    "LayerNormalization_fused_ReduceMean_0",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_convert_QuantizeLinear",
]

Add = [
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__3",
    "/tulrv6/encoder/layer.0/attention/output/dense/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/output/Add",
    "/tulrv6/encoder/layer.0/attention/output/Add_output_0_QuantizeLinear",
]

SIGMOID = [
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
]

SOFTMAX = [
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
]

MUL = [
    "tulrv6.encoder.layer.1.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/encoder/layer.1/attention/self/Slice_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.1/attention/self/Mul_2",
    "/tulrv6/encoder/layer.1/attention/self/Mul_2_output_0_QuantizeLinear",
]

DIV = [
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
]

SUB = [
    "/tulrv6/Constant_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
]

GELU = [
    "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
    "Gelu_363",
    "/tulrv6/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0_QuantizeLinear",
]

RESHAPE = [
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
]

TRANSPOSE = [
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
]

## Add and divide operators are having split(dq layer inputs to multiple mul/add nodes)
SLICE = [
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
]

REDUCE_SUM = [
    "/tulrv6/encoder/layer.8/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.8/attention/self/ReduceSum",
    "/tulrv6/encoder/layer.8/attention/self/ReduceSum_output_0_QuantizeLinear",
]


# Pattern for MHA after fusing all the small ops and removing qdq after shape ops
MHA = [
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape",
    "/tulrv6/encoder/layer.0/attention/self/Transpose",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
]

# unused but can be used to fuse GRPB seperately
GRPB = [
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear_1",
    "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear_1",
    "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/Constant_output_0_DequantizeLinear_1",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
    "/tulrv6/GatherElements_output_0_DequantizeLinear",
]

Silu_1 = [
    "input_2_DequantizeLinear__1",
    "down_blocks.0.resnets.1.nonlinearity_2.sigmoid",
    "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_QuantizeLinear",
    "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_DequantizeLinear",
    "input_2_DequantizeLinear",
    "/down_blocks.0/resnets.1/nonlinearity_2/Mul",
    "/down_blocks.0/resnets.1/nonlinearity_2/Mul_output_0_QuantizeLinear",
]

Silu_dict = {"QSilu": [Silu_1]}


dict1 = {
    "QMatMul": QDQMatmul,
    "QLayerNorm": LAYERNORM,
    "Qsigmoid": SIGMOID,
    "QMul": MUL,
    "QSoftmax": SOFTMAX,
    "QGelu": GELU,
    "QDiv": DIV,
    "QAdd": Add,
    "QSub": SUB,
    # "QReshape":RESHAPE,
    # "QTranspose":TRANSPOSE,
    "QReduceSum": REDUCE_SUM,
    # "QSlice":SLICE,
    "MHA": MHA,
    # "GRPB":GRPB,
}

dict2 = {"matmul_unit": QDQMatmul}

dict_PSF_a16 = {
    "QMHAGRPB": mha_pattern_in_PSF,
    "QLayerNorm": LAYERNORM,
    "QMatMulAddGelu": matmul_add_gelu_nodes,
    "QMatMulAdd": matmul_add_nodes,
    "QMatMul": QDQMatmul,
    "QSkipAdd": Add,
}

dict_PSF_a8 = {
    "QMHAGRPB": PSF_a8w8_MHAGRPB,
    "QLayerNorm": PSF_a8w8_LayerNorm,
}

dict_mxgan_a16 = {
    "QMHAGRPB": mxgan_MHAGRPB,
    "QMatMulAddGelu": mxgan_QMatMulADDGELU,
    "QMatMulAdd": mxgan_QMatMulADD,
    "QSkipAdd": mxgan_SkipAdd,
}

dict_mxgan_a8 = {
    "QMHAGRPB": mxgan_a8w8_MHAGRPB,
    "QMatMulAddGelu": mxgan_a8w8_MatMulAdd_Gelu,
}

layername_dict = {
    "QMHAGRPB": "MHAGRPB",
    "QLayerNorm": "LayerNorm",
    "QMatMulAddGelu": "MatMulAddGelu",
    "QMatMulAdd": "MatMulAdd",
    "QMatMul": "MatMul",
    "QSkipAdd": "Add",
}

# m3uec
dict_m3uec_a16 = {
    "QConv": [m3uec_conv_nodes, m3uec_conv_nodes_1, m3uec_conv_nodes_2],
    "QMHAWINDOW": [m3uec_spatial_MHA_nodes],
    "QMHACHANNEL": [m3uec_channel_MHA_nodes],
    "QReshapeTranspose": [m3uec_reshape_transpose_reshape],
    "QGlobalAvgPool": [m3uec_global_average_pool],
}

# mtea0a
dict_PSQ_a16 = {"DQAdd": [mtea0a_add]}

dict_mswbjvw_conv = {"QConcateOPs": [mswbjvw_CONV]}
dict_mswbjvw_lstm = {"QLstm": [mswbjvw_LSTM]}


def get_conv_attributes(g, nodes):
    inp_zp = g.tensormap[g.nodemap[nodes[0]].input[2]].numpy.astype(np.uint16)
    for nodename in nodes:
        node = g.nodemap[nodename]
        if "conv".lower() in node.op_type.lower():
            act_input_shape = g.tensormap[node.input[0]].shape
            kernel_shape = g.tensormap[node.input[1]].shape
            output_shape = g.tensormap[node.output[0]].shape
            # wt_name = node.prevnodes[1].input[0]
            # breakpoint()

            wt_name = g.nodemap[g.producedby[node.input[1]][0]].input[0]
    return (
        list(act_input_shape),
        list(kernel_shape),
        list(output_shape),
        int(inp_zp),
        wt_name,
    )


def get_merged_conv_attributes(g, nodes):
    dict = {}
    attrs = []

    list_wt_name = []
    cnt = 0
    width = 0
    for nodename in nodes:
        node = g.nodemap[nodename]

        if "conv".lower() in node.op_type.lower():
            dict[cnt] = {}
            # breakpoint()
            # print(node.op_type.lower())
            in_channel_pad = 8
            ## checking order of convs
            if cnt == 0:
                inp_shape = g.tensormap[node.input[0]].shape
                width = inp_shape[3]
                act_input_shape = []
                for i, value in enumerate(inp_shape):
                    if i == 1:
                        act_input_shape.append(in_channel_pad)
                    else:
                        act_input_shape.append(value)
                dict[0]["graphID"] = width
                dict[0]["inChannels"] = in_channel_pad

                in_ker_shape = g.tensormap[node.input[1]].shape
                kernel_shape = []
                for i, value in enumerate(in_ker_shape):
                    if i == 1:
                        kernel_shape.append(in_channel_pad)
                    else:
                        kernel_shape.append(value)
                dict[0]["outChannels"] = kernel_shape[0]
            else:
                act_input_shape = g.tensormap[node.input[0]].shape
                kernel_shape = g.tensormap[node.input[1]].shape

            output_shape = g.tensormap[node.output[0]].shape
            wt_name = node.prevnodes[1].input[0]
            list_wt_name.append(wt_name)

            # Get next nodes
            nn1 = node.nextnodes[0]
            nn2 = nn1.nextnodes[0]
            nn3 = nn2.nextnodes[0]

            if "pool".lower() in nn3.op_type.lower():
                output_shape = g.tensormap[nn3.output[0]].shape

            inp_zp = int(
                g.tensormap[node.prevnodes[1].input[2]].numpy.astype(np.uint16)
            )
            # print(g.tensormap[node.prevnodes[1]].dtype)
            # breakpoint()
            dict[cnt]["opType"] = "conv"
            dict[cnt]["opIfmDtype"] = "uint16"
            dict[cnt]["opWtsDtype"] = "uint16"
            dict[cnt]["opOfmDtype"] = "uint16"
            # print(node.attr["group"])
            dict[cnt]["group"] = node.attr["group"]

            dict[cnt]["input_shape"] = act_input_shape
            dict[cnt]["output_shape"] = output_shape
            dict[cnt]["weight_shape"] = kernel_shape
            dict[cnt]["zero_point"] = inp_zp

            if width != 320:
                dict[cnt]["width"] = width

            attrs.append(dict[cnt])
            cnt += 1

    attr_json_obj = json.dumps(attrs, indent=2)

    return (
        list(act_input_shape),
        list(kernel_shape),
        list(output_shape),
        int(inp_zp),
        list_wt_name,
        attr_json_obj,
    )


def get_lstm_attributes(g, nodes):
    dict = {}
    attrs = []

    list_wt_name = []
    list_scale = []
    list_zp = []
    cnt = 0
    for nodename in nodes:
        node = g.nodemap[nodename]

        if "lstm".lower() in node.op_type.lower():
            # breakpoint()
            if cnt == 0:
                inp_shape = g.tensormap[node.input[0]].shape
            w_shape = g.tensormap[node.input[1]].shape
            r_shape = g.tensormap[node.input[2]].shape
            b_shape = g.tensormap[node.input[3]].shape
            output_shape = g.tensormap[node.output[0]].shape
            inp_scale = float(
                g.tensormap[node.prevnodes[0].input[1]].numpy.astype(np.float32)
            )
            w_scale = float(
                g.tensormap[node.prevnodes[1].input[1]].numpy.astype(np.float32)
            )
            r_scale = float(
                g.tensormap[node.prevnodes[2].input[1]].numpy.astype(np.float32)
            )
            b_scale = float(
                g.tensormap[node.prevnodes[3].input[1]].numpy.astype(np.float32)
            )
            out_scale = float(
                g.tensormap[node.nextnodes[0].input[1]].numpy.astype(np.float32)
            )
            list_scale.append(inp_scale)
            list_scale.append(w_scale)
            list_scale.append(r_scale)
            list_scale.append(b_scale)
            list_scale.append(out_scale)
            inp_zp = int(
                g.tensormap[node.prevnodes[0].input[2]].numpy.astype(np.uint16)
            )
            w_zp = int(g.tensormap[node.prevnodes[1].input[2]].numpy.astype(np.uint16))
            r_zp = int(g.tensormap[node.prevnodes[2].input[2]].numpy.astype(np.uint16))
            b_zp = int(g.tensormap[node.prevnodes[3].input[2]].numpy.astype(np.uint16))
            out_zp = int(
                g.tensormap[node.nextnodes[0].input[1]].numpy.astype(np.uint16)
            )
            list_zp.append(inp_zp)
            list_zp.append(w_zp)
            list_zp.append(r_zp)
            list_zp.append(b_zp)
            list_zp.append(out_zp)
            wt_name = node.prevnodes[1].input[0]
            list_wt_name.append(wt_name)
            wt_name = node.prevnodes[2].input[0]
            list_wt_name.append(wt_name)
            wt_name = node.prevnodes[3].input[0]
            list_wt_name.append(wt_name)

    return (
        list(inp_shape),
        list(w_shape),
        list(r_shape),
        list(b_shape),
        list(output_shape),
        list_scale,
        list_zp,
        list_wt_name,
    )


def get_global_avg_pool_attrs(g, nodes):
    for nodename in nodes:
        node = g.nodemap[nodename]
        if "GlobalAveragePool".lower() in node.op_type.lower():
            input_shape = g.tensormap[node.input[0]].shape
            output_shape = g.tensormap[node.output[0]].shape
            zp = g.tensormap[node.prevnodes[0].input[2]].numpy
            return (list(input_shape), list(output_shape), int(zp))


def check_kernel_shape(g, nodename):
    shape_dict = {}
    shape_dict["MatMul"] = {
        "uint16_t_uint8_t_uint16_t": [
            [64, 128, 128],
            [64, 1024, 1024],
            [64, 1024, 3072],
            [64, 1024, 4096],
            [64, 4096, 1024],
            [256, 512, 512],
            [256, 512, 1536],
            [256, 512, 2048],
            [256, 2048, 512],
            [832, 256, 256],
            [832, 256, 768],
            [832, 256, 1024],
            [832, 1024, 256],
            [3136, 128, 128],
            [3136, 128, 384],
            [3136, 128, 512],
            [3136, 512, 128],
        ]
    }
    shape_dict["SkipAdd"] = {
        "uint16_t_uint8_t_uint16_t": [[64, 1024], [224, 512], [832, 256], [3200, 128]]
    }
    shape_dict["Layernorm"] = {
        "uint16_t_uint16_t_uint16_t": [[64, 1024], [256, 512], [832, 256], [3136, 128]]
    }
    node = g.nodemap[nodename]
    if "QMatMul" in node.op_type:
        act_input = node.input[0]
        act_input_dim = g.tensormap[act_input].shape
        act_output_dim = g.tensormap[node.output[0]].shape
        N = act_output_dim[-1]
        shapes_list = shape_dict["MatMul"]["uint16_t_uint8_t_uint16_t"]
        if len(act_input_dim) == 4:
            print("saw 4 dims")
            print(nodename)
        elif len(act_input_dim) == 3:
            M = act_input_dim[0] * act_input_dim[1]
            K = act_input_dim[2]
            modified_input_shape = [1, M, K]
            node_shape = [M, K, N]
            found = 0
            near = 1000000000
            for i in shapes_list:
                if i[0] == M and i[1] == K and i[2] == N:
                    found = 1
                    near = 0
                    break
                else:
                    if i[0] > M and i[0] - M < near:
                        near = i[0] - M
                        found = 1
            modified_m = M + near
            modified_input_shape[1] = modified_m
            g.tensormap[act_input].shape = modified_input_shape
            g.tensormap[node.output[0]].shape = [1, modified_m, N]
    elif "SkipAdd" in node.op_type:
        act_input = node.input[0]
        act_input_dim = g.tensormap[act_input].shape
        # print(f"act input_dim  {act_input_dim}")
        act_output_dim = g.tensormap[node.output[0]].shape
        N = act_output_dim[-1]
        shapes_list = shape_dict["SkipAdd"]["uint16_t_uint8_t_uint16_t"]

        if len(act_input_dim) == 4:
            print("saw 4 dims")
        elif len(act_input_dim) == 3:
            M = act_input_dim[0] * act_input_dim[1]
            K = act_input_dim[2]
            modified_input_shape = [1, M, K]
            found = 0
            near = 1000000000
            for i in shapes_list:
                if i[0] == M and i[1] == K:
                    found = 1
                    near = 0
                    break

                else:
                    if i[0] > M and i[0] - M < near:
                        near = i[0] - M
                        found = 1
                        # break
            if found == 1:
                modified_m = M + near
                modified_input_shape[1] = modified_m
                g.tensormap[act_input].shape = modified_input_shape
                g.tensormap[node.input[1]].shape = modified_input_shape
                g.tensormap[node.output[0]].shape = [1, modified_m, K]
            else:
                print(" Shape Not found ")

    elif "LayerNorm" in node.op_type:
        act_input = node.input[0]
        act_input_dim = g.tensormap[act_input].shape
        # print(f"act input_dim  {act_input_dim}")
        act_output_dim = g.tensormap[node.output[0]].shape
        N = act_output_dim[-1]
        shapes_list = shape_dict["Layernorm"]["uint16_t_uint16_t_uint16_t"]

        if len(act_input_dim) == 4:
            print("saw 4 dims")
        elif len(act_input_dim) == 3:
            M = act_input_dim[0] * act_input_dim[1]
            K = act_input_dim[2]
            modified_input_shape = [1, M, K]
            found = 0
            near = 1000000000
            for i in shapes_list:
                if i[0] == M and i[1] == K:
                    found = 1
                    near = 0
                    break

                else:
                    if i[0] > M and i[0] - M < near:
                        near = i[0] - M
                        found = 1
                        # break

            modified_m = M + near
            modified_input_shape[1] = modified_m
            g.tensormap[act_input].shape = modified_input_shape
            # g.tensormap[node.input[1]].shape= modified_input_shape
            g.tensormap[node.output[0]].shape = [1, modified_m, K]
    return g
    # return g


def MHAChannel_q_params(g, nodes):
    softmax_input_params = []
    softmax_output_params = []
    QKT_input_qparams = []
    QKT_output_qparams = []
    VSQKT_input_qparams = []
    VSQKT_output_qparams = []
    MUL_input_qparams = []
    MUL_weight_qparams = []
    MUL_output_qparams = []

    for node in nodes:
        if "mul" == g.nodemap[node].op_type.lower():
            mul_node = g.nodemap[node]
            mul_prevnodes = mul_node.prevnodes
            for prev in mul_prevnodes:
                input_1 = prev.input[0]
                if input_1 not in g.initials:
                    MUL_input_qparams.extend(prev.input[1:])
                else:
                    MUL_weight_qparams.extend(prev.input)
            MUL_output_qparams.extend(mul_node.nextnodes[0].input[1:])

        if "softmax" in g.nodemap[node].op_type.lower():
            softmax_node = g.nodemap[node]
            softmax_input_params.extend(softmax_node.prevnodes[0].input[1:])
            softmax_output_params.extend(softmax_node.nextnodes[0].input[1:])
            if (
                "matmul"
                in softmax_node.prevnodes[0].prevnodes[0].prevnodes[0].op_type.lower()
            ):
                qkt_matmul_node = softmax_node.prevnodes[0].prevnodes[0].prevnodes[0]
                qkt_parents = qkt_matmul_node.prevnodes
                for i in qkt_parents:
                    QKT_input_qparams.extend(i.input[1:])
                qkt_child = qkt_matmul_node.nextnodes
                for i in qkt_child:
                    QKT_output_qparams.extend(i.input[1:])
                input_1_shape = g.tensormap[qkt_matmul_node.input[0]].shape
                input_2_shape = g.tensormap[qkt_matmul_node.input[1]].shape
                common = [i for i in input_1_shape if i in input_2_shape]
                qkt_k_dim = common[-1]

            if (
                "matmul"
                in softmax_node.nextnodes[0].nextnodes[0].nextnodes[0].op_type.lower()
            ):
                VSQKT_matmul_node = softmax_node.nextnodes[0].nextnodes[0].nextnodes[0]
                VSQKT_parents = VSQKT_matmul_node.prevnodes
                for i in VSQKT_parents:
                    VSQKT_input_qparams.extend(i.input[1:])
                VSQKT_child = VSQKT_matmul_node.nextnodes
                for i in VSQKT_child:
                    VSQKT_output_qparams.extend(i.input[1:])
                input_1_shape = g.tensormap[VSQKT_matmul_node.input[0]].shape
                input_2_shape = g.tensormap[VSQKT_matmul_node.input[1]].shape
                common = [i for i in input_1_shape if i in input_2_shape]
                VSQKT_k_dim = common[-1]
    return (
        softmax_input_params,
        softmax_output_params,
        QKT_input_qparams,
        QKT_output_qparams,
        VSQKT_input_qparams,
        VSQKT_output_qparams,
        qkt_k_dim,
        VSQKT_k_dim,
        MUL_input_qparams,
        MUL_weight_qparams,
        MUL_output_qparams,
    )


def get_MHA_qparams(g, nodes):
    softmax_input_params = []
    softmax_output_params = []
    QKT_input_qparams = []
    QKT_output_qparams = []
    VSQKT_input_qparams = []
    VSQKT_output_qparams = []
    MUL_input_qparams = []
    MUL_weight_qparams = []
    MUL_output_qparams = []

    for node in nodes:
        if "mul" == g.nodemap[node].op_type.lower():
            mul_node = g.nodemap[node]
            mul_prevnodes = mul_node.prevnodes
            for prev in mul_prevnodes:
                input_1 = prev.input[0]
                if input_1 not in g.initials:
                    MUL_input_qparams.extend(prev.input[1:])
                else:
                    MUL_weight_qparams.extend(prev.input)
            MUL_output_qparams.extend(mul_node.nextnodes[0].input[1:])

        if "softmax" in g.nodemap[node].op_type.lower():
            softmax_node = g.nodemap[node]
            softmax_input_params.extend(softmax_node.prevnodes[0].input[1:])
            softmax_output_params.extend(softmax_node.nextnodes[0].input[1:])
        if "matmul" in g.nodemap[node].op_type.lower():
            if (
                g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                == "Reshape"
            ):
                qkt_matmul_node = g.nodemap[node]
                qkt_parents = qkt_matmul_node.prevnodes
                for i in qkt_parents:
                    QKT_input_qparams.extend(i.input[1:])
                qkt_child = qkt_matmul_node.nextnodes
                for i in qkt_child:
                    QKT_output_qparams.extend(i.input[1:])
                input_1_shape = g.tensormap[qkt_matmul_node.input[0]].shape
                input_2_shape = g.tensormap[qkt_matmul_node.input[1]].shape
                common = [i for i in input_1_shape if i in input_2_shape]
                qkt_k_dim = common[-1]

            else:
                VSQKT_matmul_node = g.nodemap[node]
                VSQKT_parents = VSQKT_matmul_node.prevnodes
                for i in VSQKT_parents:
                    VSQKT_input_qparams.extend(i.input[1:])
                VSQKT_child = VSQKT_matmul_node.nextnodes
                for i in VSQKT_child:
                    VSQKT_output_qparams.extend(i.input[1:])
                input_1_shape = g.tensormap[VSQKT_matmul_node.input[0]].shape
                input_2_shape = g.tensormap[VSQKT_matmul_node.input[1]].shape
                common = [i for i in input_1_shape if i in input_2_shape]
                VSQKT_k_dim = common[-1]

    return (
        softmax_input_params,
        softmax_output_params,
        QKT_input_qparams,
        QKT_output_qparams,
        VSQKT_input_qparams,
        VSQKT_output_qparams,
        qkt_k_dim,
        VSQKT_k_dim,
        MUL_input_qparams,
        MUL_weight_qparams,
        MUL_output_qparams,
    )


def extract_mkn(inp1shape, inp2shape, outputshape):  # 2 4d inputs -> a,b,c,k a,d,k,N
    if len(inp1shape) == 4:
        M = inp1shape[0] * inp1shape[1] * inp1shape[2]
        K = inp1shape[-1]
        N = outputshape[-1]

        return M, K, N


def get_mzdk5MHA_qparams(g, nodes, mzdk5MHA_Shapes_dict, design_param):
    softmax_input_params = []
    softmax_output_params = []
    QKT_input_qparams = []
    QKT_output_qparams = []
    VSQKT_input_qparams = []
    VSQKT_output_qparams = []
    MUL_input_qparams = []
    MUL_weight_qparams = []
    MUL_output_qparams = []
    unsupported_MKN_shapes = [[4096, 64, 4096], [1024, 64, 1024], [256, 64, 256]]  # QKT
    mul_nodes = []
    softmax_nodes = []
    qkt_matmul_nodes = []
    smv_matmul_nodes = []
    qkt_inputs_shapes = []
    fuse = True
    mkn_dict = {}
    for node in nodes:
        if "mul" == g.nodemap[node].op_type.lower():
            mul_node = g.nodemap[node]
            mul_prevnodes = mul_node.prevnodes
            for prev in mul_prevnodes:
                input_1 = prev.input[0]
                if input_1 not in g.initials:
                    MUL_input_qparams.extend(prev.input[1:])
                else:
                    MUL_weight_qparams.extend(prev.input)
                mul_nodes.append(prev.name)
            mul_nodes.append(node)
            mul_nodes.append(mul_node.nextnodes[0].name)
            MUL_output_qparams.extend(mul_node.nextnodes[0].input[1:])

        if "softmax" in g.nodemap[node].op_type.lower():
            # print(node)
            softmax_node = g.nodemap[node]
            softmax_input_params.extend(softmax_node.prevnodes[0].input[1:])
            softmax_output_params.extend(softmax_node.nextnodes[0].input[1:])
            softmax_nodes.extend(
                [softmax_node.prevnodes[0].name, node, softmax_node.nextnodes[0].name]
            )

        if "matmul" in g.nodemap[node].op_type.lower():
            if (
                len(g.nodemap[node].nextnodes)
                and len(g.nodemap[node].nextnodes[0].nextnodes)
                and len(g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes)
                and g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                == "Mul"
            ):
                qkt_matmul_node = g.nodemap[node]
                qkt_parents = qkt_matmul_node.prevnodes
                for i in qkt_parents:
                    QKT_input_qparams.extend(i.input[1:])
                    qkt_matmul_nodes.append(i.name)

                qkt_matmul_nodes.append(node)
                qkt_child = qkt_matmul_node.nextnodes
                for i in qkt_child:
                    QKT_output_qparams.extend(i.input[1:])
                    qkt_matmul_nodes.append(i.name)
                input_1_shape = g.tensormap[qkt_matmul_node.input[0]].shape
                input_2_shape = g.tensormap[qkt_matmul_node.input[1]].shape
                common = [i for i in input_1_shape if i in input_2_shape]
                qkt_k_dim = common[-1]

                m, k, n = extract_mkn(
                    input_1_shape,
                    input_2_shape,
                    g.tensormap[qkt_matmul_node.output[0]].shape,
                )
                if [m, k, n] in unsupported_MKN_shapes:
                    fuse = False
                    # str=str(m)+"_"+str(k)+"_"+str(n)
                    # mkn_dict[str]=mkn_dict.get(str,0)+1

            else:
                VSQKT_matmul_node = g.nodemap[node]
                VSQKT_parents = VSQKT_matmul_node.prevnodes
                for i in VSQKT_parents:
                    VSQKT_input_qparams.extend(i.input[1:])
                    smv_matmul_nodes.append(i.name)
                VSQKT_child = VSQKT_matmul_node.nextnodes
                smv_matmul_nodes.append(node)
                for i in VSQKT_child:
                    VSQKT_output_qparams.extend(i.input[1:])
                    smv_matmul_nodes.append(i.name)
                input_1_shape = g.tensormap[VSQKT_matmul_node.input[0]].shape
                input_2_shape = g.tensormap[VSQKT_matmul_node.input[1]].shape
                common = [i for i in input_1_shape if i in input_2_shape]
                VSQKT_k_dim = common[-1]

    if fuse == False:
        # QKT MatMul
        qkt_matmul_nodes_outputs = [
            g.nodemap[node].output[0] for node in qkt_matmul_nodes
        ]
        g.fuse_subgraph_node_names(
            qkt_matmul_nodes, "QMatMulDynamic", qkt_matmul_nodes[0], True
        )
        g.nodemap[qkt_matmul_nodes[0]].set_attr("nodes", qkt_matmul_nodes_outputs)
        g.nodemap[qkt_matmul_nodes[0]].set_attr("is_qkt", True)
        g.nodemap[qkt_matmul_nodes[0]].set_attr(
            "orig_output_shape", g.tensormap[qkt_matmul_nodes_outputs[-1]].shape
        )
        g.nodemap[qkt_matmul_nodes[0]].set_attr("design_param", design_param)

        # Mul  + SoftMax
        softmax_fullnodes = mul_nodes + softmax_nodes
        softmax_fullnodes_outputs = [
            g.nodemap[node].output[0] for node in softmax_fullnodes
        ]
        g.fuse_subgraph_node_names(
            softmax_fullnodes, "QMulSoftmax", softmax_fullnodes[0], True
        )
        g.nodemap[softmax_fullnodes[0]].set_attr("nodes", softmax_fullnodes_outputs)
        g.nodemap[softmax_fullnodes[0]].set_attr(
            "orig_output_shape", g.tensormap[softmax_fullnodes_outputs[-1]].shape
        )
        g.nodemap[softmax_fullnodes[0]].set_attr("design_param", design_param)

        # SMV MatMul
        smv_matmul_nodes_outputs = [
            g.nodemap[node].output[0] for node in smv_matmul_nodes
        ]
        g.fuse_subgraph_node_names(
            smv_matmul_nodes, "QMatMulDynamic", smv_matmul_nodes[0], True
        )
        g.nodemap[smv_matmul_nodes[0]].set_attr("nodes", smv_matmul_nodes_outputs)
        g.nodemap[smv_matmul_nodes[0]].set_attr("is_qkt", False)
        g.nodemap[smv_matmul_nodes[0]].set_attr(
            "orig_output_shape", g.tensormap[smv_matmul_nodes_outputs[-1]].shape
        )
        g.nodemap[smv_matmul_nodes[0]].set_attr("design_param", design_param)

    return (
        fuse,
        mzdk5MHA_Shapes_dict,
        softmax_input_params,
        softmax_output_params,
        QKT_input_qparams,
        QKT_output_qparams,
        VSQKT_input_qparams,
        VSQKT_output_qparams,
        qkt_k_dim,
        VSQKT_k_dim,
        MUL_input_qparams,
        MUL_weight_qparams,
        MUL_output_qparams,
    )


def get_MHAGRPB_params(g, nodes):
    correct_matmul = True
    QKT_input_qparams = []
    QKT_output_qparams = []
    VSQKT_input_qparams = []
    VSQKT_output_qparams = []
    softmax_input_qparams = []
    softmax_output_qparams = []
    params = []
    ini = []
    sub_scale = []
    add_scale = []
    sigmoid_params = []
    div_params = []
    grpb_matmul_add_out_params = []
    for node in nodes:
        if g.nodemap[node].op_type == "MatMul":
            correct_matmul = True
            parents = g.nodemap[node].prevnodes
            for i in parents:
                if i.op_type == "DequantizeLinear" and len(i.prevnodes) < 1:
                    correct_matmul = False
            QKT_matmul = False
            if correct_matmul == True:
                # print(node)
                if (
                    g.nodemap[node].prevnodes[0].prevnodes[0].prevnodes[0].op_type
                    == g.nodemap[node].prevnodes[1].prevnodes[0].prevnodes[0].op_type
                ):
                    for i in parents:
                        QKT_input_qparams.append(i.input[1])
                        QKT_input_qparams.append(i.input[2])
                    i = g.nodemap[node].nextnodes[0]
                    QKT_output_qparams.append(i.input[1])
                    QKT_output_qparams.append(i.input[2])

                elif (
                    g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                    == "Transpose"
                ):
                    # print(node)
                    for i in parents:
                        VSQKT_input_qparams.append(i.input[1])
                        VSQKT_input_qparams.append(i.input[2])
                    i = g.nodemap[node].nextnodes[0]
                    VSQKT_output_qparams.append(i.input[1])
                    VSQKT_output_qparams.append(i.input[2])

        if g.nodemap[node].op_type == "Softmax":
            i = g.nodemap[node].prevnodes[0]
            softmax_input_qparams.append(i.input[1])
            softmax_input_qparams.append(i.input[2])
            if (
                g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                == "MatMul"
            ):
                i = g.nodemap[node].nextnodes[0].nextnodes[0]
            elif (
                g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                == "QuantizeLinear"
            ):
                i = g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].nextnodes[0]
            softmax_output_qparams.append(i.input[1])
            softmax_output_qparams.append(i.input[2])

        if g.nodemap[node].op_type == "Sub":
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    sub_scale.extend(parent.input)

        if g.nodemap[node].op_type == "Div":
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    div_params.extend(parent.input)

        if g.nodemap[node].op_type == "Add":
            correct_add = False
            grpb_matmul_add = False
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if (
                    len(parent.prevnodes)
                    and parent.prevnodes[0].prevnodes[0].op_type == "Mul"
                ):
                    correct_add = True
            if correct_add == True:
                for parent in parents:
                    if (
                        parent.op_type == "DequantizeLinear"
                        and len(parent.prevnodes) < 1
                    ):
                        add_scale.extend(parent.input)
                        # print(add_scale)
            if correct_add == False:
                for parent in parents:
                    if (
                        parent.op_type == "DequantizeLinear"
                        and len(parent.prevnodes) < 1
                    ):
                        if g.tensormap[parent.input[0]].shape == (8,):
                            grpb_matmul_add = True
                if grpb_matmul_add == True:
                    nextnode = g.nodemap[node].nextnodes[0]
                    grpb_matmul_add_out_params.extend(nextnode.input[1:])

        if g.nodemap[node].op_type == "Sigmoid":
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    sigmoid_params.extend(parent.input[1:])
            nextnode = g.nodemap[node].nextnodes[0]
            sigmoid_params.extend(nextnode.input[1:])

    return (
        QKT_input_qparams,
        QKT_output_qparams,
        VSQKT_input_qparams,
        VSQKT_output_qparams,
        softmax_input_qparams,
        softmax_output_qparams,
        sub_scale,
        add_scale,
        sigmoid_params,
        div_params,
        grpb_matmul_add_out_params,
    )


# Get input and output shape of gemm
def get_gemv_input_output_shape(g, nodes, tensormap):
    out_shape = ""
    in_shape = ""
    for node in nodes:
        if g.nodemap[node].op_type == "Gemm":
            parents = g.nodemap[node].prevnodes
            mm_shape = tensormap[g.nodemap[node].output[0]].shape
            in_shape = tensormap[g.nodemap[node].input[0]].shape

    if len(in_shape) == 2:
        for i in range(2):  ##Make 4D tensor
            in_shape.append(1)

    return out_shape, in_shape


# Check if Matmul contains wts as one of the input
# if both the inputs are activations, we are not fusing them right now
def check_if_wts_matmul(g, nodes, tensormap):
    fuse = False
    mm_shape = ""
    in_shape = ""
    for node in nodes:
        if g.nodemap[node].op_type == "MatMul" and g.nodemap[node].name != "MatMul_791":
            parents = g.nodemap[node].prevnodes
            mm_shape = tensormap[g.nodemap[node].output[0]].shape
            in_shape = tensormap[g.nodemap[node].input[0]].shape
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    fuse = True

    return fuse, mm_shape, in_shape


def check_if_ele_mul(g, nodes, tensormap):
    fuse = True
    mm_shape = ""

    for node in nodes:
        if (
            g.nodemap[node].op_type == "Mul"
        ):  # and g.nodemap[node].name == "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.mult_1" or g.nodemap[node].name == "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.mult_1":
            parents = g.nodemap[node].prevnodes
            mm_shape = tensormap[g.nodemap[node].output[0]].shape
            #
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    fuse = False

    return fuse, mm_shape


def check_if_mladfsoftmax(g, nodes, tensormap):
    # softmax pattern includes deq, softmax and q
    fuse = True
    sm_output_shape = tensormap[g.nodemap[nodes[1]].output[0]].shape

    if sm_output_shape != [1, 4096, 4096] and sm_output_shape != [4096, 4096]:
        return False
    # if shape_infer in fuse_layers function is not called, do following specific steps
    if not tensormap[g.nodemap[nodes[0]].input[0]].shape:
        # set the output and input tensor shape for fused op to pass DoD shape check
        g.tensormap[g.nodemap[nodes[0]].input[0]].shape = sm_output_shape
        g.tensormap[g.nodemap[nodes[2]].output[0]].shape = sm_output_shape
        # set the last node shape for set_attr of g in fuse_layer function
        tensormap[g.nodemap[nodes[2]].output[0]].shape = sm_output_shape

    return fuse


def check_if_mladfmul(g, nodes, tensormap):
    fuse = True
    for node in nodes:
        if g.nodemap[node].op_type == "Mul":
            if node not in PSS_PST_MUL_nodes:
                fuse = False
    return fuse


def check_if_wts_add(g, nodes):
    fuse = True
    fuse_badd = False
    for node in nodes:
        if g.nodemap[node].op_type == "Add" and g.nodemap[node].name != "Add_775":
            inputs = g.nodemap[node].input
            parents = g.nodemap[node].prevnodes

            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    if len(parent.prevnodes) < 1 and parent.input[0] not in g.input:
                        fuse = False
                    elif len(parent.prevnodes) < 1 and parent.input[0] in g.input:
                        fuse = True
            if inputs[0] == g.nodemap[nodes[0]].output[0]:
                continue
            else:
                nodes[0], nodes[1] = nodes[1], nodes[0]
    if g.nodemap[node].name == "Add_775":
        fuse = False
    if fuse == True and g.tensormap[inputs[0]].shape != g.tensormap[inputs[1]].shape:
        fuse_badd = True
        if (
            g.tensormap[inputs[1]].shape[-1] == 1
            and g.tensormap[inputs[1]].shape[-2] == 1
        ):
            nodes[0], nodes[1] = nodes[1], nodes[0]

    return fuse, nodes, fuse_badd


def check_if_wts_add_profile(g, nodes):
    fuse = True

    for node in nodes:
        if g.nodemap[node].op_type == "Add" and g.nodemap[node].name != "Add_775":
            inputs = g.nodemap[node].input
            parents = g.nodemap[node].prevnodes

            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    if len(parent.prevnodes) < 1 and parent.input[0] not in g.input:
                        fuse = False
                    elif len(parent.prevnodes) < 1 and parent.input[0] in g.input:
                        fuse = True

    return fuse


def remove_abovedq_belowq_for_op(g, op_type):
    """
    Function to remove Dq layers above the op and Q layer below a given op_type like Transpose/Reshape/Slice etc
    - As QDQ on shape ops is not necessary
    """

    rm_node_list = []
    for node in g.nodemap:
        if g.nodemap[node].op_type == op_type:
            if g.nodemap[node].prevnodes[0].op_type == "DequantizeLinear":
                above_dq = g.nodemap[node].prevnodes[0].name

                rm_node_list.append(above_dq)
            if g.nodemap[node].nextnodes[0].op_type == "QuantizeLinear":
                belowq = g.nodemap[node].nextnodes[0].name

                rm_node_list.append(belowq)

    for rm_node in rm_node_list:
        g.skip_node(rm_node)
        g.graph_reorder_nodes()
    return g


def add_3_nodes(g, node_name, actual_output_list, nodes_to_remove):
    node = g.nodemap[node_name]

    new_output_list = []
    # get above quantlinear
    parent1 = node.prevnodes[0]
    new_output_list.insert(0, parent1.output[0])

    parent2 = g.nodemap[g.producedby[parent1.input[0]][0]]

    new_output_list.insert(0, parent2.output[0])
    parent3 = g.nodemap[g.producedby[parent2.input[0]][0]]
    new_output_list.insert(0, parent3.output[0])
    new_output_list.extend(actual_output_list)
    if parent3.op_type == "Reshape":
        node.set_attr("Fuse", "False")
        nodes_to_remove.extend([parent1.name, parent2.name, parent3.name])
        node.input[0] = parent3.input[0]

    else:
        nodes_to_remove.extend([parent1.name, parent2.name, parent3.name])
        node.set_attr("Fuse", "True")
        # node.input[0]=parent3.input[0]
    return g, list(set(nodes_to_remove)), new_output_list


def reorder_concat_nodes(g, nodes):
    new_nodes = []
    for n_m in nodes:
        if g.nodemap[n_m].op_type == "Concat":
            print(len(g.nodemap[n_m].input))
            if len(g.nodemap[n_m].input) == 2:
                print(g.tensormap[g.nodemap[n_m].input[0]].shape)
                print(g.tensormap[g.nodemap[n_m].input[1]].shape)
            for inp in g.nodemap[n_m].input:
                new_nodes.append(g.producedby[inp][0])
            new_nodes.append(n_m)
            new_nodes.append(g.nodemap[n_m].nextnodes[0].name)

    return new_nodes


def duplicate_layer(m, g, node_optype, save=False):
    """
    Duplicate a layer with multiple outputs, to facilitate fusion at a node with multiple outputs
    """

    node_keys = list(g.nodemap.keys())

    for node_name in node_keys:
        if (g.nodemap[node_name].op_type == node_optype) and (
            len(g.nodemap[node_name].nextnodes) > 1
        ):
            node = g.nodemap[node_name]

            orig_out_tensor = g.tensormap[node.output[0]]
            for i in range(1, len(node.nextnodes)):
                """
                1.Create new node
                2.new node's next node will be one of the next nodes
                3. Add new node in nodemap
                """

                new_node = copy.copy(node)
                new_node.name = node_name + "__" + str(i)
                new_node.nextnodes = [node.nextnodes[i]]
                if new_node.name not in g.nodemap.keys():
                    g.nodemap[new_node.name] = new_node
                """
                    4. Create output tensor
                    5. add it to tensormap
                    6. add output tensor name in new_node's output
                    """
                new_output_tensor = copy.copy(orig_out_tensor)
                new_output_tensor.name = orig_out_tensor.name + "_" + str(i)
                new_node.output = [new_output_tensor.name]
                """
                    7. Add tensor in tensormap
                    8. update preoducedby[new_tensor] with new_node
                    9. update consumedby[new_tensor] with one of the next nodes (ith node's name)
                    10. update the corresponding input of the consumer node with the created output tensor
                    """
                if new_output_tensor.name not in g.tensormap.keys():
                    g.tensormap[new_output_tensor.name] = new_output_tensor

                if new_output_tensor.name not in g.producedby:
                    g.producedby[new_output_tensor.name] = [new_node.name]
                # print(node.nextnodes[i].name)

                if new_output_tensor.name not in g.consumedby:
                    g.consumedby[new_output_tensor.name] = [node.nextnodes[i].name]

                    con_node = g.nodemap[node.nextnodes[i].name]
                    for j in range(len(con_node.input)):
                        if (
                            con_node.input[j] == node.output[0]
                        ):  # check old node's output
                            con_node.input[j] = new_node.output[
                                0
                            ]  # update new node's output
                            # new node's output consumed by update
                """
                    11. Update the consumed by of input tensor of the new_node ( currently it has the old node only )
                    """
                input_tensor_to_orig_node = node.input[0]

                g.consumedby[input_tensor_to_orig_node].append(new_node.name)
                """
                    12. update the prevnode's nextnodes
                    """
                if node.prevnodes:
                    prevnode = node.prevnodes[0]

                    prevnode.nextnodes.extend([new_node])

            zerothnextnode = node.nextnodes[0]
            node.nextnodes = [zerothnextnode]
            node.name = node_name

    g.graph_reorder_nodes()
    if save:
        g.save_model("PSF_v1.0_QReshape_dup.onnx", rawmodel=m.mproto)
    return g


def change_output_dtype(g):
    """
    Change the data type of output of Quantize and Dequantize layers according to zp and scale
    """
    nodes = g.nodemap.keys()
    for node_name in nodes:
        node = g.nodemap[node_name]
        if node.op_type == "QuantizeLinear":
            for input in node.input:
                if "z_p" in input or "zero_point" in input:
                    data_type = g.tensormap[input].numpy.dtype
                else:
                    data_type = np.int8
            g.tensormap[node.output[0]].dtype = data_type
        if node.op_type == "DequantizeLinear":
            for input in node.input:
                # if node.prevnodes==[] and g.tensormap[input].dtype!=np.int8 and 'scale' not in input and 'zero_point' not in input:
                #     g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)

                # if "zp" in input or "zero_point" in input:
                #     data_type = g.tensormap[input].dtype
                # g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)
                if "scale" in input:
                    data_type = g.tensormap[input].dtype
                else:
                    data_type = np.float32
            g.tensormap[node.output[0]].dtype = data_type
    return g


def removenode(g, op_type):
    rm_node_list = []
    for node in g.nodemap:
        if g.nodemap[node].op_type == op_type:
            rm_node_list.append(g.nodemap[node].name)

    for rm_node in rm_node_list:
        g.skip_node(rm_node)
        g.graph_reorder_nodes()
    return g


def loadmodel(input_path):
    m = onnx_tool.Model(input_path)
    g = m.graph
    g.graph_reorder_nodes()
    return m, g


def count_ops(g):
    # Utility to count op_types
    # Takes onnx_tool graph object
    # should load the model using loadmodel and pass g to this function
    # Return a dictionary
    op_count_dictionary = {}
    for node in g.nodemap:
        if g.nodemap[node].op_type in op_count_dictionary:
            op_count_dictionary[g.nodemap[node].op_type] += 1
        else:
            op_count_dictionary[g.nodemap[node].op_type] = 1
    return op_count_dictionary


def add_domain(g):
    # Utility to add domain name com.amd for each fused QOP
    ops = [
        "QMatMul",
        "QMatMulAdd",
        "QMatMulAddGelu",
        "QMHAGRPB",
        "QLayerNorm",
        "QSkipAdd",
        "QELWEMUL_qdq",
        "QGroupNorm",
        "QGrpNormTrans",
        "QConv",
        "IConv",
        "QConcateOPs",
        "QLstm",
        "QMHACHANNEL",
        "QMHAWINDOW",
        "QReshapeTranspose",
        "QGlobalAvgPool",
        "QGemmv",
        "DQAdd",
        "QMHA",
        "mzdk5MHA",
        "QConv2MatMul",
        "QSilu",
        "QSlice",
        "QConcat",
        "QResize",
        "QGelu",
        "QMatMulDynamic",
        "QMulSoftmax",
        "QBroadcastAdd",
        "DeQuantOP",
        "QuantOP",
        "xcom-conv2d",
        "Mladfsoftmax",
        "MLADFMATMULA16W8",
        "MLADFMATMULA16A16",
        "Mladfelwmul",
    ]
    for name, node in g.nodemap.items():
        if node.op_type in ops:
            node.domain = "com.amd"
    return g


def get_node_names(g, op_type):
    return g.nodemap.keys()


def get_savepath(model_name, key=None):
    curr_path = os.getcwd()
    if key:
        if not os.path.exists(os.path.join(curr_path, model_name, key, "subgraphs")):
            os.makedirs(os.path.join(curr_path, model_name, key, "subgraphs"))
        save_path = os.path.join(curr_path, model_name, key, "subgraphs")
        if not os.path.exists(
            os.path.join(curr_path, model_name, key, "fused_subgraphs")
        ):
            os.makedirs(os.path.join(curr_path, model_name, key, "fused_subgraphs"))
        fuse_path = os.path.join(curr_path, model_name, key, "fused_subgraphs")
        return save_path, fuse_path
    else:
        if not os.path.exists(os.path.join(curr_path, model_name, "synth_models")):
            os.makedirs(os.path.join(curr_path, model_name, "synth_models"))
        return os.path.join(curr_path, model_name, "synth_models")


def get_layer_name_for_save(model_name, found_nodes, key, count=0):
    save_path, fuse_path = get_savepath(model_name, key)
    if "Silu" in key:
        st = found_nodes[0]
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "MatMul" in key:
        st = found_nodes[2]

        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QMHA" == key:
        st = found_nodes[2]

        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "LayerNorm" in key:
        st = found_nodes[3]
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "MHAGRPB" in key:
        st = "MHAGRPB_Layer" + str(count)
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "SkipAdd" in key:
        st = found_nodes[2]
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "conv" in key.lower():
        st = found_nodes[3]
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QReshapeTranspose" in key:
        st = found_nodes[1]  # First Reshape
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QGlobalAvgPool" in key:
        st = found_nodes[7]  # Global average pool node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QGelu" == key:
        st = found_nodes[1]  # First Reshape
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QGroupNorm" in key:
        st = found_nodes[0]  # Gemm node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QSlice" in key:
        st = found_nodes[0]  # Slice node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QConcat" in key:
        st = found_nodes[0]  # Concat node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QuantOP" in key:
        st = found_nodes[0]  # Concat node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "DeQuantOP" in key:
        st = found_nodes[0]  # Concat node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QResize" in key:
        st = found_nodes[0]  # Resize node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QGemmv" in key:
        st = found_nodes[3]  # Gemm node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QMHAWINDOW" in key or "QMHACHANNEL" in key:
        st = key + "_" + str(count)
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "mzdk5MHA" in key:
        st = key + "_" + str(count)
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "DQAdd" in key or "Broadcast" in key:
        st = found_nodes[-1]  # Add node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "QELWEMUL_qdq" in key:
        st = found_nodes[2]  # Add node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "Mladfsoftmax" in key:
        st = found_nodes[2]  # Add node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "Mladfelwmul" in key:
        st = found_nodes[2]  # Add node name
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )


def check_datatypes(m, g, prompt_len, precision, conv_key):
    # This function only works for a8w8
    # TODO add support for A16W8
    #   - Move the if precision  condition inside each optype
    verbose = False  # Set to True to enable prints
    dtype_list = []
    if precision == "a8w8":
        for n_m in g.nodemap.keys():
            layer_dict = {}
            node = g.nodemap[n_m]

            if "LayerNorm".lower() in node.op_type.lower():
                if g.tensormap[node.input[0]].dtype != np.uint16:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Input"
                    layer_dict["tensor name"] = node.input[0]
                    layer_dict["dtype"] = g.tensormap[node.input[0]].dtype
                    layer_dict["Expected dtype"] = "uint16"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

                if g.tensormap[node.output[0]].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint8"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

            elif "MHAGRPB" in node.op_type:
                for inp in node.input[:3]:
                    if g.tensormap[inp].dtype != np.uint8:
                        layer_dict["Op Type"] = node.op_type
                        layer_dict["layer name"] = n_m
                        layer_dict["inp/op"] = "Input"

                        layer_dict["tensor name"] = inp
                        layer_dict["dtype"] = g.tensormap[inp].dtype
                        layer_dict["Expected dtype"] = "uint8"

                        dtype_list.append(layer_dict)
                        layer_dict = {}
                if g.tensormap[node.input[3]].dtype != np.uint16:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Input"

                    layer_dict["tensor name"] = inp
                    layer_dict["dtype"] = g.tensormap[inp].dtype
                    layer_dict["Expected dtype"] = "uint16"

                    dtype_list.append(layer_dict)
                    layer_dict = {}

                if g.tensormap[node.output[0]].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint8"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

            elif "SkipAdd".lower() in node.op_type.lower():
                for inp in node.input[0:2]:
                    if g.tensormap[inp].dtype != np.uint8:
                        layer_dict["Op Type"] = node.op_type
                        layer_dict["layer name"] = n_m
                        layer_dict["inp/op"] = "Input"

                        layer_dict["tensor name"] = inp
                        layer_dict["dtype"] = g.tensormap[inp].dtype
                        layer_dict["Expected dtype"] = "uint8"

                        dtype_list.append(layer_dict)
                        layer_dict = {}
                if g.tensormap[node.output[0]].dtype != np.uint16:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint16"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

            elif "MatMul".lower() in node.op_type.lower():
                inp = node.input[0]
                if g.tensormap[inp].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Input"

                    layer_dict["tensor name"] = inp
                    layer_dict["dtype"] = g.tensormap[inp].dtype
                    layer_dict["Expected dtype"] = "uint8"

                    dtype_list.append(layer_dict)
                    layer_dict = {}
                if g.tensormap[node.output[0]].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint8"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

        #### Enable for debugging
        # if len(dtype_list):
        #     if verbose:
        #         # print(dtype_list)
        #         keys = dtype_list[0].keys()
        #         with open("unsupported_dtype.csv", "w", newline="") as output_file:
        #             dict_writer = csv.DictWriter(output_file, keys)
        #             dict_writer.writeheader()
        #             dict_writer.writerows(dtype_list)

        #     if verbose:
        #         print(
        #             Fore.RED
        #             + f"Model has unsupported data types, please refer to 'unsupported_dtype.csv' "
        #         )
        #         print(Fore.GREEN + f"Converting unsupported datatypes ...")

    # if force_dtype_change:
    g = update_output_shapes_dtypes(m, g, prompt_len, precision, conv_key)
    return g


def get_NCHW(shape):
    if not isinstance(shape, list):
        [N, C, H, W] = list(shape)
    else:
        [N, C, H, W] = shape

    if shape[-1] == shape[-2]:  # NCHW
        H, W = shape[-2], shape[-1]
        C = shape[1]
        N = shape[0]
    elif shape[1] == shape[2]:  # NHWC
        H, W = shape[1], shape[2]
        C = shape[-1]
        N = shape[0]
    else:
        # Need to add a functionality if H!=W
        pass
    return N, C, H, W


def update_output_shapes_dtypes(m, g, prompt_len, precision, conv_key):
    count = 0
    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].domain == "com.amd":
            if "gemm" in g.nodemap[n_m].op_type.lower():
                g.nodemap[n_m].op_type = "QConv2MatMul"
            if "skipadd" in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                if "model" in node.attr.keys() and node.attr["model"] == "mzdk5":
                    g.tensormap[node.input[0]].dtype = np.uint16
                    g.tensormap[node.output[0]].dtype = np.uint16

    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].domain == "com.amd":
            node = g.nodemap[n_m]
            if "mzdk5MHA".lower() in node.op_type.lower():
                [N, C, H, W] = g.tensormap[node.input[0]].shape
                g.tensormap[node.input[0]].shape = [N * C * H, W]
                [N, C, H, W] = g.tensormap[node.input[1]].shape
                g.tensormap[node.input[1]].shape = [W, H * N * C]
                [N, C, H, W] = g.tensormap[node.input[2]].shape
                g.tensormap[node.input[2]].shape = [N * C * H, W]
                parents = node.prevnodes

            if "QGelu" == node.op_type:
                g.tensormap[node.output[0]].dtype = "BFLOAT16"

            if "QSlice" == node.op_type:
                g.tensormap[node.output[0]].dtype = "BFLOAT16"
            if "QConcat" == node.op_type:
                g.tensormap[node.output[0]].dtype = "BFLOAT16"

            if "DeQuantOP" == node.op_type:
                g.tensormap[node.output[0]].dtype = "BFLOAT16"

            if "QGroupNorm" == node.op_type:
                # g.tensormap[node.input[0]].dtype = "BFLOAT16"
                # g.tensormap[node.output[0]].dtype = "BFLOAT16"
                [N, C, H, W] = get_NCHW(g.tensormap[node.input[0]].shape)
                g.tensormap[node.input[0]].shape = [N, H, W, C]
                g.tensormap[node.output[0]].shape = [N, H, W, C]

            if "QGrpNormTrans" == node.op_type:
                [N, C, H, W] = get_NCHW(g.tensormap[node.input[0]].shape)
                g.tensormap[node.input[0]].shape = [N, H, W, C]
                g.tensormap[node.output[0]].shape = [1, H * W, C]
                node.op_type = "QGroupNorm"

            if "conv2matmul".lower() in node.op_type.lower():
                g.tensormap[node.input[0]].dtype = np.uint16
                g.consumedby[node.input[0]] = node.name
                if len(g.tensormap[node.input[0]].shape) == 4:
                    [N, C, H, W] = get_NCHW(g.tensormap[node.input[0]].shape)
                    g.tensormap[node.input[0]].shape = [N, H, W, C]
                node.set_attr("input_format", "NHWC")
            if (
                "conv".lower() in node.op_type.lower()
                and "matmul" not in node.op_type.lower()
            ):
                if (
                    "model" in node.attr.keys() and node.attr["model"] == "mzdk5"
                ):  # For case where IConv in mzdk5 needs to be converted to Conv2MatMul
                    if "from_iconv" in node.attr.keys():
                        node.op_type = "QConv2MatMul"
                if len(g.tensormap[node.input[0]].shape) == 4:
                    [N, C, H, W] = get_NCHW(g.tensormap[node.input[0]].shape)
                    g.tensormap[node.input[0]].shape = [N, H, W, C]
                if len(g.tensormap[node.output[0]].shape) == 4:
                    [N, C, H, W] = get_NCHW(g.tensormap[node.output[0]].shape)
                    g.tensormap[node.output[0]].shape = [N, H, W, C]
                node.set_attr("input_format", str("NHWC"))

            elif "QConcateOPs".lower() in node.op_type.lower():
                orig_input_shape = g.tensormap[g.input[0]].shape
                # breakpoint()
                pad_shape = []
                pad_shape.append(orig_input_shape[2])
                pad_shape.append(orig_input_shape[0])
                pad_shape.append(orig_input_shape[3])
                if orig_input_shape[3] != 320:
                    pad_shape.append(4)
                else:
                    pad_shape.append(8)

                g.tensormap[node.input[0]].shape = pad_shape
                g.tensormap[node.input[0]].dtype = np.uint16

                orig_output_shape = g.tensormap[node.output[0]].shape

                op_pad_shape = []
                op_pad_shape.append(orig_output_shape[2])
                op_pad_shape.append(orig_output_shape[0])
                op_pad_shape.append(orig_output_shape[3])
                op_pad_shape.append(orig_output_shape[1])
                g.tensormap[node.output[0]].shape = op_pad_shape
            elif "QLayerNorm".lower() in node.op_type.lower():
                if precision == "a8w8":
                    g.tensormap[node.input[0]].dtype = "BFLOAT16"
                    for output in node.output:
                        g.tensormap[output].dtype = np.uint8
                # mxgan and mxpzi shoul go to else
                elif precision == "a16w8":
                    lrn_is_uint16 = node.attr["lrn_is_uint16"]
                    if lrn_is_uint16 == 0:
                        g.tensormap[node.input[0]].dtype = "BFLOAT16"
                    else:
                        g.tensormap[node.input[0]].dtype = np.uint16

                    for output in node.output:
                        g.tensormap[output].dtype = np.uint16
                    # if "model" in node.attr.keys() and node.attr["model"] == "mzdk5":
                    #     if g.tensormap[node.input[0]].shape == [
                    #         1,
                    #         64,
                    #         64,
                    #         320,
                    #     ] or g.tensormap[node.input[0]].shape == [1, 4096, 320]:
                    #         node.domain = ""

            if "silu".lower() in node.op_type.lower():
                g.tensormap[node.input[0]].dtype = "BFLOAT16"

            elif "MHAGRPB" in node.op_type:
                if precision == "a8w8":
                    g.tensormap[node.input[3]].dtype = "BFLOAT16"
                    g.tensormap[node.output[0]].dtype = np.uint8
                elif precision == "a16w8":
                    g.tensormap[node.input[3]].dtype = "BFLOAT16"

                    g.tensormap[node.output[0]].dtype = np.uint16

            elif "SkipAdd".lower() in node.op_type.lower():
                if precision == "a8w8":
                    for inp in node.input:
                        g.tensormap[inp].dtype = np.uint8
                    g.tensormap[node.output[0]].dtype = "BFLOAT16"

                elif precision == "a16w8":
                    ### Go back to this if some error occurs
                    # for inp in node.input[:1]:
                    #     if inp in g.producedby:
                    #         producers = g.producedby[inp]

                    #         # if prevnode is add, the current node should not be offloaded
                    #         for p in producers:
                    #             if "SkipAdd".lower() in g.nodemap[p].op_type.lower():
                    #                 g.nodemap[n_m].domain = ""

                    # # if any of the next node is not layernorm, the curent add node should not be offloaded
                    # for nextnode in g.nodemap[n_m].nextnodes:
                    #     if "layernorm".lower() not in nextnode.op_type.lower():
                    #         g.nodemap[n_m].domain = ""

                    # if g.nodemap[n_m].domain == "com.amd":
                    #     for inp in node.input:
                    #         g.tensormap[inp].dtype = np.uint16
                    #     g.tensormap[node.output[0]].dtype = "BFLOAT16"
                    if "model" in node.attr.keys() and node.attr["model"] != "mzdk5":
                        output_uint16_flag = node.attr["cmat_uint16"]
                        input_uint16_flag = node.attr["amat_uint16"]
                        if output_uint16_flag:
                            g.tensormap[node.output[0]].dtype = np.uint16
                        else:
                            g.tensormap[node.output[0]].dtype = "BFLOAT16"

                        if input_uint16_flag:
                            for inp in node.input:
                                g.tensormap[inp].dtype = np.uint16
                        else:
                            for inp in node.input[:1]:
                                producers = g.producedby[inp]
                                for p in producers:
                                    if (
                                        "SkipAdd".lower()
                                        in g.nodemap[p].op_type.lower()
                                    ):
                                        g.tensormap[inp].dtype = "BFLOAT16"
            elif (
                "MatMul".lower() in node.op_type.lower()
                and "conv" not in node.op_type.lower()
                and "dynamic" not in node.op_type.lower()
            ):
                if precision == "a8w8":
                    g.tensormap[node.input[0]].dtype = np.uint8
                    M = (
                        g.tensormap[node.input[0]].shape[0]
                        * g.tensormap[node.input[0]].shape[1]
                    )
                    N = g.tensormap[node.input[1]].shape[-1]

                    # A8W8 Matmul requires second dim to be atleast 128.
                    # Currently PSF model last matmul has shape 512x26
                    # This has to be padded to 512x128.
                    A8W8_Matmul_Min_N = 128
                    if N < A8W8_Matmul_Min_N:
                        N = A8W8_Matmul_Min_N
                        wts = g.tensormap[node.input[1]].numpy.copy()
                        padded_wts = np.pad(
                            wts, ((0, 0), (0, A8W8_Matmul_Min_N - wts.shape[-1]))
                        )
                        g.tensormap[
                            node.input[1]
                        ] = onnx_tool.tensor.create_initial_Tensor(
                            node.input[1], padded_wts
                        )

                        bias = g.tensormap[node.input[2]].numpy.copy()
                        padded_bias = np.pad(
                            bias, (0, A8W8_Matmul_Min_N - bias.shape[-1])
                        )
                        g.tensormap[
                            node.input[2]
                        ] = onnx_tool.tensor.create_initial_Tensor(
                            node.input[2], padded_bias
                        )

                    g.tensormap[node.output[0]].shape = (1, M, N)

                    g.tensormap[node.output[0]].dtype = np.uint8

                elif precision == "a16w8":
                    g.tensormap[node.input[0]].dtype = np.uint16
                    if (
                        g.tensormap[node.input[0]].shape[0] == 1
                        and len(g.tensormap[node.input[0]].shape) == 3
                    ):
                        M = g.tensormap[node.input[0]].shape[1]

                        N = g.tensormap[node.input[1]].shape[-1]
                        g.tensormap[node.output[0]].shape = (1, M, N)
                    elif (
                        g.tensormap[node.input[0]].shape[0] == 1
                        and len(g.tensormap[node.input[0]].shape) == 2
                    ):
                        M = g.tensormap[node.input[0]].shape[-1]
                        N = g.tensormap[node.input[1]].shape[-1]
                        g.tensormap[node.output[0]].shape = (
                            1,
                            g.tensormap[node.output[0]].shape[0],
                            N,
                        )
                        g.tensormap[node.input[0]].shape = (
                            1,
                            g.tensormap[node.input[0]].shape[0],
                            M,
                        )
                    elif len(g.tensormap[node.input[0]].shape) == 4:
                        M = (
                            g.tensormap[node.input[0]].shape[1]
                            * g.tensormap[node.input[0]].shape[2]
                        )
                        N = g.tensormap[node.input[1]].shape[-1]
                        K = g.tensormap[node.input[0]].shape[-1]

                        g.tensormap[node.input[0]].shape = [1, M, K]
                        g.tensormap[node.output[0]].shape = [1, M, N]

                    else:
                        M = (
                            g.tensormap[node.input[0]].shape[0]
                            * g.tensormap[node.input[0]].shape[1]
                        )
                        N = g.tensormap[node.input[1]].shape[-1]
                        g.tensormap[node.output[0]].shape = (
                            g.tensormap[node.input[0]].shape[0],
                            g.tensormap[node.input[0]].shape[1],
                            N,
                        )
                    g.tensormap[node.output[0]].dtype = np.uint16

            elif "DQAdd".lower() in node.op_type.lower():
                g.tensormap[node.output[0]].dtype = "BFLOAT16"
    for n_m in g.nodemap:
        node = g.nodemap[n_m]
        if g.nodemap[n_m].domain == "com.amd":
            if g.nodemap[n_m].op_type == "QGroupNorm":
                if g.tensormap[node.input[0]].dtype == "BFLOAT16":
                    g.tensormap[node.input[3]].numpy[5] = 0
                    g.tensormap[node.input[3]].numpy[4] = 0
                else:
                    g.tensormap[node.input[3]].numpy[5] = 1
                if g.tensormap[node.output[0]].dtype == "BFLOAT16":
                    g.tensormap[node.input[3]].numpy[2] = 0
                else:
                    g.tensormap[node.input[3]].numpy[2] = 1
            # if g.nodemap[n_m].op_type == "LayerNorm":
            if "LayerNorm" in g.nodemap[n_m].op_type:
                if g.tensormap[node.input[0]].dtype == "BFLOAT16":
                    g.tensormap[node.input[3]].numpy[5] = 0
                    g.tensormap[node.input[3]].numpy[4] = 0
    for n_m in g.nodemap:
        if g.nodemap[n_m].domain == "com.amd":
            if g.nodemap[n_m].op_type == "QConv2MatMul":
                if len(g.nodemap[n_m].nextnodes):
                    next_node = g.nodemap[n_m].nextnodes[0]
                    if next_node.domain == "":
                        g.nodemap[n_m].domain = ""
                    else:
                        g.consumedby[g.nodemap[n_m].input[0]] = [n_m]
                        g.tensormap[g.nodemap[n_m].input[0]].dtype = np.uint16

                if (
                    "from_iconv" in g.nodemap[n_m].attr
                ):  # The conv2matmuls that are comming from iconv, we can enable or disable them here
                    from_iconv = g.nodemap[n_m].attr["from_iconv"]
                    if from_iconv == 1:
                        g.nodemap[n_m].domain = "com.amd"
                # if g.tensormap[g.nodemap[n_m].input[0]].shape == [
                #     1,
                #     77,
                #     1024,
                # ]:  # TODO : remove this hardcoding
                #     g.tensormap[g.nodemap[n_m].input[0]].shape = [1, 128, 1040]
            ####################Skip Add Changes for mzdk5 model ###################
            # Add can support bf16-uint16-bf16,uin16-uint16-bf16(default),uint16-uint16-uint16
            # In the first loop above, Add is set to uint16 and uint16,
            # In the 2nd loop, all the other ops's dtypes are adjusted (this would change Add's inp or output dtypes)
            # Based on the changed input and output datatypes, In this current loop we adjust Add's qdq tensor coefficients
            if "SkipAdd".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                if "model" in node.attr.keys():
                    if node.attr["model"] == "mzdk5":
                        inp1_dtype = g.tensormap[node.input[0]].dtype
                        inp2_dtype = g.tensormap[node.input[1]].dtype
                        out_dtype = g.tensormap[node.output[0]].dtype
                        elt_qdq_tensor = g.tensormap[node.input[2]].numpy
                        if inp1_dtype == "BFLOAT16":
                            elt_qdq_tensor[6] = 0  # bf16
                        else:
                            elt_qdq_tensor[6] = 1
                        if inp2_dtype == "BFLOAT16":
                            elt_qdq_tensor[8] = 0  # bf16
                        else:
                            elt_qdq_tensor[8] = 1
                        if out_dtype == "BFLOAT16":
                            elt_qdq_tensor[7] = 0  # bf16
                        else:
                            elt_qdq_tensor[7] = 1
                        g.tensormap[node.input[2]].numpy = elt_qdq_tensor
                        if len(g.tensormap[node.output[0]].shape) == 4:
                            [N, C, H, W] = get_NCHW(g.tensormap[node.output[0]].shape)
                            g.tensormap[node.output[0]].shape = [N, H, W, C]
                        if len(g.tensormap[node.output[0]].shape) == 3:
                            N = g.tensormap[node.output[0]].shape[0]
                            HW = g.tensormap[node.output[0]].shape[1]
                            C = g.tensormap[node.output[0]].shape[2]
                            g.tensormap[node.output[0]].shape = [N, HW, C]

                        # if inp2_dtype == "BFLOAT16" and inp1_dtype != "BFLOAT16":
                        #     node.input[0], node.input[1] = node.input[1], node.input[0]
                        #     node.prevnodes[0], node.prevnodes[1] = (
                        #         node.prevnodes[1],
                        #         node.prevnodes[0],
                        #     )
                        #     elt_qdq_tensor = g.tensormap[node.input[2]].numpy
                        #     elt_qdq_tensor[6] = 0  # bf16
                        #     if out_dtype == "BFLOAT16":
                        #         elt_qdq_tensor[7] = 0  # bf16
                        #     else:
                        #         elt_qdq_tensor[7] = 1
                        # elif inp1_dtype == "BFLOAT16":
                        #     elt_qdq_tensor = g.tensormap[node.input[2]].numpy
                        #     elt_qdq_tensor[6] = 0  # bf16
                        #     if out_dtype == "BFLOAT16":
                        #         elt_qdq_tensor[7] = 0  # bf16
                        #     else:
                        #         elt_qdq_tensor[7] = 1
                        # elif inp2_dtype != "BFLOAT16" and inp1_dtype != "BFLOAT16":
                        #     elt_qdq_tensor = g.tensormap[node.input[2]].numpy
                        #     elt_qdq_tensor[6] = 1  # UINT16
                        #     elt_qdq_tensor[8] = 1  # UINT16
                        #     if out_dtype == "BFLOAT16":
                        #         elt_qdq_tensor[7] = 0  # UINT16
                        #     else:
                        #         elt_qdq_tensor[7] = 1
                        #     g.tensormap[node.input[2]].numpy = elt_qdq_tensor

            if g.nodemap[n_m].op_type == "IConv":  # mzdk5 1st conv
                if g.tensormap[g.nodemap[n_m].input[0]].shape == [
                    1,
                    224,
                    224,
                    3,
                ]:  # TODO : remove this hardcoding
                    g.tensormap[g.nodemap[n_m].input[0]].shape = [1, 3, 224, 224]
                    node = g.nodemap[n_m]
                    node.set_attr("input_format", str("NCHW"))

    return g


def get_mha_inputs(g, Mha_input_dict):
    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].domain == "com.amd":
            node = g.nodemap[n_m]
            # search for inputs of MHAGRPB (on the basis of matmul shapes)
            # If matmul add --- 2shapes
            #   --shape with 1152 N dim -> Query
            #   --Shape with 768 N dim -> Value
            # if MatMul only --1 shape
            #   -- shape with 1152 N dim-> Key
            if (
                "MatMulAdd".lower() in node.op_type.lower()
                and "Gelu".lower() not in node.op_type.lower()
            ):
                if (
                    len(node.nextnodes)
                    and "MHAGRPB".lower() in node.nextnodes[0].op_type.lower()
                ):
                    if node.nextnodes[0].name not in Mha_input_dict.keys():
                        Mha_input_dict[node.nextnodes[0].name] = {}
                    if g.tensormap[node.output[0]].shape[2] == 1152:
                        Mha_input_dict[node.nextnodes[0].name]["Q"] = node.output[0]
                    else:
                        Mha_input_dict[node.nextnodes[0].name]["V"] = node.output[0]

            elif (
                "MatMul".lower() in node.op_type.lower()
                and "gelu" not in node.op_type.lower()
                and "add" not in node.op_type.lower()
            ):
                if (
                    len(node.nextnodes) > 0
                    and "MHAGRPB".lower() in node.nextnodes[0].op_type.lower()
                ):
                    if node.nextnodes[0].name not in Mha_input_dict.keys():
                        Mha_input_dict[node.nextnodes[0].name] = {}
                    if g.tensormap[node.output[0]].shape[2] == 1152:
                        Mha_input_dict[node.nextnodes[0].name]["K"] = node.output[0]

                    for inp in node.input:
                        if len(g.tensormap[inp].shape) == 1:
                            actual_tensor_bias = g.tensormap[inp].numpy.data
                            pad_tensor_bias = np.zeros((1, g.tensormap[inp].shape[0]))
                            pad_tensor_bias[
                                :, : actual_tensor_bias.shape[0]
                            ] = actual_tensor_bias
                            g.tensormap[inp] = onnx_tool.tensor.create_initial_Tensor(
                                inp,
                                pad_tensor_bias,
                            )
                            g.initials.append(inp)

    return g, Mha_input_dict


def get_node_dtype(g, nodename):
    node = g.nodemap[nodename]
    if "matmul" in node.op_type.lower():
        inputs = node.input
        out_dtype = g.tensormap[inputs[-1]].dtype
        in_dtype = g.tensormap[inputs[0]].dtype

    elif "MHAGRPB".lower() in node.op_type.lower():
        mha_input_dict = {}
        mha_input_dict = get_mha_inputs(g, mha_input_dict)
        inputs = node.input
        in_dtype = g.tensormap[inputs[0]].dtype

        out_dtype = g.tensormap[inputs[-1]].dtype

    elif "Layernorm".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype
        in_dtype = g.tensormap[node.input[0]].dtype
    elif (
        "QGroupNorm".lower() in node.op_type.lower()
        or "QGrpNormTrans".lower() in node.op_type.lower()
    ):
        out_dtype = g.tensormap[node.input[-1]].dtype
    elif "QELWEMUL_qdq".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype
    elif "Conv".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype
    elif "Gemmv".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype
    elif "QConcateOPs".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype

    elif (
        "MHACHANNEL" in node.op_type
        or "MHAWINDOW" in node.op_type
        or "MHA" in node.op_type
    ):
        out_dtype = g.tensormap[node.input[-1]].dtype
    elif "QGelu" == node.op_type:
        out_dtype = g.tensormap[node.input[-1]].dtype

    elif "QMulSoftmax" in node.op_type:
        out_dtype = g.tensormap[node.input[-1]].dtype

    elif "Mladfsoftmax".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype

    elif "Mladfelwmul".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype

    return out_dtype


def concat_wgts_mswbjvw(m, g, wts_name_list):
    # Pad initializers
    actual_tensor_izr1 = g.tensormap[wts_name_list[0]].numpy.data
    padded_tensor_izr1 = np.zeros((16, 8, 3, 3)).astype(np.uint16)
    padded_tensor_izr1[:, : actual_tensor_izr1.shape[1], :, :] = actual_tensor_izr1
    g.tensormap[wts_name_list[0]] = onnx_tool.tensor.create_initial_Tensor(
        wts_name_list[0], padded_tensor_izr1
    )

    return g, wts_name_list[0]


def add_concat_qdq(
    g, n_m, scale, zp
):  # if next node is concat, change the output scale and zp
    return scale, zp
    try:
        nextnode = g.nodemap[g.consumedby[g.nodemap[n_m].output[0]][0]]
        if nextnode.op_type == "QConcat":  # next node is Qconcat that means it is fused
            # breakpoint()
            concat_inputs = nextnode.input
            concat_initializers = 0
            for inp in concat_inputs:
                # count no. of initializers, if high initializers, that means concat op did not pass the change_inputs function, so collect the scale and zp from note #inputs itself or else collect them from "output_q_params" attribute
                if inp in g.initials:
                    concat_initializers += 1

            if concat_initializers > 2:
                scale = g.tensormap[concat_inputs[-2]].numpy
                zp = g.tensormap[concat_inputs[-1]].numpy
            else:
                concat_output_q_params = nextnode.attr["output_q_params"]
                scale = concat_output_q_params[0]
                zp = concat_output_q_params[1]
            g.nodemap[n_m].set_attr("modified_output_scale", scale.item())
            g.nodemap[n_m].set_attr("modified_output_zp", zp.item())

        elif (
            nextnode.op_type == "DequantizeLinear"
            and nextnode.nextnodes[0].op_type == "Concat"
        ):  # if concat op is not fused then collect sc&zp by traversing to the required node
            scale = g.tensormap[nextnode.nextnodes[0].nextnodes[0].input[1]].numpy
            zp = g.tensormap[nextnode.nextnodes[0].nextnodes[0].input[2]].numpy
            g.nodemap[n_m].set_attr("modified_output_scale", scale.item())
            g.nodemap[n_m].set_attr("modified_output_zp", zp.item())
    except:
        scale = scale
        zp = zp
        print(f"no change in scale and zp {scale} {zp}")

    return scale, zp


def change_inputs(m, g, precision, conv_key):
    remove_list = []
    Mha_input_dict = {}
    graph_input = g.tensormap[g.input[0]].shape

    prompt_len = graph_input[1]
    if prompt_len == 77:
        prompt_len_modified = 128
    else:
        prompt_len_modified = prompt_len
    # get a dictionary of MHA layers with respective Q,K,V input names
    g, Mha_input_dict = get_mha_inputs(g, Mha_input_dict)

    ## Below snippet adds QDQ params to nodes
    # Each optype has differnt QDQ coefficients that are packed into QDQ tensors and passed as input to the node

    mha_count = 0
    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].domain == "com.amd":
            if "ReshapeTranspose".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                qdq_tensor = np.zeros((16)).astype(np.int32)
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                node.input = [
                    node.input[0],
                    n_m + "_qdq_",
                ]  # Remove all the inputs and add a dummy buffer
            if "QGelu" == g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                # TODO Pass appropriate scale and zp
                # Adding qparams as attrs
                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())

                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))

                ## TODO
                ## Implemnet generic way to get scale and zp
                ## This only works for mzdk5
                # quantizeLayer = g.nodemap[n_m].prevnodes[0].prevnodes[0]
                # gelu_in_scale = g.tensormap[quantizeLayer.input[1]].numpy
                # gelu_in_zp = g.tensormap[quantizeLayer.input[2]].numpy
                # print(f"Gelu scale and zp {gelu_in_scale} {gelu_in_zp}")
                # node.set_attr("modified_input_scale", str(gelu_in_scale))
                # node.set_attr("modified_input_zp", str(gelu_in_zp))

                # gelu_qdq = LRN(gelu_in_scale, gelu_in_zp)
                gelu_qdq = LRN(input_list[0], input_list[1])
                lrn_c0, lrn_c1 = gelu_qdq.cal_coeff()

                gelu_qdq_tensor = np.zeros((16)).astype(np.uint16)
                gelu_qdq_tensor[3] = lrn_c1
                gelu_qdq_tensor[4] = lrn_c0
                gelu_qdq_tensor[5] = 1  # Enable dequant at input

                g.tensormap[n_m + "gelu_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "gelu_qdq_", gelu_qdq_tensor
                )
                g.initials.append(n_m + "gelu_qdq_")

                modified_input.append(node.input[0])
                modified_input.append(n_m + "gelu_qdq_")
                node.input = modified_input
            if "QSlice".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                if len(node.input) == 9:
                    node.set_attr("slice_idx", [1])
                    node.output = [node.output[1]]
                else:
                    node.set_attr("slice_idx", [0])
                qdq_tensor = np.zeros((16)).astype(np.uint8)
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                node.input = [
                    node.input[0],
                    n_m + "_qdq_",
                ]  # Remove all the inputs and add a dummy buffer

            if "QConcat" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                for inp in inputs:
                    if (
                        "scale" not in g.tensormap[inp].name
                        and "zero_point" not in g.tensormap[inp].name
                    ):
                        modified_input.append(inp)
                g.nodemap[n_m].input = []
                g.nodemap[n_m].input = modified_input

            if "QResize".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                qdq_tensor = np.zeros((16)).astype(np.uint8)
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                node.input = [
                    node.input[0],
                    n_m + "_qdq_",
                ]  # Remove all the inputs and add a dummy buffer
            if "QGlobalAvgPool".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                qdq_tensor = np.zeros((32)).astype(np.int32)
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                node.input = [
                    node.input[0],
                    n_m + "_qdq_",
                ]  # Remove all the inputs and add a dummy buffer

            if "QBroadcastAdd" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []
                # input ordering
                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)
                # Adding qparams as attrs
                # Add
                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_inps.append(input_list[2].item())
                convert_params_inps.append(input_list[3].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                eltadd = EltwiseAdd(
                    input_list[0], input_list[1], input_list[2], input_list[3]
                )
                output_coeff = LRN(
                    1 / input_list[-2].item(), input_list[-1].item()
                ).cal_coeff()
                elt_c0, elt_c1, elt_c2, elt_c3 = eltadd.cal_coeff()
                elt_qdq_tensor = np.zeros((32)).astype(np.uint16)
                elt_qdq_tensor[1] = elt_c0
                elt_qdq_tensor[0] = elt_c1
                elt_qdq_tensor[3] = elt_c2
                elt_qdq_tensor[2] = elt_c3
                elt_qdq_tensor[5] = output_coeff[0]
                elt_qdq_tensor[4] = output_coeff[1]

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", elt_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                # if add_in_parent and "skipadd" not in parents[0].op_type.lower():
                #     modified_input[0], modified_input[1] = (
                #         modified_input[0],
                #         modified_input[1],
                #     )
                #     node.prevnodes[0], node.prevnodes[1] = (
                #         node.prevnodes[1],
                #         node.prevnodes[0],
                #     )
                # add in parent and add is the first parent

                modified_input.append(n_m + "_qdq_")
                node.input = modified_input
            if "QGrpNormTrans" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                modified_input.append(node.input[0])
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype
                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                c = 0
                input_dict = {}
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                # print("Len of input list : ", len(input_list))
                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)

                gamma_tensor = dequantize_bf16(args[23], args[24], args[25])
                # gamma_tensor = np.zeros(g.tensormap[g.nodemap[n_m].input[0]].shape[1]).astype(np.int16)
                g.tensormap[n_m + "_gamma_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_gamma_", gamma_tensor
                )
                g.initials.append(n_m + "_gamma_")
                modified_input.append(n_m + "_gamma_")

                beta_tensor = dequantize_bf16(args[30], args[31], args[32])
                # beta_tensor = dequantize_bf16(args[-5], args[-4], args[-3])
                # beta_tensor = np.zeros(g.tensormap[g.nodemap[n_m].input[0]].shape[1]).astype(np.int16)
                g.tensormap[n_m + "_beta_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_beta_", beta_tensor
                )
                g.initials.append(n_m + "_beta_")
                modified_input.append(n_m + "_beta_")

                lrnlayer = LRN(1 / input_list[-2], input_list[-1])

                lrn_c0, lrn_c1 = lrnlayer.cal_coeff()

                lrnlayer_inp = LRN(input_list[0], input_list[1])

                lrn_c0_inp, lrn_c1_inp = lrnlayer_inp.cal_coeff()

                lrn_qdq_tensor = np.zeros((16)).astype(np.int32)
                lrn_qdq_tensor[0] = lrn_c0
                lrn_qdq_tensor[1] = lrn_c1
                lrn_qdq_tensor[2] = 0

                lrn_qdq_tensor[3] = lrn_c0_inp
                lrn_qdq_tensor[4] = lrn_c1_inp
                # Check input data
                lrn_qdq_tensor[5] = 1  # Assume the input is uint16

                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_Dtype", str(node_dtype))

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", lrn_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                # for inp in inputs:
                #     if (
                #         ("scale" not in inp)
                #         and ("zero_point" not in inp)
                #         and ("ort" not in inp)
                #     ):
                #         modified_input.append(inp)
                modified_input.append(n_m + "_qdq_")
                # g.tensormap[modified_input[1]].numpy = gamma_tensor
                # g.tensormap[modified_input[2]].numpy = beta_tensor
                # g.tensormap[modified_input[3]].numpy = lrn_qdq_tensor
                node.input = modified_input
            if "QGroupNorm" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                modified_input.append(node.input[0])
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype
                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                c = 0
                input_dict = {}
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                # print("Len of input list : ", len(input_list))
                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)

                gamma_tensor = dequantize_bf16(args[23], args[24], args[25])
                # gamma_tensor = np.zeros(g.tensormap[g.nodemap[n_m].input[0]].shape[1]).astype(np.int16)
                g.tensormap[n_m + "_gamma_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_gamma_", gamma_tensor
                )
                g.initials.append(n_m + "_gamma_")
                modified_input.append(n_m + "_gamma_")

                beta_tensor = dequantize_bf16(args[-5], args[-4], args[-3])
                # beta_tensor = np.zeros(g.tensormap[g.nodemap[n_m].input[0]].shape[1]).astype(np.int16)
                g.tensormap[n_m + "_beta_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_beta_", beta_tensor
                )
                g.initials.append(n_m + "_beta_")
                modified_input.append(n_m + "_beta_")

                lrnlayer = LRN(1 / input_list[-2], input_list[-1])

                lrn_c0, lrn_c1 = lrnlayer.cal_coeff()

                lrnlayer_inp = LRN(input_list[0], input_list[1])

                lrn_c0_inp, lrn_c1_inp = lrnlayer_inp.cal_coeff()

                lrn_qdq_tensor = np.zeros((16)).astype(np.int32)
                lrn_qdq_tensor[0] = lrn_c0
                lrn_qdq_tensor[1] = lrn_c1
                lrn_qdq_tensor[2] = 0

                lrn_qdq_tensor[3] = lrn_c0_inp
                lrn_qdq_tensor[4] = lrn_c1_inp
                # Check input data
                lrn_qdq_tensor[5] = 1  # Assume the input is uint16

                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_Dtype", str(node_dtype))

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", lrn_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                # for inp in inputs:
                #     if (
                #         ("scale" not in inp)
                #         and ("zero_point" not in inp)
                #         and ("ort" not in inp)
                #     ):
                #         modified_input.append(inp)
                modified_input.append(n_m + "_qdq_")
                # g.tensormap[modified_input[1]].numpy = gamma_tensor
                # g.tensormap[modified_input[2]].numpy = beta_tensor
                # g.tensormap[modified_input[3]].numpy = lrn_qdq_tensor
                node.input = modified_input
            if (
                "QConv".lower() in g.nodemap[n_m].op_type.lower()
                and "QConv2MatMul".lower() not in g.nodemap[n_m].op_type.lower()
            ):
                node = g.nodemap[n_m]
                node_dtype = get_node_dtype(g, n_m)
                if not isinstance(node.attr["wt_name"], str):
                    wt_name = node.attr["wt_name"].decode("utf-8")
                else:
                    wt_name = node.attr["wt_name"]
                node.input = [
                    node.input[0],
                    node.input[node.input.index(wt_name)],
                ]  # Remove all the inputs and store only act input and weights
                node.set_attr("Node_dtype", str(node_dtype))
                node.set_attr("input_format", str("NHWC"))

            if "QConcateOPs".lower() in g.nodemap[n_m].op_type.lower():
                # orig_input_shape = g.tensormap[g.input[0]].shape
                # padded_inp_shape =
                # wts
                node = g.nodemap[n_m]
                # breakpoint()
                node_dtype = get_node_dtype(g, n_m)
                g, wt_name = concat_wgts_mswbjvw(m, g, node.attr["list_wt_name"])

                node.input = [
                    node.input[0],
                ]  # Remove all the inputs and store only act input and weights
                node.input.extend(node.attr["list_wt_name"])
                node.set_attr("Node_dtype", str(node_dtype))

            if "QLstm".lower() in g.nodemap[n_m].op_type.lower():
                # orig_input_shape = g.tensormap[g.input[0]].shape
                # padded_inp_shape =
                # wts
                node = g.nodemap[n_m]
                node_dtype = get_node_dtype(g, n_m)

                node.input = [
                    node.input[0],
                ]  # Remove all the inputs and store only act input and weights
                node.input.extend(node.attr["list_wt_name"])
                node.set_attr("Node_dtype", str(node_dtype))

            if (
                "MatMulAdd".lower() in g.nodemap[n_m].op_type.lower()
                and "gelu" not in g.nodemap[n_m].op_type.lower()
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))

                if node_dtype == np.uint8:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = compute_qdq_coeff_matmul_bias(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[-5],
                        args[-4],
                        args[-3],
                        args[-2],
                        args[-1],
                    )
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    if len(args) == 18:
                        args[-5] = args[-9]
                        args[-4] = args[-8]
                        args[-3] = args[-7]

                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[-5],
                        args[-4],
                        args[-3],
                        args[-2],
                        args[-1],
                    )
                    is_int16 = 1

                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

                node.input.pop(2)

            if "QSilu" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                # print(f"Input list {input_list[-2]} {input_list[-1]}")

                # TODO Pass appropriate scale and zp
                # Adding qparams as attrs
                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_inps.append(input_list[2].item())
                convert_params_inps.append(input_list[3].item())

                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                silu_layer = LRN(1 / input_list[-2], input_list[-1])

                scale, zp = silu_layer.cal_coeff()

                lrn_qdq_tensor = np.zeros((16)).astype(np.int16)
                lrn_qdq_tensor[0] = zp
                lrn_qdq_tensor[1] = scale
                lrn_qdq_tensor[2] = 1  # Enable out quant

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", lrn_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                modified_input.append(node.input[0])
                modified_input.append(n_m + "_qdq_")
                # g.tensormap[modified_input[1]].numpy = scale_tensor
                # g.tensormap[modified_input[2]].numpy = beta_tensor
                node.input = modified_input

            if "QConv2MatMul".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                modified_input = []
                input_dict = OrderedDict()

                node_dtype = get_node_dtype(g, n_m)
                if not isinstance(node.attr["wt_name"], str):
                    wt_name = node.attr["wt_name"].decode("utf-8")
                else:
                    wt_name = node.attr["wt_name"]

                inputs = node.input
                node.set_attr("Node_dtype", str(node_dtype))
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                convert_params_inps = []
                convert_params_ops = []

                convert_params_inps.append(args[0].item())
                convert_params_inps.append(args[1].item())
                convert_params_ops.append(args[-2].item())
                convert_params_ops.append(args[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                if len(args) == 28:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                    ) = dq_uint16A_uint8W_conv_q_param_gen_withoutbias(
                        args[0],
                        args[1],
                        args[11],
                        args[12],
                        args[13],
                        args[17],
                        args[18],
                    )
                    weight_zp = args[13].astype(np.int32)
                elif len(args) == 19:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                    ) = dq_uint16A_uint8W_conv_q_param_gen_withoutbias(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[8],
                        args[9],
                    )
                    weight_zp = args[4].astype(np.int32)

                # this is taken bu Qconv2matmul pattern

                else:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                    ) = dq_uint16A_uint8W_conv_q_param_gen_withoutbias(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[-2],
                        args[-1],
                    )
                    weight_zp = args[4].astype(np.int32)

                node.set_attr("C1", str(c1))
                node.set_attr("C2", str(c2))
                node.set_attr("shift_conv", str(shift_qb))
                node.set_attr("shift_final", str(shift_out))
                node.set_attr("input_format", str("NHWC"))

                input_zp = args[1]
                node.input = [
                    node.input[0],
                    node.input[node.input.index(wt_name)],
                ]  # Remove all the inputs and store only act input and weights
                is_int16 = 1
                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(node.input[node.input.index(wt_name)])
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = shift_qb

                qdq_tensor[10] = 1  # weight_zp

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                modified_input.append(node.input[0])
                modified_input.append(node.input[node.input.index(wt_name)])
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            if "QGemmv".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))

                (
                    c0,
                    c1,
                    c2,
                    shift_qb,
                    shift_out,
                    matmul_shift,
                ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                    args[0],
                    args[1],
                    np.transpose(args[2], (1, 0)),
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    args[7],
                    args[8],
                    args[9],
                )
                is_int16 = 1
                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                modified_input.append(node.input[0])
                modified_input.append(
                    node.input[3]
                )  # weights index in inputs, need to make it generic
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input
                node.set_attr("input_format", str("NHWC"))

            if "IConv".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]

                modified_input = []
                input_dict = OrderedDict()

                node_dtype = get_node_dtype(g, n_m)
                if not isinstance(node.attr["wt_name"], str):
                    wt_name = node.attr["wt_name"].decode("utf-8")
                else:
                    wt_name = node.attr["wt_name"]

                inputs = node.input
                node.set_attr("Node_dtype", str(node_dtype))
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                node.set_attr("input_format", str("NHWC"))

                if len(args) == 28:
                    args[17], args[18] = add_concat_qdq(g, n_m, args[17], args[18])
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                    ) = dq_uint16A_uint8W_conv_q_param_gen(
                        args[0],
                        args[1],
                        args[11],
                        args[12],
                        args[13],
                        np.array(args[14]),
                        args[15],
                        args[16],
                        args[17],
                        args[18],
                    )
                    weight_zp = args[13].astype(np.int32)
                elif len(args) == 19:
                    args[8], args[9] = add_concat_qdq(g, n_m, args[8], args[9])

                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                    ) = dq_uint16A_uint8W_conv_q_param_gen(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        np.array(args[5]),
                        args[6],
                        args[7],
                        args[8],
                        args[9],
                    )
                    weight_zp = args[4].astype(np.int32)
                elif len(args) == 10:
                    args[8], args[9] = add_concat_qdq(g, n_m, args[8], args[9])

                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                    ) = dq_uint16A_uint8W_conv_q_param_gen(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        np.array(args[5]),
                        args[6],
                        args[7],
                        args[8],
                        args[9],
                    )
                    weight_zp = args[4].astype(np.int32)

                node.set_attr("C1", str(c1))
                node.set_attr("C2", str(c2))
                node.set_attr("shift_conv", str(shift_qb))
                node.set_attr("shift_final", str(shift_out))

                input_zp = args[1]
                node.input = [
                    node.input[0],
                    node.input[node.input.index(wt_name)],
                ]  # Remove all the inputs and store only act input and weights
                is_int16 = 1
                # C0 Tensor
                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(node.input[node.input.index(wt_name)])
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = shift_qb

                qdq_tensor[10] = weight_zp  # weight_zp
                qdq_tensor[11] = input_zp  # activation zp
                # QDQ Tensor
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                # Update Inputs
                modified_input.append(node.input[0])
                modified_input.append(node.input[node.input.index(wt_name)])
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            # MatMulAddGelu
            if "MatMulAddGelu".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )

                args = list(input_dict.values())

                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))

                if node_dtype == np.uint8:
                    if len(inputs) == 19:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = compute_qdq_coeff_matmul_bias(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[9],
                            args[10],
                            args[11],
                            args[12],
                            args[13],
                        )
                    else:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = compute_qdq_coeff_matmul_bias(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[13],
                            args[14],
                            args[15],
                            args[16],
                            args[17],
                        )
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    if len(inputs) == 19:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[9],
                            args[10],
                            args[11],
                            args[12],
                            args[13],
                        )
                    else:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[13],
                            args[14],
                            args[15],
                            args[16],
                            args[17],
                        )
                    is_int16 = 1

                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                # qdq_tensor[0] = 0
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                gelu_qdq = EltwiseAdd(args[-4], args[-3], (1 / args[-2]), args[-1])
                c0_scale_a, c0_zp_a, c0_scale_b, c0_zp_b = gelu_qdq.cal_coeff()
                gelu_qdq_tensor = np.zeros((16)).astype(np.int32)
                gelu_qdq_tensor[0] = c0_zp_a
                gelu_qdq_tensor[1] = c0_scale_a
                gelu_qdq_tensor[2] = c0_zp_b
                gelu_qdq_tensor[3] = c0_scale_b
                gelu_qdq_tensor[4] = is_int16

                g.tensormap[n_m + "gelu_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "gelu_qdq_", gelu_qdq_tensor
                )
                g.initials.append(n_m + "gelu_qdq_")

                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                modified_input.append(n_m + "gelu_qdq_")
                node.input = modified_input
                node.input.pop(2)

            # MatMul (PSS/PST) a16w8
            if "MLADFMATMULA16W8" == g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                if node_dtype == np.uint8:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = qdq_matmul_uint8_uint8_cstm(args)
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = qdq_matmul_uint16_uint8_cstm(args)
                    is_int16 = 1
                # C0 Tensor
                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")

                # QDQ size : 2
                qdq_tensor = np.zeros((2)).astype(np.int32)
                qdq_tensor[0] = c1
                qdq_tensor[1] = c2

                # Kernel param size : 16
                SV_M = 16
                SV_K = 128
                SV_N = 16
                K_dim = g.tensormap[g.nodemap[n_m].input[0]].shape[-1]
                k_iter = K_dim // SV_K
                # shift_gemm_out = 1
                # shift_qdq_out = 1
                np_kernel_params = np.zeros(16).astype(np.int32)
                np_kernel_params[0] = SV_M
                np_kernel_params[1] = SV_K
                np_kernel_params[2] = SV_N
                np_kernel_params[3] = 0x2000
                np_kernel_params[4] = 0x6000
                np_kernel_params[5] = 0x3800
                np_kernel_params[6] = k_iter
                np_kernel_params[7] = shift_out
                np_kernel_params[8] = matmul_shift

                # Tensors (DQD and Kernel Params)
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.tensormap[n_m + "_kparams_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_kparams_", np_kernel_params
                )
                g.initials.append(n_m + "_qdq_")
                g.initials.append(n_m + "_kparams_")

                # Update inputs
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                modified_input.append(n_m + "_kparams_")
                node.input = modified_input

            # MatMul (PSS/PST) a16w16 Activations
            if "MLADFMATMULA16A16" == g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                for inp in inputs:
                    if inp in g.initials:
                        input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                # Kernel param size : 16
                SV_M = 16
                SV_K = 256
                SV_N = 8
                K_dim = g.tensormap[g.nodemap[n_m].input[0]].shape[-1]
                k_iter = K_dim // SV_K
                # Get list of coeff
                coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                    args[0],  # QKT_input_1_scale,
                    args[1],  # QKT_input_1_zp,
                    K_dim,
                    args[2],  # QKT_input_2_scale,
                    args[3],  # QKT_input_2_zp,
                    args[4],  # QKT_output_scale,
                    args[5],  # QKT_output_zp,
                )
                # QDQ + RTP Buffer
                np_kernel_params = np.zeros(16).astype(np.int32).reshape([1, 16])
                np_kernel_params[0, 0] = SV_M
                np_kernel_params[0, 1] = SV_K
                np_kernel_params[0, 2] = SV_N
                np_kernel_params[0, 3] = k_iter
                np_kernel_params[0, 4] = 0x2000
                np_kernel_params[0, 5] = 0x4800
                np_kernel_params[0, 6] = 0x3800
                np_kernel_params[0, 7] = 0x3C00
                np_kernel_params[0, 8] = 0x4000
                np_kernel_params[0, 9] = 0x4400
                # C0 is int64
                np_kernel_params.reshape(16).view(np.int64)[5] = coeff_qkt[0]
                # C1/C2/C3 are int32
                np_kernel_params[0, 12] = coeff_qkt[3]  # C1
                np_kernel_params[0, 13] = coeff_qkt[1]  # C2
                np_kernel_params[0, 14] = coeff_qkt[2]  # C3
                matmul_shift = coeff_qkt[6]
                shift_out = coeff_qkt[5]
                np_kernel_params[0, 15] = matmul_shift | shift_out << 16
                qdq_tensor_name = n_m + "_qdq_kparams_"
                g.tensormap[qdq_tensor_name] = onnx_tool.tensor.create_initial_Tensor(
                    qdq_tensor_name, np_kernel_params.reshape(16)
                )
                # Update initializers
                g.initials.append(qdq_tensor_name)
                # Update inputs
                for inp in inputs:
                    if inp in g.dynamics:
                        modified_input.append(inp)
                modified_input.append(qdq_tensor_name)
                node.input = modified_input

            # MatMul
            if (
                "QMatMul" in g.nodemap[n_m].op_type
                and "Add" not in g.nodemap[n_m].op_type
                and "conv" not in g.nodemap[n_m].op_type.lower()
                and "dynamic" not in g.nodemap[n_m].op_type.lower()
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                if node_dtype == np.uint8:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = qdq_matmul_uint8_uint8_cstm(args)
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = qdq_matmul_uint16_uint8_cstm(args)
                    is_int16 = 1
                # C0 Tensor
                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = c2
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16
                # QDQ Tensor
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                # Update inputs
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            # LayerNorm
            if "QLayerNorm" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)
                # TODO Pass appropriate scale and zp
                # Adding qparams as attrs
                lrn_is_uint16 = 0
                prevnodes = node.prevnodes
                if prevnodes:
                    for prev in prevnodes:
                        if (
                            "conv" in prev.op_type.lower()
                            or "globalavgpool" in prev.op_type.lower()
                            or "matmuladd" in prev.op_type.lower()
                        ):
                            lrn_is_uint16 = 1
                else:
                    lrn_is_uint16 = 1  # For subgraph test

                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node.set_attr("lrn_is_uint16", lrn_is_uint16)

                lrnlayer = LRN(1 / input_list[-2], input_list[-1])

                lrn_c0, lrn_c1 = lrnlayer.cal_coeff()

                lrnlayer_inp = LRN(input_list[0], input_list[1])

                lrn_c0_inp, lrn_c1_inp = lrnlayer_inp.cal_coeff()

                lrn_qdq_tensor = np.zeros((16)).astype(np.int32)
                lrn_qdq_tensor[0] = lrn_c0
                lrn_qdq_tensor[1] = lrn_c1
                if act_input_dtype == np.uint8:
                    lrn_qdq_tensor[2] = 0
                else:
                    lrn_qdq_tensor[2] = 1
                lrn_qdq_tensor[3] = lrn_c0_inp
                lrn_qdq_tensor[4] = lrn_c1_inp
                lrn_qdq_tensor[5] = lrn_is_uint16

                node_dtype = get_node_dtype(g, n_m)
                if node_dtype == np.uint8:
                    lrn_qdq_tensor[2] = 0
                else:
                    lrn_qdq_tensor[2] = 1
                node.set_attr("Node_Dtype", str(node_dtype))

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", lrn_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                scale_tensor = dequantize_bf16(
                    input_list[2], input_list[3], input_list[4]
                )

                beta_tensor = dequantize_bf16(
                    input_list[5], input_list[6], input_list[7]
                )
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_qdq_")
                g.tensormap[modified_input[1]].numpy = scale_tensor

                g.tensormap[modified_input[2]].numpy = beta_tensor
                node.input = modified_input

            if "skipadd".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []
                # input ordering
                parents = node.prevnodes
                add_in_parent = 0
                lrn_in_sibling = 0
                # default
                cmat_uint16 = 0
                amat_uint16 = 1
                # check add's parent and sibling
                if parents:
                    for parent in parents:
                        if "skipadd" in parent.op_type.lower():
                            add_in_parent = 1
                            siblings = parent.nextnodes
                            for sibling in siblings:
                                if "LayerNorm".lower() in sibling.op_type.lower():
                                    lrn_in_sibling = 1

                #################m7h4xjg change#################################
                for inp in inputs:
                    if inp in g.dynamics:
                        consumers = g.consumedby[inp]
                        for consumer in consumers:
                            if g.nodemap[consumer].op_type == "QLayerNorm":
                                lrn_in_sibling = 1

                if lrn_in_sibling:  # add_in_parent and
                    amat_uint16 = 0
                # check add's nextnodes
                nexts = node.nextnodes
                lrn_in_opt = 0
                if nexts:
                    for next_node in nexts:
                        if "LayerNorm".lower() in next_node.op_type.lower():
                            lrn_in_opt = 1
                    if not lrn_in_opt:
                        cmat_uint16 = 1

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)
                # Adding qparams as attrs
                # Add
                if add_in_parent and "skipadd" not in parents[0].op_type.lower():
                    input_list[0], input_list[1], input_list[2], input_list[3] = (
                        input_list[2],
                        input_list[3],
                        input_list[0],
                        input_list[1],
                    )
                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_inps.append(input_list[2].item())
                convert_params_inps.append(input_list[3].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node.set_attr("amat_uint16", amat_uint16)
                node.set_attr("cmat_uint16", cmat_uint16)
                eltadd = EltwiseAdd(
                    input_list[0], input_list[1], input_list[2], input_list[3]
                )
                input_list[-2], input_list[-1] = add_concat_qdq(
                    g, n_m, input_list[-2], input_list[-1]
                )
                output_coeff = LRN(
                    1 / input_list[-2].item(), input_list[-1].item()
                ).cal_coeff()
                elt_c0, elt_c1, elt_c2, elt_c3 = eltadd.cal_coeff()
                elt_qdq_tensor = np.zeros((16)).astype(np.int32)
                elt_qdq_tensor[0] = elt_c0
                elt_qdq_tensor[1] = elt_c1
                elt_qdq_tensor[2] = elt_c2
                elt_qdq_tensor[3] = elt_c3
                elt_qdq_tensor[4] = output_coeff[0]
                elt_qdq_tensor[5] = output_coeff[1]
                elt_qdq_tensor[6] = amat_uint16
                elt_qdq_tensor[7] = cmat_uint16

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", elt_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                for inp in inputs:
                    if inp not in g.initials:
                        # if (
                        #     ("scale" not in inp)
                        #     and ("zero_point" not in inp)
                        #     and ("ort" not in inp)
                        # ):
                        modified_input.append(inp)
                if add_in_parent and "skipadd" not in parents[0].op_type.lower():
                    modified_input[0], modified_input[1] = (
                        modified_input[1],
                        modified_input[0],
                    )
                    node.prevnodes[0], node.prevnodes[1] = (
                        node.prevnodes[1],
                        node.prevnodes[0],
                    )
                    # add in parent and add is the first parent
                # if "model" in node.attr.keys() and "mzdk5" in node.attr["model"]:
                #     modified_input = []
                #     modified_input.append(inputs[0])
                #     modified_input.append(inputs[3])

                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            if "QELWEMUL_qdq" == g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                # Adding qparams as attrs
                # inp1: bfloat16 inp2: uint16 op:uint16
                convert_params_inps = []
                convert_params_ops = []

                convert_params_inps.append(input_list[2].item())
                convert_params_inps.append(input_list[3].item())
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())

                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                # breakpoint()
                # quantizeLayer = g.nodemap[n_m].prevnodes[0].prevnodes[0]
                # slice_scale = g.tensormap[quantizeLayer.input[1]].numpy
                # slice_zp = g.tensormap[quantizeLayer.input[2]].numpy
                # print(f"slice scale and zp {slice_scale} {slice_zp}")
                # node.set_attr("modified_b_input_scale", str(slice_scale))
                # node.set_attr("modified_b_input_zp", str(slice_zp))

                # slice_scale =
                # slice_zp =
                # in_a_qdq = LRN(slice_scale, slice_zp)
                in_a_qdq = LRN(input_list[0], input_list[1])
                a_scale, a_zp = in_a_qdq.cal_coeff()

                in_b_qdq = LRN(input_list[2], input_list[3])
                b_scale, b_zp = in_b_qdq.cal_coeff()

                out_qdq = LRN(1 / input_list[-2], input_list[-1])
                out_scale, out_zp = out_qdq.cal_coeff()

                mul_qdq_tensor = np.zeros((16)).astype(np.int32)
                mul_qdq_tensor[0] = b_scale
                mul_qdq_tensor[1] = b_zp
                mul_qdq_tensor[2] = a_scale
                mul_qdq_tensor[3] = a_zp
                mul_qdq_tensor[4] = out_scale
                mul_qdq_tensor[5] = out_zp
                mul_qdq_tensor[6] = 0  # is_matA_uint16
                mul_qdq_tensor[7] = 1  # is_matC_uint16

                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_Dtype", str(node_dtype))

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", mul_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_qdq_")
                # node.input = modified_input
                node.input = [modified_input[1], modified_input[0], modified_input[2]]
            if (
                "QuantOP" == g.nodemap[n_m].op_type
                or "DeQuantOP" == g.nodemap[n_m].op_type
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                # Adding qparams as attrs
                # inp1: bfloat16 inp2: uint16 op:uint16
                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())

                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_inps)

                # slice_scale =
                # slice_zp =
                if "DeQuantOP" == g.nodemap[n_m].op_type:
                    in_a_qdq = LRN(input_list[0], input_list[1])
                else:
                    in_a_qdq = LRN(1 / input_list[0], input_list[1])

                a_scale, a_zp = in_a_qdq.cal_coeff()

                mul_qdq_tensor = np.zeros((16)).astype(np.int32)
                mul_qdq_tensor[0] = a_zp
                mul_qdq_tensor[1] = a_scale

                # node_dtype = get_node_dtype(g, n_m)
                # node.set_attr("Node_Dtype", str(node_dtype))

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", mul_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                # modified_input.append(node.input[0])
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            if "Mladfsoftmax" == g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype

                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)

                # Adding qparams as attrs
                # inp1: bfloat16 inp2: uint16 op:uint16
                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())

                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)

                # here we only need to create tensor, dd op will configure rtp data tensor again.
                rtp_tensor = np.zeros((64)).astype(np.uint8)
                rtp_tensor[62] = 131
                rtp_tensor[63] = 199
                g.tensormap[n_m + "_rtp_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_rtp_", rtp_tensor
                )
                g.initials.append(n_m + "_rtp_")
                for inp in inputs:
                    modified_input.append(inp)
                modified_input.append(n_m + "_rtp_")
                node.input = modified_input

            if "Mladfelwmul" == g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []

                act_input_dtype = g.tensormap[node.input[0]].dtype
                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)
                if inputs[0] in g.initials:
                    const_input = inputs[0]
                    const_input_inlist = input_list[0]
                    const_scale = input_list[1].item()
                    const_zp = input_list[2].item()
                    ifm = inputs[3]
                    ifm_scale = input_list[3].item()
                    ifm_zp = input_list[4].item()
                    ofm_scale = input_list[5].item()
                    ofm_zp = input_list[6].item()
                else:
                    ifm = inputs[0]
                    ifm_scale = input_list[0].item()
                    ifm_zp = input_list[1].item()
                    const_input = inputs[3]
                    const_input_inlist = input_list[2]
                    const_scale = input_list[3].item()
                    const_zp = input_list[4].item()
                    ofm_scale = input_list[5].item()
                    ofm_zp = input_list[6].item()
                tensor_sz = 1
                for number in g.tensormap[ifm].shape:
                    tensor_sz *= number
                qdq_params_gen = mladfelwmul_qdq_param_gen(
                    ifm_scale,
                    const_scale,
                    ofm_scale,
                    ifm_zp,
                    const_zp,
                    ofm_zp,
                    tensor_sz,
                )
                qdq_param = qdq_params_gen.get_params_array()
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_Dtype", str(node_dtype))
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_param
                )
                g.initials.append(n_m + "_qdq_")
                sec_input = None
                if g.tensormap[const_input].shape == ():
                    const_input_arr = np.array(
                        [const_input_inlist.item()],
                        dtype=g.tensormap[const_input].dtype,
                    )
                    g.tensormap[
                        const_input + "_to_arr_"
                    ] = onnx_tool.tensor.create_initial_Tensor(
                        const_input + "_to_arr_", const_input_arr
                    )
                    g.initials.append(const_input + "_to_arr_")
                    sec_input = const_input + "_to_arr_"
                else:
                    sec_input = const_input
                modified_input.append(ifm)
                modified_input.append(sec_input)
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            if (
                g.nodemap[n_m].op_type.lower() == "QMHA".lower()
                or g.nodemap[n_m].op_type.lower() == "mzdk5MHA".lower()
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                if not isinstance(node.attr["QKT_output_qparams"][0], str):
                    MUL_input_scale = np.asarray(
                        g.tensormap[
                            node.attr["MUL_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    MUL_input_zp = np.asarray(
                        g.tensormap[
                            node.attr["MUL_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    MUL_weight_value = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][0]].numpy
                    ).astype(np.float32)
                    MUL_weight_scale = np.asarray(
                        g.tensormap[
                            node.attr["MUL_weight_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    MUL_weight_zp = np.asarray(
                        g.tensormap[
                            node.attr["MUL_weight_qparams"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)

                    MUL_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["MUL_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    MUL_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["MUL_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    QKT_input_1_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_input_1_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    QKT_input_2_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_input_2_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][3].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)

                    QKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_input_1_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_input_1_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_input_2_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_input_2_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][3].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)

                else:
                    MUL_input_scale = np.asarray(
                        g.tensormap[node.attr["MUL_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    MUL_input_zp = np.asarray(
                        g.tensormap[node.attr["MUL_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    MUL_weight_value = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][0]].numpy
                    ).astype(np.float32)

                    MUL_weight_scale = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][1]].numpy
                    ).astype(np.float32)
                    MUL_weight_zp = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][2]].numpy
                    ).astype(np.int32)

                    MUL_output_scale = np.asarray(
                        g.tensormap[node.attr["MUL_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    MUL_output_zp = np.asarray(
                        g.tensormap[node.attr["MUL_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    QKT_input_1_scale = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    QKT_input_1_zp = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    QKT_input_2_scale = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][2]].numpy
                    ).astype(np.float32)
                    QKT_input_2_zp = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][3]].numpy
                    ).astype(np.int32)
                    QKT_output_scale = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    VSQKT_input_1_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    VSQKT_input_1_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    VSQKT_input_2_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][2]].numpy
                    ).astype(np.float32)
                    VSQKT_input_2_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][3]].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    QKT_k_dim = np.asarray(node.attr["QKT_K_dim"]).astype(np.int32)
                    VSQKT_k_dim = np.asarray(node.attr["VSQKT_K_dim"]).astype(np.int32)

                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(g.tensormap[inputs[4]].numpy.item())
                convert_params_inps.append(g.tensormap[inputs[5]].numpy.item())
                convert_params_inps.append(g.tensormap[inputs[1]].numpy.item())
                convert_params_inps.append(g.tensormap[inputs[2]].numpy.item())
                convert_params_inps.append(g.tensormap[inputs[7]].numpy.item())
                convert_params_inps.append(g.tensormap[inputs[8]].numpy.item())

                convert_params_ops.append(g.tensormap[inputs[-2]].numpy.item())
                convert_params_ops.append(g.tensormap[inputs[-1]].numpy.item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                mul_float_value = (MUL_weight_value - MUL_weight_zp) * (
                    MUL_weight_scale
                )
                if g.nodemap[n_m].op_type.lower() == "mzdk5MHA".lower():
                    VSQKT_output_scale, VSQKT_output_zp = add_concat_qdq(
                        g, n_m, VSQKT_output_scale, VSQKT_output_zp
                    )

                    coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                        QKT_input_1_scale,
                        QKT_input_1_zp,
                        QKT_k_dim,
                        QKT_input_2_scale,
                        QKT_input_2_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )

                    coeff_smv = qdq_act_matmul_uint16_uint16_cstm(
                        VSQKT_input_1_scale,
                        VSQKT_input_1_zp,
                        VSQKT_k_dim,
                        VSQKT_input_2_scale,
                        VSQKT_input_2_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    qdq_sm_in = LRN(
                        QKT_output_scale * mul_float_value * 1.442695041, QKT_output_zp
                    ).cal_coeff()
                    is_qkt_smv_int16 = 1
                    qdq_sm_out = LRN(
                        1 / softmax_output_scale, softmax_output_zp
                    ).cal_coeff()

                    qdq_mul_in = [0, 0]
                    qdq_mul_out = [0, 0]

                    qdq_params = mha_channel_qdq_params_fill(
                        coeff_qkt,
                        coeff_smv,
                        qdq_sm_in,
                        qdq_sm_out,
                        qdq_mul_in,
                        qdq_mul_out,
                        is_qkt_smv_int16,
                        smv_swap=0,
                    )

                else:
                    coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                        MUL_input_scale * mul_float_value,
                        MUL_input_zp,
                        QKT_k_dim,
                        QKT_input_2_scale,
                        QKT_input_2_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )

                    coeff_smv = qdq_act_matmul_uint16_uint16_cstm(
                        VSQKT_input_1_scale,
                        VSQKT_input_1_zp,
                        VSQKT_k_dim,
                        VSQKT_input_2_scale,
                        VSQKT_input_2_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 1

                    qdq_sm_in = LRN(
                        QKT_output_scale * 1.442695041, QKT_output_zp
                    ).cal_coeff()

                    qdq_sm_out = LRN(
                        1 / softmax_output_scale, softmax_output_zp
                    ).cal_coeff()

                    qdq_mul_in = [0, 0]
                    qdq_mul_out = [0, 0]

                    qdq_params = mha_channel_qdq_params_fill(
                        coeff_qkt,
                        coeff_smv,
                        qdq_sm_in,
                        qdq_sm_out,
                        qdq_mul_in,
                        qdq_mul_out,
                        is_qkt_smv_int16,
                        smv_swap=1,
                    )

                qdq_params = qdq_params.reshape((6, 16))
                g.tensormap[
                    n_m + "_mha_np_qdq"
                ] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_mha_np_qdq", qdq_params
                )
                g.initials.append(n_m + "_mha_np_qdq")
                # Q,K,V,qdq_tensor

                if node.op_type == "mzdk5MHA":
                    node.input = [
                        node.input[3],
                        node.input[6],
                        node.input[0],
                        n_m + "_mha_np_qdq",
                    ]
                else:
                    node.input = [
                        node.input[3],
                        node.input[0],
                        node.input[6],
                        n_m + "_mha_np_qdq",
                    ]

            if "QMatMulDynamic".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                QKT_k_dim = g.tensormap[inputs[0]].shape[-1]
                args[4], args[5] = add_concat_qdq(g, n_m, args[4], args[5])
                # QDQ param buff
                qdq_params = np.zeros(16).astype(np.int32).reshape([1, 16])

                # Get list of coeff
                coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                    args[0],  # QKT_input_1_scale,
                    args[1],  # QKT_input_1_zp,
                    QKT_k_dim,
                    args[2],  # QKT_input_2_scale,
                    args[3],  # QKT_input_2_zp,
                    args[4],  # QKT_output_scale,
                    args[5],  # QKT_output_zp,
                )
                if node.attr.get("is_qkt", False):
                    qdq_params.reshape(16).view(np.int64)[0] = coeff_qkt[0]
                    qdq_params[0, 2] = coeff_qkt[1]
                    qdq_params[0, 3] = coeff_qkt[2]
                    qdq_params[0, 4] = coeff_qkt[3]
                    qdq_params[0, 5] = 16
                    qdq_params[0, 6] = 64
                    # TODO: This might be incorrect, recheck
                    qdq_params[0, 7] = coeff_qkt[4]
                    qdq_params[0, 8] = coeff_qkt[5]
                    qdq_params[0, 9] = coeff_qkt[6]
                else:
                    qdq_params.reshape(16).view(np.int64)[0] = coeff_qkt[0]
                    # node.domain = ""
                    qdq_params[0, 2] = coeff_qkt[1]
                    qdq_params[0, 3] = coeff_qkt[2]
                    qdq_params[0, 4] = coeff_qkt[3]
                    qdq_params[0, 5] = 16
                    qdq_params[0, 6] = 128
                    # TODO: This might be incorrect, recheck
                    qdq_params[0, 7] = coeff_qkt[4]
                    qdq_params[0, 8] = coeff_qkt[5]
                    qdq_params[0, 9] = coeff_qkt[6]

                # QDQ Tensor
                qdq_t_name = n_m + "_qdq_"
                g.tensormap[qdq_t_name] = onnx_tool.tensor.create_initial_Tensor(
                    qdq_t_name, qdq_params
                )
                g.initials.append(qdq_t_name)
                # Inputs
                node.input = [inputs[0], inputs[3], qdq_t_name]

            if "QMulSoftmax".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                # QDQ param buff
                qdq_params = np.zeros(96).astype(np.int32).reshape(6, 16)
                # Get values
                mul_value = g.tensormap[inputs[3]].numpy
                mul_scale = g.tensormap[inputs[4]].numpy
                mul_zp = g.tensormap[inputs[5]].numpy
                sm_in_scale = g.tensormap[inputs[1]].numpy
                sm_in_zp = g.tensormap[inputs[2]].numpy
                sm_out_scale = g.tensormap[inputs[10]].numpy
                sm_out_zp = g.tensormap[inputs[11]].numpy
                multiplier = (mul_value - mul_zp) * mul_scale
                # SM Scale/ZP
                qdq_sm_in = LRN(
                    sm_in_scale * multiplier * 1.442695041, sm_in_zp
                ).cal_coeff()
                qdq_sm_out = LRN(1 / sm_out_scale, sm_out_zp).cal_coeff()
                # Update Const buffer
                qdq_params[4, 0] = qdq_sm_in[1]
                qdq_params[4, 1] = qdq_sm_in[0]
                qdq_params[5, 0] = qdq_sm_out[1]
                qdq_params[5, 1] = qdq_sm_out[0]
                # QDQ Tensor
                qdq_t_name = n_m + "_qdq_"
                g.tensormap[qdq_t_name] = onnx_tool.tensor.create_initial_Tensor(
                    qdq_t_name, qdq_params
                )
                g.initials.append(qdq_t_name)
                # Update Inputs
                node.input = [inputs[0], qdq_t_name]

            if (
                "MHACHANNEL".lower() in g.nodemap[n_m].op_type.lower()
                or "MHAWINDOW".lower() in g.nodemap[n_m].op_type.lower()
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                if not isinstance(node.attr["QKT_output_qparams"][0], str):
                    MUL_input_scale = np.asarray(
                        g.tensormap[
                            node.attr["MUL_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    MUL_input_zp = np.asarray(
                        g.tensormap[node.attr["MUL_input_zp"][1].decode("utf-8")].numpy
                    ).astype(np.int32)
                    MUL_weight_value = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][0]].numpy
                    ).astype(np.float32)
                    MUL_weight_scale = np.asarray(
                        g.tensormap[
                            node.attr["MUL_weight_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    MUL_weight_zp = np.asarray(
                        g.tensormap[
                            node.attr["MUL_weight_qparams"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)

                    MUL_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["MUL_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    MUL_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["MUL_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    QKT_input_1_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_input_1_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    QKT_input_2_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_input_2_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_input_qparams"][3].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)

                    QKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_input_1_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_input_1_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_input_2_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_input_2_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_input_qparams"][3].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)

                else:
                    MUL_input_scale = np.asarray(
                        g.tensormap[node.attr["MUL_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    MUL_input_zp = np.asarray(
                        g.tensormap[node.attr["MUL_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    MUL_weight_value = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][0]].numpy
                    ).astype(np.float32)

                    MUL_weight_scale = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][1]].numpy
                    ).astype(np.float32)
                    MUL_weight_zp = np.asarray(
                        g.tensormap[node.attr["MUL_weight_qparams"][2]].numpy
                    ).astype(np.int32)

                    MUL_output_scale = np.asarray(
                        g.tensormap[node.attr["MUL_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    MUL_output_zp = np.asarray(
                        g.tensormap[node.attr["MUL_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    QKT_input_1_scale = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    QKT_input_1_zp = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    QKT_input_2_scale = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][2]].numpy
                    ).astype(np.float32)
                    QKT_input_2_zp = np.asarray(
                        g.tensormap[node.attr["QKT_input_qparams"][3]].numpy
                    ).astype(np.int32)
                    QKT_output_scale = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    VSQKT_input_1_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    VSQKT_input_1_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    VSQKT_input_2_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][2]].numpy
                    ).astype(np.float32)
                    VSQKT_input_2_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_input_qparams"][3]].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    QKT_k_dim = np.asarray(node.attr["QKT_K_dim"]).astype(np.int32)
                    VSQKT_k_dim = np.asarray(node.attr["VSQKT_K_dim"]).astype(np.int32)

                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(g.tensormap[inputs[1]].numpy.item())
                convert_params_inps.append(g.tensormap[inputs[2]].numpy.item())

                convert_params_ops.append(g.tensormap[inputs[-2]].numpy.item())
                convert_params_ops.append(g.tensormap[inputs[-1]].numpy.item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                if node_dtype == np.uint8:
                    coeff_qkt = qdq_act_matmul_uint8_uint8_cstm(
                        QKT_input_1_scale,
                        QKT_input_1_zp,
                        QKT_k_dim,
                        QKT_input_2_scale,
                        QKT_input_2_scale,
                        QKT_output_scale,
                        QKT_output_zp,
                    )
                    coeff_smv = qdq_act_matmul_uint8_uint8_cstm(
                        VSQKT_input_1_scale,
                        VSQKT_input_1_zp,
                        VSQKT_k_dim,
                        VSQKT_input_2_scale,
                        VSQKT_input_2_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 0

                elif node_dtype == np.uint16:
                    coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                        QKT_input_1_scale,
                        QKT_input_1_zp,
                        QKT_k_dim,
                        QKT_input_2_scale,
                        QKT_input_2_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )

                    coeff_smv = qdq_act_matmul_uint16_uint16_cstm(
                        VSQKT_input_1_scale,
                        VSQKT_input_1_zp,
                        VSQKT_k_dim,
                        VSQKT_input_2_scale,
                        VSQKT_input_2_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 1

                # additional scaling to emulate exp using exp2.
                qdq_sm_in = LRN(
                    QKT_output_scale * 1.442695041, QKT_output_zp
                ).cal_coeff()

                qdq_sm_out = LRN(
                    1 / softmax_output_scale, softmax_output_zp
                ).cal_coeff()
                mul_float_value = (MUL_weight_value - MUL_weight_zp) * (
                    MUL_weight_scale
                )

                qdq_mul_in = LRN(
                    mul_float_value * MUL_input_scale, MUL_input_zp
                ).cal_coeff()

                qdq_mul_out = LRN(1 / MUL_output_scale, MUL_output_zp).cal_coeff()

                qdq_params = mha_channel_qdq_params_fill(
                    coeff_qkt,
                    coeff_smv,
                    qdq_sm_in,
                    qdq_sm_out,
                    qdq_mul_in,
                    qdq_mul_out,
                    is_qkt_smv_int16,
                )

                qdq_params = qdq_params.reshape((6, 16))

                # Key s,zp, query s,zp, qkt_out s,zp
                g.tensormap[
                    n_m + "_mha_np_qdq"
                ] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_mha_np_qdq", qdq_params
                )

                g.initials.append(n_m + "_mha_np_qdq")
                node.input = [node.input[0], n_m + "_mha_np_qdq"]

            if "MHAGRPB".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = OrderedDict()
                if Mha_input_dict:
                    modified_input[0] = inputs[
                        inputs.index(Mha_input_dict[n_m]["Q"])
                    ]  # Query output

                    modified_input[1] = inputs[
                        inputs.index(Mha_input_dict[n_m]["K"])
                    ]  # Key

                    modified_input[2] = inputs[
                        inputs.index(Mha_input_dict[n_m]["V"])
                    ]  # value

                else:
                    # For subgraph, as we do not have matmuls to define the Q,K,V inputs we are relying on the input sequence
                    # input sequence is inturn decided by the way the order of nodenames in the pattern matching code
                    # order of nodenames in the pattern matching code
                    #     -- V is only one of size [1,prompt_len,768], so no issue
                    #     -- K,Q -- ambiguity occurs here as both of them have same shapes. but in all the patterns in pattern dictionary (k appears before Q)
                    # TODO: come up with beter solution for K,Q ambiguity (what if the order changes in upcomming new models)
                    index = 0
                    for i in node.input:
                        if g.tensormap[i].shape == [1, prompt_len, 768]:
                            v = i
                        elif (
                            g.tensormap[i].shape == [1, prompt_len, 1152] and index == 0
                        ):  # K appears first
                            k = i
                            index += 1
                        elif (
                            g.tensormap[i].shape == [1, prompt_len, 1152] and index == 1
                        ):  # q appears after K with same shape
                            q = i
                    modified_input[0] = q
                    modified_input[1] = k
                    modified_input[2] = v

                for inp in inputs:
                    if (
                        g.tensormap[inp].shape == [1, 1, 1, prompt_len]
                        or g.tensormap[inp].shape == (1, 1, 1, prompt_len)
                        or g.tensormap[inp].shape == [1, 1, 1, prompt_len_modified]
                        or g.tensormap[inp].shape == (1, 1, 1, prompt_len_modified)
                    ):
                        if prompt_len < 128:
                            g.tensormap[inp].shape[-1] = prompt_len_modified

                        modified_input[3] = inp
                        attention_mask_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        attention_mask_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    # TODO how to remove this hardcoding below (98,8)
                    if g.tensormap[inp].shape == (96, 8) or g.tensormap[inp].shape == [
                        96,
                        8,
                    ]:
                        modified_input[4] = inp
                        g.tensormap[inp].numpy = g.tensormap[inp].numpy
                        grpb_matmul_wts = g.tensormap[inp].numpy
                        # TODO Check if the wts are uint8
                        grpb_matmul_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_matmul_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    if g.tensormap[inp].shape == (
                        1,
                        12,
                        prompt_len,
                        prompt_len,
                    ) or g.tensormap[inp].shape == [
                        1,
                        12,
                        prompt_len_modified,
                        prompt_len,
                    ]:
                        modified_input[7] = inp + str(mha_count)
                        # TODO Check dtype
                        grpb_bias = g.tensormap[inp].numpy.data
                        grpb_bias = np.asarray(grpb_bias).reshape(
                            12, prompt_len, prompt_len
                        )
                        t1 = onnx_tool.tensor.create_initial_Tensor(
                            inp + str(mha_count),
                            grpb_bias,
                        )  # 12xprompt_lenxprompt_len
                        g.tensormap[inp + str(mha_count)] = t1
                        g.initials.append(inp + str(mha_count))
                        mha_count += 1
                        grpb_mul_ini_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_mul_ini_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    # collect bias of size 8 from GRPB block the s,zp here
                    if g.tensormap[inp].shape == (8,) or g.tensormap[inp].shape == [
                        8,
                    ]:
                        # TODO Check dtype
                        grpb_matmul_bias = g.tensormap[inp].numpy

                        grpb_matmul_bias_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_matmul_bias_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    if g.tensormap[inp].shape == (1, 12, 1, 1) or g.tensormap[
                        inp
                    ].shape == [1, 12, 1, 1]:
                        # TODO Check dtype
                        grpb_mul_1 = g.tensormap[inp].numpy

                        grpb_mul1_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_mul1_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                key_scale = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[1]) + 1]].numpy
                ).astype(np.float32)
                key_zp = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[1]) + 2]].numpy
                ).astype(np.int32)
                query_scale = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[0]) + 1]].numpy
                ).astype(np.float32)
                query_zp = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[0]) + 2]].numpy
                ).astype(np.int32)
                v_scale = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[2]) + 1]].numpy
                ).astype(np.float32)
                v_zp = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[2]) + 2]].numpy
                ).astype(np.int32)
                if not isinstance(node.attr["QKT_output_qparams"][0], str):
                    QKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    sigmoid_input_scale = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    sigmoid_input_zp = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    sigmoid_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    sigmoid_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][3].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    sub_wts = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_sub_params"][0].decode("utf-8")
                        ].numpy
                    )
                    sub_scale = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_sub_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    sub_zp = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_sub_params"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    add_wts = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_add_params"][0].decode("utf-8")
                        ].numpy
                    )
                    add_scale = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_add_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    add_zp = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_add_params"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    div_wts = np.asarray(
                        g.tensormap[node.attr["div_params"][0].decode("utf-8")].numpy
                    )
                    div_scale = np.asarray(
                        g.tensormap[node.attr["div_params"][1].decode("utf-8")].numpy
                    ).astype(np.float32)
                    div_zp = np.asarray(
                        g.tensormap[node.attr["div_params"][2].decode("utf-8")].numpy
                    ).astype(np.int32)
                    grpb_matmul_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["grpb_matmul_add_out_params"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    grpb_matmul_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["grpb_matmul_add_out_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)

                else:
                    QKT_output_scale = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    sigmoid_input_scale = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][0]].numpy
                    ).astype(np.float32)
                    sigmoid_input_zp = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][1]].numpy
                    ).astype(np.int32)
                    sigmoid_output_scale = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][2]].numpy
                    ).astype(np.float32)
                    sigmoid_output_zp = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][3]].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    sub_wts = np.asarray(
                        g.tensormap[node.attr["GRPB_sub_params"][0]].numpy
                    )
                    sub_scale = np.asarray(
                        g.tensormap[node.attr["GRPB_sub_params"][1]].numpy
                    ).astype(np.float32)
                    sub_zp = np.asarray(
                        g.tensormap[node.attr["GRPB_sub_params"][2]].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    add_wts = np.asarray(
                        g.tensormap[node.attr["GRPB_add_params"][0]].numpy
                    )
                    add_scale = np.asarray(
                        g.tensormap[node.attr["GRPB_add_params"][1]].numpy
                    ).astype(np.float32)
                    add_zp = np.asarray(
                        g.tensormap[node.attr["GRPB_add_params"][2]].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    div_wts = np.asarray(g.tensormap[node.attr["div_params"][0]].numpy)
                    div_scale = np.asarray(
                        g.tensormap[node.attr["div_params"][1]].numpy
                    ).astype(np.float32)
                    div_zp = np.asarray(
                        g.tensormap[node.attr["div_params"][2]].numpy
                    ).astype(np.int32)
                    grpb_matmul_output_scale = np.asarray(
                        g.tensormap[node.attr["grpb_matmul_add_out_params"][0]].numpy
                    ).astype(np.float32)
                    grpb_matmul_output_zp = np.asarray(
                        g.tensormap[node.attr["grpb_matmul_add_out_params"][1]].numpy
                    ).astype(np.float32)
                # MHAGRPB qparam attribute
                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(query_scale.item())
                convert_params_inps.append(query_zp.item())
                convert_params_inps.append(key_scale.item())
                convert_params_inps.append(key_zp.item())
                convert_params_inps.append(v_scale.item())
                convert_params_inps.append(v_zp.item())
                convert_params_inps.append(attention_mask_scale.item())
                convert_params_inps.append(attention_mask_zp.item())
                convert_params_ops.append(g.tensormap[inputs[-2]].numpy.item())
                convert_params_ops.append(g.tensormap[inputs[-1]].numpy.item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                if node_dtype == np.uint8:
                    coeff_qkt = qdq_act_matmul_uint8_uint8_cstm(
                        query_scale,
                        query_zp,
                        np.asarray(96).astype(np.int32),
                        key_scale,
                        key_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )
                    coeff_smv = qdq_act_matmul_uint8_uint8_cstm(
                        softmax_output_scale,
                        softmax_output_zp,
                        np.asarray(512).astype(np.int32),
                        v_scale,
                        v_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 0
                    # print(prompt_len_modified)
                elif node_dtype == np.uint16:
                    # TODO : check change mxpzi

                    coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                        query_scale,
                        query_zp,
                        np.asarray(96).astype(np.int32),
                        key_scale,
                        key_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )

                    coeff_smv = qdq_act_matmul_uint16_uint16_cstm(
                        softmax_output_scale,
                        softmax_output_zp,
                        np.asarray(prompt_len_modified).astype(np.int32),
                        v_scale,
                        v_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 1

                div_val = (div_wts.astype(np.float32) - div_zp) * div_scale.astype(
                    np.float32
                )
                qdq_sm_in = LRN((QKT_output_scale / div_val), QKT_output_zp).cal_coeff()

                qdq_sm_out = LRN(
                    1 / softmax_output_scale, softmax_output_zp
                ).cal_coeff()

                qdq_params = mha_qdq_params_fill(
                    coeff_qkt, coeff_smv, qdq_sm_in, qdq_sm_out, is_qkt_smv_int16
                )
                if node_dtype == np.uint8:
                    (
                        c0_gate_linear,
                        c1_gate_linear,
                        c2_gate_linear,
                        shift_qb_gate_linear,
                        shift_out_gate_linear,
                        matmul_shift_date_linear,
                    ) = compute_qdq_coeff_matmul_bias(
                        query_scale,
                        query_zp,
                        grpb_matmul_wts,
                        grpb_matmul_scale,
                        grpb_matmul_zp,
                        grpb_matmul_bias,
                        grpb_matmul_bias_scale,
                        grpb_matmul_bias_zp,
                        grpb_matmul_output_scale,
                        grpb_matmul_output_zp,
                    )
                    is_grpb_int16 = 0
                elif node_dtype == np.uint16:
                    (
                        c0_gate_linear,
                        c1_gate_linear,
                        c2_gate_linear,
                        shift_qb_gate_linear,
                        shift_out_gate_linear,
                        matmul_shift_date_linear,
                    ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                        query_scale,
                        query_zp,
                        grpb_matmul_wts,
                        grpb_matmul_scale,
                        grpb_matmul_zp,
                        grpb_matmul_bias,
                        grpb_matmul_bias_scale,
                        grpb_matmul_bias_zp,
                        grpb_matmul_output_scale,
                        grpb_matmul_output_zp,
                    )
                    is_grpb_int16 = 1

                coeff_grbp = [
                    c1_gate_linear,
                    c2_gate_linear,
                    shift_qb_gate_linear,
                    shift_out_gate_linear,
                    matmul_shift_date_linear,
                ]
                gprb_vec64 = grpb_qgprb_vec64_fill(
                    c0_gate_linear, coeff_qkt[0], coeff_smv[0]
                )

                gprb_vec32 = gprb_vec32_fill(
                    coeff_grbp,
                    grpb_matmul_output_scale,
                    grpb_matmul_output_zp,
                    grpb_mul_ini_scale,
                    grpb_mul_ini_zp,
                    grpb_mul_1,
                    grpb_mul1_scale,
                    grpb_mul1_zp,
                    sub_wts,
                    sub_scale,
                    sub_zp,
                    add_wts,
                    add_scale,
                    add_zp,
                    is_grpb_int16,
                )

                g.tensormap[
                    n_m + "_mha_np_grpb_vec"
                ] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_mha_np_grpb_vec", gprb_vec64
                )
                # Key s,zp, query s,zp, qkt_out s,zp
                g.tensormap[
                    n_m + "_mha_np_grpb_qdq"
                ] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_mha_np_grpb_qdq", gprb_vec32
                )
                g.tensormap[
                    n_m + "_mha_np_qdq"
                ] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_mha_np_qdq", qdq_params
                )
                g.initials.append(n_m + "_mha_np_grpb_vec")
                g.initials.append(n_m + "_mha_np_grpb_qdq")
                g.initials.append(n_m + "_mha_np_qdq")
                modified_input[5] = n_m + "_mha_np_grpb_vec"
                modified_input[6] = n_m + "_mha_np_grpb_qdq"
                modified_input[8] = n_m + "_mha_np_qdq"

                modified_input_list = []

                for i in range(0, 9):
                    modified_input_list.append(modified_input[i])
                node.input = modified_input_list

            if "DQAdd".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                input_q_params = [
                    float(g.tensormap[node.input[1]].numpy),
                    float(g.tensormap[node.input[2]].numpy),
                    float(g.tensormap[node.input[4]].numpy),
                    float(g.tensormap[node.input[5]].numpy),
                ]
                node.set_attr("input_q_params", input_q_params)
                # Get QDQ Params
                eltadd = EltwiseAdd(
                    input_q_params[0],
                    input_q_params[1],
                    input_q_params[2],
                    input_q_params[3],
                )
                elt_c0, elt_c1, elt_c2, elt_c3 = eltadd.cal_coeff()
                # default
                cmat_uint16 = 0
                amat_uint16 = 1
                dq_add_params = np.zeros((16)).astype(np.int32)
                dq_add_params[0] = elt_c0
                dq_add_params[1] = elt_c1
                dq_add_params[2] = elt_c2
                dq_add_params[3] = elt_c3
                dq_add_params[6] = amat_uint16
                dq_add_params[7] = cmat_uint16
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", dq_add_params
                )
                g.initials.append(n_m + "_qdq_")
                node.input = [node.input[0], node.input[3], n_m + "_qdq_"]

    g = check_datatypes(m, g, prompt_len_modified, precision, conv_key)
    const_inputs = []
    for n_m in g.nodemap.keys():
        if "Constant" in g.nodemap[n_m].op_type:
            remove_list.append(n_m)
            node = g.nodemap[n_m]
            g.tensormap[node.output[0]] = onnx_tool.tensor.Tensor(node.attr["value"])

            g.initials.append(node.output[0])

    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].op_type in layername_dict.keys():
            g.nodemap[n_m].op_type = layername_dict[g.nodemap[n_m].op_type]
    return g


# def change_fused_subgraphs():

# if __name__ == "__main__":
#     file = "before_change_inputs.onnx"
#     m, g = loadmodel(file)
#     g = change_inputs(m, g)
