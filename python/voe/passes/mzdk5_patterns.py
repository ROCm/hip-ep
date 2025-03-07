##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
patterns = {
    "BaseConv": [
        [
            {
                "name": "/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "post_quant_conv", 0]],
            },
            {
                "name": "post_quant_conv.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "post_quant_conv", 1]],
            },
            {
                "name": "post_quant_conv.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "post_quant_conv", 2]],
            },
            {
                "name": "post_quant_conv",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "/Mul_output_0_DequantizeLinear", 0],
                    [1, "post_quant_conv.weight_DequantizeLinear", 0],
                    [2, "post_quant_conv.bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "/post_quant_conv/Conv_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/post_quant_conv/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "post_quant_conv", 0]],
                "outport": [],
            },
        ],
    ],
    "mzdk5MHA": [
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_2.0",
                        1,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_1.0",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_1.0",
                        1,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_2.0",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_2.0/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_1.0",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_2.0/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_2.0",
                        0,
                    ]
                ],
                "outport": [],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_1.0",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul", 0]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Constant_10_output_0_DequantizeLinear__15",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul", 1]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/matmul_1.0/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Constant_10_output_0_DequantizeLinear__15",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul", 0]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.softmax_1.0",
                        0,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.softmax_1.0",
                "op": "Softmax",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.softmax_1.0",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/softmax_1.0/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.matmul_2.0",
                        0,
                    ]
                ],
            },
        ]
    ],
    "QGrpNormTrans": [
        [
            {
                "name": "/conv_in/Conv_output_0_DequantizeLinear/duplicated_token_5",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [0, "/conv_in/Conv_output_0_DequantizeLinear/duplicated_token_5", 0]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#1", 0]],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Constant_1_output_0_DequantizeLinear/duplicated_token_21",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#1", 1]],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Constant_2_output_0_DequantizeLinear/duplicated_token_13",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#1", 2]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#1",
                "op": "InstanceNormalization",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/resnets.0/norm1/Constant_1_output_0_DequantizeLinear/duplicated_token_21",
                        0,
                    ],
                    [
                        2,
                        "/down_blocks.0/resnets.0/norm1/Constant_2_output_0_DequantizeLinear/duplicated_token_13",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#1", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#2", 0]],
            },
            {
                "name": "/conv_in/Conv_output_0_DequantizeLinear/duplicated",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#0-1", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#0-1",
                "op": "Shape",
                "attrs": [],
                "inport": [
                    [0, "/conv_in/Conv_output_0_DequantizeLinear/duplicated", 0]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#2", 1]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#2",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "down_blocks.0.resnets.0.norm1#0-1", 0],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#2", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#3", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#3", 1]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#3",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "down_blocks.0.resnets.0.norm1.weight_DequantizeLinear", 0],
                ],
                "outport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Mul_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#3", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Mul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Mul_output_0_QuantizeLinear", 0]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#4.end", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#4.end", 1]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#4.end",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Mul_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "down_blocks.0.resnets.0.norm1.bias_DequantizeLinear", 0],
                ],
                "outport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Add_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#4.end", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.1/norm/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.1/norm/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Add_output_0_QuantizeLinear", 0]
                ],
                "outport": [[0, "/down_blocks.0/attentions.1/Transpose", 0]],
            },
            {
                "name": "/down_blocks.0/attentions.1/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.1/norm/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.1/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.1/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/down_blocks.0/attentions.1/Transpose", 0]],
                "outport": [],
            },
        ],
    ],
    "QGroupNorm": [
        [
            {
                "name": "/conv_in/Conv_output_0_DequantizeLinear/duplicated_token_5",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [0, "/conv_in/Conv_output_0_DequantizeLinear/duplicated_token_5", 0]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#1", 0]],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Constant_1_output_0_DequantizeLinear/duplicated_token_21",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#1", 1]],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Constant_2_output_0_DequantizeLinear/duplicated_token_13",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#1", 2]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#1",
                "op": "InstanceNormalization",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/resnets.0/norm1/Constant_1_output_0_DequantizeLinear/duplicated_token_21",
                        0,
                    ],
                    [
                        2,
                        "/down_blocks.0/resnets.0/norm1/Constant_2_output_0_DequantizeLinear/duplicated_token_13",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#1", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#2", 0]],
            },
            {
                "name": "/conv_in/Conv_output_0_DequantizeLinear/duplicated",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#0-1", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#0-1",
                "op": "Shape",
                "attrs": [],
                "inport": [
                    [0, "/conv_in/Conv_output_0_DequantizeLinear/duplicated", 0]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#2", 1]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#2",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "down_blocks.0.resnets.0.norm1#0-1", 0],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#2", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#3", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#3", 1]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#3",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "down_blocks.0.resnets.0.norm1.weight_DequantizeLinear", 0],
                ],
                "outport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Mul_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#3", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Mul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Mul_output_0_QuantizeLinear", 0]
                ],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#4.end", 0]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.0.norm1#4.end", 1]],
            },
            {
                "name": "down_blocks.0.resnets.0.norm1#4.end",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.0/norm1/Mul_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "down_blocks.0.resnets.0.norm1.bias_DequantizeLinear", 0],
                ],
                "outport": [
                    [0, "/down_blocks.0/resnets.0/norm1/Add_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.0/norm1/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.0.norm1#4.end", 0]],
                "outport": [],
            },
        ]
    ],
    "QConv2MatMul": [
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v_convs.0",
                        0,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v_convs.0.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v_convs.0",
                        1,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v_convs.0",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v_convs.0.weight_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_v_convs.0/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_v_convs.0/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v_convs.0",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_v_convs.0/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_v_convs.0/Conv_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_v_convs.0/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_v_convs.0/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1",
                "op": "Unsqueeze",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_1",
                        0,
                    ]
                ],
                "outport": [],
            },
        ],
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_output_0_DequantizeLinear__10",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k_convs.0",
                        0,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k_convs.0.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k_convs.0",
                        1,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k_convs.0",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_output_0_DequantizeLinear__10",
                        0,
                    ],
                    [
                        1,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k_convs.0.weight_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_k_convs.0/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_k_convs.0/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k_convs.0",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_k_convs.0/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_k_convs.0/Conv_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_k_convs.0/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_k_convs.0/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze",
                "op": "Unsqueeze",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Unsqueeze",
                        0,
                    ]
                ],
                "outport": [],
            },
        ],
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_output_0_DequantizeLinear__5",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_convs.0",
                        0,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_convs.0.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_convs.0",
                        1,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_convs.0",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_output_0_DequantizeLinear__5",
                        0,
                    ],
                    [
                        1,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_convs.0.weight_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q_convs.0/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q_convs.0/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_convs.0",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q_convs.0/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q_convs.0/Conv_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q_convs.0/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q_convs.0/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Transpose_2",
                        0,
                    ]
                ],
                "outport": [],
            },
        ],
    ],
    "QSlice": [
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Shape",
                "op": "Shape",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Gather",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Gather",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Shape",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Add",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Gather",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Div",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Div",
                "op": "Div",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Add",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Mul_1",
                        0,
                    ],
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1",
                        1,
                    ],
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Mul_1",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Div",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1",
                        2,
                    ]
                ],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1",
                "op": "Slice",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Div",
                        0,
                    ],
                    [
                        2,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Mul_1",
                        0,
                    ],
                ],
                "outport": [],
            },
        ],
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice",
                "op": "Slice",
                "attrs": [],
                "inport": [],
                "outport": [],
            }
        ],
    ],
    "QMatMulAdd": [
        [
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/query/MatMul", 0]
                ],
            },
            {
                "name": "onnx::MatMul_2195_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/query/MatMul", 1]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "onnx::MatMul_2195_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/query/MatMul", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/query/Add", 1]],
            },
            {
                "name": "tulrv6.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/query/Add", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "tulrv6.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/query/Add", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.1/norm/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.1/norm/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/down_blocks.0/attentions.1/Transpose", 0]],
            },
            {
                "name": "/down_blocks.0/attentions.1/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.1/norm/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.1/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.1/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/down_blocks.0/attentions.1/Transpose", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/query/MatMul", 0]
                ],
            },
            {
                "name": "onnx::MatMul_2195_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/query/MatMul", 1]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "onnx::MatMul_2195_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/query/MatMul", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/query/Add", 1]],
            },
            {
                "name": "tulrv6.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/query/Add", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "tulrv6.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/query/Add", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "138_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_157", 0]],
            },
            {
                "name": "1068_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_157", 1]],
            },
            {
                "name": "MatMul_157",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "138_DequantizeLinear", 0],
                    [1, "1068_DequantizeLinear", 0],
                ],
                "outport": [[0, "273_QuantizeLinear", 0]],
            },
            {
                "name": "273_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_157", 0]],
                "outport": [[0, "273_DequantizeLinear", 0]],
            },
            {
                "name": "273_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "273_QuantizeLinear", 0]],
                "outport": [[0, "Add_158", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_158", 1]],
            },
            {
                "name": "Add_158",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "273_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "274_QuantizeLinear", 0]],
            },
            {
                "name": "274_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_158", 0]],
                "outport": [],
            },
        ],
    ],
    "QGemmv": [
        [
            {
                "name": "/down_blocks.0/resnets.1/nonlinearity_2/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.1.time_emb_proj", 0]],
            },
            {
                "name": "down_blocks.0.resnets.1.time_emb_proj.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.1.time_emb_proj", 1]],
            },
            {
                "name": "down_blocks.0.resnets.1.time_emb_proj.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.1.time_emb_proj", 2]],
            },
            {
                "name": "down_blocks.0.resnets.1.time_emb_proj",
                "op": "Gemm",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/nonlinearity_2/Mul_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "down_blocks.0.resnets.1.time_emb_proj.weight_DequantizeLinear",
                        0,
                    ],
                    [
                        2,
                        "down_blocks.0.resnets.1.time_emb_proj.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.1.time_emb_proj", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/down_blocks.0/resnets.1/Unsqueeze_1", 0]],
            },
            {
                "name": "/down_blocks.0/resnets.1/Unsqueeze_1",
                "op": "Unsqueeze",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/time_emb_proj/Gemm_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/Unsqueeze_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.1/Unsqueeze_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/down_blocks.0/resnets.1/Unsqueeze_1", 0]],
                "outport": [],
            },
        ],
    ],
    "QMatMul": [
        [
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/key/MatMul", 0]
                ],
            },
            {
                "name": "onnx::MatMul_2196_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/key/MatMul", 1]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/key/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__1",
                        0,
                    ],
                    [1, "onnx::MatMul_2196_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/key/MatMul", 0]],
                "outport": [],
            },
        ]
    ],
    "QSilu": [
        [
            {
                "name": "input_2_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "down_blocks.0.resnets.1.nonlinearity_2.sigmoid", 0]],
            },
            {
                "name": "down_blocks.0.resnets.1.nonlinearity_2.sigmoid",
                "op": "Sigmoid",
                "attrs": [],
                "inport": [[0, "input_2_DequantizeLinear__1", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "down_blocks.0.resnets.1.nonlinearity_2.sigmoid", 0]],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/down_blocks.0/resnets.1/nonlinearity_2/Mul", 1]],
            },
            {
                "name": "input_2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/down_blocks.0/resnets.1/nonlinearity_2/Mul", 0]],
            },
            {
                "name": "/down_blocks.0/resnets.1/nonlinearity_2/Mul",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "input_2_DequantizeLinear", 0],
                    [
                        1,
                        "/down_blocks.0/resnets.1/nonlinearity_2/sigmoid/Sigmoid_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/resnets.1/nonlinearity_2/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/resnets.1/nonlinearity_2/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/down_blocks.0/resnets.1/nonlinearity_2/Mul", 0]],
                "outport": [],
            },
        ]
    ],
    "QGelu": [
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Gelu_fused_Erf_0", 0]],
            },
            {
                "name": "Gelu_fused_Erf_0",
                "op": "Gelu",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/gelu_1/Mul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/gelu_1/Mul_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Gelu_fused_Erf_0", 0]],
                "outport": [],
            },
        ]
    ],
    "QSkipAdd": [
        [
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__3",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/output/Add", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/output/dense/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/output/Add", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/output/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/output/dense/Add_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__3",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/output/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/output/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/output/Add", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "412_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "412_QuantizeLinear", 0]],
                "outport": [],
            },
            {
                "name": "138_DequantizeLinear__3",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [],
            },
            {
                "name": "Add_265",
                "op": "Add",
                "attrs": [],
                "inport": [],
                "outport": [[0, "412_QuantizeLinear", 0]],
            },
            {
                "name": "412_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_265", 0]],
                "outport": [[0, "412_DequantizeLinear", 0]],
            },
        ],
    ],
    "QELWEMUL_qdq": [
        [
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.mult_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/gelu_1/Mul_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.mult_1",
                        1,
                    ]
                ],
            },
            {
                "name": "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.mult_1",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/Slice_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/gelu_1/Mul_1_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/mult_1/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/down_blocks.0/attentions.0/transformer_blocks.0/ff/net.0/mult_1/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.mult_1",
                        0,
                    ]
                ],
                "outport": [],
            },
        ]
    ],
    "QLayerNorm": [
        [
            {
                "name": "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "LayerNormalization_fused_ReduceMean_0", 0]],
            },
            {
                "name": "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "LayerNormalization_fused_ReduceMean_0", 1]],
            },
            {
                "name": "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "LayerNormalization_fused_ReduceMean_0", 2]],
            },
            {
                "name": "LayerNormalization_fused_ReduceMean_0",
                "op": "LayerNormalization",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear", 0],
                    [1, "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear", 0],
                    [2, "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear", 0],
                ],
                "outport": [
                    [0, "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "LayerNormalization_fused_ReduceMean_0", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [],
            },
        ],
        [
            {
                "name": "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "LayerNormalization_242", 0]],
            },
            {
                "name": "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "LayerNormalization_242", 1]],
            },
            {
                "name": "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "LayerNormalization_242", 2]],
            },
            {
                "name": "LayerNormalization_242",
                "op": "LayerNormalization",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear", 0],
                    [1, "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear", 0],
                    [2, "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear", 0],
                ],
                "outport": [
                    [0, "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "LayerNormalization_242", 0]],
                "outport": [],
            },
        ],
    ],
    "QResize": [
        [
            {
                "name": "/up_blocks.0/resnets.2/add_2/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "up_blocks.0.upsamplers.0.interpolate_1", 0]],
            },
            {
                "name": "up_blocks.0.upsamplers.0.interpolate_1",
                "op": "Resize",
                "attrs": [],
                "inport": [
                    [0, "/up_blocks.0/resnets.2/add_2/Add_output_0_DequantizeLinear", 0]
                ],
                "outport": [
                    [
                        0,
                        "/up_blocks.0/upsamplers.0/interpolate_1/Resize_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/up_blocks.0/upsamplers.0/interpolate_1/Resize_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "up_blocks.0.upsamplers.0.interpolate_1", 0]],
                "outport": [],
            },
        ],
    ],
    "DeQuantOP": [
        [
            {
                "name": "conv_in.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [],
            }
        ]
    ],
    "QuantOP": [
        [
            {
                "name": "/conv_in/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [],
            }
        ]
    ],
    "QConcat": [
        [
            {
                "name": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.concat_1",
                "op": "Concat",
                "attrs": [],
                "inport": [],
                "outport": [],
            }
        ]
    ],
}
