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
    "MLADFMATMULA16A16": [
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
    "Mladfsoftmax": [
        [
            {
                "name": "/Cast_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "encoder.mid_block.attentions.0.module_softmax", 0]],
            },
            {
                "name": "encoder.mid_block.attentions.0.module_softmax",
                "op": "Softmax",
                "attrs": [],
                "inport": [[0, "/Cast_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/module_softmax/Softmax_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/module_softmax/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "encoder.mid_block.attentions.0.module_softmax", 0]],
                "outport": [],
            },
        ]
    ],
    "Mladfelwmul": [
        [
            {
                "name": "decoder.mid_block.attentions.0.group_norm.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "decoder.mid_block.attentions.0.group_norm#3", 1]],
            },
            {
                "name": "/decoder/mid_block/attentions.0/group_norm/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "decoder.mid_block.attentions.0.group_norm#3", 0]],
            },
            {
                "name": "decoder.mid_block.attentions.0.group_norm#3",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/decoder/mid_block/attentions.0/group_norm/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "decoder.mid_block.attentions.0.group_norm.weight_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/decoder/mid_block/attentions.0/group_norm/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/decoder/mid_block/attentions.0/group_norm/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "decoder.mid_block.attentions.0.group_norm#3", 0]],
                "outport": [],
            },
        ],
    ],
}
