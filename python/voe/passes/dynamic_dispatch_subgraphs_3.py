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
    "QConcateOPs": [
        [
            {
                "name": "data_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv1", 0]],
            },
            {
                "name": "conv1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv1", 1]],
            },
            {
                "name": "conv1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv1", 2]],
            },
            {
                "name": "conv1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "data_DequantizeLinear", 0],
                    [1, "conv1/weight_DequantizeLinear", 0],
                    [2, "conv1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv1", 0]],
                "outport": [[0, "conv1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv1_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool1", 0]],
            },
            {
                "name": "pool1",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv1_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool1_QuantizeLinear", 0]],
            },
            {
                "name": "pool1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool1", 0]],
                "outport": [[0, "pool1_DequantizeLinear", 0]],
            },
            {
                "name": "pool1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool1_QuantizeLinear", 0]],
                "outport": [[0, "conv2", 0]],
            },
            {
                "name": "conv2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv2", 1]],
            },
            {
                "name": "conv2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv2", 2]],
            },
            {
                "name": "conv2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool1_DequantizeLinear", 0],
                    [1, "conv2/weight_DequantizeLinear", 0],
                    [2, "conv2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv2", 0]],
                "outport": [[0, "conv2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv2_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool2", 0]],
            },
            {
                "name": "pool2",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv2_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool2_QuantizeLinear", 0]],
            },
            {
                "name": "pool2_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool2", 0]],
                "outport": [[0, "pool2_DequantizeLinear", 0]],
            },
            {
                "name": "pool2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool2_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1s", 0]],
            },
            {
                "name": "conv3_1s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1s", 1]],
            },
            {
                "name": "conv3_1s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1s", 2]],
            },
            {
                "name": "conv3_1s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool2_DequantizeLinear", 0],
                    [1, "conv3_1s/weight_DequantizeLinear", 0],
                    [2, "conv3_1s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_1s_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1s", 0]],
                "outport": [[0, "conv3_1s_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1s_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1d", 0]],
            },
            {
                "name": "conv3_1d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1d", 1]],
            },
            {
                "name": "conv3_1d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_1s_DequantizeLinear", 0],
                    [1, "conv3_1d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_1d_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1d", 0]],
                "outport": [[0, "conv3_1d_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1d_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1", 0]],
            },
            {
                "name": "conv3_1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1", 1]],
            },
            {
                "name": "conv3_1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1", 2]],
            },
            {
                "name": "conv3_1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_1d_DequantizeLinear", 0],
                    [1, "conv3_1/weight_DequantizeLinear", 0],
                    [2, "conv3_1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1", 0]],
                "outport": [[0, "conv3_1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1_relu_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2s", 0]],
            },
            {
                "name": "conv3_2s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2s", 1]],
            },
            {
                "name": "conv3_2s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2s", 2]],
            },
            {
                "name": "conv3_2s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_1_relu_DequantizeLinear", 0],
                    [1, "conv3_2s/weight_DequantizeLinear", 0],
                    [2, "conv3_2s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_2s_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2s", 0]],
                "outport": [[0, "conv3_2s_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2s_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2d", 0]],
            },
            {
                "name": "conv3_2d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2d", 1]],
            },
            {
                "name": "conv3_2d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_2s_DequantizeLinear", 0],
                    [1, "conv3_2d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_2d_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2d", 0]],
                "outport": [[0, "conv3_2d_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2d_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2", 0]],
            },
            {
                "name": "conv3_2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2", 1]],
            },
            {
                "name": "conv3_2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2", 2]],
            },
            {
                "name": "conv3_2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_2d_DequantizeLinear", 0],
                    [1, "conv3_2/weight_DequantizeLinear", 0],
                    [2, "conv3_2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2", 0]],
                "outport": [[0, "conv3_2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool3", 0]],
            },
            {
                "name": "pool3",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv3_2_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool3_QuantizeLinear", 0]],
            },
            {
                "name": "pool3_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool3", 0]],
                "outport": [[0, "pool3_DequantizeLinear", 0]],
            },
            {
                "name": "pool3_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool3_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1s", 0]],
            },
            {
                "name": "conv4_1s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1s", 1]],
            },
            {
                "name": "conv4_1s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1s", 2]],
            },
            {
                "name": "conv4_1s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool3_DequantizeLinear", 0],
                    [1, "conv4_1s/weight_DequantizeLinear", 0],
                    [2, "conv4_1s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_1s_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1s", 0]],
                "outport": [[0, "conv4_1s_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1s_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1d", 0]],
            },
            {
                "name": "conv4_1d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1d", 1]],
            },
            {
                "name": "conv4_1d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_1s_DequantizeLinear", 0],
                    [1, "conv4_1d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_1d_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1d", 0]],
                "outport": [[0, "conv4_1d_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1d_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1", 0]],
            },
            {
                "name": "conv4_1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1", 1]],
            },
            {
                "name": "conv4_1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1", 2]],
            },
            {
                "name": "conv4_1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_1d_DequantizeLinear", 0],
                    [1, "conv4_1/weight_DequantizeLinear", 0],
                    [2, "conv4_1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1", 0]],
                "outport": [[0, "conv4_1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1_relu_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2s", 0]],
            },
            {
                "name": "conv4_2s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2s", 1]],
            },
            {
                "name": "conv4_2s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2s", 2]],
            },
            {
                "name": "conv4_2s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_1_relu_DequantizeLinear", 0],
                    [1, "conv4_2s/weight_DequantizeLinear", 0],
                    [2, "conv4_2s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_2s_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2s", 0]],
                "outport": [[0, "conv4_2s_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2s_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2d", 0]],
            },
            {
                "name": "conv4_2d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2d", 1]],
            },
            {
                "name": "conv4_2d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_2s_DequantizeLinear", 0],
                    [1, "conv4_2d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_2d_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2d", 0]],
                "outport": [[0, "conv4_2d_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2d_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2", 0]],
            },
            {
                "name": "conv4_2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2", 1]],
            },
            {
                "name": "conv4_2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2", 2]],
            },
            {
                "name": "conv4_2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_2d_DequantizeLinear", 0],
                    [1, "conv4_2/weight_DequantizeLinear", 0],
                    [2, "conv4_2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2", 0]],
                "outport": [[0, "conv4_2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool4", 0]],
            },
            {
                "name": "pool4",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv4_2_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool4_QuantizeLinear", 0]],
            },
            {
                "name": "pool4_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool4", 0]],
                "outport": [[0, "pool4_DequantizeLinear", 0]],
            },
            {
                "name": "pool4_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool4_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1s", 0]],
            },
            {
                "name": "conv5_1s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1s", 1]],
            },
            {
                "name": "conv5_1s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1s", 2]],
            },
            {
                "name": "conv5_1s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool4_DequantizeLinear", 0],
                    [1, "conv5_1s/weight_DequantizeLinear", 0],
                    [2, "conv5_1s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_1s_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1s", 0]],
                "outport": [[0, "conv5_1s_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1s_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1d", 0]],
            },
            {
                "name": "conv5_1d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1d", 1]],
            },
            {
                "name": "conv5_1d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_1s_DequantizeLinear", 0],
                    [1, "conv5_1d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_1d_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1d", 0]],
                "outport": [[0, "conv5_1d_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1d_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1", 0]],
            },
            {
                "name": "conv5_1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1", 1]],
            },
            {
                "name": "conv5_1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1", 2]],
            },
            {
                "name": "conv5_1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_1d_DequantizeLinear", 0],
                    [1, "conv5_1/weight_DequantizeLinear", 0],
                    [2, "conv5_1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1", 0]],
                "outport": [[0, "conv5_1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1_relu_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2s", 0]],
            },
            {
                "name": "conv5_2s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2s", 1]],
            },
            {
                "name": "conv5_2s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2s", 2]],
            },
            {
                "name": "conv5_2s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_1_relu_DequantizeLinear", 0],
                    [1, "conv5_2s/weight_DequantizeLinear", 0],
                    [2, "conv5_2s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_2s_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2s", 0]],
                "outport": [[0, "conv5_2s_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2s_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2d", 0]],
            },
            {
                "name": "conv5_2d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2d", 1]],
            },
            {
                "name": "conv5_2d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_2s_DequantizeLinear", 0],
                    [1, "conv5_2d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_2d_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2d", 0]],
                "outport": [[0, "conv5_2d_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2d_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2", 0]],
            },
            {
                "name": "conv5_2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2", 1]],
            },
            {
                "name": "conv5_2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2", 2]],
            },
            {
                "name": "conv5_2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_2d_DequantizeLinear", 0],
                    [1, "conv5_2/weight_DequantizeLinear", 0],
                    [2, "conv5_2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2", 0]],
                "outport": [[0, "conv5_2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2_relu_QuantizeLinear", 0]],
                "outport": [[0, "convfeat", 0]],
            },
            {
                "name": "convfeat/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "convfeat", 1]],
            },
            {
                "name": "convfeat/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "convfeat", 2]],
            },
            {
                "name": "convfeat",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_2_relu_DequantizeLinear", 0],
                    [1, "convfeat/weight_DequantizeLinear", 0],
                    [2, "convfeat/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "convfeat_QuantizeLinear", 0]],
            },
            {
                "name": "convfeat_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "convfeat", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "data_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv1", 0]],
            },
            {
                "name": "conv1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv1", 1]],
            },
            {
                "name": "conv1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv1", 2]],
            },
            {
                "name": "conv1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "data_DequantizeLinear", 0],
                    [1, "conv1/weight_DequantizeLinear", 0],
                    [2, "conv1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv1_QuantizeLinear", 0]],
            },
            {
                "name": "conv1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv1", 0]],
                "outport": [[0, "conv1_DequantizeLinear", 0]],
            },
            {
                "name": "conv1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv1_QuantizeLinear", 0]],
                "outport": [[0, "conv1_relu", 0]],
            },
            {
                "name": "conv1_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv1_DequantizeLinear", 0]],
                "outport": [[0, "conv1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv1_relu", 0]],
                "outport": [[0, "conv1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv1_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool1", 0]],
            },
            {
                "name": "pool1",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv1_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool1_QuantizeLinear", 0]],
            },
            {
                "name": "pool1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool1", 0]],
                "outport": [[0, "pool1_DequantizeLinear", 0]],
            },
            {
                "name": "pool1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool1_QuantizeLinear", 0]],
                "outport": [[0, "conv2", 0]],
            },
            {
                "name": "conv2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv2", 1]],
            },
            {
                "name": "conv2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv2", 2]],
            },
            {
                "name": "conv2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool1_DequantizeLinear", 0],
                    [1, "conv2/weight_DequantizeLinear", 0],
                    [2, "conv2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv2_QuantizeLinear", 0]],
            },
            {
                "name": "conv2_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv2", 0]],
                "outport": [[0, "conv2_DequantizeLinear", 0]],
            },
            {
                "name": "conv2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv2_QuantizeLinear", 0]],
                "outport": [[0, "conv2_relu", 0]],
            },
            {
                "name": "conv2_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv2_DequantizeLinear", 0]],
                "outport": [[0, "conv2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv2_relu", 0]],
                "outport": [[0, "conv2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv2_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool2", 0]],
            },
            {
                "name": "pool2",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv2_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool2_QuantizeLinear", 0]],
            },
            {
                "name": "pool2_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool2", 0]],
                "outport": [[0, "pool2_DequantizeLinear", 0]],
            },
            {
                "name": "pool2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool2_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1s", 0]],
            },
            {
                "name": "conv3_1s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1s", 1]],
            },
            {
                "name": "conv3_1s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1s", 2]],
            },
            {
                "name": "conv3_1s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool2_DequantizeLinear", 0],
                    [1, "conv3_1s/weight_DequantizeLinear", 0],
                    [2, "conv3_1s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_1s_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1s", 0]],
                "outport": [[0, "conv3_1s_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1s_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1d", 0]],
            },
            {
                "name": "conv3_1d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1d", 1]],
            },
            {
                "name": "conv3_1d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_1s_DequantizeLinear", 0],
                    [1, "conv3_1d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_1d_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1d", 0]],
                "outport": [[0, "conv3_1d_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1d_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1", 0]],
            },
            {
                "name": "conv3_1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1", 1]],
            },
            {
                "name": "conv3_1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_1", 2]],
            },
            {
                "name": "conv3_1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_1d_DequantizeLinear", 0],
                    [1, "conv3_1/weight_DequantizeLinear", 0],
                    [2, "conv3_1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_1_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1", 0]],
                "outport": [[0, "conv3_1_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1_QuantizeLinear", 0]],
                "outport": [[0, "conv3_1_relu", 0]],
            },
            {
                "name": "conv3_1_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv3_1_DequantizeLinear", 0]],
                "outport": [[0, "conv3_1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1_relu", 0]],
                "outport": [[0, "conv3_1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_1_relu_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2s", 0]],
            },
            {
                "name": "conv3_2s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2s", 1]],
            },
            {
                "name": "conv3_2s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2s", 2]],
            },
            {
                "name": "conv3_2s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_1_relu_DequantizeLinear", 0],
                    [1, "conv3_2s/weight_DequantizeLinear", 0],
                    [2, "conv3_2s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_2s_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2s", 0]],
                "outport": [[0, "conv3_2s_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2s_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2d", 0]],
            },
            {
                "name": "conv3_2d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2d", 1]],
            },
            {
                "name": "conv3_2d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_2s_DequantizeLinear", 0],
                    [1, "conv3_2d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_2d_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2d", 0]],
                "outport": [[0, "conv3_2d_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2d_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2", 0]],
            },
            {
                "name": "conv3_2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2", 1]],
            },
            {
                "name": "conv3_2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv3_2", 2]],
            },
            {
                "name": "conv3_2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv3_2d_DequantizeLinear", 0],
                    [1, "conv3_2/weight_DequantizeLinear", 0],
                    [2, "conv3_2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv3_2_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2", 0]],
                "outport": [[0, "conv3_2_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2_QuantizeLinear", 0]],
                "outport": [[0, "conv3_2_relu", 0]],
            },
            {
                "name": "conv3_2_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv3_2_DequantizeLinear", 0]],
                "outport": [[0, "conv3_2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv3_2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2_relu", 0]],
                "outport": [[0, "conv3_2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv3_2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv3_2_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool3", 0]],
            },
            {
                "name": "pool3",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv3_2_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool3_QuantizeLinear", 0]],
            },
            {
                "name": "pool3_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool3", 0]],
                "outport": [[0, "pool3_DequantizeLinear", 0]],
            },
            {
                "name": "pool3_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool3_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1s", 0]],
            },
            {
                "name": "conv4_1s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1s", 1]],
            },
            {
                "name": "conv4_1s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1s", 2]],
            },
            {
                "name": "conv4_1s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool3_DequantizeLinear", 0],
                    [1, "conv4_1s/weight_DequantizeLinear", 0],
                    [2, "conv4_1s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_1s_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1s", 0]],
                "outport": [[0, "conv4_1s_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1s_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1d", 0]],
            },
            {
                "name": "conv4_1d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1d", 1]],
            },
            {
                "name": "conv4_1d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_1s_DequantizeLinear", 0],
                    [1, "conv4_1d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_1d_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1d", 0]],
                "outport": [[0, "conv4_1d_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1d_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1", 0]],
            },
            {
                "name": "conv4_1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1", 1]],
            },
            {
                "name": "conv4_1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_1", 2]],
            },
            {
                "name": "conv4_1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_1d_DequantizeLinear", 0],
                    [1, "conv4_1/weight_DequantizeLinear", 0],
                    [2, "conv4_1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_1_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1", 0]],
                "outport": [[0, "conv4_1_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1_QuantizeLinear", 0]],
                "outport": [[0, "conv4_1_relu", 0]],
            },
            {
                "name": "conv4_1_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv4_1_DequantizeLinear", 0]],
                "outport": [[0, "conv4_1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1_relu", 0]],
                "outport": [[0, "conv4_1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_1_relu_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2s", 0]],
            },
            {
                "name": "conv4_2s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2s", 1]],
            },
            {
                "name": "conv4_2s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2s", 2]],
            },
            {
                "name": "conv4_2s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_1_relu_DequantizeLinear", 0],
                    [1, "conv4_2s/weight_DequantizeLinear", 0],
                    [2, "conv4_2s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_2s_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2s", 0]],
                "outport": [[0, "conv4_2s_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2s_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2d", 0]],
            },
            {
                "name": "conv4_2d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2d", 1]],
            },
            {
                "name": "conv4_2d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_2s_DequantizeLinear", 0],
                    [1, "conv4_2d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_2d_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2d", 0]],
                "outport": [[0, "conv4_2d_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2d_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2", 0]],
            },
            {
                "name": "conv4_2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2", 1]],
            },
            {
                "name": "conv4_2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv4_2", 2]],
            },
            {
                "name": "conv4_2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv4_2d_DequantizeLinear", 0],
                    [1, "conv4_2/weight_DequantizeLinear", 0],
                    [2, "conv4_2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv4_2_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2", 0]],
                "outport": [[0, "conv4_2_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2_QuantizeLinear", 0]],
                "outport": [[0, "conv4_2_relu", 0]],
            },
            {
                "name": "conv4_2_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv4_2_DequantizeLinear", 0]],
                "outport": [[0, "conv4_2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv4_2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2_relu", 0]],
                "outport": [[0, "conv4_2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv4_2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv4_2_relu_QuantizeLinear", 0]],
                "outport": [[0, "pool4", 0]],
            },
            {
                "name": "pool4",
                "op": "MaxPool",
                "attrs": [],
                "inport": [[0, "conv4_2_relu_DequantizeLinear", 0]],
                "outport": [[0, "pool4_QuantizeLinear", 0]],
            },
            {
                "name": "pool4_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "pool4", 0]],
                "outport": [[0, "pool4_DequantizeLinear", 0]],
            },
            {
                "name": "pool4_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "pool4_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1s", 0]],
            },
            {
                "name": "conv5_1s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1s", 1]],
            },
            {
                "name": "conv5_1s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1s", 2]],
            },
            {
                "name": "conv5_1s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "pool4_DequantizeLinear", 0],
                    [1, "conv5_1s/weight_DequantizeLinear", 0],
                    [2, "conv5_1s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_1s_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1s", 0]],
                "outport": [[0, "conv5_1s_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1s_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1d", 0]],
            },
            {
                "name": "conv5_1d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1d", 1]],
            },
            {
                "name": "conv5_1d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_1s_DequantizeLinear", 0],
                    [1, "conv5_1d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_1d_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1d", 0]],
                "outport": [[0, "conv5_1d_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1d_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1", 0]],
            },
            {
                "name": "conv5_1/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1", 1]],
            },
            {
                "name": "conv5_1/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_1", 2]],
            },
            {
                "name": "conv5_1",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_1d_DequantizeLinear", 0],
                    [1, "conv5_1/weight_DequantizeLinear", 0],
                    [2, "conv5_1/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_1_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1", 0]],
                "outport": [[0, "conv5_1_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1_QuantizeLinear", 0]],
                "outport": [[0, "conv5_1_relu", 0]],
            },
            {
                "name": "conv5_1_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv5_1_DequantizeLinear", 0]],
                "outport": [[0, "conv5_1_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_1_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1_relu", 0]],
                "outport": [[0, "conv5_1_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_1_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_1_relu_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2s", 0]],
            },
            {
                "name": "conv5_2s/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2s", 1]],
            },
            {
                "name": "conv5_2s/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2s", 2]],
            },
            {
                "name": "conv5_2s",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_1_relu_DequantizeLinear", 0],
                    [1, "conv5_2s/weight_DequantizeLinear", 0],
                    [2, "conv5_2s/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_2s_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2s_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2s", 0]],
                "outport": [[0, "conv5_2s_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2s_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2s_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2d", 0]],
            },
            {
                "name": "conv5_2d/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2d", 1]],
            },
            {
                "name": "conv5_2d",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_2s_DequantizeLinear", 0],
                    [1, "conv5_2d/weight_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_2d_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2d_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2d", 0]],
                "outport": [[0, "conv5_2d_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2d_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2d_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2", 0]],
            },
            {
                "name": "conv5_2/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2", 1]],
            },
            {
                "name": "conv5_2/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "conv5_2", 2]],
            },
            {
                "name": "conv5_2",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_2d_DequantizeLinear", 0],
                    [1, "conv5_2/weight_DequantizeLinear", 0],
                    [2, "conv5_2/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "conv5_2_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2", 0]],
                "outport": [[0, "conv5_2_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2_QuantizeLinear", 0]],
                "outport": [[0, "conv5_2_relu", 0]],
            },
            {
                "name": "conv5_2_relu",
                "op": "Relu",
                "attrs": [],
                "inport": [[0, "conv5_2_DequantizeLinear", 0]],
                "outport": [[0, "conv5_2_relu_QuantizeLinear", 0]],
            },
            {
                "name": "conv5_2_relu_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2_relu", 0]],
                "outport": [[0, "conv5_2_relu_DequantizeLinear", 0]],
            },
            {
                "name": "conv5_2_relu_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "conv5_2_relu_QuantizeLinear", 0]],
                "outport": [[0, "convfeat", 0]],
            },
            {
                "name": "convfeat/weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "convfeat", 1]],
            },
            {
                "name": "convfeat/bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "convfeat", 2]],
            },
            {
                "name": "convfeat",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "conv5_2_relu_DequantizeLinear", 0],
                    [1, "convfeat/weight_DequantizeLinear", 0],
                    [2, "convfeat/bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "convfeat_QuantizeLinear", 0]],
            },
            {
                "name": "convfeat_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "convfeat", 0]],
                "outport": [],
            },
        ],
    ],
    "QMHAGRPB": [
        [
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_2", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_2", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_1", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_1", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul_1", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_1", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_1", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_2", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_2", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Transpose", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Transpose", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
                        0,
                    ],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Div", 0]],
            },
            {
                "name": "/tulrv6/Constant_12_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Div", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div",
                "op": "Div",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Constant_12_output_0_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Div", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add", 0]],
            },
            {
                "name": "/tulrv6/Mul_output_0_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Mul_output_0_convert_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_3", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                        0,
                    ]
                ],
            },
            {
                "name": "onnx::MatMul_2204_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                        1,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
                        0,
                    ],
                    [1, "onnx::MatMul_2204_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add", 1]
                ],
            },
            {
                "name": "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_3", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_3",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_3", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/ReduceSum", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/ReduceSum",
                "op": "ReduceSum",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/ReduceSum", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sigmoid", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
                "op": "Sigmoid",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sigmoid", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
                        0,
                    ],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice",
                "op": "Slice",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_3", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice_1", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_1",
                "op": "Slice",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice_1", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_2", 0]],
            },
            {
                "name": "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_2", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_2",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_2", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sub", 0]],
            },
            {
                "name": "/tulrv6/Constant_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sub", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sub",
                "op": "Sub",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Constant_output_0_DequantizeLinear__1", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sub", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_3", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_3",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_3", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_2", 0]],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_2", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_2",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_2", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_4", 0]],
            },
            {
                "name": "/tulrv6/GatherElements_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_4", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/GatherElements_output_0_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_4", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_3", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_3",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_3", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Softmax", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax",
                "op": "Softmax",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Softmax", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul_1", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul_1", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_3", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_3", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_4", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_4", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "274_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Reshape_173", 0]],
            },
            {
                "name": "Reshape_173",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "274_DequantizeLinear", 0]],
                "outport": [[0, "297_QuantizeLinear", 0]],
            },
            {
                "name": "297_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_173", 0]],
                "outport": [[0, "297_DequantizeLinear", 0]],
            },
            {
                "name": "297_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "297_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_174", 0]],
            },
            {
                "name": "Transpose_174",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "297_DequantizeLinear", 0]],
                "outport": [[0, "298_QuantizeLinear", 0]],
            },
            {
                "name": "298_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_174", 0]],
                "outport": [
                    [0, "298_DequantizeLinear", 0],
                    [0, "298_DequantizeLinear__1", 0],
                ],
            },
            {
                "name": "298_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "298_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_201", 0]],
            },
            {
                "name": "276_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Reshape_186", 0]],
            },
            {
                "name": "Reshape_186",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "276_DequantizeLinear", 0]],
                "outport": [[0, "316_QuantizeLinear", 0]],
            },
            {
                "name": "316_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_186", 0]],
                "outport": [[0, "316_DequantizeLinear", 0]],
            },
            {
                "name": "316_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "316_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_200", 0]],
            },
            {
                "name": "Transpose_200",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "316_DequantizeLinear", 0]],
                "outport": [[0, "336_QuantizeLinear", 0]],
            },
            {
                "name": "336_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_200", 0]],
                "outport": [[0, "336_DequantizeLinear", 0]],
            },
            {
                "name": "336_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "336_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_201", 1]],
            },
            {
                "name": "MatMul_201",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "298_DequantizeLinear", 0],
                    [1, "336_DequantizeLinear", 0],
                ],
                "outport": [[0, "337_QuantizeLinear", 0]],
            },
            {
                "name": "337_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_201", 0]],
                "outport": [[0, "337_DequantizeLinear", 0]],
            },
            {
                "name": "337_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "337_QuantizeLinear", 0]],
                "outport": [[0, "Div_203", 0]],
            },
            {
                "name": "1062_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Div_203", 1]],
            },
            {
                "name": "Div_203",
                "op": "Div",
                "attrs": [],
                "inport": [
                    [0, "337_DequantizeLinear", 0],
                    [1, "1062_DequantizeLinear", 0],
                ],
                "outport": [[0, "339_QuantizeLinear", 0]],
            },
            {
                "name": "339_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Div_203", 0]],
                "outport": [[0, "339_DequantizeLinear", 0]],
            },
            {
                "name": "339_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "339_QuantizeLinear", 0]],
                "outport": [[0, "339_convert_QuantizeLinear", 0]],
            },
            {
                "name": "339_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "339_DequantizeLinear", 0]],
                "outport": [[0, "339_convert_DequantizeLinear", 0]],
            },
            {
                "name": "339_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "339_convert_QuantizeLinear", 0]],
                "outport": [[0, "Add_204", 0]],
            },
            {
                "name": "110_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_204", 1]],
            },
            {
                "name": "Add_204",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "339_convert_DequantizeLinear", 0],
                    [1, "110_convert_DequantizeLinear", 0],
                ],
                "outport": [[0, "340_QuantizeLinear", 0]],
            },
            {
                "name": "340_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_204", 0]],
                "outport": [[0, "340_DequantizeLinear", 0]],
            },
            {
                "name": "340_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "340_QuantizeLinear", 0]],
                "outport": [[0, "Add_248", 0]],
            },
            {
                "name": "298_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "298_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_214", 0]],
            },
            {
                "name": "1077_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_214", 1]],
            },
            {
                "name": "MatMul_214",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "298_DequantizeLinear__1", 0],
                    [1, "1077_DequantizeLinear", 0],
                ],
                "outport": [[0, "351_QuantizeLinear", 0]],
            },
            {
                "name": "351_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_214", 0]],
                "outport": [[0, "351_DequantizeLinear", 0]],
            },
            {
                "name": "351_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "351_QuantizeLinear", 0]],
                "outport": [[0, "Add_215", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_215", 1]],
            },
            {
                "name": "Add_215",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "351_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "352_QuantizeLinear", 0]],
            },
            {
                "name": "352_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_215", 0]],
                "outport": [[0, "352_DequantizeLinear", 0]],
            },
            {
                "name": "352_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "352_QuantizeLinear", 0]],
                "outport": [[0, "Reshape_223", 0]],
            },
            {
                "name": "Reshape_223",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "352_DequantizeLinear", 0]],
                "outport": [[0, "366_QuantizeLinear", 0]],
            },
            {
                "name": "366_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_223", 0]],
                "outport": [[0, "366_DequantizeLinear", 0]],
            },
            {
                "name": "366_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "366_QuantizeLinear", 0]],
                "outport": [[0, "ReduceSum_225", 0]],
            },
            {
                "name": "ReduceSum_225",
                "op": "ReduceSum",
                "attrs": [],
                "inport": [[0, "366_DequantizeLinear", 0]],
                "outport": [[0, "368_QuantizeLinear", 0]],
            },
            {
                "name": "368_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "ReduceSum_225", 0]],
                "outport": [[0, "368_DequantizeLinear", 0]],
            },
            {
                "name": "368_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "368_QuantizeLinear", 0]],
                "outport": [[0, "Sigmoid_226", 0]],
            },
            {
                "name": "Sigmoid_226",
                "op": "Sigmoid",
                "attrs": [],
                "inport": [[0, "368_DequantizeLinear", 0]],
                "outport": [[0, "369_QuantizeLinear", 0]],
            },
            {
                "name": "369_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Sigmoid_226", 0]],
                "outport": [
                    [0, "369_DequantizeLinear", 0],
                    [0, "369_DequantizeLinear__1", 0],
                ],
            },
            {
                "name": "369_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "369_QuantizeLinear", 0]],
                "outport": [[0, "Slice_237", 0]],
            },
            {
                "name": "Slice_237",
                "op": "Slice",
                "attrs": [],
                "inport": [[0, "369_DequantizeLinear", 0]],
                "outport": [[0, "380_QuantizeLinear", 0]],
            },
            {
                "name": "380_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Slice_237", 0]],
                "outport": [[0, "380_DequantizeLinear", 0]],
            },
            {
                "name": "380_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "380_QuantizeLinear", 0]],
                "outport": [[0, "Mul_244", 0]],
            },
            {
                "name": "369_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "369_QuantizeLinear", 0]],
                "outport": [[0, "Slice_240", 0]],
            },
            {
                "name": "Slice_240",
                "op": "Slice",
                "attrs": [],
                "inport": [[0, "369_DequantizeLinear__1", 0]],
                "outport": [[0, "383_QuantizeLinear", 0]],
            },
            {
                "name": "383_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Slice_240", 0]],
                "outport": [[0, "383_DequantizeLinear", 0]],
            },
            {
                "name": "383_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "383_QuantizeLinear", 0]],
                "outport": [[0, "Mul_241", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Mul_241", 1]],
            },
            {
                "name": "Mul_241",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "383_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "384_QuantizeLinear", 0]],
            },
            {
                "name": "384_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Mul_241", 0]],
                "outport": [[0, "384_DequantizeLinear", 0]],
            },
            {
                "name": "384_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "384_QuantizeLinear", 0]],
                "outport": [[0, "Sub_243", 0]],
            },
            {
                "name": "107_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Sub_243", 1]],
            },
            {
                "name": "Sub_243",
                "op": "Sub",
                "attrs": [],
                "inport": [
                    [0, "384_DequantizeLinear", 0],
                    [1, "107_DequantizeLinear__1", 0],
                ],
                "outport": [[0, "386_QuantizeLinear", 0]],
            },
            {
                "name": "386_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Sub_243", 0]],
                "outport": [[0, "386_DequantizeLinear", 0]],
            },
            {
                "name": "386_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "386_QuantizeLinear", 0]],
                "outport": [[0, "Mul_244", 1]],
            },
            {
                "name": "Mul_244",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "380_DequantizeLinear", 0],
                    [1, "386_DequantizeLinear", 0],
                ],
                "outport": [[0, "387_QuantizeLinear", 0]],
            },
            {
                "name": "387_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Mul_244", 0]],
                "outport": [[0, "387_DequantizeLinear", 0]],
            },
            {
                "name": "387_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "387_QuantizeLinear", 0]],
                "outport": [[0, "Add_246", 0]],
            },
            {
                "name": "130_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_246", 1]],
            },
            {
                "name": "Add_246",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "387_DequantizeLinear", 0],
                    [1, "130_DequantizeLinear", 0],
                ],
                "outport": [[0, "389_QuantizeLinear", 0]],
            },
            {
                "name": "389_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_246", 0]],
                "outport": [[0, "389_DequantizeLinear", 0]],
            },
            {
                "name": "389_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "389_QuantizeLinear", 0]],
                "outport": [[0, "Mul_247", 0]],
            },
            {
                "name": "271_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Mul_247", 1]],
            },
            {
                "name": "Mul_247",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "389_DequantizeLinear", 0],
                    [1, "271_DequantizeLinear", 0],
                ],
                "outport": [[0, "390_QuantizeLinear", 0]],
            },
            {
                "name": "390_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Mul_247", 0]],
                "outport": [[0, "390_DequantizeLinear", 0]],
            },
            {
                "name": "390_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "390_QuantizeLinear", 0]],
                "outport": [[0, "390_convert_QuantizeLinear", 0]],
            },
            {
                "name": "390_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "390_DequantizeLinear", 0]],
                "outport": [[0, "390_convert_DequantizeLinear", 0]],
            },
            {
                "name": "390_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "390_convert_QuantizeLinear", 0]],
                "outport": [[0, "Add_248", 1]],
            },
            {
                "name": "Add_248",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "340_DequantizeLinear", 0],
                    [1, "390_convert_DequantizeLinear", 0],
                ],
                "outport": [[0, "391_QuantizeLinear", 0]],
            },
            {
                "name": "391_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_248", 0]],
                "outport": [[0, "391_DequantizeLinear", 0]],
            },
            {
                "name": "391_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "391_QuantizeLinear", 0]],
                "outport": [[0, "Softmax_249", 0]],
            },
            {
                "name": "Softmax_249",
                "op": "Softmax",
                "attrs": [],
                "inport": [[0, "391_DequantizeLinear", 0]],
                "outport": [[0, "392_QuantizeLinear", 0]],
            },
            {
                "name": "392_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Softmax_249", 0]],
                "outport": [[0, "392_DequantizeLinear", 0]],
            },
            {
                "name": "392_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "392_QuantizeLinear", 0]],
                "outport": [[0, "392_convert_QuantizeLinear", 0]],
            },
            {
                "name": "392_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "392_DequantizeLinear", 0]],
                "outport": [[0, "392_convert_DequantizeLinear", 0]],
            },
            {
                "name": "392_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "392_convert_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_250", 0]],
            },
            {
                "name": "279_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Reshape_198", 0]],
            },
            {
                "name": "Reshape_198",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "279_DequantizeLinear", 0]],
                "outport": [[0, "334_QuantizeLinear", 0]],
            },
            {
                "name": "334_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_198", 0]],
                "outport": [[0, "334_DequantizeLinear", 0]],
            },
            {
                "name": "334_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "334_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_199", 0]],
            },
            {
                "name": "Transpose_199",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "334_DequantizeLinear", 0]],
                "outport": [[0, "335_QuantizeLinear", 0]],
            },
            {
                "name": "335_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_199", 0]],
                "outport": [[0, "335_DequantizeLinear", 0]],
            },
            {
                "name": "335_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "335_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_250", 1]],
            },
            {
                "name": "MatMul_250",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "392_convert_DequantizeLinear", 0],
                    [1, "335_DequantizeLinear", 0],
                ],
                "outport": [[0, "393_QuantizeLinear", 0]],
            },
            {
                "name": "393_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_250", 0]],
                "outport": [[0, "393_DequantizeLinear", 0]],
            },
            {
                "name": "393_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "393_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_251", 0]],
            },
            {
                "name": "Transpose_251",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "393_DequantizeLinear", 0]],
                "outport": [[0, "394_QuantizeLinear", 0]],
            },
            {
                "name": "394_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_251", 0]],
                "outport": [[0, "394_DequantizeLinear", 0]],
            },
            {
                "name": "394_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "394_QuantizeLinear", 0]],
                "outport": [[0, "Reshape_263", 0]],
            },
            {
                "name": "Reshape_263",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "394_DequantizeLinear", 0]],
                "outport": [[0, "409_QuantizeLinear", 0]],
            },
            {
                "name": "409_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_263", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_2", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_2", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_1", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_1", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul_1", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_1", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_1", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_2", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_2", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Transpose", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Transpose", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
                        0,
                    ],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Div", 0],
                    [0, "/tulrv6/encoder/layer.0/attention/self/Div", 0],
                ],
            },
            {
                "name": "/tulrv6/Constant_12_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Div", 1],
                    [0, "/tulrv6/encoder/layer.0/attention/self/Div", 1],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div",
                "op": "Div",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Constant_12_output_0_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                        0,
                    ],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Div", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                        0,
                    ],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div",
                "op": "Div",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Constant_12_output_0_DequantizeLinear", 0],
                ],
                "outport": [],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Div", 0]],
                "outport": [],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [],
            },
            {
                "name": "/tulrv6/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Mul_output_0_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_3", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                        0,
                    ]
                ],
            },
            {
                "name": "onnx::MatMul_2204_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                        1,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
                        0,
                    ],
                    [1, "onnx::MatMul_2204_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add", 1]
                ],
            },
            {
                "name": "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_3", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_3",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_3", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/ReduceSum", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/ReduceSum",
                "op": "ReduceSum",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/ReduceSum", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sigmoid", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
                "op": "Sigmoid",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sigmoid", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
                        0,
                    ],
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice",
                "op": "Slice",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_3", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice_1", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_1",
                "op": "Slice",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Slice_1", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_2", 0]],
            },
            {
                "name": "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_2", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_2",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_2", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sub", 0]],
            },
            {
                "name": "/tulrv6/Constant_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sub", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sub",
                "op": "Sub",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/Constant_output_0_DequantizeLinear__1", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Sub", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_3", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_3",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_3", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_2", 0]],
            },
            {
                "name": "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_2", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_2",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_2", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_4", 0]],
            },
            {
                "name": "/tulrv6/GatherElements_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_4", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "/tulrv6/GatherElements_output_0_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Mul_4", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_3", 1]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_3",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Add_3", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Softmax", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax",
                "op": "Softmax",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Softmax", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul_1", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/MatMul_1", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_3", 0]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/tulrv6/encoder/layer.0/attention/self/Transpose_3", 0]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_4", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/attention/self/Reshape_4", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "279_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Reshape_198", 0]],
            },
            {
                "name": "Reshape_198",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "279_DequantizeLinear", 0]],
                "outport": [[0, "334_QuantizeLinear", 0]],
            },
            {
                "name": "334_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_198", 0]],
                "outport": [[0, "334_DequantizeLinear", 0]],
            },
            {
                "name": "334_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "334_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_199", 0]],
            },
            {
                "name": "Transpose_199",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "334_DequantizeLinear", 0]],
                "outport": [[0, "335_QuantizeLinear", 0]],
            },
            {
                "name": "335_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_199", 0]],
                "outport": [[0, "335_DequantizeLinear", 0]],
            },
            {
                "name": "335_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "335_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_250", 1]],
            },
            {
                "name": "276_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Reshape_186", 0]],
            },
            {
                "name": "Reshape_186",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "276_DequantizeLinear", 0]],
                "outport": [[0, "316_QuantizeLinear", 0]],
            },
            {
                "name": "316_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_186", 0]],
                "outport": [[0, "316_DequantizeLinear", 0]],
            },
            {
                "name": "316_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "316_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_200", 0]],
            },
            {
                "name": "Transpose_200",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "316_DequantizeLinear", 0]],
                "outport": [[0, "336_QuantizeLinear", 0]],
            },
            {
                "name": "336_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_200", 0]],
                "outport": [[0, "336_DequantizeLinear", 0]],
            },
            {
                "name": "336_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "336_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_201", 1]],
            },
            {
                "name": "274_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Reshape_173", 0]],
            },
            {
                "name": "Reshape_173",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "274_DequantizeLinear", 0]],
                "outport": [[0, "297_QuantizeLinear", 0]],
            },
            {
                "name": "297_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_173", 0]],
                "outport": [[0, "297_DequantizeLinear", 0]],
            },
            {
                "name": "297_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "297_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_174", 0]],
            },
            {
                "name": "Transpose_174",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "297_DequantizeLinear", 0]],
                "outport": [[0, "298_QuantizeLinear", 0]],
            },
            {
                "name": "298_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_174", 0]],
                "outport": [
                    [0, "298_DequantizeLinear", 0],
                    [0, "298_DequantizeLinear__1", 0],
                ],
            },
            {
                "name": "298_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "298_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_201", 0]],
            },
            {
                "name": "MatMul_201",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "298_DequantizeLinear", 0],
                    [1, "336_DequantizeLinear", 0],
                ],
                "outport": [[0, "337_QuantizeLinear", 0]],
            },
            {
                "name": "337_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_201", 0]],
                "outport": [[0, "337_DequantizeLinear", 0]],
            },
            {
                "name": "337_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "337_QuantizeLinear", 0]],
                "outport": [[0, "Div_203", 0]],
            },
            {
                "name": "1062_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Div_203", 1]],
            },
            {
                "name": "Div_203",
                "op": "Div",
                "attrs": [],
                "inport": [
                    [0, "337_DequantizeLinear", 0],
                    [1, "1062_DequantizeLinear", 0],
                ],
                "outport": [[0, "339_QuantizeLinear", 0]],
            },
            {
                "name": "339_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Div_203", 0]],
                "outport": [[0, "339_DequantizeLinear", 0]],
            },
            {
                "name": "339_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "339_QuantizeLinear", 0]],
                "outport": [[0, "Add_204", 0]],
            },
            {
                "name": "110_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_204", 1]],
            },
            {
                "name": "Add_204",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "339_DequantizeLinear", 0],
                    [1, "110_DequantizeLinear", 0],
                ],
                "outport": [[0, "340_QuantizeLinear", 0]],
            },
            {
                "name": "340_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_204", 0]],
                "outport": [[0, "340_DequantizeLinear", 0]],
            },
            {
                "name": "340_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "340_QuantizeLinear", 0]],
                "outport": [[0, "Add_248", 0]],
            },
            {
                "name": "298_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "298_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_214", 0]],
            },
            {
                "name": "1077_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_214", 1]],
            },
            {
                "name": "MatMul_214",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "298_DequantizeLinear__1", 0],
                    [1, "1077_DequantizeLinear", 0],
                ],
                "outport": [[0, "351_QuantizeLinear", 0]],
            },
            {
                "name": "351_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_214", 0]],
                "outport": [[0, "351_DequantizeLinear", 0]],
            },
            {
                "name": "351_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "351_QuantizeLinear", 0]],
                "outport": [[0, "Add_215", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_215", 1]],
            },
            {
                "name": "Add_215",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "351_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "352_QuantizeLinear", 0]],
            },
            {
                "name": "352_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_215", 0]],
                "outport": [[0, "352_DequantizeLinear", 0]],
            },
            {
                "name": "352_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "352_QuantizeLinear", 0]],
                "outport": [[0, "Reshape_223", 0]],
            },
            {
                "name": "Reshape_223",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "352_DequantizeLinear", 0]],
                "outport": [[0, "366_QuantizeLinear", 0]],
            },
            {
                "name": "366_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_223", 0]],
                "outport": [[0, "366_DequantizeLinear", 0]],
            },
            {
                "name": "366_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "366_QuantizeLinear", 0]],
                "outport": [[0, "ReduceSum_225", 0]],
            },
            {
                "name": "ReduceSum_225",
                "op": "ReduceSum",
                "attrs": [],
                "inport": [[0, "366_DequantizeLinear", 0]],
                "outport": [[0, "368_QuantizeLinear", 0]],
            },
            {
                "name": "368_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "ReduceSum_225", 0]],
                "outport": [[0, "368_DequantizeLinear", 0]],
            },
            {
                "name": "368_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "368_QuantizeLinear", 0]],
                "outport": [[0, "Sigmoid_226", 0]],
            },
            {
                "name": "Sigmoid_226",
                "op": "Sigmoid",
                "attrs": [],
                "inport": [[0, "368_DequantizeLinear", 0]],
                "outport": [[0, "369_QuantizeLinear", 0]],
            },
            {
                "name": "369_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Sigmoid_226", 0]],
                "outport": [
                    [0, "369_DequantizeLinear", 0],
                    [0, "369_DequantizeLinear__1", 0],
                ],
            },
            {
                "name": "369_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "369_QuantizeLinear", 0]],
                "outport": [[0, "Slice_237", 0]],
            },
            {
                "name": "Slice_237",
                "op": "Slice",
                "attrs": [],
                "inport": [[0, "369_DequantizeLinear", 0]],
                "outport": [[0, "380_QuantizeLinear", 0]],
            },
            {
                "name": "380_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Slice_237", 0]],
                "outport": [[0, "380_DequantizeLinear", 0]],
            },
            {
                "name": "380_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "380_QuantizeLinear", 0]],
                "outport": [[0, "Mul_244", 0]],
            },
            {
                "name": "369_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "369_QuantizeLinear", 0]],
                "outport": [[0, "Slice_240", 0]],
            },
            {
                "name": "Slice_240",
                "op": "Slice",
                "attrs": [],
                "inport": [[0, "369_DequantizeLinear__1", 0]],
                "outport": [[0, "383_QuantizeLinear", 0]],
            },
            {
                "name": "383_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Slice_240", 0]],
                "outport": [[0, "383_DequantizeLinear", 0]],
            },
            {
                "name": "383_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "383_QuantizeLinear", 0]],
                "outport": [[0, "Mul_241", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Mul_241", 1]],
            },
            {
                "name": "Mul_241",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "383_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "384_QuantizeLinear", 0]],
            },
            {
                "name": "384_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Mul_241", 0]],
                "outport": [[0, "384_DequantizeLinear", 0]],
            },
            {
                "name": "384_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "384_QuantizeLinear", 0]],
                "outport": [[0, "Sub_243", 0]],
            },
            {
                "name": "107_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Sub_243", 1]],
            },
            {
                "name": "Sub_243",
                "op": "Sub",
                "attrs": [],
                "inport": [
                    [0, "384_DequantizeLinear", 0],
                    [1, "107_DequantizeLinear__1", 0],
                ],
                "outport": [[0, "386_QuantizeLinear", 0]],
            },
            {
                "name": "386_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Sub_243", 0]],
                "outport": [[0, "386_DequantizeLinear", 0]],
            },
            {
                "name": "386_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "386_QuantizeLinear", 0]],
                "outport": [[0, "Mul_244", 1]],
            },
            {
                "name": "Mul_244",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "380_DequantizeLinear", 0],
                    [1, "386_DequantizeLinear", 0],
                ],
                "outport": [[0, "387_QuantizeLinear", 0]],
            },
            {
                "name": "387_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Mul_244", 0]],
                "outport": [[0, "387_DequantizeLinear", 0]],
            },
            {
                "name": "387_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "387_QuantizeLinear", 0]],
                "outport": [[0, "Add_246", 0]],
            },
            {
                "name": "130_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_246", 1]],
            },
            {
                "name": "Add_246",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "387_DequantizeLinear", 0],
                    [1, "130_DequantizeLinear", 0],
                ],
                "outport": [[0, "389_QuantizeLinear", 0]],
            },
            {
                "name": "389_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_246", 0]],
                "outport": [[0, "389_DequantizeLinear", 0]],
            },
            {
                "name": "389_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "389_QuantizeLinear", 0]],
                "outport": [[0, "Mul_247", 0]],
            },
            {
                "name": "271_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Mul_247", 1]],
            },
            {
                "name": "Mul_247",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [0, "389_DequantizeLinear", 0],
                    [1, "271_DequantizeLinear", 0],
                ],
                "outport": [[0, "390_QuantizeLinear", 0]],
            },
            {
                "name": "390_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Mul_247", 0]],
                "outport": [[0, "390_DequantizeLinear", 0]],
            },
            {
                "name": "390_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "390_QuantizeLinear", 0]],
                "outport": [[0, "Add_248", 1]],
            },
            {
                "name": "Add_248",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "340_DequantizeLinear", 0],
                    [1, "390_DequantizeLinear", 0],
                ],
                "outport": [[0, "391_QuantizeLinear", 0]],
            },
            {
                "name": "391_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_248", 0]],
                "outport": [[0, "391_DequantizeLinear", 0]],
            },
            {
                "name": "391_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "391_QuantizeLinear", 0]],
                "outport": [[0, "Softmax_249", 0]],
            },
            {
                "name": "Softmax_249",
                "op": "Softmax",
                "attrs": [],
                "inport": [[0, "391_DequantizeLinear", 0]],
                "outport": [[0, "392_QuantizeLinear", 0]],
            },
            {
                "name": "392_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Softmax_249", 0]],
                "outport": [[0, "392_DequantizeLinear", 0]],
            },
            {
                "name": "392_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "392_QuantizeLinear", 0]],
                "outport": [[0, "MatMul_250", 0]],
            },
            {
                "name": "MatMul_250",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "392_DequantizeLinear", 0],
                    [1, "335_DequantizeLinear", 0],
                ],
                "outport": [[0, "393_QuantizeLinear", 0]],
            },
            {
                "name": "393_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_250", 0]],
                "outport": [[0, "393_DequantizeLinear", 0]],
            },
            {
                "name": "393_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "393_QuantizeLinear", 0]],
                "outport": [[0, "Transpose_251", 0]],
            },
            {
                "name": "Transpose_251",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "393_DequantizeLinear", 0]],
                "outport": [[0, "394_QuantizeLinear", 0]],
            },
            {
                "name": "394_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Transpose_251", 0]],
                "outport": [[0, "394_DequantizeLinear", 0]],
            },
            {
                "name": "394_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "394_QuantizeLinear", 0]],
                "outport": [[0, "Reshape_263", 0]],
            },
            {
                "name": "Reshape_263",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "394_DequantizeLinear", 0]],
                "outport": [[0, "409_QuantizeLinear", 0]],
            },
            {
                "name": "409_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_263", 0]],
                "outport": [],
            },
        ],
    ],
    "QMHA": [
        [
            {
                "name": "/text_model/encoder/layers.0/self_attn/k_proj/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/q_proj/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Mul", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/v_proj/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_1", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/k_proj/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Mul",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/q_proj/Add_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/text_model/encoder/layers.0/self_attn/Constant_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/v_proj/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Constant_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Mul", 1]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Mul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Mul", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Mul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_1", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Transpose", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Mul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Mul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_2", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_1", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_2",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Mul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Transpose", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_2", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_1", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_4", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_2", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_5", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_4",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_2",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_5",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_4", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_2", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_5", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_3", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_3", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_4", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_3",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_3",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_4",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_3", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_3", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_4", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "text_model.encoder.layers.0.self_attn.bmm_1", 1]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "text_model.encoder.layers.0.self_attn.bmm_1", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "text_model.encoder.layers.0.self_attn.bmm_2", 0]],
            },
            {
                "name": "text_model.encoder.layers.0.self_attn.bmm_1",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "text_model.encoder.layers.0.self_attn.bmm_2",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "text_model.encoder.layers.0.self_attn.bmm_1", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "text_model.encoder.layers.0.self_attn.bmm_2", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_6", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_6", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_6",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_6",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_6", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_6", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Add", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_8", 0]],
            },
            {
                "name": "onnx::Add_4070_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Add", 1]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_6_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "onnx::Add_4070_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_8",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_6_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Add", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_8", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_7", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_7", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_7",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_7",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_8_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_7", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_7", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "text_model.encoder.layers.0.self_attn.softmax", 0]],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_9", 0]],
            },
            {
                "name": "text_model.encoder.layers.0.self_attn.softmax",
                "op": "Softmax",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_9",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Reshape_9_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "text_model.encoder.layers.0.self_attn.softmax", 0]],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Reshape_9_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/text_model/encoder/layers.0/self_attn/Reshape_9", 0]],
                "outport": [],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_5", 0]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_5",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/text_model/encoder/layers.0/self_attn/Transpose_5", 0]
                ],
                "outport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "text_model.encoder.layers.0.self_attn.bmm_2", 1]],
            },
        ]
    ],
    "QMHACHANNEL": [
        [
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__1",
                        0,
                    ],
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__2",
                        0,
                    ],
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Pow_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2", 1]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Pow_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul", 1]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax",
                "op": "Softmax",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__2",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_2_output_0_DequantizeLinear__2",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Gather_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
                        0,
                    ],
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1",
                        1,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_6",
                        0,
                    ]
                ],
                "outport": [],
            },
        ]
    ],
    "QMHAWINDOW": [
        [
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_6_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_2_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__1",
                        0,
                    ],
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__2",
                        0,
                    ],
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_7_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Constant_33_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4", 1]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4",
                "op": "Mul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_9_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Constant_33_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__1",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_8_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_10_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul", 1]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax",
                "op": "Softmax",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__2",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9",
                "op": "Gather",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_8_output_0_DequantizeLinear__2",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Gather_9_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1", 1]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_5_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_12",
                        0,
                    ]
                ],
                "outport": [],
            },
        ]
    ],
    "QLstm": [
        [
            {
                "name": "im2seq_reshape_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_0", 0]],
            },
            {
                "name": "lstm_0/W_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_0", 1]],
            },
            {
                "name": "lstm_0/R_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_0", 2]],
            },
            {
                "name": "lstm_0/B_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_0", 3]],
            },
            {
                "name": "lstm_0",
                "op": "LSTM",
                "attrs": [],
                "inport": [
                    [0, "im2seq_reshape_DequantizeLinear", 0],
                    [1, "lstm_0/W_DequantizeLinear", 0],
                    [2, "lstm_0/R_DequantizeLinear", 0],
                    [3, "lstm_0/B_DequantizeLinear", 0],
                ],
                "outport": [[0, "lstm_0_QuantizeLinear", 0]],
            },
            {
                "name": "lstm_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_0", 0]],
                "outport": [[0, "lstm_0_DequantizeLinear", 0]],
            },
            {
                "name": "lstm_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_0_QuantizeLinear", 0]],
                "outport": [[0, "lstm_0_transpose", 0]],
            },
            {
                "name": "lstm_0_transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "lstm_0_DequantizeLinear", 0]],
                "outport": [[0, "lstm_0_transpose_QuantizeLinear", 0]],
            },
            {
                "name": "lstm_0_transpose_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_0_transpose", 0]],
                "outport": [[0, "lstm_0_transpose_DequantizeLinear", 0]],
            },
            {
                "name": "lstm_0_transpose_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_0_transpose_QuantizeLinear", 0]],
                "outport": [[0, "lstm_0_reshape", 0]],
            },
            {
                "name": "lstm_0_reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "lstm_0_transpose_DequantizeLinear", 0]],
                "outport": [[0, "lstm_0_reshape_QuantizeLinear", 0]],
            },
            {
                "name": "lstm_0_reshape_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_0_reshape", 0]],
                "outport": [[0, "lstm_0_reshape_DequantizeLinear", 0]],
            },
            {
                "name": "lstm_0_reshape_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_0_reshape_QuantizeLinear", 0]],
                "outport": [[0, "lstm_1", 0]],
            },
            {
                "name": "lstm_1/W_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_1", 1]],
            },
            {
                "name": "lstm_1/R_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_1", 2]],
            },
            {
                "name": "lstm_1/B_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "lstm_1", 3]],
            },
            {
                "name": "lstm_1",
                "op": "LSTM",
                "attrs": [],
                "inport": [
                    [0, "lstm_0_reshape_DequantizeLinear", 0],
                    [1, "lstm_1/W_DequantizeLinear", 0],
                    [2, "lstm_1/R_DequantizeLinear", 0],
                    [3, "lstm_1/B_DequantizeLinear", 0],
                ],
                "outport": [[0, "lstm_1_QuantizeLinear", 0]],
            },
            {
                "name": "lstm_1_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_1", 0]],
                "outport": [[0, "lstm_1_DequantizeLinear", 0]],
            },
            {
                "name": "lstm_1_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_1_QuantizeLinear", 0]],
                "outport": [[0, "lstm_1_transpose", 0]],
            },
            {
                "name": "lstm_1_transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "lstm_1_DequantizeLinear", 0]],
                "outport": [[0, "lstm_1_transpose_QuantizeLinear", 0]],
            },
            {
                "name": "lstm_1_transpose_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_1_transpose", 0]],
                "outport": [[0, "lstm_1_transpose_DequantizeLinear", 0]],
            },
            {
                "name": "lstm_1_transpose_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_1_transpose_QuantizeLinear", 0]],
                "outport": [[0, "lstm_1_reshape", 0]],
            },
            {
                "name": "lstm_1_reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "lstm_1_transpose_DequantizeLinear", 0]],
                "outport": [[0, "lstm_1_reshape_QuantizeLinear", 0]],
            },
            {
                "name": "lstm_1_reshape_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "lstm_1_reshape", 0]],
                "outport": [],
            },
        ]
    ],
    "BaseConv": [
        [
            {
                "name": "/convs.0/norm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "/convs.0/norm/Add_1_output_0_DequantizeLinear", 0]],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv", 0]
                ],
            },
            {
                "name": "image_encoder.blocks.0.0.spatial_block.conv1.fn.dw.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv", 1]
                ],
            },
            {
                "name": "image_encoder.blocks.0.0.spatial_block.conv1.fn.dw.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv", 2]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "image_encoder.blocks.0.0.spatial_block.conv1.fn.dw.weight_DequantizeLinear",
                        0,
                    ],
                    [
                        2,
                        "image_encoder.blocks.0.0.spatial_block.conv1.fn.dw.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/dw/Conv_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1", 0]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1", 0]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Reshape_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/blocks.0/blocks.0.0/spatial_block/conv1/fn/Transpose_1", 0]
                ],
                "outport": [],
            },
        ],
        [
            {
                "name": "/convs.1/norm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/convs.1/Reshape", 0]],
            },
            {
                "name": "/convs.1/Reshape",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "/convs.1/norm/Add_1_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/convs.1/Reshape_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Reshape_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Reshape", 0]],
                "outport": [[0, "/convs.1/Reshape_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Reshape_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Reshape_output_0_QuantizeLinear", 0]],
                "outport": [[0, "/convs.1/Transpose", 0]],
            },
            {
                "name": "/convs.1/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "/convs.1/Reshape_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/convs.1/Transpose_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Transpose", 0]],
                "outport": [[0, "/convs.1/Transpose_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Transpose_output_0_QuantizeLinear", 0]],
                "outport": [[0, "/convs.1/proj/Conv", 0]],
            },
            {
                "name": "image_encoder.convs.1.proj.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/convs.1/proj/Conv", 1]],
            },
            {
                "name": "image_encoder.convs.1.proj.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/convs.1/proj/Conv", 2]],
            },
            {
                "name": "/convs.1/proj/Conv",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "/convs.1/Transpose_output_0_DequantizeLinear", 0],
                    [1, "image_encoder.convs.1.proj.weight_DequantizeLinear", 0],
                    [2, "image_encoder.convs.1.proj.bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "/convs.1/proj/Conv_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.1/proj/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/proj/Conv", 0]],
                "outport": [[0, "/convs.1/proj/Conv_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/convs.1/proj/Conv_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/proj/Conv_output_0_QuantizeLinear", 0]],
                "outport": [[0, "/convs.1/Transpose_1", 0]],
            },
            {
                "name": "/convs.1/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "/convs.1/proj/Conv_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/convs.1/Transpose_1_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Transpose_1", 0]],
                "outport": [[0, "/convs.1/Transpose_1_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Transpose_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Transpose_1_output_0_QuantizeLinear", 0]],
                "outport": [[0, "/convs.1/Reshape_3", 0]],
            },
            {
                "name": "/convs.1/Reshape_3",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "/convs.1/Transpose_1_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/convs.1/Reshape_3_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.1/Reshape_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.1/Reshape_3", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "input_image_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/convs.0/proj/Conv", 0]],
            },
            {
                "name": "image_encoder.convs.0.proj.weight_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/convs.0/proj/Conv", 1]],
            },
            {
                "name": "image_encoder.convs.0.proj.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/convs.0/proj/Conv", 2]],
            },
            {
                "name": "/convs.0/proj/Conv",
                "op": "Conv",
                "attrs": [],
                "inport": [
                    [0, "input_image_DequantizeLinear", 0],
                    [1, "image_encoder.convs.0.proj.weight_DequantizeLinear", 0],
                    [2, "image_encoder.convs.0.proj.bias_DequantizeLinear", 0],
                ],
                "outport": [[0, "/convs.0/proj/Conv_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.0/proj/Conv_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.0/proj/Conv", 0]],
                "outport": [[0, "/convs.0/proj/Conv_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/convs.0/proj/Conv_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.0/proj/Conv_output_0_QuantizeLinear", 0]],
                "outport": [[0, "/convs.0/Transpose", 0]],
            },
            {
                "name": "/convs.0/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [[0, "/convs.0/proj/Conv_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/convs.0/Transpose_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.0/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.0/Transpose", 0]],
                "outport": [[0, "/convs.0/Transpose_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/convs.0/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.0/Transpose_output_0_QuantizeLinear", 0]],
                "outport": [[0, "/convs.0/Reshape_1", 0]],
            },
            {
                "name": "/convs.0/Reshape_1",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "/convs.0/Transpose_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/convs.0/Reshape_1_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/convs.0/Reshape_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/convs.0/Reshape_1", 0]],
                "outport": [],
            },
        ],
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
    "QMatMulAddGelu": [
        [
            {
                "name": "424_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_278", 0]],
            },
            {
                "name": "1082_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_278", 1]],
            },
            {
                "name": "MatMul_278",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "424_DequantizeLinear", 0],
                    [1, "1082_DequantizeLinear", 0],
                ],
                "outport": [[0, "426_QuantizeLinear", 0]],
            },
            {
                "name": "426_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_278", 0]],
                "outport": [[0, "426_DequantizeLinear", 0]],
            },
            {
                "name": "426_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "426_QuantizeLinear", 0]],
                "outport": [[0, "426_convert_QuantizeLinear", 0]],
            },
            {
                "name": "426_convert_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "426_DequantizeLinear", 0]],
                "outport": [[0, "426_convert_DequantizeLinear", 0]],
            },
            {
                "name": "426_convert_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "426_convert_QuantizeLinear", 0]],
                "outport": [[0, "Add_279", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_279", 1]],
            },
            {
                "name": "Add_279",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "426_convert_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "427_QuantizeLinear", 0]],
            },
            {
                "name": "427_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_279", 0]],
                "outport": [[0, "427_DequantizeLinear", 0]],
            },
            {
                "name": "427_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "427_QuantizeLinear", 0]],
                "outport": [[0, "Gelu_fused_Erf_0", 0]],
            },
            {
                "name": "Gelu_fused_Erf_0",
                "op": "Gelu",
                "attrs": [],
                "inport": [[0, "427_DequantizeLinear", 0]],
                "outport": [[0, "435_QuantizeLinear", 0]],
            },
            {
                "name": "435_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Gelu_fused_Erf_0", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "/tulrv6/encoder/layer.0/attention/output/LayerNorm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/intermediate/dense/MatMul", 0]
                ],
            },
            {
                "name": "onnx::MatMul_2209_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [0, "/tulrv6/encoder/layer.0/intermediate/dense/MatMul", 1]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/dense/MatMul",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/attention/output/LayerNorm/Add_1_output_0_DequantizeLinear",
                        0,
                    ],
                    [1, "onnx::MatMul_2209_DequantizeLinear", 0],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/intermediate/dense/MatMul", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/tulrv6/encoder/layer.0/intermediate/dense/Add", 1]],
            },
            {
                "name": "tulrv6.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/tulrv6/encoder/layer.0/intermediate/dense/Add", 0]],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/dense/Add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "tulrv6.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/tulrv6/encoder/layer.0/intermediate/dense/Add", 0]],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "Gelu_363", 0]],
            },
            {
                "name": "Gelu_363",
                "op": "Gelu",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/tulrv6/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/tulrv6/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Gelu_363", 0]],
                "outport": [],
            },
        ],
        [
            {
                "name": "424_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_278", 0]],
            },
            {
                "name": "1082_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "MatMul_278", 1]],
            },
            {
                "name": "MatMul_278",
                "op": "MatMul",
                "attrs": [],
                "inport": [
                    [0, "424_DequantizeLinear", 0],
                    [1, "1082_DequantizeLinear", 0],
                ],
                "outport": [[0, "426_QuantizeLinear", 0]],
            },
            {
                "name": "426_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "MatMul_278", 0]],
                "outport": [[0, "426_DequantizeLinear", 0]],
            },
            {
                "name": "426_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "426_QuantizeLinear", 0]],
                "outport": [[0, "Add_279", 0]],
            },
            {
                "name": "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "Add_279", 1]],
            },
            {
                "name": "Add_279",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [0, "426_DequantizeLinear", 0],
                    [
                        1,
                        "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
                        0,
                    ],
                ],
                "outport": [[0, "427_QuantizeLinear", 0]],
            },
            {
                "name": "427_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Add_279", 0]],
                "outport": [[0, "427_DequantizeLinear", 0]],
            },
            {
                "name": "427_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "427_QuantizeLinear", 0]],
                "outport": [[0, "Gelu_229", 0]],
            },
            {
                "name": "Gelu_229",
                "op": "Gelu",
                "attrs": [],
                "inport": [[0, "427_DequantizeLinear", 0]],
                "outport": [[0, "435_QuantizeLinear", 0]],
            },
            {
                "name": "435_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Gelu_229", 0]],
                "outport": [],
            },
        ],
    ],
    "QGlobalAvgPool": [
        [
            {
                "name": "/blocks.3/blocks.3.0/channel_block/ffn/Add_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "/Transpose", 0]],
            },
            {
                "name": "/Transpose",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.3/blocks.3.0/channel_block/ffn/Add_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [[0, "/Transpose_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/Transpose_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/Transpose", 0]],
                "outport": [[0, "/Transpose_output_0_DequantizeLinear", 0]],
            },
            {
                "name": "/Transpose_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/Transpose_output_0_QuantizeLinear", 0]],
                "outport": [[0, "Reshape_5129", 0]],
            },
            {
                "name": "Reshape_5129",
                "op": "Reshape",
                "attrs": [],
                "inport": [[0, "/Transpose_output_0_DequantizeLinear", 0]],
                "outport": [[0, "/Transpose_output_0_reshape_QuantizeLinear", 0]],
            },
            {
                "name": "/Transpose_output_0_reshape_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "Reshape_5129", 0]],
                "outport": [[0, "/Transpose_output_0_reshape_DequantizeLinear", 0]],
            },
            {
                "name": "/Transpose_output_0_reshape_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [[0, "/Transpose_output_0_reshape_QuantizeLinear", 0]],
                "outport": [[0, "/avgpool/GlobalAveragePool", 0]],
            },
            {
                "name": "/avgpool/GlobalAveragePool",
                "op": "GlobalAveragePool",
                "attrs": [],
                "inport": [[0, "/Transpose_output_0_reshape_DequantizeLinear", 0]],
                "outport": [
                    [0, "/avgpool/GlobalAveragePool_output_0_QuantizeLinear", 0]
                ],
            },
            {
                "name": "/avgpool/GlobalAveragePool_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/avgpool/GlobalAveragePool", 0]],
                "outport": [
                    [0, "/avgpool/GlobalAveragePool_output_0_DequantizeLinear", 0]
                ],
            },
            {
                "name": "/avgpool/GlobalAveragePool_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [0, "/avgpool/GlobalAveragePool_output_0_QuantizeLinear", 0]
                ],
                "outport": [[0, "/Flatten", 0]],
            },
            {
                "name": "/Flatten",
                "op": "Flatten",
                "attrs": [],
                "inport": [
                    [0, "/avgpool/GlobalAveragePool_output_0_DequantizeLinear", 0]
                ],
                "outport": [[0, "/Flatten_output_0_QuantizeLinear", 0]],
            },
            {
                "name": "/Flatten_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [[0, "/Flatten", 0]],
                "outport": [],
            },
        ]
    ],
    "QReshapeTranspose": [
        [
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/norm/Add_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/norm/Add_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1",
                "op": "Transpose",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_3_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_QuantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5",
                "op": "Reshape",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_1_output_0_DequantizeLinear",
                        0,
                    ]
                ],
                "outport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5_output_0_QuantizeLinear",
                        0,
                    ]
                ],
            },
            {
                "name": "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5_output_0_QuantizeLinear",
                "op": "QuantizeLinear",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_5",
                        0,
                    ]
                ],
                "outport": [],
            },
        ]
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
    "DQAdd": [
        [
            {
                "name": "/text_model/embeddings/token_embedding/Gather_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "text_model.embeddings.embedding_add", 0]],
            },
            {
                "name": "/text_model/embeddings/position_embedding/Gather_output_0_DequantizeLinear",
                "op": "DequantizeLinear",
                "attrs": [],
                "inport": [],
                "outport": [[0, "text_model.embeddings.embedding_add", 1]],
            },
            {
                "name": "text_model.embeddings.embedding_add",
                "op": "Add",
                "attrs": [],
                "inport": [
                    [
                        0,
                        "/text_model/embeddings/token_embedding/Gather_output_0_DequantizeLinear",
                        0,
                    ],
                    [
                        1,
                        "/text_model/embeddings/position_embedding/Gather_output_0_DequantizeLinear",
                        0,
                    ],
                ],
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
}
