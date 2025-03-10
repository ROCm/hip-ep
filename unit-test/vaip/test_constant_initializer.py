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
import numpy as np
import onnx
import sys
from onnx import helper, TensorProto


def create_sample_onnx_model_with_all_constant_initializers():
    # Define initializers for all ONNX data types
    initializers = [
        helper.make_tensor("const_int8_scalar", TensorProto.INT8, [], [-1]),
        helper.make_tensor("const_uint8_scalar", TensorProto.UINT8, [], [2]),
        helper.make_tensor("const_int16_scalar", TensorProto.INT16, [], [-3]),
        helper.make_tensor("const_uint16_scalar", TensorProto.UINT16, [], [4]),
        helper.make_tensor("const_int32_scalar", TensorProto.INT32, [], [-5]),
        helper.make_tensor("const_uint32_scalar", TensorProto.UINT32, [], [6]),
        helper.make_tensor("const_int64_scalar", TensorProto.INT64, [], [-7]),
        helper.make_tensor("const_uint64_scalar", TensorProto.UINT64, [], [8]),
        helper.make_tensor("const_float_scalar", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor(
            "const_double_scalar", TensorProto.DOUBLE, [], [np.float64(2.0)]
        ),
        # helper.make_tensor("const_bool_scalar", TensorProto.BOOL, [], [True]),
        # helper.make_tensor("const_string_scalar", TensorProto.STRING, [], ["hello"]),
        helper.make_tensor("const_uint8", TensorProto.UINT8, [2], [3, 4]),
        helper.make_tensor("const_int8", TensorProto.INT8, [2], [-1, -2]),
        helper.make_tensor("const_uint16", TensorProto.UINT16, [2], [5, 6]),
        helper.make_tensor("const_int16", TensorProto.INT16, [2], [-3, -4]),
        helper.make_tensor("const_uint32", TensorProto.UINT32, [2], [10, 12]),
        helper.make_tensor("const_int32", TensorProto.INT32, [2], [-5, -6]),
        helper.make_tensor("const_uint64", TensorProto.UINT64, [2], [13, 14]),
        helper.make_tensor("const_int64", TensorProto.INT64, [2], [-7, -8]),
        helper.make_tensor("const_float", TensorProto.FLOAT, [2], [1.0, 2.0]),
        helper.make_tensor("const_double", TensorProto.DOUBLE, [2], [1.1, 2.2]),
        # helper.make_tensor("const_string", TensorProto.STRING, [2], ["hello", "world"]),
        helper.make_tensor("const_bool", TensorProto.BOOL, [2], [True, False]),
        helper.make_tensor("const_uint32", TensorProto.UINT32, [2], [7, 8]),
        helper.make_tensor("const_uint64", TensorProto.UINT64, [2], [9, 10]),
    ]

    # Create a dummy node that does nothing but is required to create a valid ONNX graph
    node = helper.make_node(
        op_type="concat",  # Operator
        inputs=[
            "const_int8_scalar",
            "const_uint8_scalar",
            "const_int16_scalar",
            "const_uint16_scalar",
            "const_int32_scalar",
            "const_uint32_scalar",
            "const_int64_scalar",
            "const_uint64_scalar",
            "const_float_scalar",
            "const_double_scalar",
            # "const_bool_scalar",
            # "const_string_scalar",
            "const_uint8",
            "const_int8",
            "const_uint16",
            "const_int16",
            "const_uint32",
            "const_int32",
            "const_uint64",
            "const_int64",
            "const_float",
            "const_double",
            # "const_string",
            # "const_bool",
        ],  # Input
        outputs=["out"],  # Output
        domain="com.xilinx",  # Domain
    )
    node.attribute.append(helper.make_attribute("data_type", "float"))
    node.attribute.append(helper.make_attribute("shape", [1]))
    # Create the graph (model)
    graph = helper.make_graph(
        nodes=[node],  # nodes
        name="SampleModelWithAllConstantInitializers",
        inputs=[],  # inputs
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
        initializer=initializers,  # initializers
    )

    # Create the model
    model = helper.make_model(
        graph, producer_name="sample_model_with_all_constant_initializers"
    )
    # onnx.checker.check_model(model)

    # Save the model to a file
    onnx.save(model, sys.argv[1])


create_sample_onnx_model_with_all_constant_initializers()
print(
    "Sample ONNX model with all types of constant initializers has been created and saved."
)
