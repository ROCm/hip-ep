import numpy as np
import onnx
import sys
from onnx import helper, TensorProto


def create_sample_onnx_model_with_multiple_outputs():
    # define the input tensors
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [1, 224, 224, 3]
    )
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 16, 3])

    # define the output tensors
    # output1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT,
    #                                        [1, 16, 16, 3])
    # output2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT,
    #                                       [1, 16, 3])

    # sample_mutil_outputs_op has three outputs, And all of the output is optional
    # here is the second output is optional sample
    node = helper.make_node(
        op_type="sample_multi_outputs_op",
        name="multi_outputs_abc",
        inputs=["input1", "input2"],
        outputs=["output1", "", "output2"],
        domain="com.xilinx",
    )

    # fill in the first output attr `shape_0`&`data_type_0` , for shape infer
    node.attribute.append(helper.make_attribute("shape_0", [1, 16, 16, 3]))
    node.attribute.append(helper.make_attribute("data_type_0", "float32"))

    # the second output is optional , `shape_1` & `data_type_2` need to omitted.

    # fill in the third output attr `shape_2`&`data_type_2` , for shape infer
    node.attribute.append(helper.make_attribute("shape_2", [1, 16, 3]))
    node.attribute.append(helper.make_attribute("data_type_2", "float32"))

    relu1 = helper.make_node(
        op_type="relu",
        name="relu1",
        inputs=["output1"],
        outputs=["relu_out_1"],
        domain="com.xilinx",
    )
    relu1.attribute.append(helper.make_attribute("shape", [1, 16, 16, 3]))
    relu1.attribute.append(helper.make_attribute("data_type", "float32"))

    relu2 = helper.make_node(
        op_type="relu",
        name="relu2",
        inputs=["output2"],
        outputs=["relu_out_2"],
        domain="com.xilinx",
    )
    relu2.attribute.append(helper.make_attribute("shape", [1, 16, 3]))
    relu2.attribute.append(helper.make_attribute("data_type", "float32"))

    add = helper.make_node(
        op_type="add",
        name="add",
        inputs=["relu_out_1", "relu_out_2"],
        outputs=["add_1"],
        domain="com.xilinx",
    )
    add.attribute.append(helper.make_attribute("shape", [1, 16, 16, 3]))
    add.attribute.append(helper.make_attribute("data_type", "float32"))

    add_1 = helper.make_tensor_value_info("add_1", TensorProto.FLOAT, [1, 16, 16, 3])

    graph = helper.make_graph(
        nodes=[node, relu1, relu2, add],
        name="sample_model_with_multi_outputs",
        inputs=[input1, input2],
        outputs=[add_1],
    )
    model = helper.make_model(graph, producer_name="sample_multi_outputs")

    onnx.save(model, sys.argv[1])


create_sample_onnx_model_with_multiple_outputs()
print("Sample ONNX model with multilpe outputs custom op has been created and saved")
