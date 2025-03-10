import numpy as np
import onnx
import sys
from onnx import helper, TensorProto


def create_sample_onnx():
    # Define initializers for all ONNX data types

    # Create a dummy node that does nothing but is required to create a valid ONNX graph
    node = helper.make_node(
        op_type="relu",  # Operator
        inputs=["input"],  # Input
        outputs=["output"],  # Output
        domain="com.xilinx",  # Domain
    )
    node.attribute.append(helper.make_attribute("data_type", "float"))
    node.attribute.append(helper.make_attribute("shape", [1]))
    # Create the graph (model)
    graph = helper.make_graph(
        nodes=[node],  # nodes
        name="SampleModelWithAllConstantInitializers",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
        ],  # inputs
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        initializer=[],  # initializers
    )

    # Create the model
    model = helper.make_model(
        graph, producer_name="sample_model_with_all_constant_initializers"
    )
    # onnx.checker.check_model(model)

    # Save the model to a file
    onnx.save(model, sys.argv[1])


create_sample_onnx()
print("Sample ONNX model")
