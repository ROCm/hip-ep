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
import sys
import onnx
from . import onnx_graph as ogm
from . import partitioner


def create_idx_node_map(onnx_model):
    return {i: node.name for i, node in enumerate(onnx_model.graph.node)}


def create_node_idx_map(onnx_model):
    return {node.name: i for i, node in enumerate(onnx_model.graph.node)}


def reverse_map(d):
    return {val: key for key, val in d.items()}


def create_adj_list(onnx_model, node_idx_map):
    ogm_graph = ogm.ONNXGraph(onnx_model)

    input_tensors = ogm_graph.getPrimaryInputs()
    stack = []
    visited_nodes = set()
    adj_list = {}
    for tensor in input_tensors:
        child_ops = ogm_graph.getSuccOp(tensor.name)
        for op in child_ops.keys():
            stack.append(op)
            visited_nodes.add(op)

    while stack:
        op = stack.pop()
        idx = node_idx_map[op]
        next_ops = ogm_graph.getSuccOp(op)
        next_op_ids = [node_idx_map[next_op] for next_op in next_ops.keys()]
        adj_list[idx] = next_op_ids

        for next_op in next_ops.keys():
            if next_op not in visited_nodes:
                stack.append(next_op)
                visited_nodes.add(next_op)

    return adj_list


def create_adj_list2(onnx_model, node_idx_map):
    ogm_graph = ogm.ONNXGraph(onnx_model)
    adj_list = {}
    for node in onnx_model.graph.node:
        node_id = node_idx_map[node.name]
        next_nodes = ogm_graph.getSuccOp(node.name)
        next_node_ids = [node_idx_map[name] for name, op in next_nodes.items()]
        adj_list[node_id] = next_node_ids

    return adj_list


def label_node_property(
    onnx_model, node_idx_map, supported_ops, excluded_op_names, _model
):
    res = {}

    for node in onnx_model.graph.node:
        idx = node_idx_map[node.name]
        xnode = _model.graph.nodemap[node.name]

        # Exclude if already marked excluded.
        if (
            node.name in excluded_op_names
            and excluded_op_names[node.name] == node.op_type
        ):
            res[idx] = "CPU"
            continue

        if node.op_type in supported_ops and node.domain == "com.amd":
            res[idx] = "AIE"

        else:
            res[idx] = "CPU"

    return res


def partition_onnx_model(onnx_model, idx_node_map, _model):
    supported_ops = {
        "LayerNorm",
        "QGroupNorm",
        "QGrpNormTrans",
        "MatMulAdd",
        "MatMul",
        "Add",
        "MatMulAddGelu",
        "MHAGRPB",
        "QConv",
        "IConv",
        "QELWEMUL_qdq",
        "QConcateOPs",
        "QLstm",
        "QReshapeTranspose",
        "QGlobalAvgPool",
        "QMHACHANNEL",
        "QMHAWINDOW",
        "QMHA",
        "DQAdd",
        "QConv2MatMul",
        "mzdk5MHA",
        "QSilu",
        "QSlice",
        "QConcat",
        "QResize",
        "QGelu",
        "QMatMulDynamic",
        "QMulSoftmax",
        "xcom-conv2d",
        "QBroadcastAdd",
        "Mladfsoftmax",
        "QuantOP",
        "DeQuantOP",
        "MLADFMATMULA16W8",  # PSS/PST
        "MLADFMATMULA16A16",  # PSS/PST
        "Mladfelwmul",  # PSS/PST
    }
    excluded_op_names = {
        "1024_DequantizeLinear": "Add",  # Final Add in mxgan
        "/Gather_output_0_DequantizeLinear": "MatMul",  # Final MatMul in mxpzi, shape 1x768
        "input_1_QuantizeLinear": "QuantOP",
        "input_2_QuantizeLinear": "QuantOP",
        "output_1_DequantizeLinear": "DeQuantOP",
    }
    node_idx_map = reverse_map(idx_node_map)
    adj_list = create_adj_list2(onnx_model, node_idx_map)
    node_labels = label_node_property(
        onnx_model, node_idx_map, supported_ops, excluded_op_names, _model
    )
    subgraphs = partitioner.partition_graph(adj_list, node_labels)
    cluster = partitioner.subgraph_labels_to_clusters(subgraphs)
    return subgraphs, cluster, node_labels


if __name__ == "__main__":
    onnx_model_path = sys.argv[1]
    onnx_model = onnx.load(onnx_model_path)

    idx_node_map = create_idx_node_map(onnx_model)
    subgraphs, cluster, target_map = partition_onnx_model(onnx_model, idx_node_map)
    print("cluster:", cluster)

    for label, node_ids in cluster.items():
        for node_id in node_ids:
            print(f"{label} : {idx_node_map[node_id]}")
        print()

    print("Total #subgraphs : ", len(cluster))
