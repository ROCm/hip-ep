##
## Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import os
import numpy as np
import onnx
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model
import json
import time
import sys
from onnx_tool.model import Model
from onnx_tool.graph import Graph
import onnx_tool
from collections import Counter
from tabulate import tabulate
from colorama import init, Fore

np.random.seed(42)

from . import onnx_graph as ogm
from . import onnx_graph_partitioner

TENSOR_PACK_ALIGNMENT = 4  # Bytes


def np_ref(X, A, B):
    return (X @ A) @ B


def align_to_next(n, A):
    """Align the 'n' to a multiple of 'A'
    new_n >= n
    new_n % A = 0
    """
    return ((n + A - 1) // A) * A


def pack_tensors(tensors, alignment=1):
    """Given a list of tensors, create a new tensor with accumulated size.
    Return the new tensor and list of each input tensor to offset in output tensor
    alignment refers to required alignment for each input tensor in the new tensor.
    """
    buffer_size = 0
    res = {}

    for tensor in tensors:
        if isinstance(tensor, onnx.onnx_ml_pb2.TensorProto):
            parse_fn = ogm.parseTensorProto
        else:
            parse_fn = ogm.parseTensorGraphNode

        new_offset = buffer_size
        tensor_info = parse_fn(tensor)
        res[tensor.name] = tensor_info
        res[tensor.name]["offset"] = new_offset
        buffer_size += tensor_info["size_in_bytes"]
        buffer_size = align_to_next(buffer_size, alignment)

    return buffer_size, res


def __assign_xrtkernel_argid(buffer_packs):
    """
    This function assigns an arg_id to each buffer_pack.
    The arg_id is the index of buffer_pack sent into xrt::kernel() call.
    Any change in order will cause UB.
    Current order - input:0, output:1, scratch:2, const: 3
    """
    for i, buffer_pack in enumerate(buffer_packs):
        buffer_pack["xrt_arg_id"] = i
    return buffer_packs


def combine_dicts(dicts, throw_on_duplicates=True):
    """
    Combine multiplle dicts to one
    Throw exception if duplicate keys
    """
    new_dict = {}
    for dict_ in dicts:
        for key, value in dict_.items():
            if throw_on_duplicates and key in new_dict:
                raise Exception(f"Found duplicate key: {key}. Dicts are {dicts}")
            new_dict[key] = value
    return new_dict


def prepare_tensor_maps(tensor_map):
    """
    Reformat the raw data to more structured data for next layer
    tensor_map = {tensor_name -> [tensor size, sub_bo_dict]}
    """
    # Create a mapping from fused_tensor_name -> [buf_sz, arg_id]
    new_tensors = {}
    for i, (key, [buf_sz, tensors]) in enumerate(tensor_map.items()):
        new_tensors[key] = {
            "buffer_size": buf_sz,
            "xrt_arg_id": i,
            "packed_tensors": list(tensors.keys()),
        }

    # Create a mapping from orig_tensor_name -> [fused_tensor_name, arg_id, offset] map
    new_tensors_map = {}
    for new_tensor_label, [_buf_sz, subbos] in tensor_map.items():
        for old_tensor_label, tensor_info in subbos.items():
            if old_tensor_label not in new_tensors_map:
                new_tensors_map[old_tensor_label] = {
                    "packed_buffer_label": new_tensor_label,
                    "xrt_arg_id": new_tensors[new_tensor_label]["xrt_arg_id"],
                    **tensor_info,
                }
            else:
                raise Exception(
                    f"Found duplicate key: {key}. Dicts are {new_tensors_map}"
                )

    return new_tensors, new_tensors_map


def prepare_metadata(graph: ogm.ONNXGraph, tmp_dir: str = "."):
    input_tensors = graph.getPrimaryInputs()
    input_size, input_pack = pack_tensors(input_tensors, TENSOR_PACK_ALIGNMENT)

    output_tensors = graph.getPrimaryOutputs()
    output_size, output_pack = pack_tensors(output_tensors, TENSOR_PACK_ALIGNMENT)

    scratch_tensors = graph.getIntermediateTensors()
    scratch_size, scratch_pack = pack_tensors(scratch_tensors, TENSOR_PACK_ALIGNMENT)

    const_tensors = graph.getConstTensors()
    const_size, const_pack = pack_tensors(const_tensors, TENSOR_PACK_ALIGNMENT)

    # IMP : The order of keys below dictates the xrt buffer args of fused kernel
    new_tensors, new_tensor_map = prepare_tensor_maps(
        {
            "in": [input_size, input_pack],
            "out": [output_size, output_pack],
            "scratch": [scratch_size, scratch_pack],
            "const": [const_size, const_pack],
            "super_instr": [0, {}],
        }
    )

    # Dump const tensors to files and update the tensor map
    const_file_info = graph.writeConsts(const_tensors, tmp_dir)
    for key, (filename, filesz) in const_file_info.items():
        new_tensor_map[key].update({"file_name": filename, "file_size": filesz})

    # Get the order of ops to be executed.
    op_names = graph.topologicalSortOps()
    op_list = []
    for op_name in op_names:
        op = graph.nodev[op_name]
        op_type = op.op.op_type
        in_args = [node.name for node in op.inputs if node.name not in const_pack]
        const_args = [node.name for node in op.inputs if node.name in const_pack]
        out_args = [node.name for node in op.outputs]
        op_attrs = ogm.extract_op_attrs(op.op)
        op_list.append(
            {
                "name": op_name,
                "type": op_type,
                "in_args": in_args,
                "const_args": const_args,
                "out_args": out_args,
                "attrs": op_attrs,
            }
        )

    # Large testcase
    # tmp_sz, tmp_off = pack_tensors(input_tensors+output_tensors+scratch_tensors+const_tensors)
    # print("TMP : ", tmp_sz, tmp_off)
    # new_tensors, new_tensor_map = prepare_tensor_maps({'all':[tmp_sz, tmp_off]})

    return op_list, new_tensors, new_tensor_map, graph.aux_info


def save_tensors_to_json(filename, op_list, new_tensors, tensor_map, aux_info):
    data = {
        "op_list": op_list,
        "fused_tensors": new_tensors,
        "tensor_map": tensor_map,
        "aux_info": aux_info,
    }
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2)
    # print(f"Saved metadata to {filename}")
    return json.dumps(data, indent=2)


def nested_list_to_csv_fmt(x):
    """Convert [[a,b], [c, d]] --> 'a b, c d'"""
    # Comma separated output shape for each tensor.
    tmp = []
    for item in x:
        tmp.append(" ".join(str(subitem) for subitem in item))
    res = ",".join(tmp)
    return res


def prepare_metadef_context_json_from_subgraph(
    model, meta_json, subgraph_label, subgraph_node_ids, onnx_model_tool
):
    """subgraph_nodes: List[indices] of nodes in the model which are part of this subgraph"""
    all_value_tensors = set(
        [tensor.name for tensor in model.graph.value_info]
        + [tensor.name for tensor in model.graph.input]
        + [tensor.name for tensor in model.graph.output]
    )
    subgraph_nodes = {
        model.graph.node[i].name: model.graph.node[i] for i in subgraph_node_ids
    }

    all_producers = onnx_model_tool.graph.producedby
    all_consumers = onnx_model_tool.graph.consumedby
    all_consts = set(onnx_model_tool.graph.initials)
    all_nodes = onnx_model_tool.graph.nodemap

    subgraph_input_tensors = []
    subgraph_output_tensors = []
    # original_output_shapes = {}

    # Extract subgraph I/O, nodes and node outputshape
    node_outputs_from_orig_model = []
    input_q_params = []
    output_q_params = []
    for node_name, node in subgraph_nodes.items():
        for node_input in node.input:
            producers = all_producers.get(node_input, [])
            # Which tensors are inputs of subgraph:
            #   1. Input tensor with no producer AND not an initializer
            #   2. Input tensor whose any producer is not part of subgraph nodes
            if (not producers and node_input not in all_consts) or (
                any([producer not in subgraph_nodes for producer in producers])
            ):
                subgraph_input_tensors.append(node_input)

        for node_output in node.output:
            consumers = all_consumers.get(node_output, [])
            # Which tensors are outputs of subgraph:
            #   1. Output tensor with no consumer AND not an initializer
            #   2. Output tensor whose any cosumer is not part of subgraph nodes
            if (not consumers and node_output not in all_consts) or (
                any([consumer not in subgraph_nodes for consumer in consumers])
            ):
                subgraph_output_tensors.append(node_output)

        output_names = []
        for attr in node.attribute:
            if attr.name == "nodes" and attr.type == onnx.AttributeProto.STRINGS:
                output_names = [val.decode("utf-8") for val in attr.strings]

            # if attr.name == "output_shape" and attr.type == onnx.AttributeProto.INTS:
            #     original_output_shapes[output_names[-1]] = attr.ints

        node_outputs_from_orig_model.extend(output_names)

    subgraph_input_tensors = list(dict.fromkeys(subgraph_input_tensors))
    subgraph_output_tensors = list(dict.fromkeys(subgraph_output_tensors))

    # Extract input Q params
    input_q_params = []
    for subgraph_input in subgraph_input_tensors:
        consumers = all_consumers.get(subgraph_input, [])
        if not consumers:
            input_q_params.append([])
            continue

        current_in_qparams = []
        for consumer in consumers:
            consumer_node = all_nodes[consumer]
            input_idx = [
                i
                for i, in_tensor in enumerate(consumer_node.input)
                if in_tensor == subgraph_input
            ][0]
            node_inq_attr = [
                attr
                for attr in consumer_node.proto.attribute
                if attr.name == "input_q_params"
            ]
            if node_inq_attr:
                scale = node_inq_attr[0].floats[2 * input_idx]
                zp = node_inq_attr[0].floats[2 * input_idx + 1]
                #
                current_in_qparams = [scale, zp]
                break
        input_q_params.append(current_in_qparams)
    # Extract output Q params
    output_q_params = []
    meta_def_output_shapes = []
    for subgraph_output in subgraph_output_tensors:
        producers = all_producers.get(subgraph_output, [])
        if not producers:
            output_q_params.append([])
            continue

        assert len(producers) == 1
        producer = producers[0]
        producer_node = all_nodes[producer]
        output_idx = [
            i
            for i, out_tensor in enumerate(producer_node.proto.output)
            if out_tensor == subgraph_output
        ][0]

        node_outq_attr = [
            attr
            for attr in producer_node.proto.attribute
            if attr.name == "output_q_params"
        ]

        node_outq_params = []
        if node_outq_attr:
            scale = node_outq_attr[0].floats[2 * output_idx]
            zp = node_outq_attr[0].floats[2 * output_idx + 1]
            node_outq_params = [scale, zp]
        output_q_params.append(node_outq_params)

        node_outshape_attr = [
            attr
            for attr in producer_node.proto.attribute
            if attr.name == "orig_output_shape"
        ]
        node_outshape = (
            node_outshape_attr[output_idx].ints if node_outshape_attr else []
        )
        meta_def_output_shapes.append(node_outshape)

    # Meta-def ids
    md_elem = {}
    md_elem["id"] = f"subgraph_{subgraph_label}"
    md_elem["inputs"] = subgraph_input_tensors
    md_elem["outputs"] = subgraph_output_tensors
    md_elem["nodes"] = node_outputs_from_orig_model
    md_elem["device"] = "DOD"
    md_elem["genericParam"] = {}
    md_elem["genericParam"]["meta_json"] = meta_json
    md_elem["genericParam"]["input_q_params"] = nested_list_to_csv_fmt(input_q_params)
    md_elem["genericParam"]["output_q_params"] = nested_list_to_csv_fmt(output_q_params)
    md_elem["genericParam"]["original_output_shapes"] = nested_list_to_csv_fmt(
        meta_def_output_shapes
    )

    return md_elem


def prepare_metadef_context_json(model, meta_json, context_json, onnx_tool_model):
    # Supported Ops
    supported_ops = [
        "MHAGRPB",
        "MatMulAddGelu",
        "MatMulAdd",
        "MatMul",
        "LayerNorm",
        "Add",
    ]

    meta_def_output_names = []  # Output names for metadef
    node_output_names = []  # Output names for supported ops
    node_input_names = []  # Input names for supported ops
    unsupported_node_outputs = []  # Output names for unsupported ops
    unsupported_node_inputs = []  # Input names for unsupported ops

    ql_ops = {}  # Store qualizelinear nodes with it's o/p name
    ql_initializers = {}  # Store qualizelinear input initializers
    # Get input/output quantization params (scale/zp)
    input_q_params = []
    output_q_params = []
    # Node's original output shapes
    original_output_shapes = {}
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            ql_ops[node.output[0]] = node
        if node.op_type == "DequantizeLinear":
            ql_ops[node.input[0]] = node
        if (
            node.op_type in supported_ops
            and node.domain == "com.amd"
            and node.name != "1024_DequantizeLinear"
            and node.name != "/Gather_output_0_DequantizeLinear"
        ):
            for attr in node.attribute:
                if (
                    attr.name == "input_q_params"
                    and attr.type == onnx.AttributeProto.FLOATS
                ):
                    input_q_params.extend(attr.floats)
                elif (
                    attr.name == "output_q_params"
                    and attr.type == onnx.AttributeProto.FLOATS
                ):
                    output_q_params.extend(attr.floats)

            node_output_names.extend(node.output)
            node_input_names.extend(node.input)
            output_names = []
            for attr in node.attribute:
                if attr.name == "nodes" and attr.type == onnx.AttributeProto.STRINGS:
                    output_names = [val.decode("utf-8") for val in attr.strings]
                if (
                    attr.name == "orig_output_shape"
                    and attr.type == onnx.AttributeProto.INTS
                ):
                    original_output_shapes[output_names[-1]] = attr.ints
            meta_def_output_names.extend(output_names)
        else:
            unsupported_node_outputs.extend(node.output)
            unsupported_node_inputs.extend(node.input)

    meta_def_inputs = []
    if len(unsupported_node_outputs):
        for name in unsupported_node_outputs:
            if name in node_input_names:
                meta_def_inputs.append(str(name))
    else:  # For graphs where node inputs are graph inputs
        meta_def_inputs.extend([t.name for t in model.graph.input])

    meta_def_outputs = []
    if len(unsupported_node_inputs):
        for name in unsupported_node_inputs:
            if name in node_output_names:
                meta_def_outputs.append(str(name))
    else:  # For graphs where node inputs are graph inputs
        meta_def_outputs.extend([t.name for t in model.graph.output])

    # Update input qparams
    meta_def_inputs = list(dict.fromkeys(meta_def_inputs))
    if len(unsupported_node_outputs):
        input_q_params = []
        for name in meta_def_inputs:
            node = ql_ops[name]
            scale = onnx_tool_model.graph.tensormap[node.input[1]].numpy.astype(
                np.float32
            )
            zp = onnx_tool_model.graph.tensormap[node.input[2]].numpy.astype(np.float32)
            input_q_params.extend([scale, zp])

    # Update output qparams
    meta_def_outputs = list(dict.fromkeys(meta_def_outputs))
    if len(unsupported_node_inputs):
        output_q_params = []
        for name in meta_def_outputs:
            node = ql_ops[name]
            scale = onnx_tool_model.graph.tensormap[node.input[1]].numpy.astype(
                np.float32
            )
            zp = onnx_tool_model.graph.tensormap[node.input[2]].numpy.astype(np.float32)
            output_q_params.extend([scale, zp])

    # Add original output shapes for pad/depad
    meta_def_output_shape = []
    for name in meta_def_outputs:
        meta_def_output_shape.extend(original_output_shapes[name])

    # Meta-def ids
    md_elem = {}
    md_elem["id"] = "subgraph_1"
    md_elem["inputs"] = meta_def_inputs
    md_elem["outputs"] = meta_def_outputs
    md_elem["nodes"] = meta_def_output_names
    md_elem["device"] = "DOD"
    md_elem["genericParam"] = {}
    md_elem["genericParam"]["meta_json"] = meta_json
    md_elem["genericParam"]["input_q_params"] = " ".join(
        [str(item) for item in input_q_params]
    )
    md_elem["genericParam"]["output_q_params"] = " ".join(
        [str(item) for item in output_q_params]
    )
    md_elem["genericParam"]["original_output_shapes"] = " ".join(
        [str(item) for item in meta_def_output_shape]
    )

    # Check if context.json already exists
    meta_def = {}
    # Add meta def to context data
    meta_def["meta_def"] = [md_elem]

    try:
        with open(context_json, "w") as context_fp:
            json.dump(meta_def, context_fp, indent=2)
    except:
        context_fp.close()
        raise Exception("Unable to open file for writing")
    return meta_def_inputs, meta_def_outputs


def get_tuple_list(dictionary):
    l_t = []
    total = 0
    for key in dictionary.keys():
        l_t.append([key, dictionary[key]])
        total += dictionary[key]
    l_t.append(["Total", total])
    return l_t


def npu_offload_summary(onnx_model, target_label):
    res = []
    for node_id, node_label in target_label.items():
        if node_label == "CPU":
            continue
        res.append(onnx_model.graph.node[node_id].op_type)

    counter = Counter(res)
    return dict(counter)


def generate_transactions(model_path, verbose=False):
    # Load model
    model = onnx.load(model_path)
    # Model/Dir path
    model_dir, model_filename = os.path.split(model_path)
    # Create dir if doesn't exists
    os.makedirs(model_dir, exist_ok=True)

    # Generate context/metadef file
    context_json = os.path.join(model_dir, "context_dod.json")
    meta_json = os.path.join(model_dir, model_filename + ".json")

    # ONNX Tool model
    mcfg = {
        "constant_folding": False,
        "node_rename": False,
        "if_fixed_branch": None,
        "fixed_topk": 0,
        "verbose": False,
    }

    _model = Model(str(model_path), mcfg=mcfg)

    # DD Partitioner Flow starts
    idx_node_map = onnx_graph_partitioner.create_idx_node_map(model)
    (
        node_subgraph_labels,
        subgraph_node_cluster,
        target_label,
    ) = onnx_graph_partitioner.partition_onnx_model(model, idx_node_map, _model)
    if verbose:
        summary = npu_offload_summary(model, target_label)
        print(Fore.LIGHTYELLOW_EX + "NPU Offload Summary:\n" + "-" * 40)
        print(
            Fore.CYAN
            + tabulate(
                get_tuple_list(summary),
                headers=["OP_TYPE", "COUNT"],
                tablefmt="github",
            )
        )

    # _model = Model(str(model_path), mcfg=mcfg)

    subgraph_metadefs = []
    for subgraph_label, subgraph_nodes in subgraph_node_cluster.items():
        if subgraph_nodes and target_label[subgraph_nodes[0]] == "CPU":
            continue

        meta_json_path = os.path.join(
            model_dir, model_filename + f".subgraph_{subgraph_label}.json"
        )
        metadef_data = prepare_metadef_context_json_from_subgraph(
            model, meta_json_path, subgraph_label, subgraph_nodes, _model
        )
        subgraph_metadefs.append(metadef_data)
        # breakpoint()
        node_names = [idx_node_map[node_id] for node_id in subgraph_nodes]
        # breakpoint()
        rawgraph = _model.graph.get_onnxgraph_by_nodenames(node_names)
        ot_Graph = Graph(
            rawgraph,
            onnx_tool.utils.ModelConfig(mcfg),
        )
        subgraph_name = f"fused_dod_subgraph_{subgraph_label}.onnx"
        fused_dod = os.path.join(model_dir, subgraph_name)
        ot_Graph.input = metadef_data["inputs"]
        ot_Graph.output = metadef_data["outputs"]
        ot_Graph.save_model(fused_dod, rawmodel=_model.mproto)

        nm = onnx.load(fused_dod)
        onnx_graph = ogm.ONNXGraph(nm)
        metainfo = prepare_metadata(onnx_graph, tmp_dir=model_dir)
        json_str = save_tensors_to_json(meta_json_path, *metainfo)

    with open(os.path.join(model_dir, "context_dod.json"), "w") as context_fp:
        meta_def = {}
        meta_def["meta_def"] = subgraph_metadefs
        json.dump(meta_def, context_fp, indent=2)

    # DD Partition flow ends

    # # Prepare/save meta-def / context.json for VAI-EP
    # inames, onames = prepare_metadef_context_json(
    #     model, meta_json, context_json, _model
    # )

    # # get subgraph
    # l1, l2, l3 = _model.graph.get_subgraph(inames, onames)

    # subgraph_name = "fused_dod.onnx"
    # fused_dod = os.path.join(model_dir, subgraph_name)
    # l2.save_model(fused_dod, rawmodel=_model.mproto)

    # nm = onnx.load(fused_dod)
    # #
    # onnx_graph = ogm.ONNXGraph(nm)

    # # Get metadata
    # metainfo = prepare_metadata(onnx_graph, tmp_dir=model_dir)

    # # Save meta file
    # json_str = save_tensors_to_json(meta_json, *metainfo)
    # Delete the onnx model
    # if verbose:
    #     if os.path.exists(fused_dod):
    #         os.remove(fused_dod)
    #     if os.path.exists(model_path):
    #         os.remove(model_path)

    # return inames, onames
    return [], []


if __name__ == "__main__":
    generate_transactions(sys.argv[1])
