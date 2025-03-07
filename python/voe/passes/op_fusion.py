##
## Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import numpy
import argparse
import onnx_tool
from onnx_tool import Graph
from onnx_tool.fusion import *
from tabulate import tabulate
from colorama import init, Fore
import onnx
import copy
from onnx import helper, TensorProto
import json
from pathlib import Path
from enum import Enum

from onnx_tool.fusion import create_descs_from_nodenames, FusionPattern
from .utils import *


init(autoreset=True)

# This file contains only fusioncode


def remove_const_nodes(input_model_path):
    sgmodel = onnx.load(input_model_path)
    constant_nodes = {}
    for node in sgmodel.graph.node:
        if node.op_type == "Constant":
            const_data = [data for data in node.attribute[0].t.int64_data]

            constant_nodes[node.output[0]] = (onnx.TensorProto.INT64, const_data)

    names = []

    for node in sgmodel.graph.node:
        if node.op_type == "ReduceSum" or node.op_type == "Unsqueeze":
            for name in node.input:
                if name in constant_nodes.keys():
                    names.append(name)
    for name in names:
        type_name, data = constant_nodes[name]
        if len(data) == 1:
            sgmodel.graph.initializer.append(
                onnx.helper.make_tensor(name, type_name, dims=[1], vals=data)
            )

    sgmodel = onnx.shape_inference.infer_shapes(sgmodel)
    onnx.save(sgmodel, input_model_path)


def get_tuple_list(dictionary):
    l_t = []
    total = 0
    for key in dictionary.keys():
        l_t.append([key, dictionary[key]])
        total += dictionary[key]
    l_t.append(["Total", total])
    return l_t


def enable_pad(m, g, prompt_len, pad_prompt_len):
    # Pad initializers
    # pad_prompt_len=512
    for tns in g.initials:
        tensor = g.tensormap[tns]
        if tensor.numpy.shape == [1, prompt_len, 768] or tensor.numpy.shape == (
            1,
            prompt_len,
            768,
        ):
            izr_1_name = tensor.name
            actual_tensor_izr1 = g.tensormap[izr_1_name].numpy.data
            padded_tensor_izr1 = np.zeros((1, pad_prompt_len, 768)).astype(np.uint16)
            padded_tensor_izr1[:, : actual_tensor_izr1.shape[1], :] = actual_tensor_izr1

            g.tensormap[izr_1_name] = onnx_tool.tensor.create_initial_Tensor(
                izr_1_name,
                padded_tensor_izr1,
            )
            g.initials.append(izr_1_name)
        if tensor.numpy.shape == [
            1,
            12,
            prompt_len,
            prompt_len,
        ] or tensor.numpy.shape == (1, 12, prompt_len, prompt_len):
            izr_2_name = tensor.name
            actual_tensor_izr2 = g.tensormap[izr_2_name].numpy.data
            padded_tensor_izr2 = np.zeros(
                (1, 12, pad_prompt_len, pad_prompt_len)
            ).astype(np.uint16)
            padded_tensor_izr2[
                :, :, : actual_tensor_izr2.shape[2], : actual_tensor_izr2.shape[3]
            ] = actual_tensor_izr2
            g.tensormap[izr_2_name] = onnx_tool.tensor.create_initial_Tensor(
                izr_2_name,
                padded_tensor_izr2,
            )
            g.initials.append(izr_2_name)

    # Change Shapes for all dynamic inputs
    for tns in g.tensormap:
        tensor = g.tensormap[tns]
        if tns not in g.initials:
            if len(tensor.shape) >= 2:
                if type(tensor.shape) is tuple:
                    tensor.shape = list(tensor.shape)
                    for i in range(len(tensor.shape)):
                        if tensor.shape[i] == prompt_len:
                            tensor.shape[i] = pad_prompt_len
                elif type(tensor.shape) is list:
                    for i in range(len(tensor.shape)):
                        if tensor.shape[i] == prompt_len:
                            tensor.shape[i] = pad_prompt_len
    for n_m in g.nodemap:
        node = g.nodemap[n_m]
        if node.op_type == "Reshape":
            shape_tensor = g.tensormap[node.input[1]]
            new_numpy_tensor = np.zeros(shape_tensor.numpy.shape).astype(np.int64)
            for i in range(len(shape_tensor.numpy)):
                if shape_tensor.numpy[i] == prompt_len:
                    new_numpy_tensor[i] = pad_prompt_len
                else:
                    new_numpy_tensor[i] = shape_tensor.numpy[i]
            shape_tensor.numpy = new_numpy_tensor
    g.graph_reorder_nodes()
    return g


def get_precision_from_xclbin(xclbin):
    if "a16w8" in xclbin:
        return "a16w8"
    elif "a8w8" in xclbin:
        return "a8w8"


# 2 scenarios - 1 call from onnx runtime  2 for verification
# onnx_onnx_file=None
# from file path get the output dir (do )
"""
fuse_layer :
- Create subgraphs for all the available patterns (gen_subgraphs == True)
- Fuse the total model with all available patterns 
- counts ops before and after fusion
- Important : Make sure that in dynamic_dispatch_patterns, bigger pattern comes first
    - Example: QMHAGRPB, QMatMulAddGelu, QMatMulADD,QMatMul,QLayerNorm,QSkipAdd
"""


def fuse_layers(
    input_onnx_file,
    output_onnx_file=None,
    xclbin=None,
    gen_subgraphs=False,
    log_level="error",
):
    # TODO:remove xclbin dependency
    if xclbin:
        precision = get_precision_from_xclbin(xclbin)

    # class syntax
    class ModelType(Enum):
        PSF = 1
        mxpzi = 2
        mxgan = 3
        m3uec = 4
        PSS = 5
        PST = 6
        mzdk5 = 7
        mswbjvw = 8

    model_type = "none"
    for model in ModelType:
        if model.name.lower() in xclbin.lower():
            model_type = model.name.lower()

    if "mladf" in xclbin.lower():
        print("- Model Type: {}".format(model_type))
        model_type = ModelType.PSS.name.lower()
        print("- Model Type: {}".format(model_type))

    # Differentiate Conv depend on model type
    if (
        model_type == ModelType.m3uec.name.lower()
        or model_type == ModelType.mzdk5.name.lower()
    ):
        conv_key = "IConv"
    else:
        conv_key = "QConv"
    if "4x4_dpu" in xclbin.lower():
        conv_key = "xcom-conv2d"

    if "4x4_dpu" in xclbin.lower():
        conv_key = "xcom-conv2d"

    design_param = "4x2"
    if "4x4" in xclbin:
        design_param = "4x4"

    # PSS / PST MatMul (Pattern is same as MatMul, const inputs are different)
    # matmul_op_type = "MladfMatMul" if "pss_a16a16" in xclbin.lower() else "QMatMul"

    if log_level == "debug":
        if model_type == "none":
            print(Fore.RED + f"Model type is not availabe, setting to QConv")
        else:
            print("Found Model: {}, Conv type: {}\n".format(model_type, conv_key))

    # Remove Constant nodes
    remove_const_nodes(input_onnx_file)
    model_dir, model_name = os.path.split(input_onnx_file)
    model_name = model_name.replace(".onnx", "")

    if not output_onnx_file:
        os.makedirs(os.path.join(os.getcwd(), model_name), exist_ok=True)
        output_path = os.path.join(os.getcwd(), model_name)
        output_onnx_file = os.path.join(output_path, model_name + ".fused.onnx")
    else:
        output_path = Path(output_onnx_file).parent

    ## Original model
    m, g = loadmodel(input_onnx_file)

    # Shape infer
    # Don't call shape infer for PSS/PST
    if model_type.upper() != "PSS" and model_type.upper() != "PST":
        g.shape_infer()
    # First Remove the extra QDQ from the mxgan model
    # TODO remove this hard coding, change when we get a new mxgan model
    rm_node_list = ["424_convert_QuantizeLinear", "424_convert_DequantizeLinear"]
    for rm_node in rm_node_list:
        if rm_node in g.nodemap.keys():
            g.skip_node(rm_node)
            g.graph_reorder_nodes()

    # Count number of OPs
    if log_level == "info":
        op_count_dictionary = count_ops(g)
    # Change output dtype of Q/DQ layers based on ZP
    g = change_output_dtype(g)
    # Duplicate DQ layer
    g = duplicate_layer(m, g, "DequantizeLinear")
    # Reorder nodes
    g.graph_reorder_nodes()

    # Copy original tensormap
    import copy

    original_tensormap = copy.deepcopy(g.tensormap)

    # Pad model to 128 if prompt len<128
    # TODO This should be a seperate utility in future, as
    #      currently we are only working on attention based models this will be used.
    #      In future we may encounter models where we may need to change some other dimension.
    graph_input = g.tensormap[g.input[0]].shape
    prompt_len = graph_input[1]
    if not gen_subgraphs and "psj" in xclbin.lower():
        if prompt_len < 128:
            pad_prompt_len = 128
            g = enable_pad(m, g, prompt_len, pad_prompt_len)

    # Get all supported subgraph patterns
    # Contains QMHAGRPB, QMatMuL, QMaTMuLAdd, QMaTMuLAddGelu, QLAayerNorm,
    # QSkipAdd(elementwise add), found in PSF, mxgan, mxpzi models
    if model_type == ModelType.mzdk5.name.lower():
        from .mzdk5_patterns import patterns as fuse_patterns
    elif (
        model_type == ModelType.PSS.name.lower()
        or model_type == ModelType.PST.name.lower()
    ):
        from .pss_pst_patterns import patterns as fuse_patterns
    else:
        from .dynamic_dispatch_subgraphs_3 import patterns as fuse_patterns

    # Rename conv key based on model type
    new_dict = {}
    for key, value in fuse_patterns.items():
        if key == "BaseConv":
            new_dict[conv_key] = value
        else:
            new_dict[key] = value
    del fuse_patterns
    # Update fuse patterns
    fuse_patterns = new_dict
    # find and fuse each subgraph
    fused_nodes = []
    nodes_to_remove = []
    mzdk5MHA_Shapes_dict = {}
    for key in fuse_patterns.keys():
        for fuse_pattern in fuse_patterns[key]:
            if log_level == "debug":
                print(
                    "Pattern Key: {}, Pattern Length: {}".format(key, len(fuse_pattern))
                )
            try:
                Descs = fuse_pattern
                pattern = FusionPattern(Descs)
                subgraphs = pattern.search_pattern(g)

                if log_level == "debug":
                    print(
                        Fore.LIGHTYELLOW_EX
                        + f"Number of patterns found with {key}Pattern = ",
                        len(subgraphs),
                    )
                count = 0
                mha_count = 0
                mha_2_count = 0
                mzdk5_mha_count = 0
                for nodes in subgraphs:
                    # place holder removal
                    fuse = True
                    if fuse:
                        if "QConcateOPs" in key:
                            (
                                inp_shape,
                                wt_shape,
                                out_shape,
                                inp_zp,
                                wts_name,
                                attr_json_obj,
                            ) = get_merged_conv_attributes(g, nodes)
                            # list of weight tensor names
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    inp_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "weight_shape",
                                    wt_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "output_shape",
                                    out_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "zero_point",
                                    inp_zp,
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "list_attrs",
                                    attr_json_obj,
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "list_wt_name",
                                    wts_name,
                                )

                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                # k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr(
                                "input_shape",
                                inp_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "weight_shape",
                                wt_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "output_shape",
                                out_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "zero_point",
                                inp_zp,
                            )
                            # breakpoint()
                            g.nodemap[nodes[0]].set_attr(
                                "list_attrs",
                                attr_json_obj,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "list_wt_name",
                                wts_name,
                            )

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                        if "QLstm" in key:
                            if ModelType.mswbjvw.name.lower() in xclbin.lower():
                                continue
                            (
                                inp_shape,
                                w_shape,
                                r_shape,
                                b_shape,
                                output_shape,
                                list_scale,
                                list_zp,
                                list_wt_name,
                            ) = get_lstm_attributes(g, nodes)

                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    inp_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "weight_shape",
                                    wt_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "output_shape",
                                    out_shape,
                                )
                                # k.nodemap[nodes[0]].set_attr(
                                #     "zero_point",
                                #     inp_zp,
                                # )
                                # k.nodemap[nodes[0]].set_attr(
                                #    "Kernel_tensor_name",
                                #    wt_name,
                                # )
                                k.nodemap[nodes[0]].set_attr(
                                    "list_wt_name",
                                    wts_name,
                                )

                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                # k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr(
                                "input_shape",
                                inp_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "w_shape",
                                w_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "r_shape",
                                r_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "b_shape",
                                b_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "output_shape",
                                output_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "zero_points",
                                list_zp,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "scales",
                                list_scale,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "list_wt_name",
                                list_wt_name,
                            )

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        if conv_key in key and "conv2matmul" not in key:
                            (
                                inp_shape,
                                wt_shape,
                                out_shape,
                                inp_zp,
                                wt_name,
                            ) = get_conv_attributes(g, nodes)
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    inp_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "weight_shape",
                                    wt_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "output_shape",
                                    out_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "zero_point",
                                    inp_zp,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "Kernel_tensor_name",
                                    wt_name,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "wt_name",
                                    wt_name,
                                )

                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                # k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr(
                                "input_shape",
                                inp_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "weight_shape",
                                wt_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "output_shape",
                                out_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "zero_point",
                                inp_zp,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "wt_name",
                                wt_name,
                            )

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            if (
                                model_type == ModelType.mzdk5.name.lower()
                            ):  # For case where IConv in mzdk5 needs to be converted to Conv2MatMul
                                if wt_shape[-1] == 1 and wt_shape[-2] == 1:
                                    g.nodemap[nodes[0]].set_attr("Fuse", "True")
                                    g.nodemap[nodes[0]].set_attr("from_iconv", 1)
                                    g.nodemap[nodes[0]].set_attr("model", "mzdk5")

                        if "conv2matmul" in key.lower():
                            (
                                inp_shape,
                                wt_shape,
                                out_shape,
                                inp_zp,
                                wt_name,
                            ) = get_conv_attributes(g, nodes)
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    inp_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "weight_shape",
                                    wt_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "output_shape",
                                    out_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "zero_point",
                                    inp_zp,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "Kernel_tensor_name",
                                    wt_name,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "wt_name",
                                    wt_name,
                                )

                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                # k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr(
                                "input_shape",
                                inp_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "weight_shape",
                                wt_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "output_shape",
                                out_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "zero_point",
                                inp_zp,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "wt_name",
                                wt_name,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            # FOR mzdk5 QKV conv2matmuls, we are removing dq,transpose,q layers from above and adding the outputs of these 3 nodes to the current Conv2Matmul node
                            g, nodes_to_remove, node_outputs = add_3_nodes(
                                g, nodes[0], node_outputs, nodes_to_remove
                            )
                            g.nodemap[nodes[0]].attr["nodes"] = node_outputs

                        if "QMatMul" in key or "MLADFMATMULA16W8" in key:
                            if ModelType.mswbjvw.name.lower() in xclbin:
                                continue
                            en_fuse, mm_out_shape, mm_in_shape = check_if_wts_matmul(
                                g, nodes, original_tensormap
                            )
                            if en_fuse:
                                flag = True  # to make sure we dont find a pattern of matmul nodes that are already fused in other matmul patterns
                                for name in nodes:
                                    if name not in g.nodemap:
                                        flag = False
                                if flag == True:
                                    if gen_subgraphs:
                                        k = Graph(
                                            g.get_onnxgraph_by_nodenames(nodes),
                                            onnx_tool.utils.ModelConfig({}),
                                        )
                                        node_outputs = []
                                        for n in nodes:
                                            node_outputs.extend(k.nodemap[n].output)
                                        save_name, fuse_name = get_layer_name_for_save(
                                            model_name, nodes, key
                                        )
                                        k.save_model(save_name, rawmodel=m.mproto)
                                        k.fuse_subgraph_node_names(
                                            nodes, key, nodes[0], True
                                        )

                                        k = add_domain(k)
                                        k.save_model(fuse_name, rawmodel=m.mproto)
                                        m_k, k = loadmodel(fuse_name)
                                        # if "Add" not in key:
                                        k = change_inputs(m_k, k, precision, conv_key)
                                        k.save_model(fuse_name, rawmodel=m.mproto)

                                    else:
                                        k = Graph(
                                            g.get_onnxgraph_by_nodenames(nodes),
                                            onnx_tool.utils.ModelConfig({}),
                                        )
                                        node_outputs = []
                                        for n in nodes:
                                            node_outputs.extend(k.nodemap[n].output)

                                    g.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    g.nodemap[nodes[0]].set_attr(
                                        "design_param", design_param
                                    )

                                    g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                    g.nodemap[nodes[0]].set_attr(
                                        "orig_output_shape",
                                        mm_out_shape,
                                    )
                                    g.nodemap[nodes[0]].set_attr(
                                        "input_shape",
                                        mm_in_shape,
                                    )

                        if key == "MLADFMATMULA16A16":  # Activation MatMuls
                            if ModelType.mswbjvw.name.lower() in xclbin:
                                continue
                            en_fuse, mm_out_shape, mm_in_shape = check_if_wts_matmul(
                                g, nodes, original_tensormap
                            )
                            if not en_fuse:
                                flag = True  # to make sure we dont find a pattern of matmul nodes that are already fused in other matmul patterns
                                for name in nodes:
                                    if name not in g.nodemap:
                                        flag = False
                                if flag == True:
                                    if gen_subgraphs:
                                        k = Graph(
                                            g.get_onnxgraph_by_nodenames(nodes),
                                            onnx_tool.utils.ModelConfig({}),
                                        )
                                        node_outputs = []
                                        for n in nodes:
                                            node_outputs.extend(k.nodemap[n].output)
                                        save_name, fuse_name = get_layer_name_for_save(
                                            model_name, nodes, key
                                        )
                                        k.save_model(save_name, rawmodel=m.mproto)
                                        k.fuse_subgraph_node_names(
                                            nodes, key, nodes[0], True
                                        )

                                        k = add_domain(k)
                                        k.save_model(fuse_name, rawmodel=m.mproto)
                                        m_k, k = loadmodel(fuse_name)
                                        # if "Add" not in key:
                                        k = change_inputs(m_k, k, precision, conv_key)
                                        k.save_model(fuse_name, rawmodel=m.mproto)

                                    else:
                                        k = Graph(
                                            g.get_onnxgraph_by_nodenames(nodes),
                                            onnx_tool.utils.ModelConfig({}),
                                        )
                                        node_outputs = []
                                        for n in nodes:
                                            node_outputs.extend(k.nodemap[n].output)

                                    g.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    g.nodemap[nodes[0]].set_attr(
                                        "design_param", design_param
                                    )

                                    g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                    g.nodemap[nodes[0]].set_attr(
                                        "orig_output_shape",
                                        mm_out_shape,
                                    )
                                    g.nodemap[nodes[0]].set_attr(
                                        "input_shape",
                                        mm_in_shape,
                                    )

                        elif "QMHAGRPB" in key:
                            (
                                QKT_input_qparams,
                                QKT_output_qparams,
                                VSQKT_input_qparams,
                                VSQKT_output_qparams,
                                softmax_input_qparams,
                                softmax_output_qparams,
                                sub_scale,
                                grpb_add_params,
                                sigmoid_params,
                                div_params,
                                grpb_matmul_add_out_params,
                            ) = get_MHAGRPB_params(g, nodes)
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key, count
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                k.save_model(
                                    save_name,
                                    rawmodel=m.mproto,
                                )

                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k.attr = {}
                                k.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                k.nodemap[nodes[0]].set_attr(
                                    "QKT_input_qparams", QKT_input_qparams
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "QKT_output_qparams", QKT_output_qparams
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "VSQKT_input_qparams", VSQKT_input_qparams
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "VSQKT_output_qparams", VSQKT_output_qparams
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "softmax_input_qparams", softmax_input_qparams
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "softmax_output_qparams", softmax_output_qparams
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "GRPB_sub_params", sub_scale
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "GRPB_add_params", grpb_add_params
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "sigmoid_params", sigmoid_params
                                )
                                k.nodemap[nodes[0]].set_attr("div_params", div_params)

                                k.nodemap[nodes[0]].set_attr(
                                    "grpb_matmul_add_out_params",
                                    grpb_matmul_add_out_params,
                                )

                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                # k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                count += 1

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.attr = {}

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "QKT_input_qparams", QKT_input_qparams
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "QKT_output_qparams", QKT_output_qparams
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "VSQKT_input_qparams", VSQKT_input_qparams
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "VSQKT_output_qparams", VSQKT_output_qparams
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "softmax_input_qparams", softmax_input_qparams
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "softmax_output_qparams", softmax_output_qparams
                            )
                            g.nodemap[nodes[0]].set_attr("GRPB_sub_params", sub_scale)
                            g.nodemap[nodes[0]].set_attr(
                                "GRPB_add_params", grpb_add_params
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "sigmoid_params", sigmoid_params
                            )
                            g.nodemap[nodes[0]].set_attr("div_params", div_params)
                            g.nodemap[nodes[0]].set_attr(
                                "grpb_matmul_add_out_params", grpb_matmul_add_out_params
                            )

                        elif "LayerNorm" in key:
                            # Last layernorm of mxgan (needed to take only pattern till 5th node--- unique case(was not able to generalize))
                            if "LayerNormalization_fused_ReduceMean_8" in nodes:
                                if len(nodes) > 5:
                                    nodes = nodes[:-2]
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            if model_type == ModelType.mzdk5.name.lower():
                                g.nodemap[nodes[0]].set_attr("model", "mzdk5")

                        elif "Slice" in key:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            if model_type == ModelType.mzdk5.name.lower():
                                g.nodemap[nodes[0]].set_attr("model", "mzdk5")

                        elif "QConcat" in key:
                            # nodes = reorder_concat_nodes(g, nodes)

                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                #                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                        elif "QResize" in key:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "gemmv" in key.lower():
                            # print(nodes)
                            out_shape, in_shape = get_gemv_input_output_shape(
                                g, nodes, original_tensormap
                            )
                            fuse = True

                            if fuse:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    # k = change_inputs(m_k, k, precision)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )

                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    in_shape,
                                )

                        elif "QGelu" in key:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "QGrpNormTrans" in key:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "QGroupNorm" in key:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

                            g.nodemap[nodes[0]].set_attr("design_param", design_param)

                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "SkipAdd" in key:
                            fuse, nodes, fuse_bdd = check_if_wts_add(g, nodes)
                            if fuse:
                                if fuse_bdd:
                                    key = "QBroadcastAdd"
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    if (
                                        "Add_290" in nodes
                                    ):  # Hardcoding for the unique case in mxgan TODO : Remove this change and make it generic
                                        node_outputs.extend(
                                            [
                                                "424_convert_QuantizeLinear_Output",
                                                "424_convert_DequantizeLinear_Output",
                                            ]
                                        )

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                if fuse_bdd:
                                    g.fuse_subgraph_node_names(
                                        nodes, "QBroadcastAdd", nodes[0], True
                                    )
                                else:
                                    g.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )
                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )
                                if fuse_bdd:
                                    key = "QSkipAdd"
                                if model_type == ModelType.mzdk5.name.lower():
                                    g.nodemap[nodes[0]].set_attr("model", "mzdk5")

                        elif key == "QMHA":
                            fuse = True

                            if fuse:
                                (
                                    softmax_input_qparams,
                                    softmax_output_qparams,
                                    QKT_input_qparams,
                                    QKT_output_qparams,
                                    VSQKT_input_qparams,
                                    VSQKT_output_qparams,
                                    QKT_K_dim,
                                    VSQKT_k_dim,
                                    MUL_input_qparams,
                                    MUL_weight_qparams,
                                    MUL_output_qparams,
                                ) = get_MHA_qparams(g, nodes)
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key, mha_2_count
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    k.nodemap[nodes[0]].set_attr(
                                        "QKT_input_qparams", QKT_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "QKT_output_qparams", QKT_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_input_qparams", VSQKT_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_output_qparams", VSQKT_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "softmax_input_qparams", softmax_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "softmax_output_qparams", softmax_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "MUL_input_qparams", MUL_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "MUL_weight_qparams", MUL_weight_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "MUL_output_qparams", MUL_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr("QKT_K_dim", QKT_K_dim)
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_K_dim", VSQKT_k_dim
                                    )
                                    k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    mha_2_count += 1
                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "QKT_input_qparams", QKT_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "QKT_output_qparams", QKT_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "VSQKT_input_qparams", VSQKT_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "VSQKT_output_qparams", VSQKT_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "softmax_input_qparams", softmax_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "softmax_output_qparams", softmax_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_input_qparams", MUL_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_weight_qparams", MUL_weight_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_output_qparams", MUL_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr("QKT_K_dim", QKT_K_dim)
                                g.nodemap[nodes[0]].set_attr("VSQKT_K_dim", VSQKT_k_dim)

                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[
                                        g.nodemap[nodes[0]].output[0]
                                    ].shape,
                                    # original_tensormap[node_outputs[-1]].shape,
                                )

                        elif key == "mzdk5MHA":
                            (
                                fuse,
                                mzdk5MHA_Shapes_dict,
                                softmax_input_qparams,
                                softmax_output_qparams,
                                QKT_input_qparams,
                                QKT_output_qparams,
                                VSQKT_input_qparams,
                                VSQKT_output_qparams,
                                QKT_K_dim,
                                VSQKT_k_dim,
                                MUL_input_qparams,
                                MUL_weight_qparams,
                                MUL_output_qparams,
                            ) = get_mzdk5MHA_qparams(
                                g, nodes, mzdk5MHA_Shapes_dict, design_param
                            )
                            # breakpoint()
                            if fuse:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key, mzdk5_mha_count
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    k.nodemap[nodes[0]].set_attr(
                                        "QKT_input_qparams", QKT_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "QKT_output_qparams", QKT_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_input_qparams", VSQKT_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_output_qparams", VSQKT_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "softmax_input_qparams", softmax_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "softmax_output_qparams", softmax_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "MUL_input_qparams", MUL_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "MUL_weight_qparams", MUL_weight_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "MUL_output_qparams", MUL_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr("QKT_K_dim", QKT_K_dim)
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_K_dim", VSQKT_k_dim
                                    )
                                    k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    mzdk5_mha_count += 1
                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "QKT_input_qparams", QKT_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "QKT_output_qparams", QKT_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "VSQKT_input_qparams", VSQKT_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "VSQKT_output_qparams", VSQKT_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "softmax_input_qparams", softmax_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "softmax_output_qparams", softmax_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_input_qparams", MUL_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_weight_qparams", MUL_weight_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_output_qparams", MUL_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr("QKT_K_dim", QKT_K_dim)
                                g.nodemap[nodes[0]].set_attr("VSQKT_K_dim", VSQKT_k_dim)

                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[
                                        g.nodemap[nodes[0]].output[0]
                                    ].shape,
                                    # original_tensormap[node_outputs[-1]].shape,
                                )

                        elif "QMHACHANNEL" in key or "QMHAWINDOW" in key:
                            fuse = True
                            for node in nodes:
                                if node == "a":
                                    fuse = False
                            if fuse:
                                (
                                    softmax_input_qparams,
                                    softmax_output_qparams,
                                    QKT_input_qparams,
                                    QKT_output_qparams,
                                    VSQKT_input_qparams,
                                    VSQKT_output_qparams,
                                    QKT_K_dim,
                                    VSQKT_k_dim,
                                    MUL_input_qparams,
                                    MUL_weight_qparams,
                                    MUL_output_qparams,
                                ) = MHAChannel_q_params(g, nodes)
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key, mha_count
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    k.nodemap[nodes[0]].set_attr(
                                        "QKT_input_qparams", QKT_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "QKT_output_qparams", QKT_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_input_qparams", VSQKT_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_output_qparams", VSQKT_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "softmax_input_qparams", softmax_input_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr(
                                        "softmax_output_qparams", softmax_output_qparams
                                    )
                                    k.nodemap[nodes[0]].set_attr("QKT_K_dim", QKT_K_dim)
                                    k.nodemap[nodes[0]].set_attr(
                                        "VSQKT_K_dim", VSQKT_k_dim
                                    )
                                    # k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    mha_count += 1
                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "QKT_input_qparams", QKT_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "QKT_output_qparams", QKT_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "VSQKT_input_qparams", VSQKT_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "VSQKT_output_qparams", VSQKT_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "softmax_input_qparams", softmax_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "softmax_output_qparams", softmax_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_input_qparams", MUL_input_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_weight_qparams", MUL_weight_qparams
                                )
                                g.nodemap[nodes[0]].set_attr(
                                    "MUL_output_qparams", MUL_output_qparams
                                )
                                g.nodemap[nodes[0]].set_attr("QKT_K_dim", QKT_K_dim)
                                g.nodemap[nodes[0]].set_attr("VSQKT_K_dim", VSQKT_k_dim)

                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )

                        elif "QSilu" in key:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "QReshapeTranspose" in key:
                            fused_node_set = set(fused_nodes)
                            new_nodes = set(nodes)
                            common_nodes = fused_node_set.intersection(new_nodes)
                            if len(common_nodes) != 0:
                                continue
                            else:
                                fused_nodes.extend(nodes)

                            if (
                                g.nodemap[nodes[-1]].nextnodes[0].nextnodes[0].op_type
                                == "Transpose"
                            ):
                                continue

                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "QGlobalAvgPool" in key:
                            inp_shape, out_shape, inp_zp = get_global_avg_pool_attrs(
                                g, nodes
                            )
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    inp_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "output_shape",
                                    out_shape,
                                )
                                k.nodemap[nodes[0]].set_attr(
                                    "zero_point",
                                    inp_zp,
                                )
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)
                            g.nodemap[nodes[0]].set_attr(
                                "input_shape",
                                inp_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "output_shape",
                                out_shape,
                            )
                            g.nodemap[nodes[0]].set_attr(
                                "zero_point",
                                inp_zp,
                            )
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "DQAdd" in key:
                            fuse_dq_add = True
                            for node in nodes:
                                if (
                                    g.nodemap[node].op_type == "DequantizeLinear"
                                    and g.nodemap[node].domain == "com.microsoft"
                                ):
                                    fuse_dq_add = False
                                    break
                            if not fuse_dq_add:
                                continue
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision, conv_key)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("design_param", design_param)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "orig_output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )

                        elif "QELWEMUL_qdq" in key:
                            fuse = True
                            fuse, mm_out_shape = check_if_ele_mul(
                                g, nodes, original_tensormap
                            )
                            if fuse:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    # k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )
                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )
                        elif "QuantOP" in key or "DeQuantOP" in key:
                            fuse = True
                            if fuse:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    # k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                k.nodemap[nodes[0]].set_attr(
                                    "input_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr(
                                    "design_param", design_param
                                )
                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )
                        elif "Mladfsoftmax" in key:
                            fuse = check_if_mladfsoftmax(g, nodes, original_tensormap)
                            if fuse:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    # k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )
                        elif "Mladfelwmul" in key:
                            fuse = check_if_mladfmul(g, nodes, original_tensormap)
                            if fuse:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    # k = change_inputs(m_k, k, precision, conv_key)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []

                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "orig_output_shape",
                                    original_tensormap[node_outputs[-1]].shape,
                                )

            except Exception as error:
                if log_level == "debug":
                    print(
                        Fore.RED
                        + f"Pattern is not found with {key}-{fuse_patterns[key].index(fuse_pattern)} pattern, going to next avaiable pattern of {key} "
                    )
                    print(error)
                pass

    if gen_subgraphs:
        print("Subgraphs are stored at ", model_name + "/subgraphs")

    # Add domain name for fused nodes
    g = add_domain(g)

    # remove the nodes in the remove list
    for rm_node in nodes_to_remove:
        if rm_node in g.nodemap.keys():
            g.skip_node(rm_node)
            g.graph_reorder_nodes()
    # Change inputs/initializers
    g = change_inputs(m, g, precision, conv_key)
    # Count number of OPs
    if log_level == "info":
        op_count_dictionary_f = count_ops(g)
    # Do a topsort before
    g.graph_reorder_nodes()
    # Save fused graph
    g.save_model(output_onnx_file, rawmodel=m.mproto)

    if log_level == "info":
        print(
            Fore.LIGHTYELLOW_EX + "Opcount before fusion (original model)\n" + "-" * 40
        )
        print(
            Fore.CYAN
            + tabulate(
                get_tuple_list(op_count_dictionary),
                headers=["OP_TYPE", "COUNT"],
                tablefmt="github",
            )
        )
        print(Fore.LIGHTYELLOW_EX + "Opcount after fusion (Fused model)\n" + "-" * 40)
        print(
            Fore.CYAN
            + tabulate(
                get_tuple_list(op_count_dictionary_f),
                headers=["OP_TYPE", "COUNT"],
                tablefmt="github",
            )
        )

    if gen_subgraphs:
        print(
            Fore.LIGHTYELLOW_EX + "Fused Subgraphs are stored at: {}".format(model_name)
        )


def create_patterns(
    model_name,
    psf_a16_model=None,
    psf_a8_model=None,
    mxgan_a16_model=None,
    mxgan_a8_model=None,
    m3uec_a16_model=None,
):
    pattern_dict = {}
    m, g = loadmodel(psf_a8_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # g.save_model("PSF_w8a8.onnx")
    for key in dict_PSF_a8.keys():
        descs = create_descs_from_nodenames(g, dict_PSF_a8[key])
        if key in pattern_dict.keys():
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]
        # pattern=FusionPattern(descs)
        # subgraphs=pattern.search_pattern(g)
        # for subgraph in subgraphs:
        #     # print(subgraph)
        #     g.fuse_subgraph_node_names(subgraph, key, subgraph[0], True)

    m, g = loadmodel(psf_a16_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # pattern_dict={}
    for key in dict_PSF_a16.keys():
        descs = create_descs_from_nodenames(g, dict_PSF_a16[key])
        if key in pattern_dict:
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]

    m, g = loadmodel(mxgan_a16_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # pattern_dict={}
    for key in dict_mxgan_a16.keys():
        descs = create_descs_from_nodenames(g, dict_mxgan_a16[key])
        if key in pattern_dict:
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]

    m, g = loadmodel(mxgan_a8_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # pattern_dict={}
    g.save_model("mxgan_a8_afterdequant.onnx")
    for key in dict_mxgan_a8.keys():
        descs = create_descs_from_nodenames(g, dict_mxgan_a8[key])
        if key in pattern_dict:
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]

    with open("dynamic_dispatch_subgraphs_2.py", "w") as f:
        f.write("patterns = ")
        json.dump(pattern_dict, f, indent=2)


def sort_large_pattern_first(patterns_dict):
    len_dict = {}
    for key in patterns_dict.keys():
        if len(patterns_dict[key]) > 1:
            patterns_dict[key] = sorted(patterns_dict[key], key=len, reverse=True)
        len_dict[key] = len(patterns_dict[key][0])
    len_dict = dict(sorted(len_dict.items(), key=lambda item: item[1], reverse=True))
    new_dict = {}
    for key in len_dict.keys():
        new_dict[key] = patterns_dict[key]
    # breakpoint()
    return new_dict


def add_new_pattern(model, nodes_dict):
    if "mzdk5" in model.lower():
        from .mzdk5_patterns import patterns
    else:
        from .dynamic_dispatch_subgraphs_3 import patterns

    m, g = loadmodel(model)
    g = change_output_dtype(g)
    g.shape_infer()
    g = duplicate_layer(m, g, "DequantizeLinear")
    # g.save_model("PSS_afterdequant.onnx")  # Uncomment this line and collect nodenames from this model and place them in nodes_dict
    for key in nodes_dict.keys():
        for i in range(len(nodes_dict[key])):
            descs = create_descs_from_nodenames(g, nodes_dict[key][i])
            if key in patterns:
                patterns[key].append(descs)
            else:
                patterns[key] = [descs]
    patterns = sort_large_pattern_first(patterns)
    with open("dynamic_dispatch_subgraphs_v3.py", "w") as f:
        f.write("patterns = ")
        json.dump(patterns, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_path",
        help="input onnx model path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_model_path", help="model output path", type=str, required=False
    )

    parser.add_argument(
        "--onnx_checker", help="To validate exported onnx model", action="store_true"
    )
    parser.add_argument(
        "--model_name",
        help="To validate exported onnx model",
        default="PSF",
    )
    parser.add_argument(
        "--gen_subgraphs",
        help="Enable subgraph generation",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument("--xclbin", help="XCLBIN file name", default="", required=True)
    parser.add_argument(
        "--log_level",
        help="Enable debug prints",
        default=False,
        required=False,
        action="store_true",
    )

    args = parser.parse_args()
    print(f"{args}")

    # add_new_pattern(
    #     args.input_model_path,
    #     mzdk5_dict,
    # )

    fuse_layers(
        args.input_model_path,
        args.output_model_path,
        xclbin=args.xclbin,
        gen_subgraphs=args.gen_subgraphs,
        log_level=args.log_level,
    )
