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
    verbose=False,
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

    if verbose:
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
    g.shape_infer()
    # First Remove the extra QDQ from the mxgan model
    # TODO remove this hard coding, change when we get a new mxgan model
    rm_node_list = ["424_convert_QuantizeLinear", "424_convert_DequantizeLinear"]
    for rm_node in rm_node_list:
        if rm_node in g.nodemap.keys():
            g.skip_node(rm_node)
            g.graph_reorder_nodes()

    # Count number of OPs
    if verbose:
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
    index_dict = {}
    collect_patterns = {}
    for key in fuse_patterns.keys():
        for fuse_pattern in fuse_patterns[key]:
            if verbose:
                print(
                    "vaip/python/voe/passes/op_fusion_index.py:297:21: Pattern Key: {}, Pattern Length: {}".format(
                        key, len(fuse_pattern)
                    )
                )
            try:
                Descs = fuse_pattern
                pattern = FusionPattern(Descs)  # errors if match is not found
                subgraphs = pattern.search_pattern(g)
                if subgraphs:
                    if (
                        key in index_dict
                        and fuse_patterns[key].index(fuse_pattern)
                        not in index_dict[key]
                    ):
                        index_dict[key].append(fuse_patterns[key].index(fuse_pattern))
                    else:
                        index_dict[key] = [fuse_patterns[key].index(fuse_pattern)]
                for nodes in subgraphs:
                    pattern_key = f"{key}_{fuse_patterns[key].index(fuse_pattern)}"
                    if pattern_key not in collect_patterns:
                        k = Graph(
                            g.get_onnxgraph_by_nodenames(nodes),
                            onnx_tool.utils.ModelConfig({}),
                        )
                        collect_patterns[pattern_key] = {
                            "key": pattern_key,
                            "input": k.input,
                            "output": k.output,
                            "model": input_onnx_file,
                        }
                        print(
                            f"vaip/python/voe/passes/op_fusion_index.py:326:28: key={key} pattern_key={pattern_key}"
                        )
                        print(
                            "vaip/python/voe/passes/op_fusion_index.py:326:27: -------------Subgraph input-----------"
                        )
                        print(k.input)
                        print(
                            "vaip/python/voe/passes/op_fusion_index.py:328:27:: -------------Subgraph output----------"
                        )
                        print(k.output)

            except Exception as error:
                if verbose:
                    print(
                        Fore.RED
                        + f"Pattern is not found with {key}-{fuse_patterns[key].index(fuse_pattern)} pattern, going to next avaiable pattern of {key} "
                    )
                    print(error)
                pass
    print(f"collect_patterns={json.dumps(collect_patterns, indent=2, sort_keys=True)}")
    with open("/tmp/pattern.json", "w") as file:
        file.write(
            f"collect_patterns={json.dumps(collect_patterns, indent=2, sort_keys=True)}"
        )
    # if verbose:
    #     print(Fore.LIGHTYELLOW_EX + "Fused Pattern =", key)
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
    if verbose:
        op_count_dictionary_f = count_ops(g)
    # Do a topsort before
    g.graph_reorder_nodes()
    # Save fused graph
    g.save_model(output_onnx_file, rawmodel=m.mproto)

    if verbose:
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
        print(Fore.LIGHTYELLOW_EX + "Pattern name and pattern index\n" + "-" * 40)
        # print(Fore.CYAN + index_dict)

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
        "--verbose",
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
        verbose=args.verbose,
    )
