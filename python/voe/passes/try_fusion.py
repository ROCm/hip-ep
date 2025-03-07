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
from .op_fusion import get_tuple_list


def create_p(sdxl_model, dict_sdxl, filename="patterns.py"):
    pattern_dict = {}
    m, g = loadmodel(sdxl_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # g.save_model("PSF_w8a8.onnx")
    for key in dict_sdxl.keys():
        descs = create_descs_from_nodenames(g, dict_sdxl[key])
        if key in pattern_dict.keys():
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]
    with open(filename, "w") as f:
        f.write("patterns = ")
        json.dump(pattern_dict, f, indent=2)
    return pattern_dict


def fuse_layers(
    input_onnx_file,
    fuse_patterns,
    output_onnx_file=None,
    gen_subgraphs=False,
    verbose=False,
):
    # Remove Constant nodes
    model_dir, model_name = os.path.split(input_onnx_file)
    model_name = model_name.replace(".onnx", "")
    # remove_const_nodes(input_onnx_file, os.path.join("no_const", model_name+".onnx"))

    if not output_onnx_file:
        os.makedirs(os.path.join(os.getcwd(), model_name), exist_ok=True)
        output_path = os.path.join(os.getcwd(), model_name)
        output_onnx_file = os.path.join(output_path, model_name + ".fused.onnx")
    else:
        output_path = Path(output_onnx_file).parent

    ## Original model
    m, g = loadmodel(input_onnx_file)
    print("Done loading model", model_name)
    # Shape infer
    # g.shape_infer()

    # Count number of OPs
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

    # from .patterns import patterns as fuse_patterns

    # find and fuse each subgraph
    fused_nodes = []
    for key in fuse_patterns.keys():
        print(key)
        for fuse_pattern in fuse_patterns[key]:
            if verbose:
                print(
                    "Pattern Key: {}, Pattern Length: {}".format(key, len(fuse_pattern))
                )
            # try:
            Descs = fuse_pattern

            pattern = FusionPattern(Descs)
            subgraphs = pattern.search_pattern(g)
            if verbose:
                print(
                    Fore.LIGHTYELLOW_EX
                    + f"Number of patterns found with {key}Pattern = ",
                    len(subgraphs),
                )

            count = 0
            mha_count = 0
            mha_2_count = 0

            for nodes in subgraphs:
                print("Fusing Node ", nodes[0], len(nodes), "with ", key)
                k = Graph(
                    g.get_onnxgraph_by_nodenames(nodes),
                    onnx_tool.utils.ModelConfig({}),
                )
                node_outputs = []

                for n in nodes:
                    node_outputs.extend(k.nodemap[n].output)

                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)

        if verbose:
            print(Fore.LIGHTYELLOW_EX + "Fused Pattern =", key)
    if gen_subgraphs:
        print("Subgraphs are stored at ", model_name + "/subgraphs")

    # Count number of OPs
    op_count_dictionary_f = count_ops(g)

    # Add domain name for fused nodes
    g = add_domain(g)
    # Change inputs/initializers
    # g = change_inputs(m, g, precision, conv_key)
    # Do a topsort before
    g.graph_reorder_nodes()
    # Save fused graph
    # onnx.external_data_helper.convert_model_to_external_data(m.mproto)
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

    if gen_subgraphs:
        print(
            Fore.LIGHTYELLOW_EX + "Fused Subgraphs are stored at: {}".format(model_name)
        )


def fuse_main(
    input_model_path,
    pattern_model_path,
    input_cut_points,
    output_cut_points,
    pattern_key="MHA",
):
    m, g = loadmodel(pattern_model_path)

    g1, g2, g3 = g.get_subgraph(input_cut_points, output_cut_points)
    nos = g2.topsort_nodes(g2.nodemap.keys(), g2.input)
    dict_sdxl = {pattern_key: nos}
    patterns = create_p(pattern_model_path, dict_sdxl)
    fuse_layers(input_model_path, patterns, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_path",
        help="model where fusion is to be run",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pattern_model_path",
        help="model where pattern is to be extracted from",
        type=str,
        required=True,
    )
    parser.add_argument("--input_cut_points", nargs="+", default=[])
    parser.add_argument("--output_cut_points", nargs="+", default=[])
    args = parser.parse_args()
    print(f"{args}")
    fuse_main(
        args.input_model_path,
        args.pattern_model_path,
        args.input_cut_points,
        args.output_cut_points,
    )
