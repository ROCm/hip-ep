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
import os
import argparse
import json
import graphviz
import re


def read_json(file_path):
    # return None if file not exists.
    if file_path is None:
        print("[WARN] ", "path = None while load json.")
        return None
    if os.path.exists(file_path):
        dic = json.load(open(file_path, "r"))
        return dic
    else:
        print("[WARN] ", file_path, "not exist.")
        return None


def remove_name_subfix(name):
    # name_protect=on:
    match = re.search(r"\(([^)]+)\)", name)
    if match:
        return match.group(1)
    else:
        return name.split("_vaip_")[0]
    # name_protect=false:
    pattern = re.compile(r"_reshaped_\d+_inserted_fix_\d+$")
    match = pattern.search(name)
    if match:
        name = name[: match.start()]

    pattern = re.compile(r"_reshaped_\d+_inserted_fix_\d+_merged$")
    match = pattern.search(name)
    if match:
        name = name[: match.start()]

    pos = name.find("_recomputation_")
    if pos != -1:
        name = name[:pos]

    pos = name.find("_fix_reshaped_inserted_fix_")
    if pos != -1:
        name = name[:pos]

    pos = name.find("(TransferMatMulToConv2d)_inserted_fix")
    if pos != -1:
        name = name[:pos]

    pos = name.find("_reshaped_inserted_fix_")
    if pos != -1:
        name = name[:pos]

    pos = name.find("_inserted_fix_")
    if pos != -1:
        name = name[:pos]

    pos = name.find("_FixShifter")
    if pos != -1:
        name = name[:pos]

    suffix = "_fix"
    if name.endswith(suffix):
        name = name[: -len(suffix)]

    return name.split("_vaip_")[0]


def _get_node(nodes, node_arg):
    a = [node for node in nodes if node_arg in node.outputs]
    if len(a) == 1:
        return a[0]


class Node:
    def __init__(self, name, op_type, inputs, outputs, device):
        self.name = name
        self.op_type = op_type
        self.device = device
        self.inputs = inputs
        self.outputs = outputs
        self.tags = []
        self.tooltips = []
        self.style = {"shape": "box"}

    def get_id(self):
        return self.name.replace("::", "_").replace(":", "_")

    def add_style(self, key, value):
        self.style[key] = value

    def add_tag(self, tag):
        self.tags.append(tag)

    def add_tooltip(self, tooltip):
        self.tooltips.append(tooltip)

    def get_label(self):
        return "\n".join([*self.tags, "op=" + self.op_type, self.name])

    def get_style(self):
        ret = self.style
        if self.tooltips:
            ret["tooltip"] = "\n".join(self.tooltips)
        return ret

    def __str__(self):
        return f"Node(name={self.name}, op={self.op_type}"

    def __repr__(self):
        return f"Node(name={self.name}, op={self.op_type})"


class Edge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        assert self.src is not None
        assert self.dst is not None
        self.style = {}

    def get_style(self):
        return self.style

    def add_style(self, key, value):
        self.style[key] = value

    def __str__(self):
        return f"Edge({self.src.name},{self.dst.name})"

    def __repr__(self):
        return f"Edge({self.src.name},{self.dst.name})"

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.src == other.src and self.dst == other.dst
        return False


class Graph:
    def __init__(self, report_path, dpu_report_path, name="default"):
        self.name = name
        self.nodes = _init_nodes_from_report(report_path)
        self.edges = _init_edges_from_report(report_path, self.nodes)
        self.subgraphs = _init_subgraph_from_dpu_report(dpu_report_path)
        self._init_node_attrs_from_dpu_report()

    def _init_node_attrs_from_dpu_report(self):
        for sg in self.subgraphs:
            self._init_node_attrs_from_dpu_report_xir_node(
                sg, "xirSubgraphInputs", "XI"
            )
            self._init_node_attrs_from_dpu_report_xir_node(
                sg, "xirSubgraphOutputs", "XO"
            )
            self._init_try_fuse_body(sg)
            self._init_loop_edge(sg)
            self._init_try_fuse_return_values(sg)
            self._init_try_fuse_inputs(sg)
            self._init_try_fuse_outputs(sg)
            self._init_onnx_output_ap(sg)

    def _init_node_attrs_from_dpu_report_xir_node(self, sg, key, flag):
        xir_tensors = sg._data[key]
        for i, x in enumerate(xir_tensors):
            node_name = remove_name_subfix(x["name"])
            node = _get_node(self.nodes, node_name)
            if not node is None:
                node.add_tag(f"SG_{sg._id} {flag}_[{i}/{len(xir_tensors)}]")
                node.add_tooltip(sg.summary())

    def _init_try_fuse_end(self, sg, key, flag):
        nodes = sg._data.get("tryFuse", {}).get(key, [])
        for i, node_name in enumerate(nodes):
            node = _get_node(self.nodes, node_name)
            if not node is None:
                node.add_tag(f"SG_{sg._id} {flag}_[{i}/{len(nodes)}]")

    def _init_try_fuse_return_values(self, sg):
        self._init_try_fuse_end(sg, "returnValues", "RET")

    def _init_try_fuse_inputs(self, sg):
        self._init_try_fuse_end(sg, "inputs", "OI")

    def _init_try_fuse_outputs(self, sg):
        self._init_try_fuse_end(sg, "outputs", "OO")

    def _init_try_fuse_body(self, sg):
        colors = ["green", "yellow"]
        for node_name in sg._data.get("tryFuse", {}).get("body", []):
            node = _get_node(self.nodes, node_name)
            if not node is None:
                node.add_tag(f"BODY_{sg._id}")
                node.add_style("fillcolor", colors[sg._id % 2])
                node.add_style("style", "filled")
                node.add_tooltip(sg.summary())

    def _init_onnx_output_ap(self, sg):
        if not "onnx_output_anchor_point_is_null" == sg._data["status"]:
            return
        return_values = sg._data["tryFuse"]["returnValues"]
        onnx_output_aps = sg._data.get("onnxOutputAnchorPoints", [])

        for i in range(len(onnx_output_aps)):
            output_onnx_ap = onnx_output_aps[i]
            if "nullptr" == output_onnx_ap["name"]:
                node_name = return_values[i]
                node = _get_node(self.nodes, node_name)
                node.add_tag(f"SG{sg._id} onnx_ap not found[{i}/{len(return_values)}]")

    def _init_loop_edge(self, sg):
        loop_paths = sg._data.get("tryFuseLoopPath", [])
        loop_paths.reverse()
        if len(loop_paths) < 2:
            return
        for src, dst in list(zip(loop_paths, loop_paths[1:])):
            edge = self._get_edge(
                _get_node(self.nodes, src), _get_node(self.nodes, dst)
            )
            if not edge is None:
                edge.add_style("color", "red")
                edge.add_style("label", str(sg._id))

    def _get_edge(self, src, dst):
        a = [edge for edge in self.edges if Edge(src, dst) == edge]
        if len(a) == 1:
            return a[0]

    def __str__(self):
        return str([self.name, self.nodes, self.edges])


def _init_nodes_from_report(report_path):
    report = read_json(report_path)
    ret = []
    for item in report["nodeStat"]:
        op_type = item["opType"]
        device = item["device"]
        outputs = item["output"]
        inputs = item["input"]
        name = outputs[0] if len(outputs) > 0 else "unKnown"
        node = Node(name, op_type, inputs, outputs, device)
        ret.append(node)
    return ret


def _init_edges_from_report(report_path, nodes):
    report = read_json(report_path)
    ret = []
    for item in report["nodeStat"]:
        outputs = item["output"]
        inputs = item["input"]
        for src in inputs:
            for dst in outputs:
                src, dst = _get_node(nodes, src), _get_node(nodes, dst)
                if src is not None and dst is not None:
                    ret.append(Edge(src, dst))
    return ret


class Subgraph:  # DPU subgraph  , maybe no use
    def __init__(self, id, data):
        self._id = id
        self._data = data

    def summary(self):
        lines = [
            "SG_"
            + str(self._id)
            + " "
            + self._data["subgrpahName"]
            + " status="
            + self._data["status"]
        ]
        comment = self._data.get("comments", None)
        if comment:
            lines[0] += " " + comment + ";"
        return "\n".join(lines)


def _init_subgraph_from_dpu_report(dpu_report_path):
    dpu_report = read_json(dpu_report_path)
    return [
        Subgraph(sg_id, sg_dic)
        for (sg_id, sg_dic) in enumerate(dpu_report["subgraphs"])
    ]


def render(args, g):
    dot = graphviz.Digraph("Digraph name", comment="comment")
    dot.graph_attr["rankdir"] = "TB"
    dot.graph_attr["layout"] = "dot"

    for node in g.nodes:
        dot.node(node.get_id(), node.get_label(), **node.get_style())

    for edge in g.edges:
        dot.edge(edge.src.get_id(), edge.dst.get_id(), **edge.get_style())

    print("Rendering...")
    dot.render(args.output, format=args.format)
    print("save to", args.output)
    print("save to", args.output + "." + args.format)


def generate_graph(args):
    graph = Graph(
        name="not used yet",
        report_path=args.report_path,
        dpu_report_path=args.dpu_report_path,
    )
    return graph


def main():
    parser = argparse.ArgumentParser(description="Generate svg of ops")

    parser.add_argument(
        "--report_path",
        "-r",
        default="./vitisai_ep_report.json",
        help="path of vitisai_ep_report",
    )
    parser.add_argument(
        "--dpu_report_path",
        "-d",
        default="./vitisai_ep_dpu_report.json",
        help="path of vitisai_ep_dpu_report",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg", "bmp", "pdf", "jpg", "json"],
        default="svg",
        help="output format.",
    )
    parser.add_argument("--output", "-o", default="./output", help="The output file")

    args = parser.parse_args()

    graph = generate_graph(args)
    render(args, graph)


if __name__ == "__main__":
    main()
