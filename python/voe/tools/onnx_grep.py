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
from voe.model import *
from voe.pattern import *
import onnxruntime


def get_pattern(filename):
    file_str = open(filename, "r").read()
    pattern_builder = PatternBuilder()
    if filename.endswith(".py"):
        return pattern_builder.create_pattern_by_py(file_str)
    elif filename.endswith(".json"):
        return pattern_builder.create_by_json(file_str)
    else:
        raise RuntimeError("Unsupported pattern data type")


def onnx_grep(args):
    onnxruntime.initialize_session(
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": args.config_file}],
    )
    p = get_pattern(args.pattern)
    print(p.__str__())
    model = Model(args.model)
    graph = model.get_main_graph()
    graph.resolve(True)
    index = graph.get_node_in_topoligical_order()
    for i in index:
        node = graph.get_node(i)
        binder = p.match(node)
        if binder != None:
            print(f"find node: {node.__str__()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model", type=str)
    parser.add_argument("-p", "--pattern", type=str)
    parser.add_argument("config_file", type=str)

    args = parser.parse_args()

    if not args.model:
        raise RuntimeError("no model provided")
    if not args.pattern:
        raise RuntimeError("no pattern provided")
    onnx_grep(args)
