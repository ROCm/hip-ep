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

from voe.pattern import node, wildcard
from voe.rule_ext import Rule, Node
import os
import pkgutil
from importlib import import_module
import re


def custom_sort_key(s):
    parts = s.split("_", 1)
    if len(parts) == 2:
        prefix, rest = parts
        match = re.search(r"\d+", prefix)
        if match:
            num = int(match.group())
            return (num, rest)
    return (s, "")


def NodeInputSet2GraphFuse(nis_dict):
    nis = nis_dict.values()
    all_node_args = {n.node_arg_name() for n in nis}
    all_nodes = {ni for ni in nis if ni.as_cpp_node() is not None}

    constants = [ni.node_arg_name() for ni in nis if ni.is_constant()]
    inputs = [ni for ni in nis if not ni.is_constant() and ni.as_cpp_node() is None]
    my_node = []
    for node in all_nodes:
        if all(ni.node_arg_name() in all_node_args for ni in node.inputs()):
            my_node.append(node)
    nodes = [str(n.as_cpp_node()) for n in my_node]
    outputs = []
    for n in my_node:
        for ni in n.inputs():
            if ni.node_arg_name() not in constants:
                if str(ni.as_cpp_node()) not in nodes:
                    inputs.append(ni)
        if not any(str(c.as_cpp_node()) in nodes for c in n.get_consumers()):
            outputs.append(n)

    return inputs, outputs, constants


class GraphLabelBaseRule(Rule):
    def __init__(self, name, pattern):
        self._module_name = name
        self._pattern = pattern
        self.num = 0

    def action(self, **kwargs):
        inputs, outputs, constants = NodeInputSet2GraphFuse(kwargs)
        new_node_name = self._module_name + "_" + str(self.num)
        # This pass does not actually run the model, so filling in NMS here has no practical meaning.
        meta_def = self.try_fuse(new_node_name, inputs, outputs, constants, "NMS")
        self.num = self.num + 1
        return meta_def.fuse()

    def where(self, **_kwargs):
        return True

    def pattern(self):
        return self._pattern()


def rules():
    module_names = []
    voe = __import__("voe")
    path = os.path.join(voe.__path__[0], "generate_test_cases_pattern")
    all_modules = pkgutil.iter_modules([path])
    for loader, module_name, is_pkg in all_modules:
        if not is_pkg:
            module_names.append(module_name)
    module_names = sorted(module_names, key=custom_sort_key)
    patterns = {}

    for module_name in module_names:
        module = import_module(f"voe.generate_test_cases_pattern.{module_name}")
        pattern_func = getattr(module, "pattern", None)
        patterns[module_name] = pattern_func

    return [GraphLabelBaseRule(k, v) for k, v in patterns.items()]
