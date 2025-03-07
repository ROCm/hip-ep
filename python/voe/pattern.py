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

import voe.voe_cpp2py_export as v
from typing import Union, List, Dict, Any, Optional
from .node import Node
import json


class Pattern(object):
    def __init__(self, pattern: v.Pattern) -> None:
        self._pattern = pattern

    def match(self, node: Node) -> Optional[v.Binder]:
        return self._pattern.match(node._graph, node._node)

    def __str__(self) -> str:
        return self._pattern.__str__()


class PatternBuilder(object):
    def __init__(self) -> None:
        self._pattern_builder = v.PatternBuilder()

    def create_pattern_by_py(self, file_str: str) -> Pattern:
        return Pattern(self._pattern_builder.create_by_py(file_str))

    def create_by_json(self, file_str: str) -> Pattern:
        return Pattern(self._pattern_builder.create_by_json(file_str))


class _PatternTag(str):
    pass


class PatternJson(dict):
    def __init__(self, **kargs: Dict[str, Any]) -> None:
        super().__init__(**kargs)

    def build(self, env: Dict[str, Any]) -> _PatternTag:
        patterns = self._build(env, [])
        self["is_root"] = True
        return _PatternTag(json.dumps({"patterns": patterns}, sort_keys=True, indent=4))

    def _build(
        self, env: Dict[str, Any], patterns: List["PatternJson"]
    ) -> List["PatternJson"]:
        ## prevent a pattern from building itself more than once.
        for p in patterns:
            if p is self:
                return patterns

        if "call_node" in self:
            self._build_node(env, patterns)
        elif "wildcard" in self:
            self._build_wildcard(env, patterns)
        elif "graph_input" in self:
            self._build_graph_input(env, patterns)
        else:
            assert "not a valid pattern"

        ## put it into the enviroment if this is a named pattern or it
        ## is the root pattern.
        if "is_root" in self or "id" in self:
            patterns.extend([self])
        return patterns

    def _build_node(self, env: Dict[str, Any], patterns: List["PatternJson"]) -> None:
        self["call_node"]["args"] = self._build_node_args(
            self["call_node"]["args"], env, patterns
        )
        self._set_id(env)

    def _build_node_args(
        self, args: List[Node], env: Dict[str, Any], patterns: List["PatternJson"]
    ) -> List:
        for arg in args:
            arg["pattern"]._build(env, patterns)
        return [
            {"name": arg["pattern"]["id"]} if "id" in arg["pattern"] else arg
            for arg in args
        ]

    def _build_wildcard(self, env: Dict[str, Any], patterns: List["PatternJson"]):
        self._set_id(env)

    def _build_graph_input(self, env: Dict[str, Any], patterns: List["PatternJson"]):
        self._set_id(env)

    def _set_id(self, env):
        for k, v in env.items():
            if v is self:
                v["id"] = k


def wildcard() -> PatternJson:
    ret = PatternJson(wildcard={})
    return ret


def graph_input() -> PatternJson:
    ret = PatternJson(graph_input={})
    return ret


def node(op_type, *args: Union[List[Node], List[List[Node]]]) -> "PatternJson":
    node_args = []
    optional_args = []
    for arg in args:
        if not isinstance(arg, list):
            node_args.append(arg)
            optional_args.append(False)
        else:
            for a in arg:
                node_args.append(a)
                optional_args.append(True)
    ret = PatternJson(
        call_node={
            "op_type": op_type,
            "args": [{"pattern": arg} for arg in node_args],
            "optional_args": optional_args,
        }
    )
    return ret


def xir_const_op() -> PatternJson:
    return node("com.xilinx:const")


def is_pattern(obj) -> bool:
    return isinstance(obj, _PatternTag)


def create_subpattern_env() -> Dict:
    """
    Create an environment variable that is used to hold the states of patterns.

    :return: an environment variable.
    :rtype: Dict
    """
    return {}


def update_subpattern_env(
    parent_env: Dict[str, Any], subpattern_env: Dict[str, Any], prefix=None
) -> None:
    """
    Adding the prefix to all variables in the subpattern_env and then copy them into the parent_env.
    It is called within the subpattern function with a prefix to avoid naming clashing before return.
    It is also called without a prefix within the pattern function to merge all subpatterns' environment before calling build().

    :param parent_env: The environment variable for the parent.
    :type parent_env: Dict
    :param subpattern_env: The environment variable for the subpattern.
    :type subpattern_env: Dict
    :param prefix: The prefix to be added on each variable in parent_env.
    :type prefix: str
    """
    for key in subpattern_env:
        if subpattern_env[key] == prefix:
            continue
        elif subpattern_env[key] == parent_env:
            continue
        if not prefix:
            parent_env[key] = subpattern_env[key]
        else:
            parent_env[prefix + "_" + key] = subpattern_env[key]
