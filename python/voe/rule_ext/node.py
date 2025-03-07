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
from typing import List, Union, Optional, Any


class Node(object):
    def __init__(
        self, vaip_pass: v.Pass, graph: v.GraphWrapper, node_input: v.NodeInput
    ) -> None:
        self._vaip_pass = vaip_pass
        self._graph = graph
        self._node_input = node_input

    def const_data(self) -> List[Any]:
        return self._node_input.const_data(self._vaip_pass, self._graph)

    def create_const(self, data: Union[List[Any], float]) -> None:
        self._node_input.node().create_const(self._vaip_pass, data)

    def get_consumers(self) -> List[v.NodeInput]:
        return [
            Node(self._vaip_pass, self._graph, ni)
            for ni in self._node_input.node_arg().consumers(self._graph)
        ]

    def is_same_node(self, other) -> bool:
        output1 = self.as_cpp_node().outputs()[0]
        output2 = other.as_cpp_node().outputs()[0]
        return output1.name() == output2.name()

    def op_type(self) -> str:
        return self._node_input.node().op_type()

    def shape(self) -> List[int]:
        return self._node_input.node_arg().shape()

    def is_unknown_shape(self) -> bool:
        return self._node_input.node_arg().is_unknown_shape()

    def is_dynamic_shape(self) -> bool:
        return self._node_input.node_arg().is_dynamic_shape()

    def has_attr(self, attr_name: str) -> bool:
        return self._node_input.node().has_attr(attr_name)

    def attr(self, attr_name: str) -> Any:
        return self._node_input.node().attr(attr_name)

    def data_type(self) -> str:
        return self._node_input.data_type()

    def node_arg_name(self) -> str:
        return self.as_cpp_node_arg().name()

    def is_constant(self) -> bool:
        return self.as_cpp_node_arg().is_constant(self._graph)

    def inputs(self) -> Optional[List[v.NodeInput]]:
        if self.as_cpp_node():
            return [
                Node(self._vaip_pass, self._graph, ni)
                for ni in self.as_cpp_node().inputs()
            ]
        return None

    def outputs(self) -> Optional[List[v.NodeInput]]:
        if self.as_cpp_node():
            return [
                Node(self._vaip_pass, self._graph, ni)
                for ni in self.as_cpp_node().outputs()
            ]
        return None

    def as_cpp_node(self) -> v.Node:
        return self._node_input.node()

    def as_cpp_node_arg(self) -> v.NodeArg:
        return self._node_input.node_arg()

    def as_cpp_node_input(self) -> v.NodeInput:
        return self._node_input

    def __str__(self) -> str:
        if self.as_cpp_node():
            return str(self.as_cpp_node())
        return str(self.as_cpp_node_arg())
