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

from .node import Node
import voe.voe_cpp2py_export as v
from typing import List, Any
from voe.anchor_point import _AnchorPointTag


class Builder(object):
    def __init__(self, cpp_node_builder: v.NodeBuilder) -> None:
        self._builder = cpp_node_builder

    def clone_op_type(self, node: Node) -> v.NodeBuilder:
        return self._builder.clone_op_type(node.as_cpp_node())

    def set_op_type(self, op_type: str, domain: str) -> v.NodeBuilder:
        return self._builder.set_op_type(op_type, domain)

    def clone_data_type(self, node: Node) -> v.NodeBuilder:
        return self._builder.clone_data_type(node.as_cpp_node_input())

    def set_data_type(self, data_type: str) -> v.NodeBuilder:
        return self._builder.set_data_type(data_type)

    def clone_shape(self, node: v.NodeInput) -> v.NodeBuilder:
        return self._builder.clone_shape(node.as_cpp_node_input())

    def set_shape(self, shape: List[int]) -> v.NodeBuilder:
        return self._builder.set_shape(shape)

    def clone_inputs(self, node: Node) -> v.NodeBuilder:
        return self._builder.clone_inputs(node.as_cpp_node())

    def clone_attrs(self, node: Node) -> v.NodeBuilder:
        return self._builder.clone_attrs(node.as_cpp_node())

    def set_anchor_point1(self, node: Node) -> v.NodeBuilder:
        """
        This would replace the node entirely with the node created from the builder.
        In the function create_node, it is called as anchor_point=old_node.
        It is usually called on the builder that creates the output node to replace the output node from previous graph.

        :param node: the node to be replaced
        :type node: Node
        :return: the function would return itself for chaining
        :rtype: Builder

        """
        return self._builder.set_anchor_point_node_arg1(node.as_cpp_node_arg())

    def set_anchor_point2(self, node: Node, type: _AnchorPointTag) -> v.NodeBuilder:
        """
        This would create a new with the transformation information.
        For example, to create a Conv2 which operate on an image of NHWC layout, conv_builder.set_anchor_point2(old_input, NCHW2NHWC).
        In the function create_node, it is called as anchor_point=(old_node, type).

        :param node: the node to be replaced
        :type node: Node
        :param type: The Json representation of op type with its attributes
        :type type: _AnchorPointTag
        :return: the function would return itself for chaining
        :rtype: Builder
        """
        return self._builder.set_anchor_point2(node.as_cpp_node_input(), type)

    def set_anchor_point3(
        self, node: Node, type: _AnchorPointTag, shape: List[int]
    ) -> v.NodeBuilder:
        """
        It is identical to set_anchor_point2 + set_shape.
        In the function create_node, it is called as anchor_point=(old_node, type, new_shape).

        :param node: the node to be replaced.
        :type node: Node
        :param type: The Json representation of op type with its attributes
        :type type: _AnchorPointTag
        :param shape: new shape of the node.
        :type shape: Node
        :return: the function would return itself for chaining
        :rtype: Builder
        """
        return self._builder.set_anchor_point3(node.as_cpp_node(), type, shape)

    def set_input_args(self, inputs: Node) -> v.NodeBuilder:
        return self._builder.set_input_args([i.as_cpp_node_arg() for i in inputs])

    def copy_attr(
        self, node: Node, attr_name: str, node_attr_name: str
    ) -> v.NodeBuilder:
        attr_value = node.attr(node_attr_name)
        return self._builder.set_attr(attr_name, attr_value)

    def set_attr(self, attr_name: str, attr_value: Any) -> v.NodeBuilder:
        return self._builder.set_attr(attr_name, attr_value)

    def build(self, vaip_pass: v.Pass, graph: v.GraphWrapper) -> Node:
        return Node(vaip_pass, graph, self._builder.build().as_node_input())
