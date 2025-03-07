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
"""
a rule definition.

a rule contains a pattern, an action and optionally where clause.

When applying a rule, we search nodes for the `pattern`, if matched and
`where` clause returns true then the `action` is taken.


"""

import abc
import voe.voe_cpp2py_export as v
from voe.anchor_point import _AnchorPointTag
from typing import List, Optional, Dict, Any
from .builder import Builder
from .imp.node_builder_proxy import NodeBuilderProxy

# re-export  same_as
# pylint: disable=unused-import
# flake8: noqa
from .imp.util import same_as
from .node import Node
from .meta_def import MetaDef


class Rule(metaclass=abc.ABCMeta):
    """a virtual base class for define a rule.

    Every rule need to derived from this class and implement `action`
    and `pattern`."""

    def __init__(self) -> None:
        self._graph = None
        self._pass = None
        self._binder = None

    def initialize(
        self, graph: v.GraphWrapper, vaip_pass: v.Pass, binder: v.Binder
    ) -> None:
        """this funciton is called by c++ to intialize binder"""
        self._graph = graph
        self._pass = vaip_pass
        self._binder = binder

    @abc.abstractmethod
    def pattern(self) -> None:
        """the pattern() is searched for"""

    @abc.abstractmethod
    def action(self, **kwargs: Dict[str, v.Binder]) -> None:
        """the action is taken when the pattern() is matched"""

    def where(self, **_kwargs: Dict[str, v.Binder]) -> Any:
        """the optional where clause"""
        return True

    def _where(self, **kwargs: Dict[str, v.Binder]) -> Any:
        """this funciton is called by c++ code, and convert c++ NodeInput object to Node"""
        return self.where(
            **{k: Node(self._pass, self._graph, v) for (k, v) in kwargs.items()}
        )

    def _action(self, **kwargs: Dict[str, v.Binder]) -> Any:
        """this funciton is called by c++ code, and convert c++ NodeInput object to Node"""
        return self.action(
            **{
                k: Node(self._pass, self._graph, v)
                for (k, v) in kwargs.items()
                if not v.empty()
            }
        )

    def get_node(self, node_name: str) -> Node:
        """return the matched node"""
        assert isinstance(node_name, str)
        return Node(self._pass, self._graph, self._binder[node_name])

    # pylint: disable=too-many-arguments
    def create_node(
        self,
        op_type: Optional[str] = None,
        inputs: Optional[List[Node]] = None,
        attrs: Optional[Dict] = None,
        data_type: Optional[str] = None,
        shape: Optional[Node] = None,
        anchor_point: Optional[_AnchorPointTag] = None,
    ) -> Node:
        """create a a new node"""
        if inputs is None:
            inputs = []
        if attrs is None:
            attrs = {}
        builder = Builder(self._graph.builder(self._pass))

        ## builder will ready to build after NodeBuilderProxy.build()
        NodeBuilderProxy(
            self, builder, op_type, inputs, attrs, data_type, shape, anchor_point
        ).build()
        return builder.build(self._pass, self._graph)

    def try_fuse(
        self,
        name: str,
        inputs: List[Node],
        outputs: List[Node],
        constant_initializers1: List,
        device: str,
    ) -> Optional[v.MetaDefProto]:
        """
        Created a meta_def that recorded all the information needed for fuse.

        :param name: The name of meta_def created. Must be unique.
        :type name: str
        :param inputs: Node's inputs.
        :type inputs: List[Node]
        :param outputs: Node's outputs.
        :type outputs: List[Node]
        :param constant_initializers1: constant initaliziers.
        :type constant_initializers1: List[str]
        :param device: The device name.
        :type device: str
        :return: a created meta_def filled with fuse info.
        :rtype: MetaDefProto
        """
        inputs = [input.as_cpp_node_input() for input in inputs]
        outputs = [output.as_cpp_node_input() for output in outputs]
        metadef = self._pass.try_fuse(
            self._graph,
            name,
            inputs,
            outputs,
            constant_initializers1,
            device,
        )
        return MetaDef(self._pass, self._graph, metadef)

    def cache_dir(self) -> str:
        """
        | Get the cache directory of the model.
        | On Linux, it is /tmp/{usrname}/vaip/.cache/{model's md5sum}.
        | On Windows, it is C:/temp/{model's md5sum}.

        :return: The path of cache directory.
        :rtype: str
        """
        return self._pass.get_cache_dir()

    def has_session_option(self, option):
        """
        Check the user has passed in an option.

        :param option: session option key.
        :type option: str
        :return: If option exists.
        :rtype: bool
        """
        return self._pass.has_session_option(option)

    def get_session_option(self, option):
        """
        Get the sesion option value user passed in.

        :param option: session option key.
        :type option: str
        :return: session option value.
        :rtype: str
        """
        return self._pass.get_session_option(option)

    def get_supported_ops(self):
        return self._pass.get_supported_ops()


def is_rule(obj) -> bool:
    """whether the object is a Rule"""
    return isinstance(obj, Rule)
