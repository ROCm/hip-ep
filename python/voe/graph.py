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

from .node import *
from typing import List
import voe.voe_cpp2py_export as v


class Graph(object):
    def __init__(self, cpp_graph: v.GraphWrapper) -> None:
        self._graph = cpp_graph

    def resolve(self, is_force: bool) -> None:
        self._graph.resolve(is_force)

    def get_node_in_topoligical_order(self) -> List[int]:
        return self._graph.get_node_in_topoligical_order()

    def get_node(self, index: int) -> Node:
        return Node(self._graph, self._graph.get_node(index))
