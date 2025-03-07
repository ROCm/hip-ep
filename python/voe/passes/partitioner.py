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
import collections
import os
import time


def unique(vector):
    return list(set(vector))


def stable_unique(vector):
    return list(dict.fromkeys(vector).keys())


def child_graph_to_parent_graph(graph):
    parents = {}
    for node, children in graph.items():
        parents.setdefault(node, [])
        for child in children:
            parents.setdefault(child, []).append(node)
    return parents


class SubGraph:
    def __init__(self, label, nodes=None, next_subgraphs=None):
        self.label = label
        self.nodes = nodes if nodes is not None else []
        self.next_subgraphs = next_subgraphs if next_subgraphs is not None else []

    def __repr__(self):
        return f"SubGraph (label={self.label}, nodes={self.nodes}, next_subs={self.next_subgraphs})"


class CompositeGraph:
    """Graph with a notion of subgraphs"""

    def __init__(self, adj_list):
        self.child_graph = adj_list
        self.parent_graph = child_graph_to_parent_graph(adj_list)

        # labels : map(node_id -> subgraph label it belongs to)
        self.labels = {node: node for node in adj_list.keys()}

        # clusters : map(subgraph label -> [nodes in subgraph])
        self.clusters = {label: [] for label in self.labels.values()}
        for node, label in self.labels.items():
            self.clusters[label].append(node)

    def get_input_nodes(self):
        return tuple(
            node
            for node, parent_nodes in self.parent_graph.items()
            if len(parent_nodes) == 0
        )

    def get_subgraph_label_of_node(self, node):
        return self.labels[node]

    def get_subgraph_for_node(self, node: int) -> int:
        label = self.get_subgraph_label_of_node(node)
        return label

    def get_next_nodes_of_node(self, node):
        return self.child_graph[node]

    def get_input_subgraphs(self) -> list[int]:
        """Return the list subgraphs which are inputs in the graph"""
        nodes = self.get_input_nodes()
        subgs = [self.get_subgraph_label_of_node(node) for node in nodes]
        unique_subgs = stable_unique(subgs)
        return unique_subgs

    def get_nodes_in_subgraph(self, subg: int) -> list[int]:
        """Given a subgraph, return the list of nodes in the subgraph"""
        nodes = self.clusters[subg]
        nodes = stable_unique(nodes)
        return nodes

    def get_next_subgraphs(self, subg: int) -> list[int]:
        """Given a subgraph, return its child subgraphs"""
        nodes = self.get_nodes_in_subgraph(subg)
        next_nodes = []
        for node in nodes:
            child_nodes = self.child_graph[node]
            for child_node in child_nodes:
                if child_node not in nodes:
                    next_nodes.append(child_node)
        next_subgs = [self.get_subgraph_label_of_node(node) for node in next_nodes]
        next_subgs = stable_unique(next_subgs)
        return next_subgs

    def is_cycle_detected(
        self, subg: int, subgs_visited: set[int], subgs_in_stack: set[int]
    ) -> bool:
        subgs_visited.add(subg)
        subgs_in_stack.add(subg)
        next_subgs = self.get_next_subgraphs(subg)
        for next_subg in next_subgs:
            if next_subg in subgs_in_stack:
                return True

            if next_subg not in subgs_visited:
                if self.is_cycle_detected(next_subg, subgs_visited, subgs_in_stack):
                    return True

        subgs_in_stack.remove(subg)
        return False

    def is_dag(self) -> bool:
        subgs_visited = set()
        subgs_in_stack = set()

        input_subgs = self.get_input_subgraphs()
        for subg in input_subgs:
            if self.is_cycle_detected(subg, subgs_visited, subgs_in_stack):
                return False
        return True

    def fuse(self, subgraph1: int, subgraph2: int):
        if subgraph1 < subgraph2:
            subgraph2_nodes = self.clusters[subgraph2]
            self.clusters[subgraph1].extend(subgraph2_nodes)
            for node in subgraph2_nodes:
                self.labels[node] = subgraph1
        elif subgraph2 < subgraph1:
            subgraph1_nodes = self.clusters[subgraph1]
            self.clusters[subgraph2].extend(subgraph1_nodes)
            for node in subgraph1_nodes:
                self.labels[node] = subgraph2
        else:
            pass

    def fuse_all(self, subgraphs: list[int]):
        smallest_label = min(subgraphs)
        other_subg_nodes = []
        for subgraph in subgraphs:
            if subgraph == smallest_label:
                continue
            other_subg_nodes.extend(self.clusters[subgraph])

        for node in other_subg_nodes:
            self.labels[node] = smallest_label
        self.clusters[smallest_label].extend(other_subg_nodes)

    def try_fuse(self, subgraph1: int, subgraph2: int) -> bool:
        # Back up existing setup
        old_labels = self.labels.copy()
        subg1_nodes = list(self.clusters[subgraph1])
        subg2_nodes = list(self.clusters[subgraph2])

        self.fuse(subgraph1, subgraph2)
        if self.is_dag():
            return True
        else:
            self.labels = old_labels.copy()
            self.clusters[subgraph1] = subg1_nodes
            self.clusters[subgraph2] = subg2_nodes
            return False

    def topsort(self):
        input_nodes = self.get_input_nodes()
        stack = []
        stack.extend(input_nodes)
        visited_nodes = set(input_nodes)
        result = []
        while stack:
            node = stack[-1]
            child_nodes = self.get_next_nodes_of_node(node)
            for child_node in child_nodes:
                if child_node not in visited_nodes:
                    stack.append(child_node)
                    visited_nodes.add(child_node)

            if stack[-1] == node:
                result.append(node)
                stack.pop()
        return result[::-1]


def test(graph):
    CGraph = CompositeGraph(graph)
    assert child_graph_to_parent_graph(graph)[0] == []
    assert CGraph.get_input_nodes() == (0,)
    assert CGraph.get_subgraph_label_of_node(1) == 1
    assert CGraph.get_next_nodes_of_node(0) == [1, 2]
    assert CGraph.get_input_subgraphs() == [0]
    assert CGraph.get_next_subgraphs(0) == [1, 2]
    assert CGraph.get_nodes_in_subgraph(1) == [1]
    assert CGraph.is_dag() == True

    CGraph.fuse(0, 1)
    assert 1 not in CGraph.labels.values()
    assert len(CGraph.get_next_subgraphs(0)) == 3
    assert CGraph.is_dag() == True
    print("After fusing 0 & 1 :", CGraph.labels)

    CGraph.fuse(0, 3)
    assert CGraph.is_dag() == True
    print("After fusing 0 & 3 :", CGraph.labels)

    assert CGraph.try_fuse(0, 7) == False
    print("After try_fusing 0 & 7 :", CGraph.labels)

    CGraph.fuse(0, 7)
    assert CGraph.is_dag() == False
    print("After fusing 0 & 7 :", CGraph.labels)

    print("TESTS PASSED")


def partition_graph(adj_graph, property, optimization_flag="L1", sorted=True):
    assert optimization_flag in {"L0", "L1", "L2"}

    queue = collections.deque()
    visited_nodes = set()

    graph = CompositeGraph(adj_graph)

    if optimization_flag == "L0":
        return graph.labels

    # Initial merging of subgraphs on topsorted list of nodes.
    nodes = graph.topsort() if not sorted else list(graph.labels.keys())
    for i, node in enumerate(nodes[:-1]):
        next_node = nodes[i + 1]
        node_property = property[node]
        if node_property != "CPU" and node_property == property[next_node]:
            subg_i = graph.get_subgraph_for_node(node)
            subg_in = graph.get_subgraph_for_node(next_node)
            graph.fuse(subg_i, subg_in)

    # L1 Partition
    for node in graph.get_input_nodes():
        queue.append(node)
        visited_nodes.add(node)

    while queue:
        node = queue.popleft()
        try_fuse = True if property[node] != "CPU" else False
        sg = graph.get_subgraph_for_node(node)
        next_nodes = graph.get_next_nodes_of_node(node)
        for next_node in next_nodes:
            if try_fuse and property[node] == property[next_node]:
                next_sg = graph.get_subgraph_for_node(next_node)
                status = graph.try_fuse(sg, next_sg)

            if next_node not in visited_nodes:
                queue.append(next_node)
                visited_nodes.add(next_node)

    if optimization_flag == "L1":
        return graph.labels

    if optimization_flag == "L2":
        raise RuntimeError("L2 optimization is not implmented in Graph Partitioner")


def subgraph_labels_to_clusters(subgraphs):
    clusters = dict.fromkeys(subgraphs.values())
    for key in clusters.keys():
        clusters[key] = []
    for node, label in subgraphs.items():
        clusters[label].append(node)
    return clusters


if __name__ == "__main__":
    from test_cases import tests

    test(tests[0]["adj_graph"])

    for test in tests:
        subgraphs = partition_graph(
            test["adj_graph"], test["property"], optimization_flag="L1"
        )
        clusters = subgraph_labels_to_clusters(subgraphs)
        print("Expected Result:", test["result"])
        print("Result Obtained:", clusters)
        print("-" * 80)
