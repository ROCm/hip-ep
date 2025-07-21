<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# design of graph manipulation


For `morphizen::Graph`, there are the two stages, "consistent" and "inconsistent".
 * The "consistent" stage is the one where the graph is in a valid state
 * The "inconsistent" stage is where the graph is being modified and may not be valid as far as the graph topology is concerned.

## consistent stage

In the "consistent" stage, the graph is in a valid state. This means that all nodes and edges are properly connected, and the graph can be traversed without any issues.


`morphizen::Graph::staging_graph_` is nullptr in this stage and all internal data structures are in a valid state.

1. After `morphizen::Graph::create(...)`, the returned graph object is in the "consistent" stage.
2. After `morphizen::Graph::load(...)`, the returned graph object is in the "consistent" stage.
3. After `morphizen::Graph::clone(...)`, the returned graph object is in the "consistent" stage.
4. After `morphizen::Graph::resolve(...)`, the graph object is in the "consistent" stage.


## insistent stage

After calling any of the following methods, the graph enters the "inconsistent" stage:

1. `morphizen::Graph::add_node(...)`
2. `morphizen::Graph::remove_node(...)`
5. `morphizen::Graph::fuse(...)`
6. `morphizen::Graph::new_node_arg(...)`
7. TODO: list other methods that modify the graph structure

In the "inconsistent" stage, `morphizen::Graph::staging_graph_` is not nullptr, all modifications are made to the staging graph, any newly created `NodeArgIndex` or `NodeIndex` are not fully valid, it can be used to access the underlying `ValueInfoProto` or `NodeProto` object, but it cannot be used to traverse the graph, i.e. `staging_graph_` is not a complete graph, it cannot be traversed, it is only used to save the `ValueInfoProto` or `NodeProto` objects that are to be commited by `Graph::resolve`.

After calling `morphizen::Graph::resolve(...)`, the graph object is back in the "consistent" stage, and all modifications are applied to the graph.

Even in this stage, the graph itself still looks valid and readonly until `morphizen::Graph::resolve(...)` is called. This means that you can still read the graph structure, but you cannot modify it until the resolve method is called.

## **IMPORTANT NOTE**: `morphizen::Graph::resolve(...)`

* the `staging_graph_` is set to nullptr
* the grahp object is in the "consistent" stage
* all `NodeArgIndex` or  `NodeIndex` objects obtained before `Graph::resolve` are invalidated, including the newly created ones.
