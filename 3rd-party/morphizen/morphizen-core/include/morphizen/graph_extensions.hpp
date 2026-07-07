/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/graph.hpp"
#include <memory>
#include <string>
#include <vector>

namespace morphizen {
class MetaDefProto;
struct TryFuseError;
class IPass;
struct NodeBuilder;
} // namespace morphizen

namespace morphizen_cxx {

// Extensions to morphizen-core types
// These are implemented in morphizen-core/src/node_builder.cpp

/**
 * @brief Try to fuse operation nodes into a custom operation node.
 *
 * Similar to GraphRef::fuse(), except it does not actually change the graph.
 * Try to identify if the given list of node arguments can be fused into a
 * custom operator based on the ONNX Runtime fusion API.
 *
 * @param name The name of the custom operation.
 * @param inputs A vector of input node argument names.
 * @param outputs A vector of output node argument names.
 * @param constant_initializers A vector of constant initializer names.
 * @param device The device on which the custom operation runs.
 * @return A pair containing a unique pointer to the fused operation's
 * metadata definition (MetaDefProto) and an error code.
 *
 * If MetaDefProto is nullptr, TryFuseError contains more details.
 * MetaDefProto can be passed to GraphRef::fuse() to change the actual graph.
 */
std::pair<std::unique_ptr<morphizen::MetaDefProto>, morphizen::TryFuseError>
graph_try_fuse(const GraphConstRef& graph, const std::string& name,
               const std::vector<std::string>& inputs,
               const std::vector<std::string>& outputs,
               const std::vector<std::string>& constant_initializers,
               const std::string& device);

/**
 * @brief Fuses the subgraph based on the given meta definition.
 *
 * This function takes a MetaDefProto object and creates a virtual subgraph
 * representation without modifying the actual graph.
 *
 * @param graph The graph to work with.
 * @param meta_def The meta definition used for fusing the subgraph.
 * @return The fused subgraph as a Subgraph object.
 */
Subgraph graph_virtual_fuse(const GraphConstRef& graph,
                            const morphizen::MetaDefProto& meta_def);

/**
 * @brief Fuses the given meta_def into the graph.
 *
 * This function actually modifies the graph by fusing nodes according to
 * the MetaDefProto specification.
 *
 * @param graph The graph to modify.
 * @param meta_def The MetaDefProto to fuse.
 * @return The fused NodeRef.
 */
NodeRef graph_fuse(GraphRef& graph, const morphizen::MetaDefProto& meta_def);

/**
 * @brief Creates a NodeBuilder object.
 *
 * NodeBuilder is a high-level convenience API for building nodes.
 *
 * @param graph The graph to build nodes in.
 * @param pass The pass object to use for building the node.
 * @return The created NodeBuilder object.
 */
morphizen::NodeBuilder graph_node_builder(GraphRef& graph,
                                          morphizen::IPass& pass);

} // namespace morphizen_cxx
