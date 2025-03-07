# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import sys
import logging
import time
import json
from typing import Any, Optional
from pathlib import Path
from collections import deque
import onnx
import onnx.utils
from onnx import helper, shape_inference
from onnx.external_data_helper import load_external_data_for_model
import flexml
from importlib import metadata


def get_vitisai_root_dir() -> str:
    vitisai_root_dir = Path(__file__).parent
    return str(vitisai_root_dir)


def get_vaip_version() -> str:
    return metadata.metadata("voe")["Version"]


# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=protected-access
# pylint: disable=broad-exception-raised
def load_onnx_modelfile(
    model_filename: str = "",
    onnx_external_data_dir: str = "",
) -> Any:
    logger = logging.getLogger("vaiml.compile_onnx2mlopslib")
    model = onnx.load(model_filename, load_external_data=False)

    # per Mathias' request, the onnx_external_data_dir handed to FE stay as it is.
    # if onnx_external_data_dir != "":
    #    # need to load external data
    #    logger.info(f"    Load external data from {onnx_external_data_dir}")
    #    load_external_data_for_model(model, onnx_external_data_dir)

    return model


# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=protected-access
# pylint: disable=broad-exception-raised
def modify_batch_size(model_path, new_batch_size, output_path):
    # Load the ONNX model
    model = load_onnx_modelfile(model_path)

    # Modify the batch size of all input tensors
    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        if len(shape.dim) > 0:
            shape.dim[0].dim_value = new_batch_size

    # Modify the batch size of all value_info tensors
    for value_info in model.graph.value_info:
        shape = value_info.type.tensor_type.shape
        if len(shape.dim) > 0:
            shape.dim[0].dim_value = new_batch_size

    # Modify the batch size of all output tensors
    for output_tensor in model.graph.output:
        shape = output_tensor.type.tensor_type.shape
        if len(shape.dim) > 0:
            shape.dim[0].dim_value = new_batch_size

    # Perform shape inference to propagate the new input shape
    model = shape_inference.infer_shapes(model)

    # Save the modified model to a file
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")


# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=protected-access
# pylint: disable=broad-exception-raised
def compile_microkernel(
    model: Any,
    output_dir: str = "",
    hw_device: str = "stx",
    ai_analyzer_profiling: bool = False,
    ai_analyzer_visualization: bool = False,
    config_file: Optional[str] = None,
    visualization: bool = False,
    debug: bool = False,
) -> Any:
    logger = logging.getLogger("vaiml.compile_microkernel")

    try:
        print("    Compiling microkernel ...", flush=True)
        compiled_model = flexml.compile(
            model,
            None,  # input_shapes
            output_dir=output_dir,
            microkernel_option=1,  # microkernel only flow via custom-op
            fe_match_unsupported_kernels=True,
            device=hw_device,
            config=config_file,
            visualization=visualization,
            enable_f32_to_bf16_conversion=True,
        )
    except Exception as flexml_compile_exception:
        logger.error("Exception during flexml.compile. %s", flexml_compile_exception)
        raise flexml_compile_exception
        print("    Compilation of AIE partition completed", flush=True)
    return compiled_model


# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=protected-access
# pylint: disable=broad-exception-raised
def compile_onnx2mlopslib(
    model_filename: str = "",
    output_dir: str = "",
    max_inputs: int = 0,
    max_outputs: int = 0,
    hw_device: str = "auto",
    ai_analyzer_profiling: bool = False,
    ai_analyzer_visualization: bool = False,
    hw_output_type: str = "aie-exe",
    enable_f32_to_bf16_conversion: Optional[bool] = False,
    enable_f16_to_bf16_conversion: Optional[bool] = False,
    config_file: Optional[str] = None,
    debug: bool = False,
    override_batch_size=False,
    fast_partition_swap: bool = True,
    onnx_external_data_dir: str = "",
    microkernel_operator: str = "",
    microkernel_overlay_array: str = "4x4",
) -> Any:
    """
    This method uses the mlopslib compiler to compile the model.

    Inputs:
    hw_device:
        For internal debug only. Set the target device for flexml.compile()

    hw_output_type:
        For internal debug only. Set the output_type for flexml.compile()

    config_file:
        Json file to specify advanced internal options, all settings
        in config file will have higher precedence

    onnx_external_data_dir:
        Path only to the external data file. ONNX gets the relative file name
        from the model and appends to this path to get the final external
        data file name
    microkernel_operator
        Microkernel operator to be compiled
    microkernel_overlay_array
        4x4 or 2x4x4 overlay for microkernel tiling
    """
    logger = logging.getLogger("vaiml.compile_onnx2mlopslib")

    if model_filename.endswith("ukernel.mlir"):
        model = model_filename
        microkernel_option = 2
    else:
        if override_batch_size:
            model_dir = Path(model_filename).parent
            model_basename = Path(model_filename).stem
            new_batch_size = 1
            updated_model_filename = (
                model_dir / f"{model_basename}_batch{new_batch_size}.onnx"
            )
            modify_batch_size(model_filename, new_batch_size, updated_model_filename)
            model_filename = updated_model_filename

        model = load_onnx_modelfile(model_filename, onnx_external_data_dir)
        microkernel_option = 0
        assign_unique_names(model)
        logger.info(f"    Successfully loaded model {model_filename} ...")

    try:
        logger.info(
            f"    Compiling subgraph using vaiml.compile model_file={model_filename} onnx_external_data_dir={onnx_external_data_dir}"
        )
        flexml.set_ai_analyzer_profiling(ai_analyzer_profiling)
        flexml.set_ai_analyzer_visualization(ai_analyzer_visualization)
        compiled_model = flexml.compile(
            model,
            None,
            output_type=hw_output_type,
            max_inputs=max_inputs,
            max_outputs=max_outputs,
            output_dir=output_dir,
            dse_args="dse-hw-overlay=4x4 dse-memtile-rows=1",
            backend_args="memtile-rows=1",
            device=hw_device,
            partitioner="onnxruntime-vaip-vaiml-pass",
            enable_f32_to_bf16_conversion=enable_f32_to_bf16_conversion,
            enable_f16_to_bf16_conversion=enable_f16_to_bf16_conversion,
            fast_partition_swap=fast_partition_swap,
            config=config_file,
            microkernel_option=microkernel_option,
            onnx_external_data_dir=onnx_external_data_dir,
            microkernel_operator=microkernel_operator,
            microkernel_overlay_array=microkernel_overlay_array,
        )
    except Exception as flexml_compile_exception:
        # Only print error message if debug flag is set
        if debug:
            logger.error(
                "Exception during flexml.compile. %s", flexml_compile_exception
            )
            raise flexml_compile_exception
        else:
            compiled_model = None

    return compiled_model


def assign_unique_names(model):
    # Keep in sync with VaimlSubgraphProcessor::isNodeSupported!

    for node in model.graph.node:
        if len(node.name) == 0 and len(node.output) > 0:
            node.name = "=" + node.op_type + "->" + str(node.output[0])


def dynamic_shape_infer(model_file_name, fixed_seq_lens, batch_size, json_file_path):
    print(f"model_file_name: {model_file_name}")
    seq_len_list = fixed_seq_lens.split(",")
    seq_lens = [int(x.strip()) for x in seq_len_list]

    for seq_len in seq_lens:
        # Load the ONNX model
        model = load_onnx_modelfile(model_file_name)
        json_file_dir = Path(json_file_path + f"_{seq_len}")
        if not json_file_dir.exists():
            json_file_dir.mkdir(parents=True, exist_ok=True)
        shape_json_file = json_file_dir / f"fixed_shapes.json"

        for input_tensor in model.graph.input:
            # Extract the dimensions' names and values
            dims_info = input_tensor.type.tensor_type.shape.dim
            for dim in dims_info:
                if dim.dim_param:
                    if dim.dim_param == "batch_size":
                        dim.ClearField("dim_param")
                        dim.dim_value = batch_size
                    if dim.dim_param == "sequence_length":
                        dim.ClearField("dim_param")
                        dim.dim_value = seq_len

        for output_tensor in model.graph.output:
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_param:
                    if dim.dim_param == "batch_size":
                        dim.ClearField("dim_param")
                        dim.dim_value = batch_size
                    if dim.dim_param == "sequence_length":
                        dim.ClearField("dim_param")
                        dim.dim_value = seq_len

        inferred_model = onnx.shape_inference.infer_shapes(model)
        inferred_graph = inferred_model.graph

        known_shapes = {}
        known_dim_names = {}
        for value_info in inferred_graph.value_info:
            name = value_info.name
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            known_shapes[name] = shape

        json_data = {}
        for node in inferred_graph.node:
            for input_name in node.input:
                if input_name in known_shapes:
                    shape = known_shapes[input_name]
                    if len(shape) > 0 and shape[0] <= 0:
                        shape[0] = batch_size
                    if len(shape) > 1 and shape[1] <= 0:
                        shape[1] = seq_len
                    json_data[input_name] = shape

        with open(shape_json_file, "w") as f:
            json.dump(json_data, f, indent=4)


def update_dst_dep_graph(output_id_map, node_ids, node_idx_map, dst_dep_graph):
    for idx, node in node_idx_map.items():
        for input in node.input:
            if input in output_id_map:
                src_id = output_id_map[input]
                if (
                    (idx in node_ids)
                    and (src_id in node_ids)
                    and (not src_id in dst_dep_graph[idx]["src_ids"])
                ):
                    dst_dep_graph[idx]["src_ids"].add(src_id)
                    dst_dep_graph[idx]["dependency_count"] += 1


def update_src_dep_graph(input_id_map, node_idx_map, src_dep_graph):
    for idx, node in node_idx_map.items():
        for output in node.output:
            if output in input_id_map:
                for dst_id in input_id_map[output]:
                    if not dst_id in src_dep_graph[idx]["dst_ids"]:
                        src_dep_graph[idx]["dst_ids"].add(dst_id)


def insert_node_accor_dep(node_id_group, node_id, order_idx_map):
    insert_idx = len(node_id_group)
    for group_idx, exist_node_id in enumerate(node_id_group):
        if order_idx_map[node_id] < order_idx_map[exist_node_id]:
            insert_idx = group_idx
            break
    node_id_group.insert(insert_idx, node_id)


# bfs
def bfs(
    to_be_processed_ids, supported_node_ids, node_idx_map, src_dep_graph, dst_dep_graph
):
    logger = logging.getLogger("vaiml.find_partition")
    partition = []
    while len(to_be_processed_ids) > 0:
        node_id = to_be_processed_ids.pop(0)
        if node_id in supported_node_ids:
            partition.append(node_id)
            supported_node_ids.remove(node_id)
            dst_dep_graph[node_id][
                "dependency_count"
            ] -= 1  # dependency == -1 means the node has been added to the partition
        else:
            continue
        for dst_id in src_dep_graph[node_id]["dst_ids"]:
            if dst_id in dst_dep_graph.keys():
                if node_id in dst_dep_graph[dst_id]["src_ids"]:
                    dst_dep_graph[dst_id]["src_ids"].remove(node_id)
                    dst_dep_graph[dst_id]["dependency_count"] -= 1
                    if dst_dep_graph[dst_id]["dependency_count"] == 0:
                        to_be_processed_ids.append(dst_id)

    return partition


def has_dependency(node_id_group, node_id, src_dep_graph):
    dep_node_idx = -1
    if len(node_id_group) == 0:
        return (True, dep_node_idx)
    for idx in node_id_group:
        if node_id in src_dep_graph[idx]["dst_ids"]:
            dep_node_idx = idx
            return (True, dep_node_idx)
    return (False, dep_node_idx)


# condition: node_id depends on node_id_group
# return: (True, depend_node_order) if the first node node_id dependency chain is in node_id_group or input_node_ids
#         (False, depend_node_order) if the first node node_id dependency chain is not in node_id_group or input_node_ids
def trace_back_dependency(
    node_id_group, node_id, src_dep_graph, input_node_ids, order_idx_map
):
    dep_node_idx = -1
    if len(node_id_group) == 0:
        return (False, dep_node_idx)

    work_queue = deque()
    for idx in src_dep_graph.keys():
        if node_id in src_dep_graph[idx]["dst_ids"]:
            work_queue.append(idx)
    trace_map = {}
    while len(work_queue) > 0:
        dep_node_id = work_queue.popleft()
        if dep_node_id in input_node_ids:
            trace_map[dep_node_id] = True
        elif dep_node_id in node_id_group:
            for idx in src_dep_graph.keys():
                if dep_node_id in src_dep_graph[idx]["dst_ids"]:
                    work_queue.append(idx)
        else:
            trace_map[dep_node_id] = False

    dep_result = True
    max_dep_order = -1
    for idx, dep in trace_map.items():
        dep_result = dep_result and dep
        if not dep:
            if order_idx_map[idx] > max_dep_order:
                max_dep_order = order_idx_map[idx]
    return (dep_result, max_dep_order)


def should_add_to_partition(
    partition,
    node_id,
    input_node_ids,
    supported_node_ids,
    unsupported_node_ids,
    src_dep_graph,
    order_idx_map,
):
    logger = logging.getLogger("vaiml.should_add_to_partition")
    if node_id not in supported_node_ids:
        return False
    elif node_id in input_node_ids:
        return True
    else:
        (depened, dep_node_idx) = has_dependency(partition, node_id, src_dep_graph)
        if depened:
            return True
        else:
            (depended, dep_node_idx) = has_dependency(
                unsupported_node_ids, node_id, src_dep_graph
            )
            if depended:
                if dep_node_idx in input_node_ids:
                    return True
                else:
                    (dep_trace, dep_order) = trace_back_dependency(
                        unsupported_node_ids,
                        dep_node_idx,
                        src_dep_graph,
                        input_node_ids,
                        order_idx_map,
                    )
                    if dep_trace or (
                        (dep_order >= 0)
                        and (len(partition) > 0)
                        and (dep_order < order_idx_map[partition[0]])
                    ):
                        return True
                    else:
                        return False
    return False


# Find a maximum connected subgraph for supported nodes
def find_partition(
    to_be_processed_ids,
    input_node_ids,
    supported_node_ids,
    unsupported_node_ids,
    node_idx_map,
    src_dep_graph,
    dst_dep_graph,
    order_idx_map,
):
    logger = logging.getLogger("vaiml.find_partition")
    partition = []
    max_dep_order = -1
    while len(to_be_processed_ids) > 0:
        node_id = to_be_processed_ids.pop(0)
        if order_idx_map[node_id] > max_dep_order:
            max_dep_order = order_idx_map[node_id]
        if should_add_to_partition(
            partition,
            node_id,
            input_node_ids,
            supported_node_ids,
            unsupported_node_ids,
            src_dep_graph,
            order_idx_map,
        ):
            partition.append(node_id)
            supported_node_ids.remove(node_id)
            dst_dep_graph[node_id][
                "dependency_count"
            ] -= 1  # dependency == -1 means the node has been added to the partition
        else:
            continue
        for dst_id in src_dep_graph[node_id]["dst_ids"]:
            if dst_id in dst_dep_graph.keys():
                if node_id in dst_dep_graph[dst_id]["src_ids"]:
                    dst_dep_graph[dst_id]["src_ids"].remove(node_id)
                    dst_dep_graph[dst_id]["dependency_count"] -= 1
                    if dst_dep_graph[dst_id]["dependency_count"] == 0:
                        insert_node_accor_dep(
                            to_be_processed_ids, dst_id, order_idx_map
                        )

    return partition


def get_node_dep_list(all_node_ids, node_idx_map, src_dep_graph, all_dst_dep_graph):
    logger = logging.getLogger("vaiml.get_node_dep_list")
    num_nodes = len(all_node_ids)
    logger.info(f"Total number nodes: {num_nodes}")
    partitions = []
    to_be_processed_ids = []
    for idx in all_node_ids:
        if all_dst_dep_graph[idx]["dependency_count"] == 0:
            to_be_processed_ids.append(idx)
    partition = bfs(
        to_be_processed_ids,
        all_node_ids,
        node_idx_map,
        src_dep_graph,
        all_dst_dep_graph,
    )
    if len(partition) != num_nodes:
        logger.info(f"Found one partition contains {len(partition)} nodes")
        logger.info(
            f"Number of nodes in the dependency traveral results: {len(partition)} != {num_nodes} the total number of nodes in the graph "
        )
        logger.error("Exception graph dependency traveral.")

    order_idx_map = {}
    for idx, node_id in enumerate(partition):
        order_idx_map[node_id] = idx
    return order_idx_map


def get_supported_partitions(
    input_node_ids,
    supported_node_ids,
    unsupported_node_ids,
    node_idx_map,
    src_dep_graph,
    supported_dst_dep_graph,
    order_idx_map,
):
    logger = logging.getLogger("vaiml.get_supported_partitions")
    logger.info(f"Number of supported nodes: {len(supported_node_ids)}")
    partitions = []
    to_be_processed_ids = []
    while len(supported_node_ids) > 0:
        for idx in supported_node_ids:
            if supported_dst_dep_graph[idx]["dependency_count"] == 0:
                insert_node_accor_dep(to_be_processed_ids, idx, order_idx_map)
        if len(to_be_processed_ids) == 0:
            logger.info("No node has dependency count == 0")
            return partitions

        partition = find_partition(
            to_be_processed_ids,
            input_node_ids,
            supported_node_ids,
            unsupported_node_ids,
            node_idx_map,
            src_dep_graph,
            supported_dst_dep_graph,
            order_idx_map,
        )
        if len(partition) > 0:
            logger.info(f"Found one partition contains {len(partition)} nodes")
            # insert the partition into the partitions list based on the order_idx_map
            insert_idx = len(partitions)
            for partition_idx, exist_partition in enumerate(partitions):
                if order_idx_map[partition[0]] < order_idx_map[exist_partition[0]]:
                    insert_idx = partition_idx
                    break
            partitions.insert(insert_idx, partition)
    return partitions


def experimental_partitioner(model_path, unsupported_ops_json):
    logger = logging.getLogger("vaiml.experimental_partitioner")
    logger.info(f"Partitioning model {model_path} ...")
    logger.info(f"Unsupported ops file: {unsupported_ops_json}")
    # Load the ONNX model
    model = load_onnx_modelfile(model_path)
    assign_unique_names(model)

    # Load the JSON file with unsupported operations
    with open(unsupported_ops_json, "r") as f:
        unsupported_ops = json.load(f)

    # Get the graph from the model
    graph = model.graph

    # Create dst dependency graphs
    src_dep_graph = {}  # key is src node idx
    supported_dst_dep_graph = {}  # key is dst node idx, only store supported nodes
    all_dst_dep_graph = {}  # key is dst node idx, store all nodes
    # Create a dictionary of edges
    count_nodes = 0
    all_node_ids = set()
    input_node_ids = set()
    node_idx_map = {}
    supported_node_ids = set()
    unsupported_nodes = set()
    input_id_map = {}
    output_id_map = {}

    # initialize containers
    for idx, node in enumerate(graph.node):
        count_nodes += 1
        node_idx_map[idx] = node
        all_node_ids.add(idx)
        if node.name not in unsupported_ops:
            supported_node_ids.add(idx)
            supported_dst_dep_graph[idx] = {
                "dst_id": idx,
                "src_ids": set(),
                "dependency_count": 0,  # number of src nodes it depends on
            }
        else:
            unsupported_nodes.add(idx)
        # map input name to node idx set that consume this input
        for input in node.input:
            if not input in input_id_map:
                input_id_map[input] = []
            input_id_map[input].append(idx)
        # map output name to the node idx that produce this output
        for output in node.output:
            output_id_map[output] = idx

        all_dst_dep_graph[idx] = {
            "dst_id": idx,
            "src_ids": set(),
            "dependency_count": 0,  # number of src nodes it depends on
        }
        src_dep_graph[idx] = {
            "dst_ids": set(),
            "src_id": idx,
        }

    update_src_dep_graph(input_id_map, node_idx_map, src_dep_graph)
    update_dst_dep_graph(output_id_map, all_node_ids, node_idx_map, all_dst_dep_graph)
    for idx in all_dst_dep_graph.keys():
        if all_dst_dep_graph[idx]["dependency_count"] == 0:
            input_node_ids.add(idx)

    all_node_ids_copy = all_node_ids.copy()
    order_idx_map = get_node_dep_list(
        all_node_ids_copy, node_idx_map, src_dep_graph, all_dst_dep_graph
    )

    supported_node_ids_copy = supported_node_ids.copy()
    update_dst_dep_graph(
        output_id_map, supported_node_ids_copy, node_idx_map, supported_dst_dep_graph
    )

    device_partitions = get_supported_partitions(
        input_node_ids,
        supported_node_ids_copy,
        unsupported_nodes,
        node_idx_map,
        src_dep_graph,
        supported_dst_dep_graph,
        order_idx_map,
    )
    logger.info(f"Number of device partitions: {len(device_partitions)}")

    all_partitions = {}
    for i, partition in enumerate(device_partitions):
        partition_key = f"partition_{i}"
        name_partition = []
        for node_id in partition:
            name_partition.append(node_idx_map[node_id].name)
        all_partitions[partition_key] = name_partition

    unsupported_ops_json_folder = unsupported_ops_json.split("/")
    unsupported_ops_json_folder = unsupported_ops_json_folder[:-1]
    partition_result_json = (
        "/".join(unsupported_ops_json_folder) + "/partition_result.json"
    )
    with open(partition_result_json, "w") as f:
        json.dump(all_partitions, f, indent=4)

    logger.info(f"Partition result is saved to {partition_result_json}")
    return device_partitions
