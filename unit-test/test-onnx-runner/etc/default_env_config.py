#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

## dump env config to json file
import json
import sys

"""
All config fields:

use_memory_model: true | false
enable_vitisai_ep: true | false
batch_number: 1 (default)  [int]
sessuion_count: 1 (default) [int]


session_options:
    graph_optimization_level: 0 | 1 | 2 | 99 (default)
    custom_op_library: ["/path/to/custom_op.so", "/path/to/custom_op2.so"]
    enable_profiling: true | false

session_configs:
    ep.context_enable: 0 | 1
    ep.context_embed_mode: 0 | 1
    ep.context_file_path: "/path/to/context/file"
    ep.shared_ep_context: 0 | 1
    session.workload_type : "Default" | "Efficient"

 provider_options:
    config_file
    cache_key
    cache_dir
    xclbin
    target
    xlnx_target
    encryptionKey
    log_level: "info" | "debug" | "warning" (default) | "error" | "fatal"
    enable_cache_file_io_in_mem
    xlnx_enable_old_qdq
    xlnx_enable_py3_round
    xlnx_aie_tool_path
    xlnx_target_name
"""

default_session_configs = {
    "ep.context_enable": "1",
    "ep.context_embed_mode": "0",
    "ep.context_file_path": "default_ctx.onnx",
    "ep.shared_ep_context": "0",
}
default_provider_options = {
    "log_level": "info",
    "enable_cache_file_io_in_mem": "1",
}

default_config = {
    "enable_vitisai_ep": True,
    "session_count": 2,
    "use_memory_model": False,
    "session_options": {
        "custom_op_library": [],
        "enable_profiling": False,
        "session_configs": {
            **default_session_configs,
        },
        "provider_options": {
            **default_provider_options,
        },
    },
}
single_session = {
    **default_config,
    "skip_test": False,
    "session_count": 1,
    "session_options": {
        **default_config["session_options"],
        "session_configs": {
            **default_session_configs,
            "ep.context_enable": "1",
            "ep.context_embed_mode": "0",
            "ep.context_file_path": "single_session.onnx",
        },
    },
}

offline_compilation_config = {
    **default_config,
    "session_options": {
        **default_config["session_options"],
        "session_configs": {
            **default_session_configs,
            "ep.context_file_path": "offline_ctx.onnx",
        },
    },
}

config = {
    "default_config": default_config,
    "single_session": single_session,
    "offline_compilation_config": offline_compilation_config,
}

json.dump(config, open(sys.argv[1], "w"), indent=4)
