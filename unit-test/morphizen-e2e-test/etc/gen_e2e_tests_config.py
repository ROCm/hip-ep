#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

## dump env config to json file
import json
import sys

import platform


def is_windows():
    return platform.system() == "Windows"


single_session_gen_and_run_embed_ctx = {
    "name": "single_session_gen_and_run_embed_ctx",
    "env": {
        "ort_log_level": "warning",
        "ort_log_id": "morphizen_e2e_tests.single_session_gen_and_run_embed_ctx",
        "session_options": [
            {
                "session_configs": {
                    "ep.context_enable": "1",
                    "ep.context_embed_mode": "1",
                    "ep.context_file_path": "single_session_gen_and_run_embed_ctx.onnx",
                    "ep.shared_ep_context": "0",
                },
                "morphizen_ep_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "pt_resnet50.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    }
                ],
            },
            {
                "session_configs": {
                    "ep.context_enable": "0",
                },
                "morphizen_ep_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "single_session_gen_and_run_embed_ctx.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    }
                ],
            },
        ],
    },
}
multiple_session_gen_and_run_embed_ctx = {
    "name": "multiple_session_gen_and_run_embed_ctx",
    "env": {
        "ort_log_level": "warning",
        "ort_log_id": "morphizen_e2e_tests.multiple_session_gen_and_run_embed_ctx",
        "session_options": [
            {
                "session_configs": {
                    "ep.context_enable": "1",
                    "ep.context_embed_mode": "1",
                    "ep.context_file_path": "multiple_session_gen_and_run_embed_ctx.onnx",
                    "ep.shared_ep_context": "0",
                },
                "morphizen_ep_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "pt_resnet50.onnx",
                        "session_count": 2,
                        "run": {
                            "run_count": 1,
                        },
                    },
                    {
                        "model_path": "pt_resnet50.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    },
                ],
            },
            {
                "session_configs": {
                    "ep.context_enable": "0",
                },
                "morphizen_ep_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "multiple_session_gen_and_run_embed_ctx.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    }
                ],
            },
        ],
    },
}
single_session_gen_and_run_non_embed_no_prefix_ctx = {
    "name": "single_session_gen_and_run_non_embed_no_prefix_ctx",
    "env": {
        "ort_log_level": "warning",
        "ort_log_id": "morphizen_e2e_tests.single_session_gen_and_run_non_embed_no_prefix_ctx",
        "session_options": [
            {
                "session_configs": {
                    "ep.context_enable": "1",
                    "ep.context_embed_mode": "0",
                    "ep.context_file_path": "single_session_gen_and_run_non_embed_no_prefix_ctx.onnx",
                },
                "morphizen_ep_param": {
                    "provider_options": {
                        "log_level": "info",
                        "use_cache_key_prefix": "0",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "pt_resnet50.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    }
                ],
            },
            {
                "session_configs": {
                    "ep.context_enable": "0",
                },
                "morphizen_ep_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "single_session_gen_and_run_non_embed_no_prefix_ctx.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    }
                ],
            },
        ],
    },
}

v2_single_session_gen_and_run_embed_ctx = {
    "name": "v2_single_session_gen_and_run_embed_ctx",
    "env": {
        "ort_log_level": "warning",
        "ort_log_id": "morphizen_e2e_tests.v2_single_session_gen_and_run_embed_ctx",
        "registration": [
            {
                "name": "MorphiZenExecutionProvider",
                "library": "onnxruntime_morphizen_ep.dll",
            },
        ],
        "session_options": [
            {
                "session_configs": {
                    "ep.context_enable": "1",
                    "ep.context_embed_mode": "1",
                    "ep.context_file_path": "v2_single_session_gen_and_run_embed_ctx.onnx",
                    "ep.shared_ep_context": "0",
                },
                "v2_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "pt_resnet50.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    },
                ],
            },
            {
                "session_configs": {
                    "ep.context_enable": "0",
                },
                "v2_param": {
                    "provider_options": {
                        "log_level": "info",
                        "enable_cache_file_io_in_mem": "1",
                    }
                },
                "session": [
                    {
                        "model_path": "v2_single_session_gen_and_run_embed_ctx.onnx",
                        "session_count": 1,
                        "run": {
                            "run_count": 1,
                        },
                    },
                ],
            },
        ],
    },
}

## Only enable the V2 API path (AppendExecutionProvider_V2 + RegisterExecutionProviderLibrary).
## The legacy morphizen_ep_param path (AppendExecutionProvider_VitisAI) is not supported
## on the GitHub side.
config = [v2_single_session_gen_and_run_embed_ctx]

json.dump(config, open(sys.argv[1], "w"), indent=4)
