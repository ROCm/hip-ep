#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import ctypes
import sys

# usage:
#     python morphizen_check_version.py <path_to_onnxruntime_vitis_ai_ep.dll>
morphizen_dll = ctypes.CDLL(sys.argv[1])

morphizen_dll.morphizen_get_build_info.argtypes = []
morphizen_dll.morphizen_get_build_info.restype = ctypes.c_char_p

build_info = morphizen_dll.morphizen_get_build_info()
print("Build info: ")
print(build_info.decode())
