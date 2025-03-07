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
import sys
import os
import string

# import glog as log
import numpy as np
from voe.anchor_point import CONST
from voe.pattern import node, wildcard, xir_const_op
from voe.rule_ext import Rule, same_as
import builtins

from .utils import *
from .op_fusion import fuse_layers


verbose = False


class dynamic_dispatcher_2(Rule):
    def __init__(self):
        self.matched = False

    def pattern(self):
        if verbose:
            print("-- DynamicDispatch::pattern pass::start")
        dummy = node("DequantizeLinear", wildcard(), wildcard(), wildcard())
        return dummy.build(locals())

    def where(self, dummy, **kwargs):
        if self.matched:
            return False
        else:
            self.matched = True
            return True

    def action(self, dummy, **kwargs):
        if verbose:
            print("-- DynamicDispatch::action pass::start")
        model_path = os.path.join(self.cache_dir(), "onnx.onnx")
        fused_model_path = os.path.join(self.cache_dir(), "fused.onnx")
        xclbin = self.get_session_option("xclbin")
        log_level = "error"
        if self.has_session_option("log_level"):
            log_level = self.get_session_option("log_level")
            if log_level not in VALID_LOG_LEVELS:
                raise ValueError(f"Invalid log_level: {log_level}")
        if verbose:
            print("-- CacheDir: {}".format(self.cache_dir()))
            print("-- i/p Model Path: {}".format(model_path))
            print("-- o/p Model Path: {}".format(fused_model_path))

        fuse_layers(model_path, fused_model_path, xclbin=xclbin, log_level=log_level)
        if verbose:
            print("-- DynamicDispatch::action pass::done")
        # Always return false
        return False


def rules():
    return [dynamic_dispatcher_2()]
