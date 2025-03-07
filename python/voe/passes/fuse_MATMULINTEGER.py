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


class fuse_MATMULINTEGER(Rule):
    def pattern(self):
        self.num = 0
        p_input = wildcard()
        input_w = wildcard()
        zp = wildcard()
        matmul = node("MatMulInteger", p_input, input_w, zp, wildcard())

        return matmul.build(locals())

    def where(self, p_input, input_w, zp, matmul):
        if input_w.is_constant():
            return True
        else:
            return False

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(self, p_input, input_w, zp, matmul, **kwargs):
        self.num = self.num + 1

        inputs = [p_input, zp]
        outputs = [matmul]

        name = input_w.__str__()
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = "".join(c for c in name if c in valid_chars)
        name = filename.replace(" ", "_")  # remove spaces in filenames.
        # print("name : ", name)
        wts_bin = self.cache_dir() + "/" + str(name) + "_weight.bin"
        wts_file = self.cache_dir() + "/" + name + ".bin"
        file = open(wts_file, "wb")
        weight = np.array(input_w.const_data(), dtype=np.int8).reshape(
            input_w.shape()[0], input_w.shape()[1]
        )
        file.write(np.array(weight).tobytes())
        file.close()

        meta_def = self.try_fuse(
            "MATMULINTEGER_" + str(name), inputs, outputs, [], "MATMULINTEGER"
        )
        meta_def.set_generic_param("wts_shape_dim_0", str(input_w.shape()[0]))
        meta_def.set_generic_param("wts_shape_dim_1", str(input_w.shape()[1]))

        meta_def.set_generic_param("node_name", str(name))
        meta_def.set_generic_param("wts_file", str(wts_file))
        meta_def.set_generic_param("wts_scale", str(input_w.const_data()[0]))
        meta_def.set_generic_param("cache_dir", str(self.cache_dir()))
        device = os.getenv("DEVICE", default="")
        if device not in ["phx", "stx"]:
            print("Implementaion is not available for the device ", device)
            sys.exit()

        # Perform checks and error out
        if builtins.quant_mode == "w8a16" and device != "stx":
            print(f"{builtins.quant_mode} is not supported on {device} platform")
            raise SystemExit

        meta_def.set_generic_param("impl", str(builtins.impl))
        meta_def.set_generic_param("quant_mode", str(builtins.quant_mode))
        meta_def.set_generic_param("device", device)

        return meta_def.fuse()


def rules():
    return [fuse_MATMULINTEGER()]
