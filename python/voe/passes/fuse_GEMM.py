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

# import glog as log
import numpy as np
from voe.anchor_point import CONST
from voe.pattern import node, wildcard, xir_const_op
from voe.rule_ext import Rule, same_as
import os
import string


class fuse_GEMM(Rule):
    def pattern(self):
        self.num = 0
        p_input = wildcard()
        input_w = wildcard()
        input_b = wildcard()

        gemm = node("Gemm", p_input, input_w, input_b)
        return gemm.build(locals())

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(self, p_input, input_w, input_b, gemm, **kwargs):
        self.num = self.num + 1

        inputs = [p_input]
        outputs = [gemm]

        name = input_w.__str__()
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = "".join(c for c in name if c in valid_chars)
        name = filename.replace(" ", "_")  # remove spaces in filenames.

        bias_file = self.cache_dir() + "/" + name + ".bin"
        wts_bin = self.cache_dir() + "/" + name + "_weight.bin"
        file = open(bias_file, "wb")
        file.write(np.array(input_b.const_data(), dtype=np.single))
        file.close()

        weight_f = np.array(input_w.const_data(), dtype=np.single).reshape(
            input_w.shape()[0], input_w.shape()[1]
        )  ##Assuming transB is 1.
        weight_f = np.transpose(weight_f, (1, 0))
        weight_scale = (np.abs(weight_f).max()) / 128
        weight_q = np.clip(np.round(weight_f / weight_scale), -128, 127).astype(np.int8)
        file = open(wts_bin, "wb")
        file.write(np.array(weight_q).tobytes())
        file.close()

        meta_def = self.try_fuse("GEMM_" + str(name), inputs, outputs, [], "GEMM")
        meta_def.set_generic_param("wts_shape_dim_0", str(input_w.shape()[0]))
        meta_def.set_generic_param("wts_shape_dim_1", str(input_w.shape()[1]))

        meta_def.set_generic_param("node_name", str(name))
        meta_def.set_generic_param("bias_file", str(bias_file))
        meta_def.set_generic_param("wts_file", str(wts_bin))
        meta_def.set_generic_param("wts_scale", str(weight_scale))
        meta_def.set_generic_param("cache_dir", str(self.cache_dir()))

        return meta_def.fuse()


def rules():
    return [fuse_GEMM()]
