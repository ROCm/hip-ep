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
import string
import os

# import glog as log
import numpy as np
from voe.anchor_point import CONST
from voe.pattern import node, wildcard, xir_const_op
from voe.rule_ext import Rule, same_as
import builtins


class fuse_MATMUL(Rule):
    def pattern(self):
        f_input = wildcard()
        quant_w = wildcard()
        q_input = wildcard()
        q_input_scale = wildcard()
        q_input_zp = wildcard()

        quant_w_scale = wildcard()
        quant_w_zp = wildcard()

        input_w = node(
            "DequantizeLinear",
            quant_w,
            quant_w_scale,
            quant_w_zp,
        )

        dequant_input = node(
            "DequantizeLinear",
            q_input,
            q_input_scale,
            q_input_zp,
        )

        matmul = node("MatMul", dequant_input, input_w)

        return matmul.build(locals())

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(
        self,
        q_input,
        q_input_scale,
        q_input_zp,
        quant_w,
        quant_w_scale,
        quant_w_zp,
        matmul,
        **kwargs,
    ):
        inputs = [q_input]
        outputs = [matmul]

        if quant_w.is_constant():
            name = quant_w.__str__()
            valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
            filename = "".join(c for c in name if c in valid_chars)
            name = filename.replace(" ", "_")  # remove spaces in filenames.
            wts_bin = self.cache_dir() + "/" + str(name) + "_weight.bin"

            weight_q = np.array(quant_w.const_data(), dtype=np.int8).reshape(
                quant_w.shape()[0], quant_w.shape()[1]
            )
            file = open(wts_bin, "wb")
            file.write(np.array(weight_q).tobytes())
            file.close()

            weight_scale = quant_w_scale.const_data()[0]
            in_scale = q_input_scale.const_data()[0]
            in_zp = q_input_zp.const_data()[0]

            meta_def = self.try_fuse("MATMUL_" + str(name), inputs, outputs, [], "GEMM")
            meta_def.set_generic_param("wts_shape_dim_0", str(quant_w.shape()[1]))
            meta_def.set_generic_param("wts_shape_dim_1", str(quant_w.shape()[0]))

            meta_def.set_generic_param("node_name", str(name))
            meta_def.set_generic_param("wts_file", str(wts_bin))
            meta_def.set_generic_param("wts_scale", str(weight_scale))
            meta_def.set_generic_param("bias_file", str("null"))
            meta_def.set_generic_param("cache_dir", str(self.cache_dir()))

            if builtins.quant_mode != "w8a8":
                print(f"{builtins.quant_mode} is not supported")
                raise SystemExit

            if builtins.impl != "v1":
                print(f"{builtins.impl} implementation is not supported")
                raise SystemExit

            meta_def.set_generic_param("impl", str(builtins.impl))
            meta_def.set_generic_param("in_scale", str(in_scale))
            meta_def.set_generic_param("in_zp", str(in_zp))

            return meta_def.fuse()


def rules():
    return [fuse_MATMUL()]
