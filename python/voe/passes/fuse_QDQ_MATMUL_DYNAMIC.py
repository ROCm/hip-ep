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


class fuse_MATMUL_DYNAMIC(Rule):
    def pattern(self):
        f_input = wildcard()
        qa_input = wildcard()
        qa_input_scale = wildcard()
        qa_input_zp = wildcard()
        qb_input = wildcard()
        qb_input_scale = wildcard()
        qb_input_zp = wildcard()

        dequant_inputa = node(
            "DequantizeLinear",
            qa_input,
            qa_input_scale,
            qa_input_zp,
        )

        dequant_inputb = node(
            "DequantizeLinear",
            qb_input,
            qb_input_scale,
            qb_input_zp,
        )

        matmul = node("MatMul", dequant_inputa, dequant_inputb)

        return matmul.build(locals())

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(
        self,
        qa_input,
        qa_input_scale,
        qa_input_zp,
        qb_input,
        qb_input_scale,
        qb_input_zp,
        matmul,
        **kwargs,
    ):
        inputs = [qa_input, qb_input]
        outputs = [matmul]

        if not qb_input.is_constant():
            name = matmul.__str__()
            a_in_scale = qa_input_scale.const_data()[0]
            a_in_zp = qa_input_zp.const_data()[0]

            b_in_scale = qb_input_scale.const_data()[0]
            b_in_zp = qb_input_zp.const_data()[0]

            meta_def = self.try_fuse(
                "MATMUL_DYNAMIC_" + str(name), inputs, outputs, [], "GEMM_DYNAMIC"
            )
            meta_def.set_generic_param("node_name", str(name))
            meta_def.set_generic_param("cache_dir", str(self.cache_dir()))

            if builtins.quant_mode != "w8a8":
                print(f"{builtins.quant_mode} is not supported")
                raise SystemExit

            meta_def.set_generic_param("impl", str(builtins.impl))

            if builtins.impl != "v1":
                print(f"{builtins.impl} implementation is not supported")
                raise SystemExit

            meta_def.set_generic_param("a_in_scale", str(a_in_scale))
            meta_def.set_generic_param("a_in_zp", str(a_in_zp))
            meta_def.set_generic_param("b_in_scale", str(b_in_scale))
            meta_def.set_generic_param("b_in_zp", str(b_in_zp))

            cnt = 0
            for i in outputs:
                meta_def.set_generic_param("type_" + str(cnt), str(i.data_type()))
                cnt += 1

            return meta_def.fuse()


def rules():
    return [fuse_MATMUL_DYNAMIC()]
