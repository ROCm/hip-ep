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

from voe.pattern import node, wildcard, create_subpattern_env, update_subpattern_env


def dequant(env, prefix):
    x = wildcard()
    x_scale = wildcard()
    x_zero_point = wildcard()
    dequantizeLinear = node(
        "DequantizeLinear",
        x,
        x_scale,
        [x_zero_point],
    )
    update_subpattern_env(env, locals(), prefix)
    return dequantizeLinear


def pattern():
    env = create_subpattern_env()

    X = dequant(env, "X")
    W = dequant(env, "W")
    B = dequant(env, "B")
    conv = node("Conv", X, W, [B])
    rule = node("Relu", conv)

    quantize_y_scale4 = wildcard()
    quantize_y_zp4 = wildcard()
    y4 = node("QuantizeLinear", rule, quantize_y_scale4, [quantize_y_zp4])
    update_subpattern_env(env, locals())
    return y4.build(env)
