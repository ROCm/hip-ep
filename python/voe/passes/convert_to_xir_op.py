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
import math

from voe.pattern import node, wildcard
from voe.rule_ext import Rule, same_as
import logging

"""
   issue: #1319
   convert to xilinx op when node_arg shape is not unknown shape and not dynamic shape.
"""


def check_dynamic_shape(node):
    return (not node.is_unknown_shape()) and (not node.is_dynamic_shape())


def check_supported_op(my_pass, op_type):
    supported_ops = my_pass.get_supported_ops()
    return not supported_ops or op_type in supported_ops


"""
  test case model xxx
  From : DequantizeLinear(input)
  To  : com.xilinx:dequantizeLinear(input)
"""


class ConvertQDQ(Rule):
    def __init__(self, node_type, new_type):
        self.node_type = node_type
        self.new_type = new_type

    def pattern(self):
        x = wildcard()
        main_node = node(self.node_type, x, wildcard(), [wildcard()])
        return main_node.build(locals())

    def where(self, main_node, x, **_kwargs):
        if os.environ.get("XLNX_ENABLE_OLD_QDQ", "1") == "1":
            return False
        return check_dynamic_shape(x) and check_supported_op(self, main_node.op_type())

    def action(self, main_node, x, **_others):
        optype = self.new_type
        return self.create_node(
            op_type=optype,
            inputs=same_as(main_node),
            attrs=same_as(main_node),
            anchor_point=main_node,
        )


"""
  test case model mixer_b16_224.miil_in21k_ft_in1k.onnx
  From : LayerNormalization(input)
  To  : com.xilinx:layernorm(input)
"""


class ConvertLayerNorm(Rule):
    def pattern(self):
        x = wildcard()
        layernorm_node = node("LayerNormalization", x, wildcard(), wildcard())
        return layernorm_node.build(locals())

    def where(self, layernorm_node, **_kwargs):
        return check_dynamic_shape(layernorm_node) and check_supported_op(
            self, layernorm_node.op_type()
        )

    def action(self, layernorm_node, **_others):
        optype = "com.xilinx:layernorm"
        ret = self.create_node(
            op_type=optype,
            inputs=same_as(layernorm_node),
            attrs=same_as(layernorm_node),
            anchor_point=layernorm_node,
        )
        return ret


class ConvertInstanceNorm(Rule):
    def __init__(self, node_type):
        self.node_type = node_type

    def pattern(self):
        x = wildcard()
        instancenorm_node = node(self.node_type, x, wildcard(), wildcard())
        return instancenorm_node.build(locals())

    def where(self, instancenorm_node, **_kwargs):
        shape = instancenorm_node.shape()
        if len(shape) != 4:
            logging.info(
                "cancel xir ops conversion, dim!=4 is not support yet ,current dim is %d, op is %s",
                len(shape),
                instancenorm_node.__str__(),
            )
            return False
        return check_dynamic_shape(instancenorm_node) and check_supported_op(
            self, instancenorm_node.op_type()
        )

    def action(self, instancenorm_node, **_others):
        optype = "com.xilinx:instancenorm_nchw"
        ret = self.create_node(
            op_type=optype,
            inputs=same_as(instancenorm_node),
            attrs={"affine": False, "eps": instancenorm_node.attr("epsilon")},
            anchor_point=instancenorm_node,
        )
        return ret


class ConvertPrelu(Rule):
    def pattern(self):
        x = wildcard()
        weight = wildcard()
        prelu_node = node("PRelu", x, weight)
        return prelu_node.build(locals())

    def where(self, prelu_node, **_kwargs):
        return check_dynamic_shape(prelu_node) and check_supported_op(
            self, prelu_node.op_type()
        )

    def action(self, prelu_node, **_others):
        optype = "com.xilinx:prelu"
        ret = self.create_node(
            op_type=optype,
            inputs=same_as(prelu_node),
            attrs=same_as(prelu_node),
            anchor_point=prelu_node,
        )
        return ret


class ConvertSiso(Rule):
    def __init__(self, node_type, new_type):
        self.node_type = node_type
        self.new_type = new_type

    def pattern(self):
        x = wildcard()
        main_node = node(self.node_type, x)
        return main_node.build(locals())

    def where(self, main_node, **_kwargs):
        return check_dynamic_shape(main_node) and check_supported_op(
            self, main_node.op_type()
        )

    def action(self, main_node, **_others):
        optype = self.new_type
        ret = self.create_node(
            op_type=optype,
            inputs=same_as(main_node),
            attrs=same_as(main_node),
            anchor_point=main_node,
        )
        return ret


class ConvertSplitOp(Rule):
    def pattern(self):
        x = wildcard()
        split = wildcard()
        main_node = node("Split", x, [split])
        return main_node.build(locals())

    def where(self, main_node, **_kwargs):
        return check_dynamic_shape(main_node) and check_supported_op(
            self, main_node.op_type()
        )

    def action(self, main_node, x, **_others):
        outs = main_node.outputs()
        input_shape = x.shape()
        rank = len(input_shape)
        axis = 0
        if main_node.has_attr("axis"):
            axis = (main_node.attr("axis") + rank) % rank
        num_outputs = len(outs)
        if main_node.has_attr("num_outputs"):
            num_outputs = main_node.attr("num_outputs")

        axis_size = input_shape[axis]
        split = [math.ceil(axis_size / num_outputs)] * num_outputs
        if "split" in _others:
            split_node = _others["split"]
            split = split_node.const_data()

        begins = [sum(split[:i]) for i in range(num_outputs)]

        begin = [0] * rank
        end = input_shape
        strides = [1] * rank

        for i in range(num_outputs - 1, -1, -1):
            begin[axis] = begins[i]
            end[axis] = min(begins[i] + split[i], axis_size)

            self.create_node(
                op_type="com.xilinx:strided_slice",
                inputs=[x],
                attrs={
                    "begin": begin,
                    "end": end,
                    "strides": strides,
                },
                anchor_point=outs[i],
            )
        return True


def rules():
    return [
        # ConvertQDQ("QuantizeLinear", "com.xilinx:quantize_linear"),
        # ConvertQDQ("DequantizeLinear", "com.xilinx:dequantize_linear"),
        # ConvertQDQ("com.microsoft:QuantizeLinear", "com.xilinx:quantize_linear"),
        # ConvertQDQ("com.microsoft:DequantizeLinear", "com.xilinx:dequantize_linear"),
        # ConvertQDQ(
        #     "com.vai.quantize:VitisQuantizeLinear", "com.xilinx:quantize_linear"
        # ),
        # ConvertQDQ(
        #     "com.vai.quantize:VitisDequantizeLinear", "com.xilinx:dequantize_linear"
        # ),
        # ConvertLayerNorm(),
        # ConvertInstanceNorm("InstanceNormalization"),
        # ConvertInstanceNorm("com.vai.quantize:VitisInstanceNormalization"),
        # ConvertPrelu(),
        # ConvertSiso("Cast", "com.xilinx:cast"),
        # ConvertSiso("Neg", "com.xilinx:neg"),
        # ConvertSiso("com.microsoft:Gelu", "com.xilinx:gelu"),
        #    ConvertSplitOp(),
    ]
