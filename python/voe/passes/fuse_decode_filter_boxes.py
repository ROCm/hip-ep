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

import glog as log
import numpy as np
from voe.anchor_point import CONST
from voe.pattern import node, wildcard, xir_const_op
from voe.rule_ext import Rule, same_as


class fuse_DecodeAndFilterBoxes(Rule):
    def pattern(self):
        in1 = wildcard()
        in2 = wildcard()
        output = node("vitis.customop:DecodeAndFilterBoxes", in1, in2)
        return output.build(locals())

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(self, in1, in2, output, **kwargs):
        inputs = [in1, in2]
        outputs = [output]
        # pattern
        meta_def = self.try_fuse(
            "DecodeFilterBoxesNode", inputs, outputs, [], "DecodeFilterBoxes"
        )
        meta_def.set_generic_param("iou_threshold", "0.6")
        meta_def.fuse()


def rules():
    return [fuse_DecodeAndFilterBoxes()]
