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


class fuse_NMS(Rule):
    def pattern(self):
        p_input = wildcard()
        score = wildcard()
        max_output = wildcard()
        threshold = wildcard()
        nms = node("NonMaxSuppression", p_input, score, max_output, threshold)
        return nms.build(locals())

    def action(self, score, p_input, nms, **kwargs):
        inputs = [p_input, score]
        outputs = [nms]
        meta_def = self.try_fuse("NMS", inputs, outputs, [], "NMS")
        return meta_def.fuse()


def rules():
    return [fuse_NMS()]
