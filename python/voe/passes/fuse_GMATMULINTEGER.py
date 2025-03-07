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
import re
import sys
import os
import string

# import glog as log
import numpy as np
from glob import glob
from voe.anchor_point import CONST
from voe.pattern import node, wildcard, xir_const_op
from voe.rule_ext import Rule, same_as
import builtins


class DLLScanner:
    def __init__(self):
        self.extn_files = {}

    def read_dir(self, dir):
        for file in glob(os.path.join(dir, "*.dll")):
            name = os.path.basename(file)
            dims = [int(numeric_string) for numeric_string in re.findall(r"\d+", name)]
            self.extn_files[(dims[0], dims[1], dims[3])] = file

    def find(self, value):
        if value in self.extn_files.keys():
            return self.extn_files[value]
        else:
            return None


class fuse_GMATMULINTEGER(Rule):
    def pattern(self):
        self.num = 0
        dquantlinear = node("DynamicQuantizeLinear", wildcard())

        return dquantlinear.build(locals())

    def where(self, dquantlinear):
        device = os.getenv("DEVICE", default="")
        if device not in ["phx", "stx"]:
            print("Implementaion is not available for the device ", device)
            return False

        self.device = device

        consumers = dquantlinear.get_consumers()
        if len(consumers) != 3:
            return False

        for matmul in consumers:
            if matmul.op_type() != "MatMulInteger":
                return False

        consumer_0_in = consumers[0].inputs()
        self.inputs = [consumer_0_in[0], consumer_0_in[2]]
        self.outputs = consumers
        self.full_name = ""
        self.weight = None
        self.wts_shape_split = ""

        for matmul in consumers:
            input_w = matmul.inputs()[1]
            name = input_w.__str__()
            valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
            filename = "".join(c for c in name if c in valid_chars)
            name = filename.replace(" ", "_")  # remove spaces in filenames.

            if self.full_name == "":
                self.weight = np.array(input_w.const_data(), dtype=np.int8).reshape(
                    input_w.shape()[0], input_w.shape()[1]
                )
            else:
                self.weight = np.hstack(
                    (
                        np.array(input_w.const_data(), dtype=np.int8).reshape(
                            input_w.shape()[0], input_w.shape()[1]
                        ),
                        self.weight,
                    )
                )
            self.wts_shape_split += str(input_w.shape()[1]) + ","
            self.full_name += "_"
            self.full_name += str(name)

        return True

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(self, dquantlinear):
        self.num = self.num + 1
        meta_def = self.try_fuse(
            "GMATMULINTEGER" + str(self.full_name),
            self.inputs,
            self.outputs,
            [],
            "GMATMULINTEGER",
        )

        wts_file = self.cache_dir() + "/" + self.full_name + ".bin"
        file = open(wts_file, "wb")
        file.write(np.array(self.weight).tobytes())
        file.close()

        if builtins.impl != "v1":
            print(f"{builtins.impl} implementation is not supported")
            raise SystemExit

        meta_def.set_generic_param("impl", str(builtins.impl))
        meta_def.set_generic_param(f"wts_shape_dim_0", str(self.weight.shape[0]))
        meta_def.set_generic_param(f"wts_shape_dim_1", str(self.weight.shape[1]))
        meta_def.set_generic_param(f"wts_shape_dim_split", str(self.wts_shape_split))
        meta_def.set_generic_param(f"node_name", str(self.full_name))
        meta_def.set_generic_param(f"wts_file", str(wts_file))
        meta_def.set_generic_param("cache_dir", str(self.cache_dir()))

        return meta_def.fuse()


def rules():
    return [fuse_GMATMULINTEGER()]
