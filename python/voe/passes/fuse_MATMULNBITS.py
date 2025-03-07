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


class fuse_MATMULNBITS(Rule):
    def pattern(self):
        self.num = 0
        p_input = wildcard()
        input_w = wildcard()
        scales = wildcard()
        zp = wildcard()
        matmul = node("com.microsoft:MatMulNBits", p_input, input_w, scales, [zp])

        return matmul.build(locals())

    def where(self, p_input, input_w, scales, zp, matmul):
        if input_w.is_constant():
            return True
        else:
            return False

    # argument is passed by key, value pair
    # so the argument name should be identical to the pattern's local variable's name
    def action(self, p_input, input_w, scales, matmul, **kwargs):
        self.num = self.num + 1
        # inputs = [p_input, input_w, zp, scales]
        if not "zp" in kwargs:
            inputs = [p_input, input_w, scales]
        else:
            inputs = [p_input, input_w, scales, kwargs["zp"]]
        outputs = [matmul]
        # NXK/2
        m_k = matmul.attr("K")
        m_n = matmul.attr("N")
        m_grp_size = matmul.attr("block_size")
        name = input_w.__str__()
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = "".join(c for c in name if c in valid_chars)
        name = filename.replace(" ", "_")  # remove spaces in filenames.
        # print("name : ", name)
        wts_bin = self.cache_dir() + "/" + str(name) + "_weight.bin"
        wts_file = self.cache_dir() + "/" + name + ".bin"
        file = open(wts_file, "wb")
        weight = np.array(input_w.const_data(), dtype=np.int8).reshape(
            input_w.shape()[0], (input_w.shape()[1] * input_w.shape()[2])
        )
        # TODO: Conversion in reptitive for numpy array
        file.write(weight.tobytes())
        file.close()
        if "zp" in kwargs:
            name_zp = kwargs["zp"].__str__()
            filename_zp = "".join(c for c in name_zp if c in valid_chars)
            name_zp = filename_zp.replace(" ", "_")  # remove spaces in filenames.
            # print("name : ", name)
            zp_bin = self.cache_dir() + "/" + str(name_zp) + "_zp.bin"
            zp_file = self.cache_dir() + "/" + name_zp + ".bin"
            file_zp = open(zp_file, "wb")
            zp_shape = int((m_k * (m_n / m_grp_size)) / 2)
            zero_pt = np.array(kwargs["zp"].const_data(), dtype=np.int8).reshape(
                zp_shape
            )
            # TODO: Conversion in reptitive for numpy array
            file_zp.write(zero_pt.tobytes())
            file_zp.close()
        name1 = scales.__str__()
        filename1 = "".join(c for c in name1 if c in valid_chars)
        name1 = filename1.replace(" ", "_")
        scl_bin = self.cache_dir() + "/" + str(name1) + "_scale.bin"
        scl_file = self.cache_dir() + "/" + name1 + ".bin"
        file1 = open(scl_file, "wb")
        sc_shape = int(m_k * (m_n / m_grp_size))
        scale = np.array(scales.const_data(), dtype=np.float32).reshape(sc_shape)
        file1.write(scale.tobytes())
        file1.close()
        meta_def = self.try_fuse(
            "MATMULNBITS_" + str(name), inputs, outputs, [], "MATMULNBITS"
        )
        meta_def.set_generic_param("scl_shape_dim_0", str(scales.shape()[0]))

        meta_def.set_generic_param("node_name", str(name))
        meta_def.set_generic_param("wts_file", str(wts_file))
        meta_def.set_generic_param("scl_file", str(scl_file))
        if "zp" in kwargs:
            meta_def.set_generic_param("zp_file", str(zp_file))
        meta_def.set_generic_param("cache_dir", str(self.cache_dir()))
        meta_def.set_generic_param("K", str(matmul.attr("K")))
        meta_def.set_generic_param("N", str(matmul.attr("N")))
        meta_def.set_generic_param("bits", str(matmul.attr("bits")))
        meta_def.set_generic_param("block_size", str(matmul.attr("block_size")))
        device = os.getenv("DEVICE", default="")
        if device not in ["phx", "stx"]:
            print("Implementaion is not available for the device ", device)
            sys.exit()
        return meta_def.fuse()


def rules():
    return [fuse_MATMULNBITS()]
