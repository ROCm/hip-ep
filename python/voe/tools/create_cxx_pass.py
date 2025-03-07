#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import pathlib
import argparse

#                   vaip/python/voe/tools/..     /..    /..   /..
VAIP_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent


def under_score_name(names):
    return "_".join([name.lower() for name in names])


def under_score_upper_name(names):
    return "_".join([name.upper() for name in names])


def CamelCase(names):
    return "".join([name.lower().capitalize() for name in names])


def cmake_lists_txt_content(name):
    pass_cmake_target_name = under_score_name(["pass", *name])
    enable_name = under_score_upper_name(["enable", "vaip", "pass", *name])
    vaip_config_pass_name = "vaip-pass_" + under_score_name([*name])
    glog_macro_name = under_score_upper_name(["debug", *name])
    return f"""#
#
# The Xilinx Vitis AI Vaip in this distribution are provided under the following
# free and permissive binary-only license, but are not provided in source code
# form.  While the following free and permissive license is similar to the BSD
# open source license, it is NOT the BSD open source license nor other
# OSI-approved open source license.
#
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in binary form only, without modification, is permitted
# provided that the following conditions are met:
#
# 1. Redistributions must reproduce the above copyright notice, this list of
#   conditions and the following disclaimer in the documentation and/or other
#   materials provided with the distribution.
#
# 2. The name of Xilinx, Inc. may not be used to endorse or promote products
#   redistributed with this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE
#
vai_add_library(
  NAME
  {pass_cmake_target_name}
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp)

target_link_libraries({pass_cmake_target_name} PRIVATE vaip::core vart::util glog::glog)
vai_add_debug_command({pass_cmake_target_name}
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\\\temp\\\\$ENV{{USERNAME}}\\\\vaip\\\\.cache\\\\CUR\\\\onnx.onnx -o c:\\\\temp\\\\a.onnx -t c:\\\\temp\\\\a.txt -p {vaip_config_pass_name} -c  C:\\\\temp\\\\$ENV{{USERNAME}}\\\\vaip\\\\.cache\\\\CUR"
    ENVIRONMENT {glog_macro_name}=1
    )
target_compile_definitions({pass_cmake_target_name} PUBLIC "-D{enable_name}=1")
"""


def pass_main_cpp_content(name):
    glog_macro_name = under_score_upper_name(["debug", *name])
    struct_name = CamelCase([*name])
    pass_name = under_score_name(["vaip", "pass", *name])
    vaip_config_pass_name = "vaip-pass_" + under_score_name([*name])
    return f"""/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved. 
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include <glog/logging.h>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM({glog_macro_name}, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM({glog_macro_name}) >= n)

/**
 * test case: <???>
 *
 *
 * Replace pattern:
 *
 * From: <???>
 * To  : <???>
 */

// add the following line in your vaip_config.json
/*
    {{ "name": "{pass_name}",
       "plugin": "{vaip_config_pass_name}",
       "disabled": false
    }}
*/
namespace {{
using namespace vaip_core;
struct {struct_name} {{
   {struct_name}(IPass& self) : self_{{self}} {{}}
   std::unique_ptr<Rule> create_rule(IPass* self) {{
       auto builder = PatternBuilder();
       std::shared_ptr<Pattern> input_ = builder.wildcard();
       std::shared_ptr<Pattern> pattern_ = builder.node2("<your-op-type?>", {{input_}});
       return Rule::create_rule(
              pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {{
          //auto input = binder[input_->get_id()];
          MY_LOG(1) << "Sample log message.";
          return false; // return true if graph is modified.
      }});
   }}
   // apply the rule
   void process(IPass& self, Graph& graph) {{
      MY_LOG(1) << self_.get_pass_proto().name() << "[" << self_.get_pass_proto().plugin() << "] start processing graph";
      create_rule(&self)->apply(&graph); 
      MY_LOG(1) << self_.get_pass_proto().name() << "[" << self_.get_pass_proto().plugin() << "] finish processing graph";
    }}

   IPass& self_;
}};
}} // namespace

DEFINE_VAIP_PASS({struct_name},
                 {pass_name})
"""


def insert_content(filename, insert_point_tag, content):
    original_content = open(filename).read()
    insert_point = original_content.find(insert_point_tag)
    if insert_point == -1:
        raise f"cannot find tag {insert_point_tag} in {filename}"
    open(filename, "wt").write(
        f"{original_content[0:insert_point]}{content}{original_content[insert_point:]}"
    )


def update_vaip_cmake_lists_txt(enable, name):
    pass_directory = VAIP_ROOT / under_score_name(["vaip", "pass", *name])
    pass_name = under_score_name(["vaip", "pass", *name])
    cmake_option_name = under_score_upper_name(["enable", "vaip", "pass", *name])
    cmake_option_value = "ON" if enable else "OFF"
    pass_relative_directory = pass_directory.relative_to(VAIP_ROOT)

    cmake_list_txt_context = open(VAIP_ROOT / "CMakeLists.txt").read()
    end_tag = "# !!! DO NOT DELETE OR MODIFY THESE TWO LINES THIS LINE USED BY voe.tools.create_pass !!!!"
    insert_point = cmake_list_txt_context.find(end_tag)
    if insert_point == -1:
        raise "cannot find end tag. do not delete or modify line '#  passes' in CMakeLists.txt"

    cmake_list_txt_context_insert_content = f"""option({cmake_option_name} "enable {pass_name} or not " {cmake_option_value})
if({cmake_option_name})
   add_subdirectory({pass_relative_directory})
endif({cmake_option_name})
"""

    insert_content(
        VAIP_ROOT / "CMakeLists.txt", end_tag, cmake_list_txt_context_insert_content
    )


def update_vaip_symbols_txt(pass_directory, name):
    hook_name = under_score_name(["vaip", "pass", *name, "_hook"])

    with open(pass_directory / "symbols.txt", "wt") as symbol_txt:
        symbol_txt.write(hook_name)


def main(args):
    parser = argparse.ArgumentParser(
        description="A simple tool to create cxx pass in C++."
    )
    parser.add_argument(
        "--name", nargs="+", default=[], required=True, help="pass name"
    )
    parser.add_argument("--enable", action="store_true", help="enable it by default")

    arg_setting, unknown_args = parser.parse_known_args(args)

    ##
    update_vaip_cmake_lists_txt(arg_setting.enable, arg_setting.name)
    pass_directory = VAIP_ROOT / under_score_name(["vaip", "pass", *arg_setting.name])
    (pass_directory / "src").mkdir(exist_ok=True, parents=True)
    (pass_directory / "include").mkdir(exist_ok=True, parents=True)
    open(pass_directory / "CMakeLists.txt", "wt").write(
        cmake_lists_txt_content(arg_setting.name)
    )
    open(pass_directory / "src" / "pass_main.cpp", "wt").write(
        pass_main_cpp_content(arg_setting.name)
    )

    update_vaip_symbols_txt(pass_directory, arg_setting.name)


if __name__ == "__main__":
    main(sys.argv[1:])
