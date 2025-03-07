#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
#   and permissive binary-only license, but are not provided in source code form.  While the following free 
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in 
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific 
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC. 
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#
dev:
	python setup.py develop --user
test:
	python setup.py test

passes += const_fold_batchnorm_to_scale
passes += merge_fix

png:
	for i in $(passes); do \
		python -mvoe.vis_pass  -m voe.passes.$$i -o ./vaip/passes/$$i.dot; \
	done

.PHONY: build
build:
	p=vaip; cmake --build ~/build/build.Ubuntu.20.04.x86_64.Debug/$$p   --verbose -j $$(nproc) && cmake --install ~/build/build.Ubuntu.20.04.x86_64.Debug/$$p

test_5: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/d111d3ee33c942c0b6f1134bc6a0f56b/vaip.*.remove_reshape_fix.action_0.onnx \
	-m voe.passes.const_fold_batchnorm_to_scale2 \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/d111d3ee33c942c0b6f1134bc6a0f56b/ \

test_merge_fix: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/b67dda654698f6fa1d77db9d2d9594f3/vaip.*.add_fix_after_const.action_0.onnx \
	-m voe.passes.merge_fix \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/b67dda654698f6fa1d77db9d2d9594f3/ \


test_merge_mul: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/1fc7144c0c251d033f788d35e9a7745e/vaip.*.merge_hard_sigmoid.action_0.onnx \
	-m voe.passes.merge_mul \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/1fc7144c0c251d033f788d35e9a7745e/ \

test_around_add_transpose_reshape: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/c16d40525783acc93afa0ba77976863a/vaip.*.gc_after_layout_transform.action_0.onnx \
	-m voe.passes.add_transpose_reshape_transpose_reshape \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/c16d40525783acc93afa0ba77976863a/ \
