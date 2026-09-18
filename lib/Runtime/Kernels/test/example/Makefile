# CI aggregator for kernel unit tests. Makefile is the only entry.
#
#   make test OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> MODE=autotune [COVERAGE=1|2|3]
#
# Loops tests.manifest. Each leaf must implement the uniform API in README.md.
# Use GOAL not TARGET — leaves already use TARGET for the exe path.

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
endif

OFFLOAD  ?=
HIP_SDK  ?=
MODE     ?= autotune
COVERAGE ?= 3
GOAL     ?= test

ifeq ($(OFFLOAD),)
    $(error pass OFFLOAD=--offload-arch=<gfx>)
endif
ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# Keep in sync with tests.manifest
LEAVES := gemm/Xfp16_Wfp16_Yfp16 \
	gqa/decode/Xfp16_Wfp16_Yfp16 gqa/decode/Xfp16_Wi8_Yfp16 \
	gqa/prefill/Xfp16_Wfp16_Yfp16 gqa/prefill/Xfp16_Wi8_Yfp16 \
	matmul_nbits/Xfp16_Wu2_Yfp16 matmul_nbits/Xfp16_Wu3_Yfp16 \
	matmul_nbits/Xfp16_Wu4_Yfp16 matmul_nbits/Xfp16_Wi8_Yfp16

.PHONY: test test_custom clean $(LEAVES)

test test_custom clean:
	$(MAKE) GOAL=$@ $(LEAVES)

$(LEAVES):
	$(MAKE) -C $@ $(GOAL) OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE) COVERAGE=$(COVERAGE)
