# Standalone-development aggregator for kernel unit tests.
#
#   make test OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> MODE=lookup|autotune [COVERAGE=1|2|3]
#
# Use GOAL not TARGET — leaves already use TARGET for the exe path.

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
endif

OFFLOAD  ?=
HIP_SDK  ?=
MODE     ?= lookup
COVERAGE ?= 3
GOAL     ?= test

ifeq ($(OFFLOAD),)
    $(error pass OFFLOAD=--offload-arch=<gfx>)
endif
ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# Keep in sync with example/CMakeLists.txt and kernel_ut_main.cpp.
LEAVES := gemm/Xfp16_Wfp16_Yfp16 gemm/Xbf16_Wbf16_Ybf16 gemm/Xfp32_Wfp32_Yfp32 \
	gqa/decode/Xfp16_Wfp16_Yfp16 gqa/decode/Xfp16_Wi8_Yfp16 \
	gqa/prefill/Xfp16_Wfp16_Yfp16 gqa/prefill/Xfp16_Wi8_Yfp16 \
	matmul_nbits/Xfp16_Wu2_Yfp16 matmul_nbits/Xfp16_Wu3_Yfp16 \
	matmul_nbits/Xfp16_Wu4_Yfp16 matmul_nbits/Xfp16_Wi8_Yfp16

.PHONY: test test_custom clean $(LEAVES)

test test_custom clean:
	$(MAKE) GOAL=$@ $(LEAVES)

$(LEAVES):
	$(MAKE) -C $@ $(GOAL) OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE) COVERAGE=$(COVERAGE)
