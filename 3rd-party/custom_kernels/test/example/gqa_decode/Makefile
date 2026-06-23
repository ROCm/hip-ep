# ============================================================
# custom_kernels GQA flash decode (TPS) test + A/B benchmark.
#
# Direct mode -- compiles hip/gqa_kernel.hip directly (no lib needed) and
# links the standalone harness. Works in Windows CMD/PowerShell and WSL.
#
# Usage:
#   make test                       # build + run the model-shape matrix
#   make run                        # run (no rebuild)
#   make test ITERS=500             # more benchmark iterations
#   make test ARGS="--b 1 --h 64 --g 8 --d 64 --total 8192 --window 128 --sink 1"
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

OFFLOAD := --offload-arch=gfx1151
BUILD_DIR ?= build

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?= C:\Users\a1_iputest\Documents\Workspace\ROCm\7.1
else
    HIP_SDK ?= /mnt/c/AMD/ROCm/7.1
endif

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../include
KERNEL      := ../../../hip/gqa_kernel.hip

TARGET      := $(BUILD_DIR)/test_gqa_decode.exe

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

ITERS ?= 200
ARGS  ?= --all

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR)
    endef
endif

KERNEL_OBJS := $(BUILD_DIR)/gqa_kernel.obj

all: direct

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

direct: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_gqa_decode.cpp -o $(BUILD_DIR)/test_gqa_decode.obj
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL) -o $(BUILD_DIR)/gqa_kernel.obj
	$(HIPCC) $(OFFLOAD) $(BUILD_DIR)/test_gqa_decode.obj $(KERNEL_OBJS) -o $(TARGET)

run: direct
	$(call RUN_CMD,$(TARGET)) $(ARGS) --iters $(ITERS)

test: direct
	$(call RUN_CMD,$(TARGET)) $(ARGS) --iters $(ITERS)

clean:
	$(DO_CLEAN)

FORCE:

.PHONY: all direct run test clean FORCE
