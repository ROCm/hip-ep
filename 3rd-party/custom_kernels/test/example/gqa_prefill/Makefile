# ============================================================
# custom_kernels GQA flash prefill (TTFT) test + benchmark.
# Direct mode -- compiles the v5/v7 prefill kernels directly (no lib needed)
# and links the standalone harness. Mirrors ../gqa_decode/Makefile.
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
# v5/v7 prefill kernels now live inside gqa_kernel.hip (merged).
KERNEL      := ../../../hip/gqa_kernel.hip

TARGET      := $(BUILD_DIR)/test_gqa_prefill.exe

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

ITERS ?= 100
ARGS  ?=

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
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_gqa_prefill.cpp -o $(BUILD_DIR)/test_gqa_prefill.obj
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL) -o $(BUILD_DIR)/gqa_kernel.obj
	$(HIPCC) $(OFFLOAD) $(BUILD_DIR)/test_gqa_prefill.obj $(KERNEL_OBJS) -o $(TARGET)

run: direct
	$(call RUN_CMD,$(TARGET)) $(ARGS) --iters $(ITERS)

test: direct
	$(call RUN_CMD,$(TARGET)) $(ARGS) --iters $(ITERS)

clean:
	$(DO_CLEAN)

FORCE:

.PHONY: all direct run test clean FORCE
