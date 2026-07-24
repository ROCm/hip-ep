# ============================================================
# custom_kernels MatMulNBits bits=3 (uint3) vs bits=4 (uint4) Test
#
# Works in both WSL and Windows CMD/PowerShell.
#
# Compiles matmul_nbits_kernel.hip directly (single TU — it already
# contains the bits=3, bits=4 and bits=8 kernels) together with the test
# driver, then benchmarks + verifies both kernels on the same shape.
#
# Usage:
#   make test SIZE=128x128x128 GS=128            # single shape (with zeros)
#   make test SIZE=128x128x128 GS=128 NO_ZEROS=1 # single shape (no zeros)
#   make test_model MODEL_JSON=test_model/gpt_oss_20b.json GS=128 NO_ZEROS=1
#   make test_all                                # a handful of preset shapes
# ============================================================

# ---- OS detection ----

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

# ---- User-configurable ----

OFFLOAD := --offload-arch=gfx1150

BUILD_DIR ?= build

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?= C:\AMD\Rocm\7.1
    PYTHON  ?= python
else
    HIP_SDK ?= /mnt/c/AMD/ROCm/7.1
    PYTHON  ?= python3
endif

# ---- Paths ----

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../include
KERNEL_HIP  := ../../../hip/matmul_nbits_kernel.hip

TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe

# ---- Build settings ----

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

# ---- Test parameters ----

SIZE     ?= 128x128x128
GS       ?= 128
DATA_DIR ?= data
NO_ZEROS ?=

ifeq ($(NO_ZEROS),1)
    PY_NZ_FLAG  := --no-zeros
    CXX_NZ_FLAG := --no-zeros
else
    PY_NZ_FLAG  :=
    CXX_NZ_FLAG :=
endif

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
	if exist $(MODEL_DATA_DIR) rmdir /s /q $(MODEL_DATA_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR) $(DATA_DIR) $(MODEL_DATA_DIR)
    endef
endif

# ============================================================
# Build targets
# ============================================================

all: direct

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

# Generate ISA assembly for inspection
asm: | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) --save-temps $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(BUILD_DIR)/matmul_nbits_kernel_asm.obj
	@echo "[asm] ISA saved -- look for .s files in $(BUILD_DIR)/"

# ---- Direct mode (compile each TU separately, then link) ----

DIRECT_TEST_OBJ   := $(BUILD_DIR)/direct_test.obj
DIRECT_KERNEL_OBJ := $(BUILD_DIR)/direct_kernel.obj

direct: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_matmul_nbits_u3.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) -o $(TARGET_DIRECT)

# ============================================================
# Run / test targets
# ============================================================

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_matmul_nbits_u3_data.py $(SIZE) --group-size $(GS) --dir $(DATA_DIR) $(PY_NZ_FLAG)

run: direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

test: gendata direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

run_custom: direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(ARGS)

# ---- Preset test suite ----

test_all:
	$(MAKE) test SIZE=1x2880x2880 GS=128
	$(MAKE) test SIZE=128x2880x2880 GS=128
	$(MAKE) test SIZE=1x2880x5120 GS=128
	$(MAKE) test SIZE=128x2880x5120 GS=128
	$(MAKE) test SIZE=1x4096x2880 GS=128
	$(MAKE) test SIZE=128x4096x2880 GS=128

# ---- Model sweep mode (iterate all shapes from model config JSON) ----

MODEL_JSON     ?= test_model/gpt_oss_20b.json
MODEL_DATA_DIR ?= data_model

gendata_model:
	$(PYTHON) gen_model_data_u3.py $(MODEL_JSON) --group-size $(GS) --out-dir $(MODEL_DATA_DIR) $(PY_NZ_FLAG)

test_model: gendata_model direct
	$(call RUN_CMD,$(TARGET_DIRECT)) --model $(MODEL_JSON) --data-root $(MODEL_DATA_DIR) --group-size $(GS) $(CXX_NZ_FLAG)

# Benchmark decode (M=1) and small-prefill (M=128) shapes across a few
# typical MoE expert / lm_head sizes, no zeros (fastest path both kernels).
bench_decode:
	$(MAKE) test SIZE=1x2880x2880 GS=128 NO_ZEROS=1
	$(MAKE) test SIZE=1x2880x5120 GS=128 NO_ZEROS=1
	$(MAKE) test SIZE=1x4096x2880 GS=128 NO_ZEROS=1
	$(MAKE) test SIZE=1x2880x201088 GS=128 NO_ZEROS=1

FORCE:

.PHONY: all direct clean run run_custom gendata gendata_model test test_all test_model bench_decode asm FORCE
