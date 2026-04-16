# ============================================================
# custom_kernels MatMulNBits (WMMA Fused GEMM + Dequant) Test
#
# Works in both WSL and Windows CMD/PowerShell.
#
# Two build modes:
#   1) Library mode  — links against pre-built hip_custom_kernels.lib
#   2) Direct mode   — compiles kernel .hip source directly (no lib needed)
#
# Both modes support NO_ZEROS=1 (zeros handled at runtime via nullptr).
#
# Library mode prerequisites:
#   cd ../../../ && mkdir build && cd build
#   cmake .. -G Ninja -DHIP_ARCHITECTURES=gfx1150 -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#
# Usage:
#   make test SIZE=128x128x128 GS=128            # direct mode (with zeros)
#   make test SIZE=128x128x128 GS=128 NO_ZEROS=1 # direct mode (no zeros)
#   make test_lib SIZE=128x128x128 GS=128        # library mode (with zeros)
#   make test_real DATA_PATH=true_data/layer_0/o_proj  # true-data mode
#   make test_model MODEL_JSON=test_model/gpt_oss_20b.json GS=128 NO_ZEROS=1  # model sweep
#   make test_all                                # Run all preset cases
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

OFFLOAD := --offload-arch=gfx1151

BUILD_DIR ?= build

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?=  C:\Users\a1_iputest\Documents\Workspace\ROCm\7.1
    PYTHON  ?= python
else
    HIP_SDK ?= /mnt/c/AMD/ROCm/7.1
    PYTHON  ?= python3
endif

# ---- Paths ----

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
LLVM_AR     := $(HIP_SDK)/bin/llvm-ar.exe
INCLUDE_DIR := ../../../include
KERNEL_HIP  := ../../../hip/matmul_nbits_kernel.hip

# All build artifacts go into BUILD_DIR
KERNEL_OBJ  := $(BUILD_DIR)/matmul_nbits_kernel.obj
LIB_FILE    := $(BUILD_DIR)/hip_custom_kernels.lib
TEST_OBJ    := $(BUILD_DIR)/test_matmul_nbits.obj
TARGET      := $(BUILD_DIR)/test_matmul_nbits.exe
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe

# ---- Build settings ----

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)
LDFLAGS      := $(LIB_FILE)

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

# ---- Create build directory ----

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

# ---- Build kernel lib (hipcc + llvm-ar) ----

$(KERNEL_OBJ): $(KERNEL_HIP) | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $< -o $@

$(LIB_FILE): $(KERNEL_OBJ)
	$(LLVM_AR) rcs $@ $<

lib: $(LIB_FILE)

# Generate ISA assembly for inspection
asm: | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) --save-temps $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(BUILD_DIR)/matmul_nbits_kernel_asm.obj
	@echo "[asm] ISA saved — look for .s files in $(BUILD_DIR)/"

# ---- Library mode (links hip_custom_kernels.lib) ----

$(TEST_OBJ): test_matmul_nbits.cpp | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(OFFLOAD) -x hip -c $< -o $@ $(INCLUDES)

$(TARGET): $(TEST_OBJ) $(LIB_FILE)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) -o $@ $(LDFLAGS)

# ---- Direct mode (compile each TU separately, then link) ----

DIRECT_TEST_OBJ   := $(BUILD_DIR)/direct_test.obj
DIRECT_KERNEL_OBJ := $(BUILD_DIR)/direct_kernel.obj

direct: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_matmul_nbits.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) -o $(TARGET_DIRECT)

# ============================================================
# Run / test targets
# ============================================================

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_matmul_nbits_data.py $(SIZE) --group-size $(GS) --dir $(DATA_DIR) $(PY_NZ_FLAG)

# ---- Direct mode (default) ----

run: direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

test: gendata direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

# ---- Library mode ----

run_lib: $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

test_lib: lib gendata $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

# ---- True-data mode (load from shape.json + bin files) ----

DATA_PATH ?=

test_real: direct
	$(call RUN_CMD,$(TARGET_DIRECT)) --true-data $(DATA_PATH)

# ---- Custom args ----

run_custom: direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(ARGS)

# ---- Preset test suite ----

test_all:
	$(MAKE) test SIZE=128x128x128 GS=128
	$(MAKE) test SIZE=128x256x128 GS=128
	$(MAKE) test SIZE=256x256x256 GS=128
	$(MAKE) test SIZE=128x2880x5120 GS=128

# ---- Model sweep mode (iterate all shapes from model config JSON) ----

MODEL_JSON     ?= test_model/gpt_oss_20b.json
MODEL_DATA_DIR ?= data_model

gendata_model:
	$(PYTHON) gen_model_data.py $(MODEL_JSON) --group-size $(GS) --out-dir $(MODEL_DATA_DIR) $(PY_NZ_FLAG)

test_model: gendata_model direct
	$(call RUN_CMD,$(TARGET_DIRECT)) --model $(MODEL_JSON) --data-root $(MODEL_DATA_DIR) --group-size $(GS) $(CXX_NZ_FLAG)

# Benchmark M=512..3072 at K=2880, N=201088 (no zeros)
bench_large_n:
	$(MAKE) test SIZE=512x2880x201088 GS=128 NO_ZEROS=1
	$(MAKE) test SIZE=1024x2880x201088 GS=128 NO_ZEROS=1
	$(MAKE) test SIZE=2048x2880x201088 GS=128 NO_ZEROS=1
	$(MAKE) test SIZE=3072x2880x201088 GS=128 NO_ZEROS=1

FORCE:

.PHONY: all lib direct clean run run_lib run_custom gendata gendata_model test test_lib test_real test_all test_model bench_large_n FORCE
