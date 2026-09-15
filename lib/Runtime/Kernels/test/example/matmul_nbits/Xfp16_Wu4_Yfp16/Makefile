# ============================================================
# custom_kernels MatMulNBits bits=4 (uint4) Test
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk>
#   make test_shapes OFFLOAD=... HIP_SDK=... [SHAPES=shapes.csv]
#   make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../models/gpt_oss_20b.json]
#   make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x2880x5120 GS=128
#   make clean
#
# MODE=autotune (default) | MODE=lut -- see README.md.
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

# ---- Required from CI / CLI ----

OFFLOAD ?= --offload-arch=gfx1151
HIP_SDK ?= $(HIP_PATH)
MODE    ?= autotune

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

ifeq ($(DETECTED_OS),Windows)
    PYTHON ?= python
else
    PYTHON ?= python3
endif

# ---- Paths ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR  := ../../../../include
KERNEL_HIP   := ../../../../hip/matmul_nbits_kernel.hip
AUTOTUNE_DIR := ../../../../hip/autotune/matmul_nbits
MODEL_DIR    := ../../models

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

# ---- Build settings ----

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

# ---- Test parameters ----

SIZE     ?= 128x2880x5120
GS       ?= 128
DATA_DIR ?= data
NO_ZEROS ?=
SHAPES   ?= shapes.csv

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
# Build
# ============================================================

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

DIRECT_TEST_OBJ   := $(BUILD_DIR)/direct_test.obj
DIRECT_KERNEL_OBJ := $(BUILD_DIR)/direct_kernel.obj

direct: FORCE lut_note | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -I$(AUTOTUNE_DIR) -x hip test_matmul_nbits.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) -o $(TARGET_DIRECT)

lut_note:
ifeq ($(MODE),lut)
	@if exist "$(subst /,\,$(LUT_FB))" (echo [MODE=lut] $(LUT_FB) exists but this hipcc-only build has no flatc-free embedder yet -- see README; falling back to MODE=autotune) else (echo [MODE=lut] no LUT for $(ARCH) at $(LUT_FB); MODE=autotune)
endif

asm: | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) --save-temps $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(BUILD_DIR)/matmul_nbits_kernel_asm.obj
	@echo "[asm] ISA saved -- look for .s files in $(BUILD_DIR)/"

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_data.py $(SIZE) --group-size $(GS) --dir $(DATA_DIR) $(PY_NZ_FLAG)

# Smoke: one small fixed shape. PR / fast CI.
test:
	$(MAKE) test_custom SIZE=1x2880x2880 GS=128 OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE)

# One shape from Make variables / CLI (SIZE=, GS=, NO_ZEROS=1).
test_custom: gendata direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

# Every row in $(SHAPES) (default shapes.csv).
test_shapes: direct
	@for /f "usebackq skip=1 tokens=1,2 delims=," %%a in ("$(SHAPES)") do @$(MAKE) test_custom SIZE=%%a GS=%%b OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE)

# ---- Model sweep mode (iterate all shapes from the shared model JSON) ----

MODEL_JSON     ?= $(MODEL_DIR)/gpt_oss_20b.json
MODEL_DATA_DIR ?= data_model

gendata_model:
	$(PYTHON) gen_data.py --model $(MODEL_JSON) --group-size $(GS) --out-dir $(MODEL_DATA_DIR) $(PY_NZ_FLAG)

test_model: gendata_model direct
	$(call RUN_CMD,$(TARGET_DIRECT)) --model $(MODEL_JSON) --data-root $(MODEL_DATA_DIR) --group-size $(GS) $(CXX_NZ_FLAG)

FORCE:

.PHONY: test test_shapes test_model test_custom clean direct gendata gendata_model asm lut_note FORCE
