# ============================================================
# custom_kernels GQA flash prefill (TTFT) Test -- INT8 KV cache
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk>
#   make test_shapes OFFLOAD=... HIP_SDK=...
#   make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../../models/gpt_oss_20b.json]
#   make test_custom OFFLOAD=... HIP_SDK=... H=32 G=8 D=128 SQ=512
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

# ---- Paths ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR  := ../../../../../include
KERNEL_HIP   := ../../../../../hip/gqa_kernel.hip
AUTOTUNE_DIR := ../../../../../hip/autotune/gqa
MODEL_DIR    := ../../../models

BUILD_DIR ?= out
TARGET    := $(BUILD_DIR)/test_gqa_prefill_i8.exe

ARCH   := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

# ---- Test parameters (test_custom) ----

ITERS ?= 100
H     ?= 32
G     ?= 8
D     ?= 128
SQ    ?= 512

PYTHON  ?= python
DATA_DIR ?= data

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR) $(DATA_DIR)
    endef
endif

# ============================================================
# Build -- gqa_kernel.hip does not reference any autotune-LUT symbol, so no
# stub TU is needed (see README for what MODE=lut means here).
# ============================================================

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

TEST_OBJ   := $(BUILD_DIR)/prefill_i8_test.obj
KERNEL_OBJ := $(BUILD_DIR)/prefill_kernel.obj

$(TARGET): FORCE lut_note | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_gqa_prefill_i8.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) -o $(TARGET)

lut_note:
ifeq ($(MODE),lut)
	@if exist "$(subst /,\,$(LUT_FB))" (echo [MODE=lut] $(LUT_FB) exists but this hipcc-only build has no flatc-free embedder yet -- see README; falling back to MODE=autotune) else (echo [MODE=lut] no LUT for $(ARCH) at $(LUT_FB); MODE=autotune)
endif

asm: | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) --save-temps $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(BUILD_DIR)/gqa_kernel_asm.obj
	@echo "[asm] ISA saved -- look for .s files in $(BUILD_DIR)/"

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_data.py --h $(H) --g $(G) --d $(D) --sq $(SQ) --dir $(DATA_DIR)

# Smoke: one small fixed shape.
test: H=8
test: G=8
test: D=64
test: SQ=512
test: gendata $(TARGET)
	$(call RUN_CMD,$(TARGET)) --h 8 --g 8 --d 64 --sq 512 --iters 20 --data-dir $(DATA_DIR)

# One shape from Make variables / CLI.
test_custom: gendata $(TARGET)
	$(call RUN_CMD,$(TARGET)) --h $(H) --g $(G) --d $(D) --sq $(SQ) --iters $(ITERS) --data-dir $(DATA_DIR)

# The binary's own built-in matrix (8 shapes x 3 seq-lens); see shapes.csv for
# the representative subset. More thorough than a hand-duplicated CSV loop.
test_shapes: $(TARGET)
	$(call RUN_CMD,$(TARGET)) --all --iters $(ITERS)

# ---- Model sweep mode (gqa_prefill array from the shared model JSON) ----

MODEL_JSON ?= $(MODEL_DIR)/gpt_oss_20b.json

test_model: $(TARGET)
	$(call RUN_CMD,$(TARGET)) --model $(MODEL_JSON) --iters $(ITERS)

FORCE:
.PHONY: test test_shapes test_model test_custom clean asm lut_note gendata FORCE
