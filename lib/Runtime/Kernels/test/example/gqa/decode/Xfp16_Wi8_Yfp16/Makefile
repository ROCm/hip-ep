# ============================================================
# custom_kernels GQA flash decode (TPS) Test -- fp16 KV cache
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk>
#   make test_shapes OFFLOAD=... HIP_SDK=...
#   make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../../models/gpt_oss_20b.json]
#   make test_custom OFFLOAD=... HIP_SDK=... B=1 H=32 G=8 D=128 MAX_SEQ=4096 TOTAL=2048
#   make clean
#
# MODE=auto (default) -- lut if AUTOTUNE_DIR/lut/<arch>.fb exists, else
# autotune. MODE=lut forces lut (falls back to autotune with a warning if the
# .fb is missing). MODE=lut needs flatc + its include/ (a build tool, not part
# of this repo -- see README): FLATC=<path to flatc(.exe)>
# FLATBUFFERS_INC=<its include dir>.
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
MODE    ?= auto

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# ---- Paths ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR  := ../../../../../include
KERNEL_HIP   := ../../../../../hip/gqa_kernel.hip
AUTOTUNE_DIR := ../../../../../hip/autotune/gqa
MODEL_DIR    := ../../../models
COMMON_DIR   := ../../../common

BUILD_DIR ?= out
TARGET    := $(BUILD_DIR)/test_gqa_decode_i8.exe
RESULTS_CSV := $(BUILD_DIR)/results.csv

ARCH   := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

ifeq ($(MODE),auto)
    MODE_RESOLVED := $(if $(wildcard $(LUT_FB)),lut,autotune)
else ifeq ($(MODE),lut)
    MODE_RESOLVED := $(if $(wildcard $(LUT_FB)),lut,autotune)
else
    MODE_RESOLVED := autotune
endif

# flatc is a build tool that lives outside this repo (see README); only
# required when MODE actually resolves to lut, and only checked when a target
# that needs it (via mode_info) actually runs -- `make clean` etc. must not
# require it.
FLATC ?= flatc
FLATBUFFERS_INC ?=

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

# ---- Test parameters (test_custom) ----

ITERS   ?= 100
B       ?= 1
H       ?= 32
G       ?= 8
D       ?= 128
MAX_SEQ ?= 4096
TOTAL   ?= 2048

PYTHON  ?= python
DATA_DIR ?= data

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=gqa" && set "HIPDNN_RESULTS_LEAF=decode/Xfp16_Wi8_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=gqa && set HIPDNN_RESULTS_LEAF=decode/Xfp16_Wi8_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR) $(DATA_DIR)
    endef
endif

CSV_FRESH ?= 1
RESET_RESULTS_CSV = if exist $(subst /,\,$(RESULTS_CSV)) del $(subst /,\,$(RESULTS_CSV))

# ============================================================
# Build -- gqa_kernel.hip does not call the autotune-LUT resolver itself (only
# real/gqa.cpp does, in production); MODE=lut instead has THIS test call
# hip_gqa_autotune_resolve_decode() then dispatch through
# hip_gqa_flash_decode_configured() (see test_gqa_decode_i8.cpp, guarded by
# HIPDNN_LUT_LINKED_EXTERNALLY). MODE=autotune links no LUT at all.
# ============================================================

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

GEN_DIR := $(BUILD_DIR)/gen

$(GEN_DIR): | $(BUILD_DIR)
	@if not exist $(subst /,\,$(GEN_DIR)) mkdir $(subst /,\,$(GEN_DIR))

TEST_OBJ   := $(BUILD_DIR)/decode_i8_test.obj
KERNEL_OBJ := $(BUILD_DIR)/decode_kernel.obj

LUT_GENERATED_H := $(GEN_DIR)/gqa_autotune_generated.h
LUT_DATA_CPP    := $(GEN_DIR)/gqa_lut_data.cpp
LUT_DATA_OBJ    := $(BUILD_DIR)/lut_data.obj
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj
EMBED_SCRIPT    := $(COMMON_DIR)/embed_lut.py

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/gqa_autotune.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/gqa_autotune.fbs

$(LUT_DATA_CPP): $(LUT_FB) $(EMBED_SCRIPT) | $(GEN_DIR)
	$(PYTHON) $(EMBED_SCRIPT) $(LUT_FB) kGqaLutData $(LUT_DATA_CPP)

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/gqa_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/gqa_autotune.cpp -o $(AUTOTUNE_OBJ)

$(LUT_DATA_OBJ): $(LUT_DATA_CPP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip $(LUT_DATA_CPP) -o $(LUT_DATA_OBJ)

ifeq ($(MODE_RESOLVED),lut)
    TEST_DEFINES       := -DHIPDNN_LUT_LINKED_EXTERNALLY=1
    LUT_EXTRA_INCLUDES := -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC)
    LUT_OBJS           := $(AUTOTUNE_OBJ) $(LUT_DATA_OBJ)
else
    TEST_DEFINES       :=
    LUT_EXTRA_INCLUDES :=
    LUT_OBJS           :=
endif

$(TARGET): FORCE mode_info $(LUT_OBJS) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $(LUT_EXTRA_INCLUDES) -x hip test_gqa_decode_i8.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) $(LUT_OBJS) -o $(TARGET)

mode_info:
	@echo [MODE=$(MODE) -^> $(MODE_RESOLVED)] arch=$(ARCH) lut_fb=$(LUT_FB)
ifeq ($(MODE),lut)
ifneq ($(MODE_RESOLVED),lut)
	@echo [MODE=lut] no LUT for $(ARCH) at $(LUT_FB) -- falling back to MODE=autotune
endif
endif
ifeq ($(MODE_RESOLVED),lut)
ifeq ($(FLATBUFFERS_INC),)
	@echo ERROR: MODE=lut needs FLATBUFFERS_INC=^<flatbuffers include dir^> (flatc's include/, a build tool outside the repo -- see README) 1>&2
	@exit /b 1
endif
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
	$(PYTHON) gen_data.py --b $(B) --h $(H) --g $(G) --d $(D) --max-seq $(MAX_SEQ) --total $(TOTAL) --dir $(DATA_DIR)

# Smoke: one small fixed shape.
test: B=1
test: H=8
test: G=8
test: D=64
test: MAX_SEQ=512
test: TOTAL=512
test: gendata $(TARGET)
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET)) --b 1 --h 8 --g 8 --d 64 --max-seq 512 --total 512 --iters 20 --data-dir $(DATA_DIR)

# One shape from Make variables / CLI.
test_custom: gendata $(TARGET)
ifeq ($(CSV_FRESH),1)
	@$(RESET_RESULTS_CSV)
endif
	$(call RUN_CMD,$(TARGET)) --b $(B) --h $(H) --g $(G) --d $(D) --max-seq $(MAX_SEQ) --total $(TOTAL) --iters $(ITERS) --data-dir $(DATA_DIR)

# The binary's own built-in matrix (real models + MHA/GQA x head_dim geometry
# sweep, see shapes.csv for the representative subset); more thorough than a
# hand-duplicated CSV loop and cannot drift out of sync with the source.
test_shapes: $(TARGET)
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET)) --all --iters $(ITERS)

# ---- Model sweep mode (gqa_decode array from the shared model JSON) ----

MODEL_JSON ?= $(MODEL_DIR)/gpt_oss_20b.json

test_model: $(TARGET)
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET)) --model $(MODEL_JSON) --iters $(ITERS)

FORCE:
.PHONY: test test_shapes test_model test_custom clean asm mode_info gendata FORCE
