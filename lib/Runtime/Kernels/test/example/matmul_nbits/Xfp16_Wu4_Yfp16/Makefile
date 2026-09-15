# ============================================================
# custom_kernels MatMulNBits bits=4 (uint4) Test
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk>
#   make test_shapes OFFLOAD=... HIP_SDK=... [SHAPES=shapes.csv]
#   make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../models/gpt_oss_20b.json]
#   make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x2880x5120 GS=128
#   make clean
#
# MODE=auto (default) -- lut if AUTOTUNE_DIR/lut/<arch>.fb exists, else
# autotune. MODE=lut forces lut (falls back to autotune with a warning if the
# .fb is missing). MODE=autotune always uses the runtime sweep. See README.md.
# MODE=lut needs flatc + its include/ (a build tool, not part of this repo --
# see README): FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>.
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
COMMON_DIR   := ../../common

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe
RESULTS_CSV   := $(BUILD_DIR)/results.csv

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

# MODE resolution: auto picks lut iff the arch's .fb exists; an explicit
# MODE=lut with no .fb falls back to autotune (never a hard failure -- see
# mode_info below, which prints the fallback so it isn't silent).
ifeq ($(MODE),auto)
    MODE_RESOLVED := $(if $(wildcard $(LUT_FB)),lut,autotune)
else ifeq ($(MODE),lut)
    MODE_RESOLVED := $(if $(wildcard $(LUT_FB)),lut,autotune)
else
    MODE_RESOLVED := autotune
endif

# flatc is a build tool that lives outside this repo (see README); only
# required when MODE actually resolves to lut, and only checked when a target
# that needs it (direct, via mode_info) actually runs -- `make clean` etc.
# must not require it.
FLATC ?= flatc
FLATBUFFERS_INC ?=

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
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=matmul_nbits" && set "HIPDNN_RESULTS_LEAF=Xfp16_Wu4_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
	if exist $(MODEL_DATA_DIR) rmdir /s /q $(MODEL_DATA_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=matmul_nbits && set HIPDNN_RESULTS_LEAF=Xfp16_Wu4_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR) $(DATA_DIR) $(MODEL_DATA_DIR)
    endef
endif

CSV_FRESH ?= 1
RESET_RESULTS_CSV = if exist $(subst /,\,$(RESULTS_CSV)) del $(subst /,\,$(RESULTS_CSV))

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

GEN_DIR := $(BUILD_DIR)/gen

$(GEN_DIR): | $(BUILD_DIR)
	@if not exist $(subst /,\,$(GEN_DIR)) mkdir $(subst /,\,$(GEN_DIR))

DIRECT_TEST_OBJ   := $(BUILD_DIR)/direct_test.obj
DIRECT_KERNEL_OBJ := $(BUILD_DIR)/direct_kernel.obj

# ---- MODE=lut extras: flatc-generated header + hex-embedded .fb + the real
# resolver, all reused unmodified from hip/autotune/matmul_nbits/. ----

LUT_GENERATED_H := $(GEN_DIR)/matmul_nbits_autotune_generated.h
LUT_DATA_CPP    := $(GEN_DIR)/matmul_nbits_lut_data.cpp
LUT_DATA_OBJ    := $(BUILD_DIR)/lut_data.obj
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj
EMBED_SCRIPT    := $(COMMON_DIR)/embed_lut.py

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/matmul_nbits_autotune.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/matmul_nbits_autotune.fbs

$(LUT_DATA_CPP): $(LUT_FB) $(EMBED_SCRIPT) | $(GEN_DIR)
	$(PYTHON) $(EMBED_SCRIPT) $(LUT_FB) kMatmulNbitsLutData $(LUT_DATA_CPP)

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/matmul_nbits_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/matmul_nbits_autotune.cpp -o $(AUTOTUNE_OBJ)

$(LUT_DATA_OBJ): $(LUT_DATA_CPP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip $(LUT_DATA_CPP) -o $(LUT_DATA_OBJ)

ifeq ($(MODE_RESOLVED),lut)
    TEST_DEFINES       := -DHIPDNN_LUT_LINKED_EXTERNALLY=1
    LUT_EXTRA_INCLUDES := -I$(GEN_DIR) -I$(FLATBUFFERS_INC)
    LUT_OBJS           := $(AUTOTUNE_OBJ) $(LUT_DATA_OBJ)
else
    TEST_DEFINES       :=
    LUT_EXTRA_INCLUDES :=
    LUT_OBJS           :=
endif

direct: FORCE mode_info $(LUT_OBJS) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) -I$(AUTOTUNE_DIR) $(LUT_EXTRA_INCLUDES) -x hip test_matmul_nbits.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) $(LUT_OBJS) -o $(TARGET_DIRECT)

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
ifeq ($(CSV_FRESH),1)
	@$(RESET_RESULTS_CSV)
endif
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) $(GS) $(DATA_DIR) $(CXX_NZ_FLAG)

# Every row in $(SHAPES) (default shapes.csv). One results.csv for the whole
# sweep: reset once here, then each recursive test_custom appends (CSV_FRESH=0).
test_shapes: direct
	@$(RESET_RESULTS_CSV)
	@for /f "usebackq skip=1 tokens=1,2 delims=," %%a in ("$(SHAPES)") do @$(MAKE) test_custom SIZE=%%a GS=%%b OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE) CSV_FRESH=0

# ---- Model sweep mode (iterate all shapes from the shared model JSON) ----

MODEL_JSON     ?= $(MODEL_DIR)/gpt_oss_20b.json
MODEL_DATA_DIR ?= data_model

gendata_model:
	$(PYTHON) gen_data.py --model $(MODEL_JSON) --group-size $(GS) --out-dir $(MODEL_DATA_DIR) $(PY_NZ_FLAG)

test_model: gendata_model direct
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET_DIRECT)) --model $(MODEL_JSON) --data-root $(MODEL_DATA_DIR) --group-size $(GS) $(CXX_NZ_FLAG)

FORCE:

.PHONY: test test_shapes test_model test_custom clean direct gendata gendata_model asm mode_info FORCE
