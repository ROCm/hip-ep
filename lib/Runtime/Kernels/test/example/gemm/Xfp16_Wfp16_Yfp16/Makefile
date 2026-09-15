# ============================================================
# hip_gemm (gemm_kernel.hip) correctness + bench
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk>
#   make test_shapes OFFLOAD=... HIP_SDK=... [SHAPES=shapes.csv]
#   make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../models/gpt_oss_20b.json]
#   make test_custom OFFLOAD=... HIP_SDK=... M=128 N=4096 K=2880 TA=0 TB=1 TYPE=0
#   make clean
#
# MODE=auto (default) -- lut if AUTOTUNE_DIR/lut/<arch>.fb exists, else
# autotune. MODE=lut forces lut (falls back to autotune with a warning if the
# .fb is missing). MODE=lut needs flatc + its include/ (a build tool, not part
# of this repo -- see README): FLATC=<path to flatc(.exe)>
# FLATBUFFERS_INC=<its include dir>.
# HIPDNN_EP_DEBUG=1 logs autotune. HIPDNN_EP_GEMM_AUTOTUNE=0 uses heuristics.
#
# NOTE: out/results.csv is not yet wired for this leaf (test_gemm.cpp's
# perf/correctness path is inlined in main(), unlike matmul_nbits/gqa's
# extracted single-shape functions) -- MODE=auto/lut/autotune all build and
# run correctly, but only print to stdout. See example/README.md.
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
KERNEL_HIP   := ../../../../hip/gemm_kernel.hip
AUTOTUNE_DIR := ../../../../hip/autotune/gemm
MODEL_DIR    := ../../models
COMMON_DIR   := ../../common

BUILD_DIR ?= out
TARGET    := $(BUILD_DIR)/test_gemm.exe
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

FLATC ?= flatc
FLATBUFFERS_INC ?=

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR) -I$(AUTOTUNE_DIR)

# ---- Test parameters (test_custom) ----

M    ?= 128
N    ?= 512
K    ?= 256
TA   ?= 0
TB   ?= 1
TYPE ?= 0
SHAPES ?= shapes.csv

DATA_DIR ?= data

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=gemm" && set "HIPDNN_RESULTS_LEAF=Xfp16_Wfp16_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=gemm && set HIPDNN_RESULTS_LEAF=Xfp16_Wfp16_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR) $(DATA_DIR)
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

GEN_DIR := $(BUILD_DIR)/gen

$(GEN_DIR): | $(BUILD_DIR)
	@if not exist $(subst /,\,$(GEN_DIR)) mkdir $(subst /,\,$(GEN_DIR))

TEST_OBJ   := $(BUILD_DIR)/test_gemm.obj
KERNEL_OBJ := $(BUILD_DIR)/gemm_kernel.obj

LUT_GENERATED_H := $(GEN_DIR)/gemm_autotune_generated.h
LUT_DATA_CPP    := $(GEN_DIR)/gemm_lut_data.cpp
LUT_DATA_OBJ    := $(BUILD_DIR)/lut_data.obj
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj
EMBED_SCRIPT    := $(COMMON_DIR)/embed_lut.py

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/gemm_autotune.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/gemm_autotune.fbs

$(LUT_DATA_CPP): $(LUT_FB) $(EMBED_SCRIPT) | $(GEN_DIR)
	$(PYTHON) $(EMBED_SCRIPT) $(LUT_FB) kGemmLutData $(LUT_DATA_CPP)

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/gemm_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/gemm_autotune.cpp -o $(AUTOTUNE_OBJ)

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

$(TARGET): FORCE mode_info $(LUT_OBJS) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $(LUT_EXTRA_INCLUDES) -x hip test_gemm.cpp -o $(TEST_OBJ)
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

bench_hipblaslt: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -DHIPDNN_GEMM_BENCH_HIPBLASLT -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) -lhipblaslt -o $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(M) $(N) $(K) $(TA) $(TB) $(TYPE)

# Decode NT, prefill NT, prefill NN, square-ish NT -- vs wrap_gemm algos. Not
# CI, kept as a convenience (was already in this Makefile before normalizing).
bench_suite: bench_hipblaslt
	$(call RUN_CMD,$(TARGET)) 1 4096 2880 0 1 0
	$(call RUN_CMD,$(TARGET)) 128 4096 2880 0 1 0
	$(call RUN_CMD,$(TARGET)) 128 4096 2880 0 0 0
	$(call RUN_CMD,$(TARGET)) 256 2048 2048 0 1 0

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_data.py --m $(M) --n $(N) --k $(K) --ta $(TA) --tb $(TB) --dir $(DATA_DIR)

# Smoke: one small fixed shape (matches this Makefile's M/N/K/TA/TB defaults).
test: gendata $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(M) $(N) $(K) $(TA) $(TB) $(TYPE) --data-dir $(DATA_DIR)

# One shape from Make variables / CLI.
test_custom: gendata $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(M) $(N) $(K) $(TA) $(TB) $(TYPE) --data-dir $(DATA_DIR)

# Every row in $(SHAPES) (default shapes.csv).
test_shapes: $(TARGET)
	@for /f "usebackq skip=1 tokens=1-6 delims=," %%a in ("$(SHAPES)") do @$(MAKE) test_custom M=%%a N=%%b K=%%c TA=%%d TB=%%e TYPE=%%f OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE)

# ---- Model sweep mode (M_array x KN_pairs from the shared model JSON) ----

MODEL_JSON ?= $(MODEL_DIR)/gpt_oss_20b.json

test_model: $(TARGET)
	$(call RUN_CMD,$(TARGET)) --model $(MODEL_JSON)

FORCE:
.PHONY: test test_shapes test_model test_custom clean bench_hipblaslt bench_suite mode_info gendata FORCE
