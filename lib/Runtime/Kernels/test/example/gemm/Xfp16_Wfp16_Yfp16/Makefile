# ============================================================
# hip_gemm (gemm_kernel.hip) correctness + bench
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk>
#   make test_shapes OFFLOAD=... HIP_SDK=... [SHAPES=shapes.csv]
#   make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../models/gpt_oss_20b.json]
#   make test_custom OFFLOAD=... HIP_SDK=... M=128 N=4096 K=2880 TA=0 TB=1 TYPE=0
#   make clean
#
# MODE=autotune (default) | MODE=lut -- see README.md.
# HIPDNN_EP_DEBUG=1 logs autotune. HIPDNN_EP_GEMM_AUTOTUNE=0 uses heuristics.
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
INCLUDE_DIR  := ../../../../include
KERNEL_HIP   := ../../../../hip/gemm_kernel.hip
AUTOTUNE_DIR := ../../../../hip/autotune/gemm
MODEL_DIR    := ../../models

BUILD_DIR ?= out
TARGET    := $(BUILD_DIR)/test_gemm.exe

ARCH   := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

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
# Build
# ============================================================

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

TEST_OBJ   := $(BUILD_DIR)/test_gemm.obj
KERNEL_OBJ := $(BUILD_DIR)/gemm_kernel.obj

$(TARGET): FORCE lut_note | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) -o $(TARGET)

lut_note:
ifeq ($(MODE),lut)
	@if exist "$(subst /,\,$(LUT_FB))" (echo [MODE=lut] $(LUT_FB) exists but this hipcc-only build has no flatc-free embedder yet -- see README; falling back to MODE=autotune) else (echo [MODE=lut] no LUT for $(ARCH) at $(LUT_FB); MODE=autotune)
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
.PHONY: test test_shapes test_model test_custom clean bench_hipblaslt bench_suite lut_note gendata FORCE
