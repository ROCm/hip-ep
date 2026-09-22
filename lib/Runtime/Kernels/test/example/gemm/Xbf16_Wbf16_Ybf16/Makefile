# ============================================================
# hip_gemm (gemm_kernel.hip) correctness + bench
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> [COVERAGE=1|2|3]
#   make test_custom OFFLOAD=... HIP_SDK=... M=128 N=4096 K=2880 TA=0 TB=1
#   make clean
#
# `test` builds the exe and runs it with no positional shape, which hits the
# built-in `cases[]` matrix in test_gemm.cpp's main() -- every TA/TB/bias/
# dtype situation paired with a small set of typical shapes, NOT a full M/N/K
# sweep. COVERAGE (default 3) picks how many of those typical shapes run;
# every tier still touches every TA/TB/bias/dtype value. See
# example/README.md "Shape coverage".
#
# MODE=lookup (default) -- resolves the config from AUTOTUNE_DIR/lut/<arch>.fb,
# falling back to autotune when this arch has no table (warned, never a hard
# failure). MODE=autotune always sweeps. lookup needs flatc + its include/ (a
# build tool, not part
# of this repo -- see README): FLATC=<path to flatc(.exe)>
# FLATBUFFERS_INC=<its include dir>.
# HIPDNN_EP_DEBUG=1 logs autotune. HIPDNN_EP_GEMM_AUTOTUNE=0 uses heuristics.
#
# NOTE: out/results.csv carries one row per case, and the same detail is also
# printed to stdout. Rows whose shape reaches a fixed kernel (TA=1, or NN)
# report config=fixed-kernel: only the NT path has a config to choose.
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

# ---- Required from CI / CLI ----

OFFLOAD  ?= --offload-arch=gfx1151
HIP_SDK  ?= $(HIP_PATH)
MODE     ?= lookup
COVERAGE ?= 3

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# ---- Paths ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR  := ../../../../include
KERNEL_HIP   := ../../../../hip/gemm_kernel.hip
AUTOTUNE_DIR := ../../../../hip/autotune/gemm

BUILD_DIR ?= out
TARGET    := $(BUILD_DIR)/test_gemm.exe
RESULTS_CSV := $(BUILD_DIR)/results.csv

ARCH   := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

ifeq ($(MODE),lookup)
    MODE_RESOLVED := $(if $(wildcard $(LUT_FB)),lookup,autotune)
else
    MODE_RESOLVED := autotune
endif

# The kernel picks its own source from this; MODE only decides whether the
# real resolver and table are linked, so both must agree or a run labelled
# autotune would still be resolving from a table.
KERNEL_MODE := $(if $(filter lookup,$(MODE_RESOLVED)),lookup,online)

FLATC ?= flatc
FLATBUFFERS_INC ?=

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17 -Wno-c23-extensions
INCLUDES     := -I$(INCLUDE_DIR) -I$(AUTOTUNE_DIR)

# ---- Test parameters (test_custom) ----

M    ?= 128
N    ?= 512
K    ?= 256
TA   ?= 0
TB   ?= 1

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_GEMM_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=gemm" && set "HIPDNN_RESULTS_LEAF=Xbf16_Wbf16_Ybf16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_GEMM_AUTOTUNE_MODE=$(KERNEL_MODE) && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=gemm && set HIPDNN_RESULTS_LEAF=Xbf16_Wbf16_Ybf16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR)
    endef
endif

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

TEST_OBJ   := $(BUILD_DIR)/test_gemm.obj
KERNEL_OBJ := $(BUILD_DIR)/gemm_kernel.obj

# ---- MODE=lookup extras: flatc-generated header (kept, not python) + the real
# resolver, reused unmodified from hip/autotune/gemm/. The LUT bytes
# themselves are embedded directly into test_gemm.cpp via a one-line C23
# #embed (see the HIPDNN_LUT_LINKED_EXTERNALLY block there) -- LUT_FB_HEADER
# just gives that #embed a path macro without needing python or a generated
# .cpp; it's a single #define line written by `echo`, not data. ----

COMMON_DIR      := $(AUTOTUNE_DIR)/../common
DTYPE_GENERATED_H := $(GEN_DIR)/hipdnn_dtype_generated.h
LUT_GENERATED_H := $(GEN_DIR)/gemm_autotune_generated.h
LUT_FB_HEADER   := $(GEN_DIR)/lut_fb_path.h
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj

# gemm_autotune.fbs includes ../common/hipdnn_dtype.fbs (the shared dtype
# vocabulary, see autotune/common/README.md); flatc does not generate a
# transitively-included schema's header on its own, so it needs its own
# invocation into the SAME output dir gemm_autotune_generated.h's bare
# #include "hipdnn_dtype_generated.h" is found in.
$(DTYPE_GENERATED_H): $(COMMON_DIR)/hipdnn_dtype.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(COMMON_DIR)/hipdnn_dtype.fbs

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/gemm_autotune.fbs $(DTYPE_GENERATED_H) | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/gemm_autotune.fbs

$(LUT_FB_HEADER): | $(GEN_DIR)
	@echo #define HIPDNN_LUT_FB "$(LUT_FB)" > $(subst /,\,$(LUT_FB_HEADER))

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/gemm_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/gemm_autotune.cpp -o $(AUTOTUNE_OBJ)

ifeq ($(MODE_RESOLVED),lookup)
    TEST_DEFINES       := -DHIPDNN_LUT_LINKED_EXTERNALLY=1
    LUT_EXTRA_INCLUDES := -I$(GEN_DIR) -I$(FLATBUFFERS_INC)
    LUT_OBJS           := $(AUTOTUNE_OBJ)
    LUT_HEADER_DEP     := $(LUT_FB_HEADER)
else
    TEST_DEFINES       :=
    LUT_EXTRA_INCLUDES :=
    LUT_OBJS           :=
    LUT_HEADER_DEP      :=
endif

$(TARGET): FORCE mode_info $(LUT_OBJS) $(LUT_HEADER_DEP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $(LUT_EXTRA_INCLUDES) -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) $(LUT_OBJS) -o $(TARGET)

mode_info:
	@echo [MODE=$(MODE) -^> $(MODE_RESOLVED)] arch=$(ARCH) lut_fb=$(LUT_FB)
ifeq ($(MODE),lookup)
ifneq ($(MODE_RESOLVED),lookup)
	@echo [MODE=lookup] no LUT for $(ARCH) at $(LUT_FB) -- falling back to MODE=autotune
endif
endif
ifeq ($(MODE_RESOLVED),lookup)
ifeq ($(FLATBUFFERS_INC),)
	@echo ERROR: MODE=lookup needs FLATBUFFERS_INC=^<flatbuffers include dir^> (flatc's include/, a build tool outside the repo -- see README) 1>&2
	@exit /b 1
endif
endif

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

# Coverage-tiered sweep: the exe enumerates its own built-in cases[] matrix
# and self-samples to COVERAGE (default 3). No data-dir -- in-process rng.
test: $(TARGET)
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET)) --coverage $(COVERAGE)

# One shape from Make variables / CLI (in-process rng, no data files).
test_custom: $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(M) $(N) $(K) $(TA) $(TB)

FORCE:
.PHONY: test test_custom clean mode_info FORCE
