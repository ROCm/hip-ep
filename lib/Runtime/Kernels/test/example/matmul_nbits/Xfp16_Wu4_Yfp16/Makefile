# ============================================================
# custom_kernels MatMulNBits bits=4 (uint4) Test
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> [COVERAGE=1|2|3]
#   make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x2880x5120 GS=128
#   make clean
#
# MODE=lookup (default) -- resolves the config from AUTOTUNE_DIR/lut/<arch>.fb,
# falling back to autotune when this arch has no table (warned, never a hard
# failure). MODE=autotune always sweeps. lookup needs flatc + its include/ (a
# build tool, not part
# of this repo -- see README): FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its
# include dir>.
#
# `test` runs the exe's own built-in `sweep` namespace: every categorical
# situation (group_size x zero-points x dtype) x a small typical-shape list,
# self-sampled to COVERAGE (default 3; see example/README.md "Shape
# coverage"). Everything (inputs + CPU reference) is generated in-process --
# no python, no data files.
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
KERNEL_HIP   := ../../../../hip/matmul_nbits_kernel.hip
AUTOTUNE_DIR := ../../../../hip/autotune/matmul_nbits

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe
RESULTS_CSV   := $(BUILD_DIR)/results.csv

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))
LUT_FB := $(AUTOTUNE_DIR)/lut/$(ARCH).fb

# MODE resolution: auto picks lut iff the arch's .fb exists; an explicit
# MODE=lookup with no .fb falls back to autotune (never a hard failure -- see
# mode_info below, which prints the fallback so it isn't silent).
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

# ---- Build settings ----

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17 -Wno-c23-extensions
INCLUDES     := -I$(INCLUDE_DIR)

# ---- Test parameters (test_custom) ----

SIZE     ?= 128x2880x5120
GS       ?= 128
NO_ZEROS ?=
FP32     ?=

ifeq ($(NO_ZEROS),1)
    CXX_NZ_FLAG := --no-zeros
else
    CXX_NZ_FLAG :=
endif
ifeq ($(FP32),1)
    CXX_FP32_FLAG := --fp32
else
    CXX_FP32_FLAG :=
endif

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_MATMUL_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=matmul_nbits" && set "HIPDNN_RESULTS_LEAF=Xfp16_Wu4_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_MATMUL_AUTOTUNE_MODE=$(KERNEL_MODE) && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=matmul_nbits && set HIPDNN_RESULTS_LEAF=Xfp16_Wu4_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR)
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

# ---- MODE=lookup extras: flatc-generated header (kept, not python) + the real
# resolver, reused unmodified from hip/autotune/matmul_nbits/. The LUT bytes
# themselves are embedded directly into test_matmul_nbits.cpp via a one-line
# C23 #embed (see the HIPDNN_LUT_LINKED_EXTERNALLY block there) --
# LUT_FB_HEADER just gives that #embed a path macro without needing python or
# a generated .cpp; it's a single #define line written by `echo`, not data.

LUT_GENERATED_H := $(GEN_DIR)/matmul_nbits_autotune_generated.h
LUT_FB_HEADER   := $(GEN_DIR)/lut_fb_path.h
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/matmul_nbits_autotune.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/matmul_nbits_autotune.fbs

$(LUT_FB_HEADER): | $(GEN_DIR)
	@echo #define HIPDNN_LUT_FB "$(LUT_FB)" > $(subst /,\,$(LUT_FB_HEADER))

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/matmul_nbits_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/matmul_nbits_autotune.cpp -o $(AUTOTUNE_OBJ)

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

direct: FORCE mode_info $(LUT_OBJS) $(LUT_HEADER_DEP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) -I$(AUTOTUNE_DIR) $(LUT_EXTRA_INCLUDES) -x hip test_matmul_nbits.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) $(LUT_OBJS) -o $(TARGET_DIRECT)

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

asm: | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) --save-temps $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(BUILD_DIR)/matmul_nbits_kernel_asm.obj
	@echo "[asm] ISA saved -- look for .s files in $(BUILD_DIR)/"

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

# make test = build + run the categorical x typical-shape sweep at
# $(COVERAGE) (default 3). No data files -- everything is in-process.
test: direct
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET_DIRECT)) --coverage $(COVERAGE)

# One shape from Make variables / CLI (SIZE=, GS=, NO_ZEROS=1, FP32=1).
test_custom: direct
ifeq ($(CSV_FRESH),1)
	@$(RESET_RESULTS_CSV)
endif
	$(call RUN_CMD,$(TARGET_DIRECT)) $(SIZE) --group-size $(GS) $(CXX_NZ_FLAG) $(CXX_FP32_FLAG)

FORCE:

.PHONY: test test_custom clean direct asm mode_info FORCE
