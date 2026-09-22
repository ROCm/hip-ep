# ============================================================
# custom_kernels GQA flash prefill (TTFT) Test -- INT8 KV cache
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> [COVERAGE=1|2|3]
#   make test_custom OFFLOAD=... HIP_SDK=... H=32 G=8 D=128 SQ=512
#   make clean
#
# MODE=lookup (default) -- resolves the config from AUTOTUNE_DIR/lut/<arch>.fb,
# falling back to autotune when this arch has no table (warned, never a hard
# failure). MODE=autotune always sweeps. lookup needs flatc + its include/ (a
# build tool, not part
# of this repo -- see README): FLATC=<path to flatc(.exe)>
# FLATBUFFERS_INC=<its include dir>.
#
# `test` runs the exe with no single-shape flags, which hits its own built-in
# shape matrix (auto-enumerated in test_gqa_prefill_i8.cpp -- every
# categorical situation, always) x a small typical prompt-length list
# (thinned by COVERAGE, default 3). See example/README.md.
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
INCLUDE_DIR  := ../../../../../include
KERNEL_HIP   := ../../../../../hip/gqa_kernel.hip
AUTOTUNE_DIR := ../../../../../hip/autotune/gqa

BUILD_DIR ?= out
TARGET    := $(BUILD_DIR)/test_gqa_prefill_i8.exe
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
INCLUDES     := -I$(INCLUDE_DIR)

# ---- Test parameters (test_custom) ----

ITERS ?= 100
H     ?= 32
G     ?= 8
D     ?= 128
SQ    ?= 512

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_GQA_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=gqa" && set "HIPDNN_RESULTS_LEAF=prefill/Xfp16_Wi8_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_GQA_AUTOTUNE_MODE=$(KERNEL_MODE) && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=gqa && set HIPDNN_RESULTS_LEAF=prefill/Xfp16_Wi8_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR)
    endef
endif

CSV_FRESH ?= 1
RESET_RESULTS_CSV = if exist $(subst /,\,$(RESULTS_CSV)) del $(subst /,\,$(RESULTS_CSV))

# ============================================================
# Build -- gqa_kernel.hip does not call the autotune-LUT resolver itself, and
# neither does this test (unlike the fp16 decode/prefill leaves); MODE=lookup
# here only needs to satisfy gqa_autotune.cpp's LUT-data symbols so it links.
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

TEST_OBJ   := $(BUILD_DIR)/prefill_i8_test.obj
KERNEL_OBJ := $(BUILD_DIR)/prefill_kernel.obj

LUT_GENERATED_H := $(GEN_DIR)/gqa_autotune_generated.h
LUT_FB_HEADER   := $(GEN_DIR)/lut_fb_path.h
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/gqa_autotune.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/gqa_autotune.fbs

$(LUT_FB_HEADER): | $(GEN_DIR)
	@echo #define HIPDNN_LUT_FB "$(LUT_FB)" > $(subst /,\,$(LUT_FB_HEADER))

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/gqa_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/gqa_autotune.cpp -o $(AUTOTUNE_OBJ)

ifeq ($(MODE_RESOLVED),lookup)
    TEST_DEFINES       := -DHIPDNN_LUT_LINKED_EXTERNALLY=1
    LUT_EXTRA_INCLUDES := -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC)
    LUT_OBJS           := $(AUTOTUNE_OBJ)
    LUT_HEADER_DEP     := $(LUT_FB_HEADER)
else
    TEST_DEFINES       :=
    LUT_EXTRA_INCLUDES :=
    LUT_OBJS           :=
    LUT_HEADER_DEP      :=
endif

$(TARGET): FORCE mode_info $(LUT_OBJS) $(LUT_HEADER_DEP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $(LUT_EXTRA_INCLUDES) -x hip test_gqa_prefill_i8.cpp -o $(TEST_OBJ)
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

asm: | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) --save-temps $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(BUILD_DIR)/gqa_kernel_asm.obj
	@echo "[asm] ISA saved -- look for .s files in $(BUILD_DIR)/"

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

# Coverage-tiered sweep: the exe's own built-in 8-geometry x seq-len matrix
# (every categorical situation, always) x a small typical prompt-length list
# (thinned by COVERAGE, default 3). No single-shape flags -- that routes
# main() into the sweep branch.
test: $(TARGET)
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET)) --coverage $(COVERAGE) --iters $(ITERS)

# One shape from Make variables / CLI (in-process rng, no data files).
test_custom: $(TARGET)
ifeq ($(CSV_FRESH),1)
	@$(RESET_RESULTS_CSV)
endif
	$(call RUN_CMD,$(TARGET)) --h $(H) --g $(G) --d $(D) --sq $(SQ) --iters $(ITERS)

FORCE:
.PHONY: test test_custom clean asm mode_info FORCE
