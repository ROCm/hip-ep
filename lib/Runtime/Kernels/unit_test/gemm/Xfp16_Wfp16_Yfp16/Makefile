# ============================================================
# custom_kernels CK GEMM (fp16 operands, fp16 output) + fp16 GEMV Test
#
#   make test        HIP_SDK=<sdk> CK_SRC=<ck checkout> CK_GEN=<ck_generated> [OFFLOAD=--offload-arch=<gfx>] [COVERAGE=1|2|3]
#   make test_custom HIP_SDK=... CK_SRC=... CK_GEN=... ARGS="--coverage 1"
#   make clean
#
# ck_gemm.hip and ref_gemm.hip build against the Composable Kernel source
# hip-ep pins (cmake/deps.txt): CK_SRC is that checkout and CK_GEN the
# directory holding the ck/config.h a hip-ep configure writes to
# <build>/ck_generated.
#
# MODE=lookup (default) links the real ck_gemm_autotune.cpp resolver with
# AUTOTUNE_DIR/lut/<LUT_ARCH>.fb and needs FLATC + FLATBUFFERS_INC; a chip
# without a table falls back to autotune. MODE=autotune links an empty
# hip_ck_gemm_lut_candidates, so every case scans the instances.
#
# `test` runs every categorical situation on the shapes COVERAGE selects
# (default 3; see unit_test/README.md "Shape coverage"). Inputs and the CPU
# reference are generated in-process -- no python, no data files.
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

# ---- Required from CI / CLI ----

OFFLOAD  ?= --offload-arch=gfx11-generic
HIP_SDK  ?= $(HIP_PATH)
CK_SRC   ?=
CK_GEN   ?=
MODE     ?= lookup
COVERAGE ?= 3
ARGS     ?=

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# ---- Paths ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR  := ../../../include
KERNEL_DIR   := ../../../hip
KERNEL_HIPS  := $(KERNEL_DIR)/ck_gemm.hip $(KERNEL_DIR)/ref_gemm.hip $(KERNEL_DIR)/gemv_kernel.hip
AUTOTUNE_DIR := ../../../hip/autotune/ck_gemm

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe
RESULTS_CSV   := $(BUILD_DIR)/results.csv

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))

# A gfx*-generic ARCH runs on every member chip, so it takes the table of the
# GPU the test runs on -- this machine's (amdgpu-arch) unless LUT_ARCH=gfxNNNN
# names another. A concrete ARCH uses its own table.
ifeq ($(origin LUT_ARCH),undefined)
    ifneq ($(findstring generic,$(ARCH)),)
        LUT_ARCH := $(firstword $(shell "$(HIP_SDK)/bin/amdgpu-arch.exe"))
    else
        LUT_ARCH := $(ARCH)
    endif
endif
LUT_FB := $(AUTOTUNE_DIR)/lut/$(LUT_ARCH).fb

ifeq ($(MODE),lookup)
    MODE_RESOLVED := $(if $(wildcard $(LUT_FB)),lookup,autotune)
else
    MODE_RESOLVED := autotune
endif

FLATC ?= flatc
FLATBUFFERS_INC ?=

# ---- Build settings ----

KERNEL_FLAGS := -O2 -std=c++17 -x hip -DNDEBUG -Wno-ignored-attributes
CXXFLAGS     := -O3 -std=c++17 -Wno-c23-extensions
INCLUDES     := -I$(INCLUDE_DIR)
CK_INCLUDES  := -I$(CK_GEN) -I$(CK_SRC)/include -I$(CK_SRC)/library/include

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_KERNEL_UT_MODE=$(MODE_RESOLVED)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=gemm" && set "HIPDNN_RESULTS_LEAF=Xfp16_Wfp16_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
        if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_KERNEL_UT_MODE=$(MODE_RESOLVED) && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=gemm && set HIPDNN_RESULTS_LEAF=Xfp16_Wfp16_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
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
	@if not exist $(subst /,\,$(BUILD_DIR)) mkdir $(subst /,\,$(BUILD_DIR))
else
$(BUILD_DIR):
	@mkdir -p $@
endif

GEN_DIR := $(BUILD_DIR)/gen

$(GEN_DIR): | $(BUILD_DIR)
	@if not exist $(subst /,\,$(GEN_DIR)) mkdir $(subst /,\,$(GEN_DIR))

DIRECT_TEST_OBJ := $(BUILD_DIR)/direct_test.obj
KERNEL_OBJS     := $(patsubst $(KERNEL_DIR)/%.hip,$(BUILD_DIR)/%.obj,$(KERNEL_HIPS))

# ---- MODE=lookup extras: flatc-generated header + the real resolver, reused
# unmodified from hip/autotune/ck_gemm/. The table bytes are #embed-ed into
# test_gemm.cpp through the path macro in LUT_FB_HEADER.

LUT_GENERATED_H := $(GEN_DIR)/ck_gemm_autotune_generated.h
LUT_FB_HEADER   := $(GEN_DIR)/lut_fb_path.h
AUTOTUNE_OBJ    := $(BUILD_DIR)/autotune.obj

$(LUT_GENERATED_H): $(AUTOTUNE_DIR)/ck_gemm_autotune.fbs | $(GEN_DIR)
	$(FLATC) --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $(AUTOTUNE_DIR)/ck_gemm_autotune.fbs

$(LUT_FB_HEADER): | $(GEN_DIR)
	@echo #define HIPDNN_LUT_FB "$(LUT_FB)" > $(subst /,\,$(LUT_FB_HEADER))

$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/ck_gemm_autotune.cpp $(LUT_GENERATED_H) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) -x hip $(INCLUDES) -I$(GEN_DIR) -I$(FLATBUFFERS_INC) $(AUTOTUNE_DIR)/ck_gemm_autotune.cpp -o $(AUTOTUNE_OBJ)

ifeq ($(MODE_RESOLVED),lookup)
    TEST_DEFINES       := -DHIPDNN_LUT_LINKED_EXTERNALLY=1
    LUT_EXTRA_INCLUDES := -I$(GEN_DIR)
    LUT_OBJS           := $(AUTOTUNE_OBJ)
    LUT_HEADER_DEP     := $(LUT_FB_HEADER)
else
    TEST_DEFINES       :=
    LUT_EXTRA_INCLUDES :=
    LUT_OBJS           :=
    LUT_HEADER_DEP     :=
endif

$(BUILD_DIR)/%.obj: $(KERNEL_DIR)/%.hip | $(BUILD_DIR)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(CK_INCLUDES) $(INCLUDES) $< -o $@

direct: FORCE mode_info $(KERNEL_OBJS) $(LUT_OBJS) $(LUT_HEADER_DEP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $(LUT_EXTRA_INCLUDES) -x hip test_gemm.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(KERNEL_OBJS) $(LUT_OBJS) -o $(TARGET_DIRECT)

mode_info:
	@echo [MODE=$(MODE) -^> $(MODE_RESOLVED)] arch=$(ARCH) lut_fb=$(LUT_FB)
ifeq ($(CK_SRC),)
	@echo ERROR: pass CK_SRC=^<Composable Kernel checkout^> and CK_GEN=^<dir holding ck/config.h^> 1>&2
	@exit /b 1
endif
ifeq ($(MODE),lookup)
ifneq ($(MODE_RESOLVED),lookup)
	@echo [MODE=lookup] no LUT for $(LUT_ARCH) at $(LUT_FB) -- falling back to MODE=autotune
endif
endif
ifeq ($(MODE_RESOLVED),lookup)
ifeq ($(FLATBUFFERS_INC),)
	@echo ERROR: MODE=lookup needs FLATBUFFERS_INC=^<flatbuffers include dir^> 1>&2
	@exit /b 1
endif
endif

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

test: direct
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET_DIRECT)) --coverage $(COVERAGE)

test_custom: direct
ifeq ($(CSV_FRESH),1)
	@$(RESET_RESULTS_CSV)
endif
	$(call RUN_CMD,$(TARGET_DIRECT)) $(ARGS)

# Build only. The unit_test/ aggregator passes PKG_DIR and PKG_NAME to collect
# the exe into its hipdnn-kernel-ut package layout.
build: direct
ifneq ($(PKG_DIR),)
ifeq ($(DETECTED_OS),Windows)
	@copy /Y $(subst /,\,$(TARGET_DIRECT)) "$(subst /,\,$(PKG_DIR))\$(PKG_NAME)" >nul
else
	@cp $(TARGET_DIRECT) $(PKG_DIR)/$(PKG_NAME)
endif
endif

FORCE:

.PHONY: test test_custom clean build direct mode_info FORCE
