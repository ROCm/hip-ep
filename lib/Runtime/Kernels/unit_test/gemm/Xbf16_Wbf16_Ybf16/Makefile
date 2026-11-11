# ============================================================
# hip_gemm (Xbf16_Wbf16_Ybf16) correctness + adaptive complete-call bench
#
#   make test        HIP_SDK=<sdk> [OFFLOAD=--offload-arch=<gfx>] [COVERAGE=1|2|3] [TIMING=screen|standard|confirm]
#   make test_custom OFFLOAD=... HIP_SDK=... M=128 N=4096 K=2880 TA=0 TB=1
#   make clean
#
# MODE=lookup (default) resolves AUTOTUNE_DIR/lut/<LUT_ARCH>.fb; MODE=autotune
# always sweeps. Timing policy lives in unit_test/common/.
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

OFFLOAD  ?= --offload-arch=gfx11-generic
HIP_SDK  ?= $(HIP_PATH)
MODE     ?= lookup
COVERAGE ?= 3
TIMING   ?= standard

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR  := ../../../include
KERNEL_HIP   := ../../../hip/gemm_kernel.hip
AUTOTUNE_DIR := ../../../hip/autotune/gemm

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_gemm.exe
RESULTS_CSV   := $(BUILD_DIR)/results.csv

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))

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
KERNEL_MODE := $(if $(filter lookup,$(MODE_RESOLVED)),lookup,online)

FLATC ?= flatc
FLATBUFFERS_INC ?=

KERNEL_FLAGS := -O2 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17 -Wno-c23-extensions
INCLUDES     := -I$(INCLUDE_DIR) -I$(AUTOTUNE_DIR)

M    ?= 128
N    ?= 512
K    ?= 256
TA   ?= 0
TB   ?= 1

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_UT_TIMING=$(TIMING)" && set "HIPDNN_GEMM_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=gemm" && set "HIPDNN_RESULTS_LEAF=Xbf16_Wbf16_Ybf16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE_RESOLVED)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_UT_TIMING=$(TIMING) && set HIPDNN_GEMM_AUTOTUNE_MODE=$(KERNEL_MODE) && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=gemm && set HIPDNN_RESULTS_LEAF=Xbf16_Wbf16_Ybf16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE_RESOLVED) && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR)
    endef
endif

RESET_RESULTS_CSV = if exist $(subst /,\,$(RESULTS_CSV)) del $(subst /,\,$(RESULTS_CSV))

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

COMMON_DIR        := $(AUTOTUNE_DIR)/../common
DTYPE_GENERATED_H := $(GEN_DIR)/hipdnn_dtype_generated.h
LUT_GENERATED_H   := $(GEN_DIR)/gemm_autotune_generated.h
LUT_FB_HEADER     := $(GEN_DIR)/lut_fb_path.h
AUTOTUNE_OBJ      := $(BUILD_DIR)/autotune.obj

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
    LUT_HEADER_DEP     :=
endif

direct: FORCE mode_info $(LUT_OBJS) $(LUT_HEADER_DEP) | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $(LUT_EXTRA_INCLUDES) -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) $(LUT_OBJS) -o $(TARGET_DIRECT)

mode_info:
	@echo [MODE=$(MODE) -^> $(MODE_RESOLVED)] arch=$(ARCH) lut_fb=$(LUT_FB)
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

clean:
	$(DO_CLEAN)

test: direct
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET_DIRECT)) --coverage $(COVERAGE)

test_custom: direct
	$(call RUN_CMD,$(TARGET_DIRECT)) $(M) $(N) $(K) $(TA) $(TB)

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
