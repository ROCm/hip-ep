# ============================================================
# custom_kernels qmoe_ragged_mm (fp16 activations, uint4 weights) Test
#
#   make test        OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> [COVERAGE=1|2|3]
#   make test_custom OFFLOAD=... HIP_SDK=... ARGS="--launcher-grid-only"
#   make clean
#
# The op has no autotune table and no tuner, so MODE is accepted for the
# uniform API but only labels out/results.csv.
#
# `test` runs every categorical situation (routing x input layout x bias x
# zero points, the GEMV / WMMA16 / WMMA64 row-count boundaries, the bucket
# descriptors and the FC1 -> SwiGLU -> FC2 pipeline) on the shapes COVERAGE
# selects (default 3; see unit_test/README.md "Shape coverage"). Inputs and
# the CPU reference are generated in-process -- no python, no data files.
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
ARGS     ?=

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# ---- Paths ----

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../include
KERNEL_HIP  := ../../../hip/qmoe_ragged_mm_kernel.hip
# Token bucketing and SwiGLU, which the pipeline case drives around the MatMul.
ROUTING_HIP := ../../../hip/qmoe_kernel.hip

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe
RESULTS_CSV   := $(BUILD_DIR)/results.csv

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))

# ---- Build settings ----

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=qmoe_ragged_mm" && set "HIPDNN_RESULTS_LEAF=Xfp16_Wu4_Yfp16" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=qmoe_ragged_mm && set HIPDNN_RESULTS_LEAF=Xfp16_Wu4_Yfp16 && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE) && %CD%\$(1)"
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

DIRECT_TEST_OBJ    := $(BUILD_DIR)/direct_test.obj
DIRECT_KERNEL_OBJ  := $(BUILD_DIR)/direct_kernel.obj
DIRECT_ROUTING_OBJ := $(BUILD_DIR)/direct_routing.obj

direct: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_qmoe_ragged_mm.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(ROUTING_HIP) -o $(DIRECT_ROUTING_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) $(DIRECT_ROUTING_OBJ) -o $(TARGET_DIRECT)

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

# make test = build + run every categorical situation at $(COVERAGE)
# (default 3). No data files -- everything is in-process.
test: direct
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET_DIRECT)) --coverage $(COVERAGE)

# Pass the exe's own flags through ARGS= (e.g. --coverage 1,
# --launcher-grid-only).
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

.PHONY: test test_custom clean build direct FORCE
