# ============================================================
# custom_kernels elementwise_bcast (Add / Mul / Min / Max with per-axis
# broadcasting, fp32, fp16 and bf16) Test
#
#   make test        HIP_SDK=<sdk> [OFFLOAD=--offload-arch=gfx11-generic] [COVERAGE=1|2|3] [TIMING=standard]
#   make test_custom HIP_SDK=<sdk> ARGS="--n 8 --c 256 --h 56 --w 56 [--lhs full|n|c|h|w|scalar]
#                    [--rhs full|n|c|h|w|scalar] [--op add|mul|min|max]
#                    [--dtype fp32|fp16|bf16] [--align aligned|dense|output]"
#   make clean
#
# The op has no autotune table and no tuner, so MODE is accepted for the
# uniform API but only labels out/results.csv.
#
# `test` runs every shape (see unit_test/README.md "Shape coverage") in both
# operand orders, for every dtype and op, with aligned buffers, a misaligned
# dense operand and a misaligned output, and with extreme values. Inputs and
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

OFFLOAD  ?= --offload-arch=gfx11-generic
HIP_SDK  ?= $(HIP_PATH)
MODE     ?= lookup
COVERAGE ?= 3
TIMING ?= standard
ARGS     ?=

ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# ---- Paths ----

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../include
KERNEL_HIP  := ../../../hip/elementwise_binary_kernel.hip

BUILD_DIR ?= out
TARGET_DIRECT := $(BUILD_DIR)/test_direct.exe
RESULTS_CSV   := $(BUILD_DIR)/results.csv

ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))

# ---- Build settings ----

KERNEL_FLAGS := -O3 -std=c++17 -x hip
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_OP=elementwise_bcast" && set "HIPDNN_RESULTS_LEAF=Xany_Wany_Yany" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE)" && set "HIPDNN_UT_TIMING=$(TIMING)" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && set HIPDNN_RESULTS_CSV=$(RESULTS_CSV) && set HIPDNN_RESULTS_OP=elementwise_bcast && set HIPDNN_RESULTS_LEAF=Xany_Wany_Yany && set HIPDNN_RESULTS_ARCH=$(ARCH) && set HIPDNN_RESULTS_MODE=$(MODE) && set HIPDNN_UT_TIMING=$(TIMING) && %CD%\$(1)"
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

DIRECT_TEST_OBJ   := $(BUILD_DIR)/direct_test.obj
DIRECT_KERNEL_OBJ := $(BUILD_DIR)/direct_kernel.obj

direct: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_elementwise_bcast.cpp -o $(DIRECT_TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(DIRECT_KERNEL_OBJ)
	$(HIPCC) $(OFFLOAD) $(DIRECT_TEST_OBJ) $(DIRECT_KERNEL_OBJ) -o $(TARGET_DIRECT)

# ============================================================
# Run / test targets (uniform API)
# ============================================================

clean:
	$(DO_CLEAN)

# make test = build + run every shape and categorical situation at $(COVERAGE)
# (default 3). No data files -- everything is in-process.
test: direct
	@$(RESET_RESULTS_CSV)
	$(call RUN_CMD,$(TARGET_DIRECT)) --coverage $(COVERAGE)

# One explicit case through ARGS= (the exe's --n --c --h --w and optional
# --lhs, --rhs, --op, --dtype, --align), checked and timed like the suite.
test_custom: direct
	$(if $(strip $(ARGS)),,$(error test_custom requires one explicit case in ARGS))
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
