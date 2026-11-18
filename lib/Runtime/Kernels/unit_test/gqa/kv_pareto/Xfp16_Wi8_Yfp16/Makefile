# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Link the production kernel package, including its GQA resolver and LUT.
# make test HIP_SDK=<sdk> HIP_EP_PACKAGE=<package> COVERAGE=1 TIMING=screen
# make test_custom HIP_SDK=<sdk> HIP_EP_PACKAGE=<package> H=24 SQ=1 PAST=4095 CAPACITY=4133
# OFFLOAD selects the test's stream-gate ISA. KERNEL_ARCH selects the packaged
# kernel library, defaulting to that ISA. No production kernels or LUT bytes
# are compiled into this executable.

OFFLOAD ?= --offload-arch=gfx11-generic
HIP_SDK ?= $(HIP_PATH)
HIP_EP_PACKAGE ?=
KERNEL_ARCH ?= $(patsubst --offload-arch=%,%,$(OFFLOAD))
MODE ?= lookup
COVERAGE ?= 1
TIMING ?= standard
BUILD_DIR ?= out
PKG_DIR ?=
PKG_NAME ?= hipdnn-kernel-ut-gqa-kv-pareto-xfp16-wi8-yfp16.exe

B ?= 1
H ?= 24
G ?= 4
D ?= 256
SQ ?= 17
PAST ?= 129
CAPACITY ?= 4096
SEED ?= 1234
SCALE ?= 0.0625

SOURCE := test_gqa_kv_pareto.cpp
TARGET := $(BUILD_DIR)/test_gqa_kv_pareto.exe
RESULTS_CSV := $(abspath $(BUILD_DIR)/results.csv)
STDERR_LOG := $(abspath $(BUILD_DIR)/kernel.stderr.log)
INCLUDES := -I../../../../include -I../../../../hip/autotune/gqa
ifneq ($(ADLX_INCLUDE),)
INCLUDES += -I$(ADLX_INCLUDE)
endif
COMMON_HEADERS := $(wildcard ../../../common/*.h)
KERNEL_MODE := $(if $(filter lookup,$(MODE)),lookup,online)
CUSTOM_ARGS := --batch $(B) --heads $(H) --kv-heads $(G) --head-dim $(D) --new-tokens $(SQ) --past $(PAST) --capacity $(CAPACITY) --seed $(SEED) --scale $(SCALE)

ifneq ($(filter clean,$(MAKECMDGOALS)),clean)
ifeq ($(HIP_SDK),)
$(error pass HIP_SDK=<HIP SDK root>)
endif
ifeq ($(HIP_EP_PACKAGE),)
$(error pass HIP_EP_PACKAGE=<built hip-ep package with KV-Pareto enabled>)
endif
endif
ifneq ($(filter $(MODE),lookup autotune),$(MODE))
$(error MODE must be lookup or autotune)
endif

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /c
HIPCC := $(HIP_SDK)/bin/hipcc.exe
KERNEL_LIBRARY := $(HIP_EP_PACKAGE)/lib/custom_kernels_$(KERNEL_ARCH).lib
NATIVE_STDERR = 2> "$(STDERR_LOG)"
COMPILE_LIBRARY = $(KERNEL_LIBRARY)
COMPILE_TARGET = $@
LINK_FLAGS :=
MKDIR = if not exist "$(subst /,\,$(1))" mkdir "$(subst /,\,$(1))"
COPY = copy /y "$(subst /,\,$(1))" "$(subst /,\,$(2))"
REMOVE = if exist "$(subst /,\,$(1))" del "$(subst /,\,$(1))"
RUN = set "PATH=$(subst /,\,$(HIP_EP_PACKAGE))/bin;$(subst /,\,$(HIP_SDK))/bin;%PATH%" && set "HIPDNN_UT_TIMING=$(TIMING)" && set "HIPDNN_RESULTS_ARCH=$(KERNEL_ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE)" && set "HIPDNN_KERNEL_UT_MODE=$(MODE)" && set "HIPDNN_GQA_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_KERNEL_UT_STDERR=$(subst /,\,$(STDERR_LOG))" && "$(subst /,\,$(TARGET))" $(1)
else ifneq ($(wildcard $(HIP_SDK)/bin/hipcc.exe),)
# WSL builds with the Windows SDK, as does the top-level UT Makefile.
HIPCC := $(HIP_SDK)/bin/hipcc.exe
KERNEL_LIBRARY := $(HIP_EP_PACKAGE)/lib/custom_kernels_$(KERNEL_ARCH).lib
COMPILE_LIBRARY = $(shell wslpath -w "$(KERNEL_LIBRARY)")
COMPILE_TARGET = $(shell wslpath -w "$(abspath $@)")
LINK_FLAGS :=
HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
HIP_EP_PACKAGE_WIN := $(shell wslpath -w "$(HIP_EP_PACKAGE)")
TARGET_WIN := $(shell wslpath -w "$(abspath $(TARGET))")
RESULTS_CSV_WIN := $(shell wslpath -w "$(RESULTS_CSV)")
STDERR_LOG_WIN := $(shell wslpath -w "$(STDERR_LOG)")
MKDIR = mkdir -p "$(1)"
COPY = cp "$(1)" "$(2)"
REMOVE = rm -f "$(1)"
RUN = cmd.exe /c 'set "PATH=$(HIP_EP_PACKAGE_WIN)\bin;$(HIP_SDK_WIN)\bin;%PATH%" && set "HIPDNN_UT_TIMING=$(TIMING)" && set "HIPDNN_RESULTS_ARCH=$(KERNEL_ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE)" && set "HIPDNN_KERNEL_UT_MODE=$(MODE)" && set "HIPDNN_GQA_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_RESULTS_CSV=$(RESULTS_CSV_WIN)" && set "HIPDNN_KERNEL_UT_STDERR=$(STDERR_LOG_WIN)" && "$(TARGET_WIN)" $(1) 2> "$(STDERR_LOG_WIN)"'
else
HIPCC := $(HIP_SDK)/bin/hipcc
KERNEL_LIBRARY := $(HIP_EP_PACKAGE)/lib/libcustom_kernels_$(KERNEL_ARCH).so
NATIVE_STDERR = 2> "$(STDERR_LOG)"
COMPILE_LIBRARY = $(KERNEL_LIBRARY)
COMPILE_TARGET = $@
LINK_FLAGS := -Wl,-rpath,$(abspath $(HIP_EP_PACKAGE)/lib)
MKDIR = mkdir -p "$(1)"
COPY = cp "$(1)" "$(2)"
REMOVE = rm -f "$(1)"
RUN = LD_LIBRARY_PATH="$(HIP_EP_PACKAGE)/lib:$(HIP_SDK)/lib:$${LD_LIBRARY_PATH}" HIPDNN_UT_TIMING=$(TIMING) HIPDNN_RESULTS_ARCH=$(KERNEL_ARCH) HIPDNN_RESULTS_MODE=$(MODE) HIPDNN_KERNEL_UT_MODE=$(MODE) HIPDNN_GQA_AUTOTUNE_MODE=$(KERNEL_MODE) HIPDNN_RESULTS_CSV="$(RESULTS_CSV)" HIPDNN_KERNEL_UT_STDERR="$(STDERR_LOG)" "$(TARGET)" $(1)
endif

.PHONY: build test test_custom clean
build: $(TARGET)
ifneq ($(PKG_DIR),)
	$(call MKDIR,$(PKG_DIR))
	$(call COPY,$(TARGET),$(PKG_DIR)/$(PKG_NAME))
endif

$(TARGET): $(SOURCE) $(COMMON_HEADERS) $(KERNEL_LIBRARY)
	$(call MKDIR,$(BUILD_DIR))
	"$(HIPCC)" -O3 -std=c++17 -x hip $(OFFLOAD) $(INCLUDES) $(SOURCE) -x none "$(COMPILE_LIBRARY)" $(LINK_FLAGS) -o "$(COMPILE_TARGET)"

test: build
	$(call REMOVE,$(RESULTS_CSV))
	$(call RUN,--coverage $(COVERAGE)) $(NATIVE_STDERR)

test_custom: build
	$(call REMOVE,$(RESULTS_CSV))
	$(call RUN,$(CUSTOM_ARGS)) $(NATIVE_STDERR)

clean:
ifeq ($(OS),Windows_NT)
	if exist "$(subst /,\,$(BUILD_DIR))" rmdir /s /q "$(subst /,\,$(BUILD_DIR))"
else
	rm -rf "$(BUILD_DIR)"
endif
