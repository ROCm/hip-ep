# ============================================================
# GQA autotune sweep
#
# Enumerates every autotune candidate (prefill v5/v7/v8 tile configs, decode
# impl x split-count) for the representative model shapes in
# RdpCapture/ops_analyze/gqa/data/ and writes the full ranking plus the winner
# per shape back into the same folder. Run tools/make_gqa_xlsx.py afterwards to
# refresh the workbook that collects all four tables as sheets.
#
# Single TU: gqa_autotune_sweep.cpp #includes gqa_kernel.hip so it can call the
# internal dispatch helpers, so gqa_kernel.obj must NOT be linked in.
#
# Usage:
#   make                       # build
#   make run                   # sweep every shape -> $(OUT_CSV)
#   make run ONLY=P            # prefill shapes only
#   make run ONLY=D            # decode shapes only
#   make run ONLY=P23          # one shape
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := WSL
endif

OFFLOAD   := --offload-arch=gfx1151
BUILD_DIR ?= build

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?= $(HIP_PATH)
else
    HIP_SDK ?= /mnt/c/Users/zhanxipu/Documents/WorkSpace/Rocm/7.1
endif

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../../include
ANALYSIS    := ../../../../../../../../RdpCapture/ops_analyze/gqa/data

TARGET   := $(BUILD_DIR)/gqa_autotune_sweep.exe
SHAPES   ?= $(ANALYSIS)/gqa_representative_shapes.csv
OUT_CSV  ?= $(ANALYSIS)/gqa_autotune_results.csv
BEST_CSV ?= $(ANALYSIS)/gqa_autotune_best.csv
LOG      ?= $(BUILD_DIR)/sweep.log

# -w: the kernel TU is warning-clean under its own build flags; this harness
# compiles it as a header and would otherwise replay every warning.
CXXFLAGS := -O3 -std=c++17 -x hip -w -D_CRT_SECURE_NO_WARNINGS

ONLY      ?=
TARGET_MS ?= 40
ROUNDS    ?= 3

ifneq ($(ONLY),)
    ONLY_ARG := --only $(ONLY)
endif

all: $(TARGET)

$(BUILD_DIR):
ifeq ($(DETECTED_OS),Windows)
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
	@mkdir -p $@
endif

KERNEL_HIP_DIR := ../../../../hip

$(TARGET): gqa_autotune_sweep.cpp | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(OFFLOAD) -I$(INCLUDE_DIR) -I$(KERNEL_HIP_DIR) $< -o $@

run: $(TARGET)
	$(TARGET) --shapes $(SHAPES) --csv $(OUT_CSV) --best-csv $(BEST_CSV) \
	    --target-ms $(TARGET_MS) --rounds $(ROUNDS) $(ONLY_ARG) 2>&1 | tee $(LOG)

clean:
ifeq ($(DETECTED_OS),Windows)
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
else
	@rm -rf $(BUILD_DIR)
endif

.PHONY: all run clean
