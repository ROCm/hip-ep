# ============================================================
# hip_gemm (gemm_kernel.hip) correctness + bench
#
#   make test
#   make test M=128 N=4096 K=2880 TB=1
#   make bench_hipblaslt M=128 N=4096 K=2880 TB=1
#   make bench_suite
# ============================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

OFFLOAD := --offload-arch=gfx1151
BUILD_DIR ?= build

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?= C:\Users\zhanxipu\Documents\WorkSpace\Rocm\7p14\therock-dist-windows-gfx1151-7.14.0\therock-dist-windows-gfx1151-7.14.0
else
    HIP_SDK ?= /mnt/c/Users/zhanxipu/Documents/WorkSpace/Rocm/7p14/therock-dist-windows-gfx1151-7.14.0/therock-dist-windows-gfx1151-7.14.0
endif

HIPCC       := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../include
KERNEL_HIP  := ../../../hip/gemm_kernel.hip
TARGET      := $(BUILD_DIR)/test_gemm.exe

KERNEL_FLAGS := -O3 -std=c++17 -x hip -ffp-contract=fast
CXXFLAGS     := -O3 -std=c++17
INCLUDES     := -I$(INCLUDE_DIR)

M ?= 128
N ?= 512
K ?= 256
TA ?= 0
TB ?= 1
TYPE ?= 0

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    RUN_CMD     = set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && $(subst /,\,.\$(1))
    define DO_CLEAN
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    endef
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD     = cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && %CD%\$(1)"
    define DO_CLEAN
	rm -rf $(BUILD_DIR)
    endef
endif

all: $(TARGET)

ifeq ($(DETECTED_OS),Windows)
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
$(BUILD_DIR):
	@mkdir -p $@
endif

TEST_OBJ   := $(BUILD_DIR)/test_gemm.obj
KERNEL_OBJ := $(BUILD_DIR)/gemm_kernel.obj
# gemm_kernel.hip now calls hipdnn_ep::gemm_autotune::resolve. The real resolver
# (gemm_autotune.cpp) needs flatc-generated headers, which this standalone build
# has no flatc for, so link the flatbuffers-free stub (returns a miss -> shapes
# use the heuristic, which is what the harness wants). The production DLL links
# the real resolver instead.
AUTOTUNE_STUB := ../../../hip/autotune/gemm/tools/gemm_autotune_stub.cpp
STUB_OBJ      := $(BUILD_DIR)/gemm_autotune_stub.obj

$(TARGET): FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) $(AUTOTUNE_STUB) -o $(STUB_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) $(STUB_OBJ) -o $(TARGET)

bench_hipblaslt: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -DHIPDNN_GEMM_BENCH_HIPBLASLT -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) $(AUTOTUNE_STUB) -o $(STUB_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) $(STUB_OBJ) -lhipblaslt -o $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(M) $(N) $(K) $(TA) $(TB) $(TYPE)

# Decode NT, prefill NT, prefill NN, square-ish NT — vs wrap_gemm algos.
bench_suite: FORCE | $(BUILD_DIR)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) -DHIPDNN_GEMM_BENCH_HIPBLASLT -x hip test_gemm.cpp -o $(TEST_OBJ)
	$(HIPCC) -c $(KERNEL_FLAGS) $(OFFLOAD) $(INCLUDES) $(KERNEL_HIP) -o $(KERNEL_OBJ)
	$(HIPCC) -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) $(AUTOTUNE_STUB) -o $(STUB_OBJ)
	$(HIPCC) $(OFFLOAD) $(TEST_OBJ) $(KERNEL_OBJ) $(STUB_OBJ) -lhipblaslt -o $(TARGET)
	$(call RUN_CMD,$(TARGET)) 1 4096 2880 0 1 0
	$(call RUN_CMD,$(TARGET)) 128 4096 2880 0 1 0
	$(call RUN_CMD,$(TARGET)) 128 4096 2880 0 0 0
	$(call RUN_CMD,$(TARGET)) 256 2048 2048 0 1 0

test: $(TARGET)
	$(call RUN_CMD,$(TARGET))

run: $(TARGET)
	$(call RUN_CMD,$(TARGET)) $(M) $(N) $(K) $(TA) $(TB) $(TYPE)

clean:
	$(DO_CLEAN)

FORCE:
.PHONY: all test run bench_hipblaslt bench_suite clean FORCE
