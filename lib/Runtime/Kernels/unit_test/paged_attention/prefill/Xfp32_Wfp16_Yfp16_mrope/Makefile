# FP32 MRoPE + cast + paged append/attention; complete unfused comparison.
ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /c
endif
OFFLOAD ?= --offload-arch=gfx11-generic
HIP_SDK ?= $(HIP_PATH)
MODE ?= lookup
RANK ?= 0
COMPARE ?= 0
MROPE_LAYOUT ?= 1
MROPE_SECTIONS ?= 11,11,10
MROPE_SCALE ?= 1
POSITION_MODE ?= distinct
FLASH_SOURCE ?=
SHARED_PREFIX ?= 0
LUT_ARCH ?= $(ARCH)
FLATC ?= flatc
FLATBUFFERS_INC ?=
AUTOTUNE_DIR := ../../../../hip/autotune/paged_attention
LUT_FB ?= $(AUTOTUNE_DIR)/lut/$(LUT_ARCH).fb
COVERAGE ?= 3
TIMING ?= standard
B ?= 3
H ?= 8
G ?= 2
D ?= 128
P ?= 64
S ?= 17
M ?= 8
QLENS ?= 3,0,17
PAST ?= 16,17,18
ROTARY ?= 0
ROTARY_DIM ?= 64
INTERLEAVED ?= 0
MAX_POSITION ?= 4096
SCALE ?= 0
WINDOW ?= -1
SOFTCAP ?= 0
MAP ?= permuted
SEED ?= 1119
ERROR ?= none
PHASE ?= prefill
BUILD_DIR ?= out
HIPCC := $(HIP_SDK)/bin/hipcc.exe
INCLUDE_DIR := ../../../../include
KERNEL_HIP := ../../../../hip/paged_attention_kernel.hip
GQA_HIP := ../../../../hip/gqa_kernel.hip
COMMON_HEADERS := $(wildcard ../../../common/*.h)
PUBLIC_HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)
ARCH := $(patsubst --offload-arch=%,%,$(OFFLOAD))
GEN_DIR := $(BUILD_DIR)/generated
GENERATED_H := $(GEN_DIR)/paged_attention_autotune_generated.h
LUT_PATH_H := $(GEN_DIR)/lut_fb_path.h
AUTOTUNE_OBJ := $(BUILD_DIR)/autotune.obj
TARGET := $(BUILD_DIR)/test_paged_attention.exe
TEST_DEFINES :=
LUT_DEP :=
ifeq ($(RANK),1)
GQA_HIP := ../../../common/paged_attention_rank_adapter.hip
TEST_DEFINES += -DHIPDNN_UT_RANK_PAGED_ADAPTER=1
else
ifneq ($(wildcard $(LUT_FB)),)
TEST_DEFINES += -DHIPDNN_PAGED_LUT_EMBED=1
LUT_DEP := $(LUT_FB) $(LUT_PATH_H)
else
ifneq ($(origin LUT_FB),file)
$(error explicit LUT_FB file does not exist: $(LUT_FB))
endif
endif
endif
COMPARE_OBJ := $(BUILD_DIR)/rope.obj $(BUILD_DIR)/cast.obj
$(BUILD_DIR)/rope.obj: ../../../../hip/rope_kernel.hip $(PUBLIC_HEADERS) | $(BUILD_DIR)
	"$(HIPCC)" -c $(KERNEL_FLAGS) $(OFFLOAD) -I$(INCLUDE_DIR) $< -o $@
$(BUILD_DIR)/cast.obj: ../../../../hip/cast_kernel.hip $(PUBLIC_HEADERS) | $(BUILD_DIR)
	"$(HIPCC)" -c $(KERNEL_FLAGS) $(OFFLOAD) -I$(INCLUDE_DIR) $< -o $@
INCLUDES := -I$(INCLUDE_DIR) -I$(AUTOTUNE_DIR) -I$(GEN_DIR) -I$(FLATBUFFERS_INC)
KERNEL_MODE := $(if $(filter autotune,$(MODE)),online,lookup)
RESULTS_CSV ?= $(BUILD_DIR)/results.csv
CSV_FRESH ?= 1
ifneq ($(filter $(MODE),lookup autotune),$(MODE))
$(error MODE must be lookup or autotune)
endif
ifeq ($(HIP_SDK),)
$(error pass HIP_SDK=<maintained HIP SDK>)
endif
CXXFLAGS := -O3 -std=c++17 -x hip -Wno-deprecated-declarations
KERNEL_FLAGS := -O2 -std=c++17 -x hip -ffp-contract=fast -Wno-deprecated-declarations
ifneq ($(ADLX_INCLUDE),)
CXXFLAGS += -I$(ADLX_INCLUDE)
endif
TEST_OBJ := $(BUILD_DIR)/test.obj
KERNEL_OBJ := $(BUILD_DIR)/paged.obj
GQA_OBJ := $(BUILD_DIR)/gqa.obj
$(BUILD_DIR):
	@if not exist "$(subst /,\,$(BUILD_DIR))" mkdir "$(subst /,\,$(BUILD_DIR))"
$(GEN_DIR): | $(BUILD_DIR)
	@if not exist "$(subst /,\,$(GEN_DIR))" mkdir "$(subst /,\,$(GEN_DIR))"
$(GENERATED_H): $(AUTOTUNE_DIR)/paged_attention_autotune.fbs | $(GEN_DIR)
	"$(FLATC)" --cpp --gen-object-api --scoped-enums -o $(GEN_DIR) $<
$(LUT_PATH_H): FORCE | $(GEN_DIR)
	@echo #define HIPDNN_PAGED_LUT_FB "$(LUT_FB)" > "$(subst /,\,$(LUT_PATH_H))"
$(AUTOTUNE_OBJ): $(AUTOTUNE_DIR)/paged_attention_autotune.cpp $(AUTOTUNE_DIR)/paged_attention_autotune.h $(PUBLIC_HEADERS) $(GENERATED_H) | $(BUILD_DIR)
	"$(HIPCC)" -c $(CXXFLAGS) $(OFFLOAD) $(INCLUDES) $< -o $@
$(TEST_OBJ): test_paged_attention.cpp $(COMMON_HEADERS) $(PUBLIC_HEADERS) $(AUTOTUNE_DIR)/paged_attention_autotune.h $(LUT_DEP) | $(BUILD_DIR) $(GEN_DIR)
	"$(HIPCC)" -c $(CXXFLAGS) $(OFFLOAD) $(TEST_DEFINES) $(INCLUDES) $< -o $@
$(KERNEL_OBJ): $(KERNEL_HIP) $(PUBLIC_HEADERS) | $(BUILD_DIR)
	"$(HIPCC)" -c $(KERNEL_FLAGS) $(OFFLOAD) -I$(INCLUDE_DIR) $< -o $@
$(GQA_OBJ): $(GQA_HIP) ../../../../hip/gqa_kernel.hip $(PUBLIC_HEADERS) | $(BUILD_DIR)
	"$(HIPCC)" -c $(KERNEL_FLAGS) $(OFFLOAD) -I$(INCLUDE_DIR) $< -o $@
$(TARGET): $(TEST_OBJ) $(KERNEL_OBJ) $(GQA_OBJ) $(AUTOTUNE_OBJ) $(COMPARE_OBJ)
	"$(HIPCC)" $(OFFLOAD) $^ -o $@
RUN = set "PATH=$(subst /,\,$(HIP_SDK))\bin;%PATH%" && set "HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE=$(KERNEL_MODE)" && set "HIPDNN_UT_TIMING=$(TIMING)" && set "HIPDNN_RESULTS_CSV=$(subst /,\,$(RESULTS_CSV))" && set "HIPDNN_RESULTS_ARCH=$(ARCH)" && set "HIPDNN_RESULTS_MODE=$(MODE)" && "$(subst /,\,$(TARGET))"
test: $(TARGET)
ifeq ($(CSV_FRESH),1)
	@if exist "$(subst /,\,$(RESULTS_CSV))" del "$(subst /,\,$(RESULTS_CSV))"
endif
	$(RUN) --coverage $(COVERAGE)
test_custom: $(TARGET)
ifeq ($(CSV_FRESH),1)
	@if exist "$(subst /,\,$(RESULTS_CSV))" del "$(subst /,\,$(RESULTS_CSV))"
endif
	$(RUN) --b $(B) --h $(H) --g $(G) --d $(D) --pages $(P) --page-size $(S) --table-columns $(M) --q-lens $(QLENS) --past $(PAST) --rotary $(ROTARY) --rotary-dim $(ROTARY_DIM) --interleaved $(INTERLEAVED) --max-position $(MAX_POSITION) --scale $(SCALE) --window $(WINDOW) --softcap $(SOFTCAP) --map $(MAP) --shared-prefix $(SHARED_PREFIX) --seed $(SEED) --error $(ERROR) --phase $(PHASE) --mrope-layout $(MROPE_LAYOUT) --mrope-sections $(MROPE_SECTIONS) --mrope-scale $(MROPE_SCALE) --position-mode $(POSITION_MODE) --compare $(COMPARE)
build: $(TARGET)
ifneq ($(PKG_DIR),)
	@copy /Y "$(subst /,\,$(TARGET))" "$(subst /,\,$(PKG_DIR))/$(PKG_NAME)" >nul
endif
clean:
	@if exist "$(subst /,\,$(BUILD_DIR))" rmdir /s /q "$(subst /,\,$(BUILD_DIR))"
.PHONY: test test_custom build clean

FORCE:
.PHONY: FORCE
