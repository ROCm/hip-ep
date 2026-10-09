# Kernel unit tests: build every leaf, then run them through hipdnn-kernel-ut.exe,
# the same controller the packaging CI runs.
#
#   make test  OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> [MODE=lookup|autotune] [COVERAGE=1|2|3]
#   make build OFFLOAD=... HIP_SDK=...   (build only)
#   make clean OFFLOAD=... HIP_SDK=...
#
# `build` runs each leaf's own `make build` (the kernel sources are compiled into
# the leaf), collects the executables into out/bin/kernel-tests/<arch>/ under the
# names the CMake package installs, and builds out/bin/hipdnn-kernel-ut.exe from
# kernel_ut_main.cpp. `test` then runs, from out/bin,
#   hipdnn-kernel-ut.exe --mode <MODE> --coverage <COVERAGE>
# with the CI package's layout, exit codes and out/results.csv. Add -j to build
# the leaves in parallel.

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

OFFLOAD  ?=
HIP_SDK  ?=
MODE     ?= lookup
COVERAGE ?= 3

ifeq ($(OFFLOAD),)
    $(error pass OFFLOAD=--offload-arch=<gfx>)
endif
ifeq ($(HIP_SDK),)
    $(error pass HIP_SDK=<HIP SDK root>)
endif

# Keep in sync with example/CMakeLists.txt and kernel_ut_main.cpp.
LEAVES := gqa/decode/Xfp16_Wfp16_Yfp16 gqa/decode/Xfp16_Wi8_Yfp16 \
	gqa/prefill/Xfp16_Wfp16_Yfp16 gqa/prefill/Xfp16_Wi8_Yfp16 \
	matmul_nbits/Xfp16_Wu2_Yfp16 matmul_nbits/Xfp16_Wu3_Yfp16 \
	matmul_nbits/Xfp16_Wu4_Yfp16 matmul_nbits/Xfp16_Wi8_Yfp16

ARCH  := $(patsubst --offload-arch=%,%,$(OFFLOAD))
HIPCC := $(HIP_SDK)/bin/hipcc.exe
PKG   := out/bin
KT    := $(PKG)/kernel-tests/$(ARCH)

# A leaf's packaged executable, named as in CMakeLists.txt / kernel_ut_main.cpp:
# matmul_nbits/Xfp16_Wu4_Yfp16 -> hipdnn-kernel-ut-matmul-nbits-xfp16-wu4-yfp16.exe
ut_exe = hipdnn-kernel-ut-$(subst _,-,$(subst /,-,$(subst X,x,$(subst W,w,$(subst Y,y,$(1)))))).exe

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK_WIN := $(subst /,\,$(HIP_SDK))
    MKDIR_KT    = if not exist $(subst /,\,$(KT)) mkdir $(subst /,\,$(KT))
    RUN_UT      = cd $(subst /,\,$(PKG)) && set "PATH=$(HIP_SDK_WIN)\bin;%PATH%" && hipdnn-kernel-ut.exe $(1)
    RM_OUT      = if exist out rmdir /s /q out
else
    HIP_SDK_WIN := $(shell wslpath -w "$(HIP_SDK)")
    MKDIR_KT    = mkdir -p $(KT)
    RUN_UT      = cd $(PKG) && cmd.exe /c "set PATH=$(HIP_SDK_WIN)\bin;%PATH% && hipdnn-kernel-ut.exe $(1)"
    RM_OUT      = rm -rf out
endif

CLEAN_LEAVES := $(addprefix clean/,$(LEAVES))

.PHONY: test build clean $(LEAVES) $(CLEAN_LEAVES)

test: build
	$(call RUN_UT,--mode $(MODE) --coverage $(COVERAGE))

build: $(LEAVES) $(PKG)/hipdnn-kernel-ut.exe

$(KT):
	@$(MKDIR_KT)

$(LEAVES): | $(KT)
	$(MAKE) -C $@ build OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK) MODE=$(MODE) PKG_DIR=$(abspath $(KT)) PKG_NAME=$(call ut_exe,$@)

$(PKG)/hipdnn-kernel-ut.exe: kernel_ut_main.cpp | $(KT)
	$(HIPCC) -O3 -std=c++17 -w $(OFFLOAD) -x hip kernel_ut_main.cpp -o $@

clean: $(CLEAN_LEAVES)
	@$(RM_OUT)

$(CLEAN_LEAVES):
	$(MAKE) -C $(patsubst clean/%,%,$@) clean OFFLOAD=$(OFFLOAD) HIP_SDK=$(HIP_SDK)
