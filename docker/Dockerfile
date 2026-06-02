##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# hipdnn-ep Linux build environment. Mirrors the apt setup in
# .github/workflows/linux-build.yml so local matches CI on ubuntu-latest.
FROM ubuntu:24.04

ARG LLVM_MAJOR=22
ARG DEBIAN_FRONTEND=noninteractive
# Parking slot for the squatter `ubuntu` user/group (see block comment below).
# Override at build time with `--build-arg UBUNTU_USER_ID=<n>`. docker/run.sh
# forwards this from its own `UBUNTU_USER_ID` env (default 60001) and CI
# pins the value in .github/workflows/linux-build.yml so it can't silently
# drift. Single integer is used for both the renumbered UID and GID.
ARG UBUNTU_USER_ID=60001

# Ubuntu 24.04+ ships a default `ubuntu:ubuntu` user/group at UID/GID 1000,
# which collides with the most common Linux host user IDs (the entrypoint
# needs to claim HOST_UID:HOST_GID — typically 1000:1000 — so bind-mounted
# host files keep the right owner without chown). Park the squatter at a
# high UID/GID at build time rather than tearing it down at runtime: the
# `ubuntu` user and its home directory are preserved intact, just renumbered
# out of the way. Three steps because shadow-utils command behavior is
# asymmetric:
#   1. groupmod -g — updates /etc/group; does NOT chgrp files on disk.
#   2. usermod -u -g — updates /etc/passwd AND auto-chowns files in the
#      user's home dir whose user owner matched the old UID (the new
#      primary-GID pointer is corrected in the same call so no /etc/passwd
#      entry dangles at the freed GID).
#   3. find -gid + chgrp — group ownership of files is NOT touched by
#      either of the first two; sweep the rootfs (xdev to skip pseudo-fs)
#      and re-stamp anything still group-owned by the freed GID, otherwise
#      whatever group later claims GID 1000 silently inherits those files.
# The `1000` in `find -gid 1000` is the SOURCE id (Ubuntu 24.04's fixed
# default for the `ubuntu` user) — it stays a literal. UBUNTU_USER_ID is
# the DESTINATION id.
RUN if id ubuntu >/dev/null 2>&1; then \
        groupmod -g ${UBUNTU_USER_ID} ubuntu && \
        usermod -u ${UBUNTU_USER_ID} -g ${UBUNTU_USER_ID} ubuntu && \
        find / -xdev -gid 1000 -exec chgrp -h ${UBUNTU_USER_ID} {} +; \
    fi

# Layer 1: base toolchain for ORT / protobuf / flatbuffers / OGA sub-builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config \
        git ca-certificates curl wget gnupg lsb-release \
        python3 python3-pip python3-venv python3-dev \
        sudo gosu \
        zip unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: LLVM/MLIR/clang/lld from apt.llvm.org (pinned via LLVM_MAJOR).
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null \
    && echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_MAJOR} main" \
        > /etc/apt/sources.list.d/llvm-${LLVM_MAJOR}.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        llvm-${LLVM_MAJOR}-dev \
        llvm-${LLVM_MAJOR}-tools \
        libpolly-${LLVM_MAJOR}-dev \
        libmlir-${LLVM_MAJOR}-dev \
        mlir-${LLVM_MAJOR}-tools \
        clang-${LLVM_MAJOR} \
        lld-${LLVM_MAJOR} \
        liblld-${LLVM_MAJOR}-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 3: pip bits. ubuntu:24.04 ships cmake 3.28.3 but backend-mlir-compiler
# needs >=3.29; pip cmake lands at /usr/local/bin and wins over apt's.
RUN pip3 install --no-cache-dir --break-system-packages \
        cmake lit setuptools wheel numpy requests packaging \
    && ln -sf /usr/local/bin/lit /usr/bin/lit

# Sanity: the cmake configs the build needs are actually present.
RUN test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/llvm/LLVMConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/mlir/MLIRConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/lld/LLDConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/libPolly.a

# NOPASSWD sudo for any user the entrypoint puts into the `dev` group.
RUN echo "%dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev-nopasswd \
    && chmod 0440 /etc/sudoers.d/dev-nopasswd

ENV LLVM_MAJOR=${LLVM_MAJOR}
ENV PATH=/usr/lib/llvm-${LLVM_MAJOR}/bin:${PATH}

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
