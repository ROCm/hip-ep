/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
/*
 * C ABI between the EP and the MIGraphX backend plugin
 * (lib/Backend/MIGraphX). The EP loads the plugin at run time, so hipgpu
 * builds and runs without MIGraphX; no MIGraphX type crosses this boundary.
 *
 * Every function that can fail returns 0 on success and otherwise writes a
 * NUL-terminated message (truncated to err_size) into err.
 */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIPEP_MGX_PLUGIN_ABI_VERSION 1

typedef struct hipep_mgx_program hipep_mgx_program;

/* Returns HIPEP_MGX_PLUGIN_ABI_VERSION of the plugin. */
typedef int (*hipep_mgx_abi_version_fn)(void);

/* Parses and compiles the ONNX model for the GPU target. Input shapes are the
 * static shapes of the ORT graph inputs; they replace the shapes in the file.
 */
typedef int (*hipep_mgx_compile_fn)(const char *model_path, size_t num_inputs,
                                    const char *const *input_names,
                                    const size_t *const *input_dims,
                                    const size_t *input_ranks,
                                    hipep_mgx_program **out_program, char *err,
                                    size_t err_size);

typedef size_t (*hipep_mgx_num_outputs_fn)(const hipep_mgx_program *program);

/* Writes the static shape of output `index` into dims[0..*rank). */
typedef int (*hipep_mgx_output_shape_fn)(const hipep_mgx_program *program,
                                         size_t index, size_t *dims,
                                         size_t max_rank, size_t *rank);

/* Runs the program on caller-owned device buffers and returns after the GPU
 * work has completed. Inputs are bound by name, outputs by index. */
typedef int (*hipep_mgx_run_fn)(hipep_mgx_program *program, size_t num_inputs,
                                const char *const *input_names,
                                void *const *input_data, size_t num_outputs,
                                void *const *output_data, char *err,
                                size_t err_size);

typedef void (*hipep_mgx_release_fn)(hipep_mgx_program *program);

#ifdef __cplusplus
}
#endif
