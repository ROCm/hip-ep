<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# EP Context Documentation

This document covers both the user-facing configuration and deployment of EP (Execution Provider) context models, as well as the internal design and implementation details.

For more details, refer to the official documentation:
https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html

---

## User Guide

### Generate EP Context

```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
```

If the model is loaded from a file path(e.g. `c:\models\resnet50.onnx`), an EP
context model `c:\models\resnet50_ctx.onnx` will be generated in the same directory,
The naming format is `[model_name]_ctx.onnx`.

If the model is loaded from memory (i.e., no model path is visible to ORT),
The end users must provide a path via the session config
`ep.context_file_path`, otherwise ORT raise an exception

```c++
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Both ep_context_path and model_path are empty");
```
The `ep.context_file_path` support absolute and relative paths.
Relative path is resolved based on the `current working directory`.

Example:
```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.context_file_path", "resnet50_ep_ctx.onnx");
```

By default, context is generated in embed mode. To enable non-embed mode, add the following:

```c++
  session_options.AddConfigEntry("ep.context_embed_mode", "0");
```

If it is non-embed mode, an EP context binary file would be also generated in same as the EP context model directory,
The EP context binary file naming format is  `[EP context model name]_MORPHIZEN.bin`.

i.e. `resnet50_ep_ctx.onnx_MORPHIZEN.bin` or `[ep.context_file_path]_MORPHIZEN.bin` if ep.context_file_path set.

#### Example: EP Context Generation

Assume the `current working directory` contains:
```
A.onnx
```

##### 1. Embed mode (default)
The EP context model will generated in the same directory as the input ONNX model.
```
A.onnx
A_ctx.onnx
```
##### 2. Non-embed mode
The EP context model and EP context binary file will generated in the same directory as the input ONNX model.
```
A.onnx
A_ctx.onnx
A_ctx.onnx_MORPHIZEN.bin
```
##### 3. Non-embed mode with specify `ep.context_file_path`
**With a relative path:**

When `ep.context_file_path` is set to a relative path , the EP context model will be generated at the
specifed path relative to the current working directory. And the EP context binary file will generated
in the same directory as EP Context model directory.
```C++
session_options.AddConfigEntry("ep.context_file_path", "./test/A_rel_ctx.onnx");
```
Resulting files (in current working directory):
```
A.onnx
./test/A_rel_ctx.onnx
./test/A_rel_ctx.onnx_MORPHIZEN.bin
```
**With an absolute path:**

When `ep.context_file_path` is set to a absolute path , the EP context model will be generated at the
specifed full path. And the EP context binary file will generated in the same directory.
```C++
session_options.AddConfigEntry("ep.context_file_path", "C:\temp\A_abs_ctx.onnx");
```
Resulting files:
```
A.onnx
C:\temp\A_abs_ctx.onnx
C:\temp\A_abs_ctx.onnx_MORPHIZEN.bin
```

##### 4. Model loaded from memory
When the model is loaded from memory, you must specify `ep.context_file_path`.
```C++
session_options.AddConfigEntry("ep.context_file_path", "C:\temp\A_ctx.onnx");
```
Resulting files:
```
C:\temp\A_ctx.onnx
C:\temp\A_ctx.onnx_MORPHIZEN.bin
```

### Deploy EP Context Model

```c++
  auto session_options = Ort::SessionOptions();
```
When deploying an EP context model, ensure the necessary files are prepared based on the mode:
##### Embed mode:
Only the context model file is required:
`A_ctx.onnx`

##### Non-embed mode:
Both the context model and its associated binary are required, and must be located in the same directory.
The EP context binary file name is taked from the `ep_cache_context` attribute of the main `EPContext` node in the EP Context ONNX model.

`A_ctx.onnx`
`A_ctx.onnx_MORPHIZEN.bin`

##### Non-embed mode with load ctx model from memory:

When the EP context model (in non-embed mode) is loaded from memory,
if `ep.context_file_path` is set, the binary file will be loaded from the directory specified by this path.
otherwise,  the binary file will be loaded from the current working directory.
The binary file name is specified in the `ep_cache_context` attribute of the `EPContext` node within the EP Context ONNX model.


### Shared EP Context Binary

One or more EP context model can share a same EP context binary
file. It is only applicable for non-embed model.

Enable shared context with:
```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.context_embed_mode", "0");
  session_options.AddConfigEntry("ep.share_ep_contexts", "1");
```

MorphiZen EP would raises an exception if `ep.share_ep_contexts=1`
however `ep.context_embed_mode=1`


Models that share weights are grouped into a shared context workspace, where each directory is treated as a separate workspace.
The final ONNXRuntime session should set `ep.stop_share_ep_contexts` to indicate that it is the last session within the workspace.
It is recommended to place ONNX models from the same group in the same directory, or ensure that all `ep.context_file_path` values point to files within the same directory.

According [Implementation Guidelines for EPContext Model Generation with Weight Sharing][guide]


#### Example: Shared EP Context
Given models:
```
A.onnx
B.onnx
C.onnx
D.onnx
```

```
ep.share_ep_contexts|1
generate context for A.onnx
generate context for B.onnx
ep.stop_share_ep_contexts

ep.share_ep_contexts|1
generate context for C.onnx
generate context for D.onnx
ep.stop_share_ep_contexts

```
In the above case we when context cache is generated for `A.onnx`, the bin will be `A_ctx.onnx_MORPHIZEN.bin`
For the second call `A_ctx.onnx_MORPHIZEN.bin` will be updated

Then `C_ctx.onnx_MORPHIZEN.bin` will be generated for `C.onnx` and then when `D.onnx` is called, `C_ctx.onnx_MORPHIZEN.bin` will be updated

As results, the following files are generated.

```
A_ctx.onnx
A_ctx.onnx_MORPHIZEN.bin
B_ctx.onnx

C_ctx.onnx
C_ctx.onnx_MORPHIZEN.bin
D_ctx.onnx
```

`A_ctx.onnx` and `B_ctx.onnx` share `A_ctx.onnx_MORPHIZEN.bin`.
`C_ctx.onnx` and `D_ctx.onnx` share `C_ctx.onnx_MORPHIZEN.bin`.


#### Example : Some compilicated use cases

Given models
```
dir1/A1.onnx
dir1/A2.onnx
dir1/B1.onnx
dir1/B2.onnx

dir2/B3.onnx
```
Models `dir1/A1.onnx` and `dir1/A2.onnx` belong to the same group and share common weights through the shared context mechanism.

Models `dir1/B1.onnx`, `dir1/B2.onnx` and `dir2/B3.onnx` belong to the same group and share common weights through the shared context mechanism.

Python sample codes:
``` python
options = onnxruntime.SessionOptions()
options.add_session_config_entry("ep.context_enable", "1")
options.add_session_config_entry("ep.context_embed_mode", "0")
options.add_session_config_entry("ep.share_ep_contexts", "1")

# open workspace `dir1`, shared binray file : dir1/A1_ctx.onnx_MORPHIZEN.bin
create_ort_session("dir1/A1.onnx", options)
# close workspace `dir1`, dir1/A1_ctx.onnx and dir2/A2_ctx.onnx as a group for shared weights.
options.add_session_config_entry("ep.stop_share_ep_contexts", "1")
create_ort_session("dir1/A2.onnx", options)

# open workspace `dir1` , shared library file: dir1/B1_ctx.onnx_MORPHIZEN.bin
options.add_session_config_entry("ep.stop_share_ep_contexts", "0")
create_ort_session("dir1/B1.onnx", options)
create_ort_session("dir1/B2.onnx", options)
# close workspace `dir1`, dir1/B1.onnx dir1/B2.onnx and dir1/B3.onnx as a group for shared weights
options.add_session_config_entry("ep.stop_share_ep_contexts", "1")
# Point the EP context onnx model of `dir2/B3.onnx` to the workspace directory `dir1/`.
options.add_session_config_entry("ep.context_file_path", "dir1/B3_ctx.onnx")
create_ort_session("dir2/B3.onnx", options)
```

As results, the following files are generated.
```
dir1/A1_ctx.onnx
dir1/A1_ctx.onnx_MORPHIZEN.bin
dir1/A2_ctx.onnx
dir1/B1_ctx.onnx
dir1/B1_ctx.onnx_MORPHIZEN.bin
dir1/B2_ctx.onnx
dir1/B3_ctx.onnx
```
`dir1/A1_ctx.onnx` and `dir1/A2_ctx.onnx` share `dir1/A1_ctx.onnx_MORPHIZEN.bin`.
`dir1/B1_ctx.onnx`, `dir1/B2_ctx.onnx`  and `dir1/B3_ctx.onnx` share `dir1/B1_ctx.onnx_MORPHIZEN.bin`.


Others:
```
if you want to share, all models must in the same directory or point output ep context model to workspace directory.
dir1/A1.onnx -> dir1/A1_ctx.onnx dir1/A1_ctx.onnx_MORPHIZEN.bin
dir2/A1.onnx -> dir2/A1_ctx.onnx dir2/A1_ctx.onnx_MORPHIZEN.bin
dir1/A2.onnx -> dir1/A2_ctx.onnx dir1/A1_ctx.onnx_MORPHIZEN.bin  : shared =1 stop=1
dir2/A2.onnx -> dir2/A2_ctx.onnx dir2/A1_ctx.onnx_MORPHIZEN.bin  : shared =1 stop=1

dir1/B1.onnx -> dir1/B1_ctx.onnx dir1/B1_ctx.onnx_MORPHIZEN.bin
dir1/B2.onnx -> dir1/B2_ctx.onnx dir1/B1_ctx.onnx_VITSIAI.bin
dir1/B3.onnx -> dir1/B3_ctx.onnx dir1/B1_ctx.onnx_VITSIAI.bin : shared=1 stop=1

dir1/A1.onnx -> dir1/A1_ctx.onnx dir1/A1_ctx.onnx_VITISIA.bin
dir2/A2.onnx -> dir1/A2_ctx.onnx dir1/A1_ctx.onnx_VITISIA.bin : ep.config_file_path = dir1/A2_ctx.onnx
dir3/A3.onnx -> dir1/A3_ctx.onnx dir1/A1_ctx.onnx_VITISIA.bin : ep.config_file_path = dir2/A3_ctx.onnx  shared=1 stop=1
```

[guide]:https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#implementation-guidelines-for-epcontext-model-generation-with-weight-sharing_

---

## Internal Design

The EP context is becoming more and more complex because we keep adding new features:

1. Encryption
2. Compression
3. mmap [PR #136][136]
4. Shared EP context [PR #44][44]
5. Prebuilt EP context [PR #238][238]

### Generate EP Context Model

#### Non-embed Mode

Using `tar_file_` because of the [mmap feature][136]. `tar_file_` is created upon the EP context binary file; we don't need any extra operations.

#### Embed Mode

Using `tar_file_` because of the [mmap feature][136]. `tar_file_` is created upon `std::tmpfile`.

`tar_file_->dump_to` is used to filling the EP context node attribute.

##### morphizen_compile_model, the producer

There are two producers:

   1. `tar_file_`, unified storage mechanism for all scenarios.
   2. `prebuilt_ep_context=1`, the PR.


##### create_ep_context, the consumer

  1. Uses `tar_file_` for all scenarios. Falls back to `mem_files_` only when tmpfile() fails.

### Deploy EP Context Model

#### Non-embed Mode

  1. Uses `tar_file_` because of the [mmap feature][136].

#### Embed Mode

  1. Uses `tar_file_` with cache_key prefix for file isolation.

TODO:
  1. Add `TarFile::to_memory()` return.
  2. `MemStream` does not support write.

### Prebuilt EP Context

#### Non-embed Mode

  1. Uses `tar_file_` because of the [mmap feature][136].

#### Embed Mode

  1. Uses `tar_file_`. DO NOT USE IT.

### Deploy EP Context Model

#### Non-embed Mode

  1. Uses `tar_file_` because of the [mmap feature][136].

#### Embed Mode

  1. Uses `tar_file_` for all cache storage.

#### "ep.share_ep_contexts" = 1

All cache files are stored with cache_key prefix for proper isolation when `tar_file_` is in use. This ensures multiple EP contexts can safely share the same tar archive.

### Compression or Encryption is Enabled

Uses `tar_file_` for all storage scenarios. Encryption and compression are handled at serialization boundaries (when saving/loading EP context), not within tar_file_ itself.

When encryption is enabled:
- During save: tar stream is encrypted using `vaip_encryption::aes_encryption`
- During load: encrypted stream is decrypted using `vaip_encryption::aes_decryption`, then loaded into tar_file_

This approach keeps tar_file_ simple while supporting encryption transparently.

[136]: #136 (mmap)
[44]: #44 (shared ep context)
[238]: #238 (prebuilt ep context)
