<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# EP context design

For more details, refer to the official documentation:
https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html

## Generate EP context

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
The EP context binary file naming format is  `[EP context model name]_VITISAI.bin`.

i.e. `resnet50_ep_ctx.onnx_VITISAI.bin` or `[ep.context_file_path]_VITISAI.bin` if ep.context_file_path set.

### Example: EP Context Generation

Assume the `current working directory` contains:
```
A.onnx
```

#### 1. Embed mode (default)
The EP context model will generated in the same directory as the input ONNX model.
```
A.onnx
A_ctx.onnx
```
#### 2. Non-embed mode
The EP context model and EP context binary file will generated in the same directory as the input ONNX model.
```
A.onnx
A_ctx.onnx
A_ctx.onnx_VITISAI.bin
```
#### 3. Non-embed mode with specify `ep.context_file_path`
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
./test/A_rel_ctx.onnx_VITISAI.bin
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
C:\temp\A_abs_ctx.onnx_VITISAI.bin
```

#### 4. Model loaded from memory
When the model is loaded from memory, you must specify `ep.context_file_path`.
```C++
session_options.AddConfigEntry("ep.context_file_path", "C:\temp\A_ctx.onnx");
```
Resulting files:
```
C:\temp\A_ctx.onnx
C:\temp\A_ctx.onnx_VITISAI.bin
```

## Deploy EP context model

```c++
  auto session_options = Ort::SessionOptions();
```
When deploying an EP context model, ensure the necessary files are prepared based on the mode:
#### Embed mode:
Only the context model file is required:
`A_ctx.onnx`

#### Non-embed mode:
Both the context model and its associated binary are required, and must be located in the same directory.
The EP context binary file name is taked from the `ep_cache_context` attribute of the main `EPContext` node in the EP Context ONNX model.

`A_ctx.onnx`
`A_ctx.onnx_VITISAI.bin`

#### Non-embed mode with load ctx model from memory:

When the EP context model (in non-embed mode) is loaded from memory,
if `ep.context_file_path` is set, the binary file will be loaded from the directory specified by this path.
otherwise,  the binary file will be loaded from the current working directory.
The binary file name is specified in the `ep_cache_context` attribute of the `EPContext` node within the EP Context ONNX model.


## Shared EP context binary

One or more EP context model can share a same EP context binary
file. It is only applicable for non-embed model.

Enable shared context with:
```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.context_embed_mode", "0");
  session_options.AddConfigEntry("ep.share_ep_contexts", "1");
```

VitisAI EP would raises an exception if `ep.share_ep_contexts=1`
however `ep.context_embed_mode=1`

In this case, it is suggested that all onnx models are put into a same
directory, or all `ep.context_file_path` points to file names in a
same directory.

When it is enabled, instead of generating
`[ep_context_file_path1]_VITISAI.bin`,
`[ep_context_file_path2]_VITISAI.bin` for each individual EP context
models, a shared binary file is generated in the same directory, `VITISAI.bin`


### Example: Shared EP Context
Given models:
```
A.onnx
B.onnx
C.onnx
```

#### Without shared context(`ep.share_ep_contexts=0`)
Resulting files:
```
A.onnx
A_ctx.onnx
A_ctx.onnx_VITISAI.bin
B.onnx
B_ctx.onnx
B_ctx.onnx_VITISAI.bin
C.onnx
C_ctx.onnx
C_ctx.onnx_VITISAI.bin
```

#### With shared context(`ep.share_ep_contexts=1`)
Resulting files:
```
A.onnx
A_ctx.onnx
B.onnx
B_ctx.onnx
C.onnx
C_ctx.onnx
VITISAI.bin
```
Each of the *_ctx.onnx (`{A,B,C}_ctx.onnx`) models references the same `VITISAI.bin`.
