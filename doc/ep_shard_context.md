<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# EP context design

see https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html

## generate EP context

```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
```

If we provide an original model, e.g. `c:\models\resnet50.onnx`, an EP
context model `c:\models\resnet50_ctx.onnx` is generated,
i.e. `[model_name]_ctx.onnx`.

If the model is loaded from memory, i.e. there is no model path from
ORT point of view, end users must set session config
`ep.context_file_path`, otherwise ORT raise an exception

```c++
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Both ep_context_path and model_path are empty");
```
The `ep.context_file_path` support absolute path and relative path.
relative path is relative to current working directory.

```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.ep_context_file_path", "resnet50_ep_ctx.onnx");
```

That is for embed mode, which is the default mode. End users can
enable non-embed mode as below


```c++
  session_options.AddConfigEntry("ep.context_embed_mode", "0");
```

If it is non-embed mode, an EP context binary file would be also
generated along with the EP context model,
i.e. `resnet50_ep_ctx.onnx_VITISAI.bin`, it means
`[ep_context_file_path]_VITISAI.bin`


## shared EP context binary

One or more EP context model can share a same EP context binary
file. It is only applicable for non-embed model.


```c++
  auto session_options = Ort::SessionOptions();
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.context_embed_mode", "0");
  session_options.AddConfigEntry("ep.share_ep_contexts", "1");
```

VitisAI EP would raises an exception if `ep.share_ep_contexts=1`
however `ep.context_embed_mode=1`

In this case, it is suggested that all onnx models are put into a same
directory, or all `ep.ep_context_file_path` points to file names in a
same directory.

When it is enabled, instead of generating
`[ep_context_file_path1]_VITISAI.bin`,
`[ep_context_file_path2]_VITISAI.bin` for each individual EP context
models, a shared binary file is generated in the same directory, `VITISAI.bin`


For example, if we have 3 models as below,

```
A.onnx
B.onnx
C.onnx
```

when `ep.share_ep_contexts=0`, it would generates the following files.

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

when `ep.share_ep_contexts=1`, it would generates the following files.

```
VITISAI.bin
A.onnx
A_ctx.onnx
B.onnx
B_ctx.onnx
C.onnx
C_ctx.onnx
```

And all `{A,B,C}_ctx.onnx` refer to the same `VITISAI.bin`
