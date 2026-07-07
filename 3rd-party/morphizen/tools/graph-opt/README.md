<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->


## morphizen-graph-opt
`morphizen-graph-opt` is a useful tool for developing morphizen passes to optimize graph

### Usage :
```
 morphizen-graph-opt.exe -i <input_onnx_model> -o <output_onnx_model> -t <output_txt_file> -p <morphizen-pass_1> [morphizen-pass_2 ...] [-h]
```
`-i` <input_onnx_model> : input onnx model file

`-o` <output_onnx_model> :  output onnx model file

`-t` <output_txt_file> :  output model to txt file

`-p` <morphizen-pass_1> [morphizen-pass_2 ...] : pass list

`-h` help


### Sample with morphizen-pass_init
```
morphizen-graph-opt.exe -i pt_reset50.onnx -o pt_reset50_opt.onnx -p morphizen-pass_init

```
