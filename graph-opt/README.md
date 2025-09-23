<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->


## morphizen-graph-opt
`morphizen-graph-opt` is a useful tool for developing VAIP passes to optimize graph

### Usage :
```
 morphizen-graph-opt.exe -i <input_onnx_model> -o <output_onnx_model> -t <output_txt_file> -p <vaip-pass_1> [vaip-pass_2 ...] [-h]
```
`-i` <input_onnx_model> : input onnx model file

`-o` <output_onnx_model> :  output onnx model file

`-t` <output_txt_file> :  output model to txt file

`-p` <vaip-pass_1> [vaip-pass_2 ...] : pass list

`-h` help


### Sample with vaip-pass_init
```
morphizen-graph-opt.exe -i pt_reset50.onnx -o pt_reset50_opt.onnx -p vaip-pass_init

```
