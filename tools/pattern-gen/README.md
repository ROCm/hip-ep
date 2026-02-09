<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
## generate a new pattern


```
$BUILD/morphizen/onnxruntime_morphizen_ep/onnx_pattern_gen env \
 IGNORE_CONSTANT=1 \
 ENABLE_CONSTNAT_SHARING=0 \
 $BUILD/morphizen/onnxruntime_morphizen_ep/onnx_pattern_gen \
 -i value/Add_output_0_QuantizeLinear_Output \
 -i key/MatMul_output_0_QuantizeLinear_Output \
 -i query/Add_output_0_QuantizeLinear_Output \
 -i Mul_output_0_convert_QuantizeLinear_Output \
 -o Reshape_4_output_0_QuantizeLinear_Output \
 -f morphizen/.cache/acd89c9415eba62a3623a3af2e7e8227/onnx.onnx\
 -c ../../morphizen_pattern_zoo/src/QMHAGRPB_0.h.inc
 -m ../../morphizen_pattern_zoo/src/QMHAGRPB_0.mmd
 -j ../../morphizen_pattern_zoo/src/QMHAGRPB_0.json
 ```

 1. `IGNORE_CONSTANT` when it is 0, constant initializers are not
    shown in the generated mermaid diagram. Usually it makes diagrams
    cleaner.
 2. `ENABLE_CONSTNAT_SHARING=0` when it is 0, the generated pattern
    does not try to share a common constant initializer, which make
    the generated pattern potentially match wider range of nodes. If
    `ENABLE_CONSTNAT_SHARING=1`, the generated pattern is stricter to
    match a certain of subgraph which also share these constants. It
    does not match subgraphs which do not share constant initializers.
 3. `-i` specify subgraph input, if there are more than one inputs, we
    need to set multiple `-i` options.
 3. `-o` the subgraph output, only a single output is allowed.
 4. `-f` the sample onnx model as a template to genrate a pattern.
 5. `-c` the source file for generated c++ code.
 5. `-m` the file name for generated mermaid diagram.
 6. `-h` for see sample usage.
 7. `-j` for dump to a json file
