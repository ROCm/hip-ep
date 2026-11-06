<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OGA Export Directions for a Windowed-State Model

These directions reproduce the original windowed OGA package used by the
Qwen3.6-27B DFlash experiment:

```text
Qwen3.6-27B_gpuep_1.0_gpu_128_state_window8_hipep
```

This is not a new Hugging Face-to-ONNX export. It starts from the existing
non-windowed OGA/HIP EP package, copies that package without changing it, and
adds the windowed-state contract to its `text.onnx` decoder. The model weights,
tokenizer, vision graph, and generation settings remain the same.

The transformation:

- adds `state_window=8` to every `com.microsoft::CausalConvWithState` and
  `com.microsoft::LinearAttention` node;
- prepends a window dimension of `8` to those operators' recurrent and
  convolution state inputs and outputs; and
- synchronizes the recurrent-state names in `genai_config.json` with the names
  already present in the ONNX graph.

A window size of 8 supports speculative blocks through size 7. The window must
be larger than the largest DFlash verification block.

## What Adding Windowed State Means

The original hybrid decoder already passes convolution and recurrent state
into and out of every `CausalConvWithState` and `LinearAttention` operator.
Without windowing, each operator exposes only the latest committed state:

```text
state input -> process tokens -> one final state output
```

With `state_window=8`, each state tensor gains a leading eight-slot dimension:

```text
[original state shape] -> [8, original state shape]
```

During a multi-token verification call, HIP EP retains the intermediate state
associated with each token in that rolling window. If DFlash accepts only part
of the proposed block, `crop_to_accepted()` promotes the slot corresponding to
the last accepted token into the committed slot. The accepted prefix therefore
does not need to be executed a second time.

No learned weights, quantization parameters, attention layers, or token
embeddings change. The transformation changes only:

1. the `state_window` attribute on the existing hybrid-state operators;
2. the declared shapes of their state inputs and outputs; and
3. the OGA names used to bind those state tensors.

This works without rerunning the full exporter because ONNX describes both the
operator attributes and tensor interfaces directly. It is safe for this model
because its existing decoder already contains the exact window-capable custom
operators, and the matching HIP EP runtime implements their windowed
semantics. A model lacking those operators, or lacking matching runtime
support, would require a compatible export or implementation instead.

## 1. Prerequisites

Run from the root of the HIP EP repository in the Conda environment created by
the DFlash README. That environment must provide `onnx`.

The source must be a complete, working Qwen3.6-27B OGA package whose
`text.onnx` contains the Microsoft-domain `CausalConvWithState` and
`LinearAttention` operators.

Set the source and destination paths. Change only `MODEL_ROOT` when the models
are stored elsewhere:

```bash
export MODEL_ROOT=/absolute/path/to/oga-models
export SOURCE_MODEL="$MODEL_ROOT/Qwen3.6-27B_gpuep_1.0_gpu_128"
export WINDOWED_MODEL="$MODEL_ROOT/Qwen3.6-27B_gpuep_1.0_gpu_128_state_window8_hipep"

test -f "$SOURCE_MODEL/text.onnx"
test -f "$SOURCE_MODEL/genai_config.json"
test ! -e "$WINDOWED_MODEL"
```

The final check deliberately fails if the destination already exists. Do not
overwrite an existing model package.

## 2. Copy the OGA Package

Copy the complete package so all tokenizer, configuration, vision, and model
assets remain available:

```bash
cp -a "$SOURCE_MODEL" "$WINDOWED_MODEL"
```

The following transformation reads `text.onnx` from the source and writes a
separate file in the copied package. It never writes to the source model:

```bash
python tools/dflash-hipep/add_state_window_to_oga_model.py \
  "$SOURCE_MODEL/text.onnx" \
  "$WINDOWED_MODEL/text.windowed.onnx" \
  --state-window 8

mv "$WINDOWED_MODEL/text.windowed.onnx" "$WINDOWED_MODEL/text.onnx"
```

For this Qwen3.6-27B graph, the command must report:

```text
Updated 96 hybrid-state operators
```

## 3. Synchronize the OGA State Names

The original package can contain older recurrent-state aliases in
`genai_config.json`, while its graph exposes `past_key_values.*.conv_state`,
`past_key_values.*.recurrent_state`, and matching `present.*` outputs. OGA
binds tensors using this configuration, so update the copied package:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["WINDOWED_MODEL"]) / "genai_config.json"
config = json.loads(path.read_text(encoding="utf-8"))
decoder = config["model"]["decoder"]
decoder["inputs"]["past_conv_names"] = "past_key_values.%d.conv_state"
decoder["inputs"]["past_recurrent_names"] = "past_key_values.%d.recurrent_state"
decoder["outputs"]["present_conv_names"] = "present.%d.conv_state"
decoder["outputs"]["present_recurrent_names"] = "present.%d.recurrent_state"
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
```

This changes only the copied package's `genai_config.json`.

## 4. Verify the Exported Contract

Verify the operator count, `state_window` values, state tensor shapes, and OGA
configuration before loading the model:

```bash
python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper

root = Path(os.environ["WINDOWED_MODEL"])
model = onnx.load(root / "text.onnx", load_external_data=False)
windowed = []
for node in model.graph.node:
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    if "state_window" in attributes:
        windowed.append((node.op_type, attributes["state_window"]))

counts = Counter(op_type for op_type, _ in windowed)
assert counts == {"CausalConvWithState": 48, "LinearAttention": 48}, counts
assert all(window == 8 for _, window in windowed)

state_values = [
    value
    for value in (*model.graph.input, *model.graph.output)
    if value.name.endswith((".conv_state", ".recurrent_state"))
]
assert len(state_values) == 192, len(state_values)
assert all(value.type.tensor_type.shape.dim[0].dim_value == 8 for value in state_values)

config = json.loads((root / "genai_config.json").read_text(encoding="utf-8"))
decoder = config["model"]["decoder"]
assert decoder["inputs"]["past_conv_names"] == "past_key_values.%d.conv_state"
assert decoder["inputs"]["past_recurrent_names"] == "past_key_values.%d.recurrent_state"
assert decoder["outputs"]["present_conv_names"] == "present.%d.conv_state"
assert decoder["outputs"]["present_recurrent_names"] == "present.%d.recurrent_state"

print("PASS: 96 windowed operators and 192 windowed state interfaces")
PY
```

## 5. Runtime Validation

Exporting the graph does not add runtime support. Execute the model only with a
HIP EP build that implements the state-window operator contract and an OGA
build exposing `Generator.crop_to_accepted()`, as built in the DFlash README.

With that environment active, verify actual slot promotion across all 96
recurrent and convolution state tensors:

```bash
python tools/dflash-hipep/tests/probe_oga_window_crop_slots.py \
  --model "$WINDOWED_MODEL" \
  --hipep-root "$HIPEP_ROOT"
```

The probe must report exact promotion for 96/96 destination tensors, no changes
to the other 672 slots, and zero mismatches.

After this source package passes, derive the DFlash hidden-tap target from it
using Section 2 of `tools/dflash-hipep/README.md`.
