#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Subprocess worker: load one ONNX model, run ORT inference, print result."""

import sys
import numpy as np
import onnxruntime as ort


def make_dummy_input(inp):
    shape = []
    for dim in inp.shape:
        if isinstance(dim, str) or dim is None or dim == "" or dim == 0:
            shape.append(1)
        else:
            try:
                v = int(dim)
                shape.append(v if v > 0 else 1)
            except (ValueError, TypeError):
                shape.append(1)

    name_lower = inp.name.lower()
    dtype_str = inp.type

    if "attention_mask" in name_lower:
        return np.ones(shape, dtype=np.int64 if "int64" in dtype_str else np.int32)

    if "float16" in dtype_str:
        return np.zeros(shape, dtype=np.float16)
    elif "float" in dtype_str:
        return np.zeros(shape, dtype=np.float32)
    elif "int64" in dtype_str:
        return np.zeros(shape, dtype=np.int64)
    elif "int32" in dtype_str:
        return np.zeros(shape, dtype=np.int32)
    elif "int8" in dtype_str:
        return np.zeros(shape, dtype=np.int8)
    elif "uint8" in dtype_str:
        return np.zeros(shape, dtype=np.uint8)
    elif "bool" in dtype_str:
        return np.zeros(shape, dtype=np.bool_)
    else:
        return np.zeros(shape, dtype=np.float32)


def main():
    fpath = sys.argv[1]
    verbose = "--verbose" in sys.argv

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(fpath, sess_options=opts)
    except Exception as e:
        print(f"Session creation failed: {e}")
        sys.exit(1)

    feeds = {}
    for inp in sess.get_inputs():
        feeds[inp.name] = make_dummy_input(inp)
        if verbose:
            print(
                f"  input: {inp.name:50s} "
                f"shape={feeds[inp.name].shape} "
                f"dtype={feeds[inp.name].dtype}",
                file=sys.stderr,
            )

    try:
        outputs = sess.run(None, feeds)
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

    if verbose:
        for i, out_meta in enumerate(sess.get_outputs()):
            arr = outputs[i]
            has_nan = (
                bool(np.isnan(arr).any())
                if np.issubdtype(arr.dtype, np.floating)
                else False
            )
            print(
                f"  output: {out_meta.name:50s} "
                f"shape={arr.shape} dtype={arr.dtype} nan={has_nan}",
                file=sys.stderr,
            )

    print(f"{len(outputs)} outputs")


if __name__ == "__main__":
    main()
