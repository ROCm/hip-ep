"""
Verify extracted ONNX models: check file sizes, weights.data offsets,
graph structure (nodes, inputs, outputs), and initializer integrity.

Usage:
  python verify_extraction.py <output_dir>
  python verify_extraction.py D:\liuc\0-modelzoo\llm\Qwen2.5-14B-Instruct-fp16\Qwen2.5-14B-Instruct\space
"""

import os
import sys
import argparse

import onnx


def verify_single_op(d):
    """Verify single_op directory structure."""
    if not os.path.exists(d):
        print("  [NOT FOUND]")
        return True

    folders = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
    print(f"  {len(folders)} folders:")
    for f in folders:
        cnt = len([x for x in os.listdir(os.path.join(d, f)) if x.endswith(".onnx")])
        print(f"    {f:50s} ({cnt} onnx files)")

    all_ok = True
    for folder in folders:
        folder_path = os.path.join(d, folder)
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".onnx"):
                continue
            fpath = os.path.join(folder_path, f)
            try:
                m = onnx.load(fpath, load_external_data=False)
                n_nodes = len(m.graph.node)
                n_inits = len(m.graph.initializer)
                if n_nodes == 0:
                    print(f"    WARNING: {folder}/{f} has 0 nodes!")
                    all_ok = False
            except Exception as e:
                print(f"    ERROR loading {folder}/{f}: {e}")
                all_ok = False

    return all_ok


def verify_external_data_dir(d, label):
    """Verify a directory with external weights (single_layer or full_model)."""
    if not os.path.exists(d):
        print("  [NOT FOUND]")
        return True

    weights_path = os.path.join(d, "weights.data")
    if not os.path.exists(weights_path):
        print("  [No weights.data file]")
        onnx_files = sorted([f for f in os.listdir(d) if f.endswith(".onnx")])
        for f in onnx_files:
            sz = os.path.getsize(os.path.join(d, f))
            print(f"    {f:45s} {sz / (1024*1024):.1f} MB")
        return True

    weights_size = os.path.getsize(weights_path)
    if weights_size > 1024 ** 3:
        print(f"  weights.data: {weights_size / (1024**3):.2f} GB")
    else:
        print(f"  weights.data: {weights_size / (1024**2):.1f} MB")

    all_ok = True
    onnx_files = sorted([f for f in os.listdir(d) if f.endswith(".onnx")])

    for f in onnx_files:
        fpath = os.path.join(d, f)
        try:
            m = onnx.load(fpath, load_external_data=False)
        except Exception as e:
            print(f"    {f:45s} ERROR loading: {e}")
            all_ok = False
            continue

        n_nodes = len(m.graph.node)
        n_inputs = len(m.graph.input)
        n_outputs = len(m.graph.output)

        ext_count = 0
        emb_count = 0
        max_end = 0
        bad_count = 0

        for init in m.graph.initializer:
            ext = {e.key: e.value for e in init.external_data}
            if ext:
                ext_count += 1
                offset = int(ext.get("offset", 0))
                length = int(ext.get("length", 0))
                end = offset + length
                if end > max_end:
                    max_end = end
                if end > weights_size:
                    bad_count += 1
            else:
                emb_count += 1

        if bad_count > 0:
            status = f"ERROR: {bad_count} tensors out of bounds!"
            all_ok = False
        else:
            status = "OK"

        print(
            f"    {f:40s} nodes={n_nodes:>4d}  in={n_inputs:>3d}  "
            f"out={n_outputs:>3d}  ext={ext_count}  emb={emb_count}  "
            f"max_end={max_end:>15d}  {status}"
        )

    if all_ok:
        print(f"  >> ALL OFFSETS VALID (max_end = weights.data size: {max_end == weights_size})")
    else:
        print("  >> ERRORS FOUND!")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted ONNX models (single_op, single_layer, full_model)")
    parser.add_argument(
        "output_dir", type=str,
        help="Base output directory containing single_op/, single_layer/, full_model/",
    )
    args = parser.parse_args()

    base = args.output_dir
    if not os.path.isdir(base):
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    overall_ok = True

    print("=" * 70)
    print("single_op")
    print("=" * 70)
    ok = verify_single_op(os.path.join(base, "single_op"))
    if not ok:
        overall_ok = False
    print()

    print("=" * 70)
    print("single_layer")
    print("=" * 70)
    ok = verify_external_data_dir(os.path.join(base, "single_layer"), "single_layer")
    if not ok:
        overall_ok = False
    print()

    print("=" * 70)
    print("full_model")
    print("=" * 70)
    ok = verify_external_data_dir(os.path.join(base, "full_model"), "full_model")
    if not ok:
        overall_ok = False
    print()

    print("=" * 70)
    if overall_ok:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 70)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
