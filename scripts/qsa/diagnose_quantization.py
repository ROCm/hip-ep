#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Separate export/runtime numerical errors from calibration-free RTN loss."""
# isort: skip_file

import argparse
import gc
import json
import math
import time

import numpy as np
import torch

from evaluate_original import nll
from original_inference import Original, reference_model, stats
from prepare_baseline import REFERENCE, original_folder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["hip", "rtn-reference", "fp16-reference"], required=True
    )
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    tokens = json.loads(
        (REFERENCE / "data/selector-extended-v1/test.json").read_text()
    )[0]["input_ids"]
    ids = torch.tensor([tokens], device="cuda", dtype=torch.int64)
    started = time.perf_counter()
    original = original_folder(args.bits)
    if args.variant == "hip":
        model = Original(original, 4608)
        logits = model.forward(ids, all_logits=True)
    else:
        model = reference_model(original, quantized=args.variant == "rtn-reference")
        logits = model(ids, use_cache=False).logits
    loss = nll(logits[:, :-1], ids[:, 1:])
    folder = original / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    samples = logits[:, ::64].clone()
    np.save(folder / (args.variant + ".npy"), samples.float().cpu().numpy())
    receipt = {
        "variant": args.variant,
        "document": 0,
        "targets": 4095,
        "nll_sum": loss,
        "ppl": math.exp(loss / 4095),
        "training_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    reference = folder / "hip.npy"
    if args.variant != "hip" and reference.exists():
        receipt["sampled_logits_vs_hip"] = stats(
            samples, torch.from_numpy(np.load(reference)).cuda()
        )
    (folder / (args.variant + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    del model, logits
    gc.collect()


if __name__ == "__main__":
    main()
