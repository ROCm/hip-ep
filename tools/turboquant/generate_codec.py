#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Generate the MSE-only QR codec on CPU; no binary fixture or model is needed."""

import argparse
import hashlib
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import integrate
import torch

from write_codec import codec_bytes


def generate_constants(seed=42):
    """Match the standalone d=256, 4-bit, shared-QR recipe (200 Lloyd-Max steps)."""
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(256, 256, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    rotation = (q * signs.unsqueeze(0)).numpy()
    sigma = 1.0 / math.sqrt(256)

    def pdf(x):
        return math.exp(-x * x / (2 * sigma**2)) / math.sqrt(2 * math.pi * sigma**2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / 16 for i in range(16)]
    for _ in range(200):
        edges = (
            [lo * 3]
            + [(a + b) / 2 for a, b in zip(centroids, centroids[1:])]
            + [hi * 3]
        )
        updated = []
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            numerator = integrate.quad(lambda x: x * pdf(x), a, b)[0]
            denominator = integrate.quad(pdf, a, b)[0]
            updated.append(
                numerator / denominator if denominator > 1e-15 else centroids[i]
            )
        # Preserve the original stopping convention, including its final iterate.
        if max(abs(a - b) for a, b in zip(updated, centroids)) < 1e-10:
            break
        centroids = updated
    book = np.asarray(centroids, dtype=np.float32)
    return rotation, book, book.copy()


def generate(output_dir, seed=42, expected_sha256=None):
    import io
    import json

    rotation, cb_k, cb_v = generate_constants(seed)
    codec = codec_bytes(rotation, cb_k, cb_v)
    digest = hashlib.sha256(codec).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"Codec SHA-256 mismatch: generated {digest}, expected {expected_sha256}"
        )
    arrays = io.BytesIO()
    np.savez(arrays, rotation=rotation, cb_k=cb_k, cb_v=cb_v)
    files = {"codec.bin": codec, "constants.npz": arrays.getvalue()}
    manifest = {
        "codec": "tq4_mse_qr_v1",
        "recipe": "shared-qr-lloyd-max-v1",
        "seed": seed,
        "head_dim": 256,
        "bits": 4,
        "row_bytes": 132,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        "reproducibility": "QR can vary across platforms/library versions; retain the codec hash with measurements and cached data.",
    }
    output_dir = Path(output_dir)
    # Do not silently replace an existing measured codec or reference arrays.
    for name, data in files.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(
                f"Refusing to replace different constants: {path}; use a new output directory"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-sha256",
        help="Fail before writing if the codec differs from an archived run",
    )
    args = parser.parse_args()
    result = generate(args.output_dir, args.seed, args.expected_sha256)
    print(f"Generated codec SHA-256: {result['files']['codec.bin']}")
