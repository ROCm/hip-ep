#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Record a model's greedy decode chain so two builds (or two flag states) can
be compared exactly.

This is the correctness gate for a kernel change that lands behind a
default-off flag. The flags are latched in a function-local static on first
read, so a single process cannot measure both states -- hence dump-then-compare
across two processes rather than an in-process A/B.

Why not test/python's golden-cache suite. Its bar is cosine >= 0.90 against a
cached CPU reference, deliberately loosened because matmul_nbits autotune picks
different (BLOCK_SIZE, TILE_N) configs between runs. That slack is wider than
most kernel bugs. Comparing flag-on against flag-off on ONE build removes the
autotune variance from the comparison instead of budgeting for it, so the bar
can be near-exact: the two arms run the same kernels on the same weights and
differ only in the code under test.

Prompt tokens are fixed and generation is greedy, so a correct change reproduces
the token sequence exactly and the logits to within fp16 reassociation.

    python flag_parity.py --model <dir> --steps 24 --out off.npz
    python flag_parity.py --model <dir> --steps 24 --out on.npz   # other flag state
    python flag_parity.py --compare off.npz on.npz
"""

import argparse
import os
import sys

import numpy as np

# Fixed, boring, and long enough to fill more than one prefill chunk when asked.
_PROMPT = (
    "Explain, step by step and in plain language, how a modern computer "
    "translates a line of source code into instructions a processor can "
    "execute. Cover compilation, linking, and loading. "
)


def record(args) -> None:
    import onnxruntime_genai as og

    config = og.Config(args.model)
    model = og.Model(config)

    text = _PROMPT * max(1, args.prompt_repeat)

    # Text first, vision only if the decoder rejects it. Config sniffing is not
    # reliable here: several text-only exports in the guard set are cut from
    # vision-enabled checkpoints and still carry a "vision" section in
    # genai_config.json while prefilling perfectly well from tokens alone. The
    # models that genuinely need an image say so, with "Invalid rank for input:
    # image_features", and that is the signal worth keying on.
    tokenizer = None
    try:
        tokenizer = og.Tokenizer(model)
        tokens = tokenizer.encode(text)
        prompt_len = len(tokens)
    except Exception:
        tokens, prompt_len = [], 0

    def build_generator(use_vision):
        params = og.GeneratorParams(model)
        # do_sample off is what makes this reproducible; max_length must cover
        # the prompt and the generated tail, plus room for the image tokens a
        # vision prefill splices in.
        params.set_search_options(
            do_sample=False, max_length=prompt_len + args.steps + 2048
        )
        g = og.Generator(model, params)
        if use_vision:
            if not args.image or not os.path.isfile(args.image):
                raise SystemExit(f"{args.model} needs an image; pass --image")
            processor = model.create_multimodal_processor()
            images = og.Images.open(args.image)
            # The placeholder has to be in the prompt or the processor emits no
            # image tokens and the decoder is back to the rank error.
            g.set_inputs(processor("<start_of_image>" + text, images=images))
        else:
            g.append_tokens(tokens)
        return g

    try:
        gen = build_generator(use_vision=False)
    except Exception as e:
        if "image_features" not in str(e):
            raise
        print("text-only prefill rejected; retrying as a vision model")
        gen = build_generator(use_vision=True)

    logits, out_tokens = [], []
    for i in range(args.steps):
        if gen.is_done():
            break
        # A multimodal prefill happens inside set_inputs, and the first
        # generate_next_token is the step that consumes it -- same as the text
        # path, where append_tokens defers prefill to the same call.
        gen.generate_next_token()
        out_tokens.append(int(gen.get_next_tokens()[0]))
        # get_logits returns the last position only at decode. float64 so the
        # comparison is not itself doing fp16 arithmetic.
        logits.append(np.array(gen.get_logits(), copy=True).astype(np.float64).ravel())

    arr = np.stack(logits) if logits else np.zeros((0, 0))
    np.savez_compressed(
        args.out,
        logits=arr,
        tokens=np.array(out_tokens, dtype=np.int64),
        prompt_tokens=np.array(tokens, dtype=np.int64),
    )
    print(f"recorded {len(out_tokens)} steps, logits {arr.shape} -> {args.out}")
    print("tokens: " + " ".join(str(t) for t in out_tokens))


# How much worse than its own run-to-run noise a flag is allowed to be. The
# control measures one sample of a distribution, not its bound, so demanding
# flag <= control would fail half the time on a change that does nothing.
CONTROL_SLACK = 1.5


def _worst_cosine(la, lb):
    """(worst per-step cosine, step index) between two logit stacks."""
    cos = []
    for i in range(min(len(la), len(lb))):
        x, y = la[i], lb[i]
        dn = np.linalg.norm(x) * np.linalg.norm(y)
        cos.append(float(np.dot(x, y) / dn) if dn > 0 else 0.0)
    if not cos:
        return 0.0, -1, cos
    return min(cos), int(np.argmin(cos)), cos


def compare(a_path: str, b_path: str, min_cos: float, control_path: str = None) -> int:
    a, b = np.load(a_path), np.load(b_path)
    ta, tb = a["tokens"], b["tokens"]
    la, lb = a["logits"], b["logits"]

    n = min(len(ta), len(tb))
    if n == 0:
        print("FAIL: one side recorded no steps")
        return 1
    # Report the first divergence rather than just a count: which step it is
    # says whether the change breaks the first decode or drifts into it.
    first_bad = next((i for i in range(n) if ta[i] != tb[i]), None)
    tok_ok = first_bad is None and len(ta) == len(tb)

    worst, worst_at, cos = _worst_cosine(la, lb)

    print(f"steps compared      : {n}")
    print(
        f"token sequence      : {'identical' if tok_ok else f'DIVERGES at step {first_bad}'}"
    )
    print(f"worst logit cosine  : {worst:.8f} (step {worst_at})")
    if cos:
        print(f"mean  logit cosine  : {sum(cos) / len(cos):.8f}")

    budget = 1.0 - min_cos
    ok = tok_ok and worst >= min_cos

    # The control is a second flag-OFF run, recorded between off and on, and it
    # is here because the first run of a model is not like the ones after it.
    # Measured on HIPDNN_EP_QMOE_ROUTER_GEMV across the whole guard set, the
    # control and on dumps came out bitwise equal on all five models while
    # off-vs-on sat at 0.9981 (gemma3) and 0.9998 (gpt-oss) -- so every bit of
    # that divergence belonged to run 1, and none of it to the flag. Two full
    # primes do not remove it.
    #
    # So the verdict is taken on control-vs-on. That is still an off-vs-on
    # comparison -- the control arm has the flag unset -- but with both sides
    # past the warmup, which is the like-for-like test. off-vs-on is kept in the
    # output as the pessimistic number, and the off-vs-control gap is reported
    # so a model that is simply noisy is visibly distinct from one the flag
    # moved.
    if control_path:
        c = np.load(control_path)
        lc, tc = c["logits"], c["tokens"]
        warm, warm_at, _ = _worst_cosine(la, lc)
        prim, prim_at, _ = _worst_cosine(lc, lb)
        tok_ok = tok_ok and len(tc) == len(ta) and all(x == y for x, y in zip(tc, ta))
        print(f"first-run artefact  : {warm:.8f} (off vs control, step {warm_at})")
        print(
            f"control vs on       : {prim:.8f} (step {prim_at})"
            f"{'  [bitwise equal]' if np.array_equal(lc, lb) else ''}"
        )
        ok = tok_ok and prim >= min_cos
        if not ok and tok_ok:
            # Still short: allow it only inside the model's own measured noise.
            budget = max(budget, (1.0 - warm) * CONTROL_SLACK)
            print(
                f"noise budget        : {budget:.8f} divergence "
                f"({CONTROL_SLACK}x first-run artefact)"
            )
            ok = (1.0 - prim) <= budget

    if ok:
        print("PARITY OK")
    else:
        print(f"PARITY FAIL (need tokens identical and divergence <= {budget:.8f})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    ap.add_argument(
        "--control",
        metavar="OFF2.npz",
        help="second flag-OFF dump; when the absolute bar fails, "
        "the verdict falls back to this model's own noise",
    )
    ap.add_argument(
        "--min-cos",
        type=float,
        default=0.9999,
        help="worst per-step logit cosine accepted (default: 0.9999)",
    )
    ap.add_argument("--model")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--prompt-repeat", type=int, default=1)
    ap.add_argument("--image", help="required for vision models (gemma3, etc.)")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.min_cos, args.control)
    if not args.model or not args.out:
        ap.error("--model and --out are required when not using --compare")
    if not os.path.isdir(args.model):
        ap.error(f"no such model directory: {args.model}")
    record(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
