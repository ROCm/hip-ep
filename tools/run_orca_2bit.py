#!/usr/bin/env python3
"""
ORCA 2-bit end-to-end inference with MorphiZen (HIP) EP + profiling.

Model layout (split-session):
  orca_2bit_embeddings_w4a32.quant.onnx  -- embedding: input_ids -> hidden
  orca_2bit_1.onnx                        -- transformer body (seq=1, decode)
  orca_2bit_128.onnx                      -- transformer body (seq=128, prefill)
  orca_2bit_lm_head_w4a32.quant.onnx     -- LM head: hidden -> logits

Usage:
  python run_orca_2bit.py --model_dir C:/Users/vakulkar/Downloads/ORCA_2_bit/orca2bit-latest
                          --ep_dll    C:/Users/vakulkar/dev/prebuilt-local/bin/onnxruntime_morphizen_ep.dll
                          [--prompt "Hello!"]
                          [--max_tokens 32]
                          [--prefill_len 128]

Set environment before running (use forward slashes or quoted paths in cmd):
  set PATH=C:/Users/vakulkar/dev/therock_gfx1152_20260512/bin;...
  set THEROCK_DIST=C:/Users/vakulkar/dev/therock_gfx1152_20260512
  set HIPDNN_EP_PERF=1      (optional: enable per-op profiling)
  set HIPDNN_EP_DEBUG=1     (optional: verify GPU dispatch via [REAL] wrap_* lines)
  set HIPDNN_EP_STRICT=1    (optional: abort on compilation failure instead of CPU fallback)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ------------------- EP registration -------------------

def register_morphizen_ep(ep_dll: str):
    import onnxruntime as ort
    ort.register_execution_provider_library("MorphiZenExecutionProvider", ep_dll)
    from onnxruntime.capi._pybind_state import get_ep_devices
    devices = [d for d in get_ep_devices() if d.ep_name == "MorphiZenExecutionProvider"]
    if not devices:
        raise RuntimeError("MorphiZenExecutionProvider not available after registration")
    return devices


def make_session(model_path: str, ep_devices, ep_dll: Optional[str]):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # Disable AOT function inlining so Gelu stays atomic (Whisper lesson applies here too)
    so.add_session_config_entry("session.disable_aot_function_inlining", "1")
    so.log_severity_level = 3  # suppress INFO, keep WARNING+
    if ep_dll is not None:
        so.add_provider_for_devices(ep_devices, {})
        return ort.InferenceSession(model_path, sess_options=so)
    else:
        return ort.InferenceSession(model_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])


# ------------------- model wrappers -------------------

class OrcaSplitSession:
    def __init__(self, model_dir: Path, ep_dll: Optional[str],
                 max_seq_len: int = 1024):
        self.model_dir = model_dir
        self.max_seq_len = max_seq_len

        devices = None
        if ep_dll:
            devices = register_morphizen_ep(ep_dll)
        self.devices = devices

        # Bug #4: the GPU GatherBlockQuantized (embeddings) kernel is broken
        # ("scales tensor underflows the block grid") and returns corrupt hidden
        # states -> garbage output (e.g. "!idthidth..."). The embeddings are just
        # a quantized gather (negligible cost), so force them onto CPU and keep
        # the transformer + lm_head on the GPU EP. A proper GPU gbq fix needs
        # morphizen to represent UINT4 as packed uint8 (follow-up).
        print(f"Loading embedding model (forced CPU: GPU gbq kernel broken)...", flush=True)
        self.emb_sess = make_session(
            str(model_dir / "orca_2bit_embeddings_w4a32.quant.onnx"),
            None, None)

        # Use fixed-shape models:
        #   orca_2bit_1.onnx   -- decode (seq_len=1)
        #   orca_2bit_128.onnx -- prefill (seq_len=128, pad input to 128)
        print(f"Loading decode transformer (seq=1)...", flush=True)
        self.decode_sess = make_session(
            str(model_dir / "orca_2bit_1.onnx"), devices, ep_dll)

        print(f"Loading prefill transformer (seq=128)...", flush=True)
        self.prefill_sess = make_session(
            str(model_dir / "orca_2bit_128.onnx"), devices, ep_dll)

        print(f"Loading LM head model...", flush=True)
        self.lmhead_sess = make_session(
            str(model_dir / "orca_2bit_lm_head_w4a32.quant.onnx"),
            devices, ep_dll)

        # Read config
        import json
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        self.hidden_size    = cfg["hidden_size"]
        self.num_kv_heads   = cfg["num_key_value_heads"]
        self.num_layers     = cfg["num_hidden_layers"]
        self.head_dim       = cfg["head_dim"]

        # KV cache: [1, num_kv_heads, max_seq_len, head_dim], fp16
        self.past_keys   = [np.zeros((1, self.num_kv_heads, max_seq_len, self.head_dim),
                                      dtype=np.float16) for _ in range(self.num_layers)]
        self.past_values = [np.zeros((1, self.num_kv_heads, max_seq_len, self.head_dim),
                                      dtype=np.float16) for _ in range(self.num_layers)]

    def _kv_inputs(self, past_seq_len: int) -> dict:
        inp = {
            "past_seq_len": np.array([[past_seq_len]], dtype=np.int32),
            "total_seq_len": np.array([self.max_seq_len], dtype=np.int32),
        }
        for i in range(self.num_layers):
            inp[f"past_keys_{i}"]   = self.past_keys[i]
            inp[f"past_values_{i}"] = self.past_values[i]
        return inp

    def _update_kv(self, outputs):
        for i in range(self.num_layers):
            self.past_keys[i]   = outputs[f"present_keys_{i}"]
            self.past_values[i] = outputs[f"present_values_{i}"]

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        """Run embedding lookup. input_ids: int64 [1, seq]"""
        out = self.emb_sess.run(None, {"input_ids": input_ids})
        return out[0]  # [1, seq, hidden]

    def prefill(self, hidden: np.ndarray, past_seq_len: int) -> np.ndarray:
        """Run prefill transformer. Returns output_hidden_states."""
        inp = {"input_hidden_states": hidden}
        inp.update(self._kv_inputs(past_seq_len))
        out_names = self.prefill_sess.get_outputs()
        results = self.prefill_sess.run(None, inp)
        out_dict = {o.name: results[i] for i, o in enumerate(out_names)}
        self._update_kv(out_dict)
        return out_dict["output_hidden_states"]

    def decode_step(self, hidden: np.ndarray, past_seq_len: int) -> np.ndarray:
        """Run single-token decode transformer."""
        inp = {"input_hidden_states": hidden}
        inp.update(self._kv_inputs(past_seq_len))
        out_names = self.decode_sess.get_outputs()
        results = self.decode_sess.run(None, inp)
        out_dict = {o.name: results[i] for i, o in enumerate(out_names)}
        self._update_kv(out_dict)
        return out_dict["output_hidden_states"]

    def lm_head(self, hidden: np.ndarray) -> np.ndarray:
        """Run LM head. Returns logits [1, seq, vocab]."""
        out = self.lmhead_sess.run(None, {"output_hidden_states": hidden})
        return out[0]

    def generate(self, token_ids: list[int], max_new_tokens: int = 32,
                 prefill_len: int = 128, verbose: bool = True) -> list[int]:
        """Simple greedy generation loop."""
        generated = []
        prompt_len = len(token_ids)

        # Run prefill in chunks of prefill_len.
        # orca_2bit_128.onnx requires exactly seq_len=128 — pad with zeros.
        past_len = 0
        pos = 0
        chunk = []
        while pos < prompt_len:
            chunk_end = min(pos + prefill_len, prompt_len)
            chunk = token_ids[pos:chunk_end]
            # Pad to prefill_len (model expects exactly prefill_len tokens)
            padded = chunk + [0] * (prefill_len - len(chunk))
            ids = np.array([padded], dtype=np.int64)

            # past_seq_len is the ABSOLUTE position of the LAST token in this
            # (zero-padded) chunk, i.e. first_pos = past_seq_len - seq_len + 1.
            # The chunk's real tokens start at absolute position `pos`, so the
            # last position is pos + prefill_len - 1. Passing `past_len` (= pos)
            # here made RoPE positions run pos-127..pos (negative on chunk 0),
            # corrupting the hidden states -> garbage output. (Matches the
            # reference ort_inference.py: past_seq_len = (k+1)*prefill_len - 1.)
            prefill_pos = past_len + prefill_len - 1
            t0 = time.perf_counter()
            hidden = self.embed(ids)
            hidden = self.prefill(hidden, prefill_pos)
            t1 = time.perf_counter()

            if verbose:
                print(f"  prefill chunk [{pos}:{chunk_end}] "
                      f"(padded to {prefill_len}) in {(t1-t0)*1000:.1f}ms",
                      flush=True)
            past_len += len(chunk)
            pos = chunk_end

        # Use the last REAL token's hidden state (not padded zeros) for logits
        last_real_idx = len(chunk) - 1
        last_hidden = hidden[:, last_real_idx:last_real_idx+1, :]
        logits = self.lm_head(last_hidden)
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)
        if verbose:
            print(f"  prefill -> token {next_token}", flush=True)

        # Decode loop
        decode_times = []
        for step in range(max_new_tokens - 1):
            ids = np.array([[next_token]], dtype=np.int64)
            t0 = time.perf_counter()
            hidden = self.embed(ids)
            hidden = self.decode_step(hidden, past_len)
            logits = self.lm_head(hidden)
            t1 = time.perf_counter()

            next_token = int(np.argmax(logits[0, -1]))
            generated.append(next_token)
            past_len += 1
            decode_times.append(t1 - t0)

            if verbose and step < 5:
                print(f"  decode step {step+1}: token={next_token} "
                      f"({(t1-t0)*1000:.1f}ms)", flush=True)

        if decode_times:
            avg_ms = np.mean(decode_times) * 1000
            tps = 1000 / avg_ms
            print(f"\n=== Decode stats ===")
            print(f"  Tokens generated : {len(generated)}")
            print(f"  Avg decode time  : {avg_ms:.1f} ms/token")
            print(f"  Throughput       : {tps:.1f} tok/s")

        return generated


# ------------------- profiling output -------------------

def print_profiling_note():
    perf = os.environ.get("HIPDNN_EP_PERF", "0")
    debug = os.environ.get("HIPDNN_EP_DEBUG", "0")
    strict = os.environ.get("HIPDNN_EP_STRICT", "0")

    print("\n=== Environment ===")
    print(f"  HIPDNN_EP_PERF={perf}   (per-op GPU profiling; '1' adds ~58% overhead)")
    print(f"  HIPDNN_EP_DEBUG={debug}  (GPU dispatch verification via [REAL] wrap_* logs)")
    print(f"  HIPDNN_EP_STRICT={strict} (abort on compilation failure vs silent CPU fallback)")
    if perf == "1":
        print("\n  [PERF] profiling is ON — per-op breakdown will appear in stderr above.")
        print("         Do NOT cite throughput numbers when HIPDNN_EP_PERF=1 is set!")


# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",  required=True,
                    help="Path to orca2bit-latest directory")
    ap.add_argument("--ep_dll",     default=None,
                    help="Path to onnxruntime_morphizen_ep.dll (omit for CPU)")
    ap.add_argument("--prompt",     default="Hello, how are you?")
    ap.add_argument("--max_tokens", type=int, default=20)
    ap.add_argument("--prefill_len",type=int, default=128,
                    help="Prefill chunk size (must match model: 128 or 1)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print_profiling_note()

    # Tokenize. This is an instruct model: it MUST see the chat template's
    # special tokens (<|im_start|>...<|im_sep|>...<|im_end|>) plus the assistant
    # generation prompt. Feeding the raw string with tok.encode() puts the model
    # off-distribution and yields degenerate repetition (e.g. "!idthidth..."),
    # which looked like an EP accuracy bug but is a driver bug. Matches the
    # authoritative reference (ort_inference.py).
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        messages = [{"role": "user", "content": args.prompt}]
        chat_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        token_ids = tok.encode(chat_text, add_special_tokens=False)
        print(f"\nPrompt ({len(token_ids)} templated tokens): {args.prompt!r}")
    except Exception as e:
        print(f"WARNING: tokenizer unavailable ({e}), using dummy token IDs")
        token_ids = list(range(1, min(len(args.prompt) + 1, 10)))

    print(f"Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")

    ep_dll = str(Path(args.ep_dll).resolve()) if args.ep_dll else None
    if ep_dll:
        print(f"\nEP DLL: {ep_dll}")
        print("EP: MorphiZen (HIP GPU)")
    else:
        print("\nEP: CPU (no EP DLL specified)")

    print(f"\nLoading ORCA 2-bit split-session model from {model_dir}")
    sess = OrcaSplitSession(model_dir, ep_dll, max_seq_len=1024)

    print(f"\n=== Running inference ===")
    print(f"  prompt_len={len(token_ids)}, max_new_tokens={args.max_tokens}, "
          f"prefill_chunk={args.prefill_len}\n")

    t_start = time.perf_counter()
    new_tokens = sess.generate(token_ids, max_new_tokens=args.max_tokens,
                               prefill_len=args.prefill_len, verbose=True)
    t_end = time.perf_counter()

    print(f"\n=== Total wall time: {(t_end - t_start)*1000:.0f} ms ===")

    # Decode output
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        output_text = tok.decode(new_tokens, skip_special_tokens=True)
        print(f"\nGenerated text: {output_text!r}")
    except Exception:
        print(f"\nGenerated token IDs: {new_tokens}")

    print_profiling_note()


if __name__ == "__main__":
    main()
