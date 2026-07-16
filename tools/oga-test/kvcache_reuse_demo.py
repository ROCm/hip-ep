#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

r"""
Minimal OGA KV-cache reuse demo.

This is a standalone multi-turn OGA append demo:
1. Build each user turn with the model's chat template.
2. Append only the newly required token suffix for that turn.
3. Generate an assistant answer before moving to the next turn.

run environment: use python env

run command example:
 python kvcache_reuse_demo.py -m model_path\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml  --testAll   > dump.txt
this command run the full six group of prompts;
or
 python kvcache_reuse_demo.py -m model_path\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml > dump.txt
this command run 1 group of prompts;

for vlm model, you can provide picture by parameter of -i
python kvcache_reuse_demo.py -m model_path\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml -i some_pic.jpg --testAll   > dump.txt

"""

import argparse
import ast
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


prompts = [
    # 1
    [
        " My name is Alex, I work as a data scientist at Orbital AI.",
        " I live in Berlin and enjoy playing the piano in my free time.",
        " Where do I live and what do I enjoy doing?",
        " What is my profession?",
    ],
    # 2
    [
        ' When I say "compact summary," respond with 3 bullet points only.',
        ' Give me a compact summary of: Ancient Egypt (Egyptian: km.t) was a cradle of civilization concentrated along the lower reaches of the Nile River in Northeast Africa. It emerged from prehistoric Egypt around 3150 BC (according to conventional Egyptian chronology), when Upper and Lower Egypt were amalgamated by Menes, who is believed by the majority of Egyptologists to have been the same person as Narmer. The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by the "Intermediate Periods" of relative instability. These stable kingdoms existed in one of three periods: the Old Kingdom of the Early Bronze Age; the Middle Kingdom of the Middle Bronze Age; or the New Kingdom of the Late Bronze Age.',
        " Give me a compact summary of: stages of butterfly",
    ],
    # 3
    [
        " Sarah is a software engineer who leads the AI team. James is her manager.",
        " Who is Sarah's manager?",
        " What team does Sarah lead?",
    ],
    # 4
    [
        " Here is a list of items I need from the store: apples, bread, milk, eggs, tomatoes, pasta, and cheese.",
        " What was the fourth item?",
        " How many dairy items are in the list?",
    ],
    # 5
    [
        " Alice is a biologist who works at Genomix Lab. She reports to Dr. Patel, who is the head of the lab. Her colleague, Martin, specializes in data analysis.",
        " Who is Alice's manager?",
        " Who is the data analyst at Genomix Lab?",
        " What is Dr. Patel's position?",
        " What field does Alice work in?",
    ],
    # 6
    [
        " A farmer has 3 fields. Each field has 240 apple trees.",
        " If each tree produces 120 apples, how many apples does one field produce?",
        " How many apples in total across all fields?",
        " If 10% of the apples are spoiled, how many are still good?",
    ],
]

default_questions = [" My name is Alice. I live in Berlin.", " Where do I live?"]


def build_prompt_chat_template(
    model_path: str,
    tokenizer: og.Tokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> str:
    """Build a text prompt using the model directory's chat template.

    This mirrors vlm_benchmark.py, but emits text-only messages.
    """
    model_dir = Path(model_path)
    tok_cfg_path = model_dir / "tokenizer_config.json"
    jinja_path = model_dir / "chat_template.jinja"

    template_str = None
    bos = None

    if tok_cfg_path.exists():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        template_str = tok_cfg.get("chat_template")
        bos = tok_cfg.get("bos_token")

    if not template_str and jinja_path.exists():
        with open(jinja_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        # print(f"  Loaded chat template from: {jinja_path}")

    if not template_str:
        return "".join(str(message.get("content", "")) for message in messages)

    if not bos:
        template_str = template_str.replace("{{ bos_token }}", "")

    message_json = json.dumps(messages)
    return tokenizer.apply_chat_template(
        message_json,
        template_str=template_str,
        add_generation_prompt=add_generation_prompt,
    )


def encode_chat_prompt(model_path: str, tokenizer: og.Tokenizer, messages: list[dict]):
    prompt = build_prompt_chat_template(
        model_path, tokenizer, messages, add_generation_prompt=True
    )
    return [int(t) for t in tokenizer.encode(prompt)], prompt


def to_numpy(tensor):
    if hasattr(tensor, "as_numpy"):
        return tensor.as_numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.array(tensor)


def encode_prompt_inputs(
    model_path: str,
    tokenizer: og.Tokenizer,
    processor,
    images,
    messages: list[dict],
):
    prompt = build_prompt_chat_template(
        model_path, tokenizer, messages, add_generation_prompt=True
    )
    text_tokens = [int(t) for t in tokenizer.encode(prompt)]
    if processor is None:
        token_debug = {
            "text_tokens": len(text_tokens),
            "prompt_tokens": len(text_tokens),
            "image_tokens_est": 0,
        }
        return text_tokens, prompt, None, token_debug

    inputs = processor(prompt, images=images)
    input_ids = to_numpy(inputs["input_ids"]).reshape(-1)
    prompt_tokens = [int(t) for t in input_ids]
    token_debug = {
        "text_tokens": len(text_tokens),
        "prompt_tokens": len(prompt_tokens),
        "image_tokens_est": max(0, len(prompt_tokens) - len(text_tokens)),
    }
    return prompt_tokens, prompt, inputs, token_debug


def make_user_message(question: str, image_path: str | None = None) -> dict:
    if not image_path:
        return {"role": "user", "content": question}

    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }


def make_generator(model: og.Model, max_length: int, repetition_penalty: float):
    params = og.GeneratorParams(model)
    params.set_search_options(
        # max_length=max_length,
        # min_length=0,
        # do_sample=False,
        # repetition_penalty=repetition_penalty,
        max_length=max_length,
        min_length=0,
        do_sample=False,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    return og.Generator(model, params)


def generate_up_to(generator, tokenizer, limit: int, label: str, ttft_start: float):
    generated = []
    pieces = []
    stream = tokenizer.create_stream()
    ttft_ms = None

    print(f"  {label}=", end="", flush=True)
    for step in range(limit):
        generator.generate_next_token()
        if step == 0:
            ttft_ms = (time.perf_counter() - ttft_start) * 1000
        if generator.is_done():
            # print("", flush=True)
            # print(
            #     f"  {label}_debug stopped_by=eos step={step} "
            #     f"generated={len(generated)}",
            #     flush=True,
            # )
            return generated, pieces, True, ttft_ms

        token = int(generator.get_next_tokens()[0])
        generated.append(token)
        piece = stream.decode(token)
        pieces.append(piece)
        print(piece, end="", flush=True)

    # print("", flush=True)
    # print(
    #     f"  {label}_debug stopped_by=limit limit={limit} "
    #     f"generated={len(generated)} is_done={generator.is_done()}",
    #     flush=True,
    # )
    return generated, pieces, False, ttft_ms


def run_questions(
    args: argparse.Namespace,
    model: og.Model,
    tokenizer: og.Tokenizer,
    processor,
    images,
    questions: list[str],
    group_label: str,
) -> list[float]:
    generator = make_generator(model, args.max_length, args.repetition_penalty)
    messages: list[dict] = []
    committed_tokens: list[int] = []
    answers = []
    ttft_values: list[float] = []
    close_previous_assistant = False

    try:
        print("\n" + "=" * 60)
        # print(f"OGA KV-cache reuse demo: {group_label}")
        # print(
        #     f"  max_length={args.max_length} answer_limit={args.answer_limit} "
        #     f"turns={len(questions)}"
        # )

        for i, question in enumerate(questions, start=1):
            image_path = args.image_path if i == 1 else None
            messages.clear()
            messages.append(make_user_message(question, image_path))
            prompt_tokens, prompt, inputs, token_debug = encode_prompt_inputs(
                args.model_path,
                tokenizer,
                processor if image_path else None,
                images if image_path else None,
                messages,
            )
            append_tokens = prompt_tokens
            if close_previous_assistant:
                close_tokens = [int(t) for t in tokenizer.encode("<|im_end|>\n")]
                append_tokens = close_tokens + append_tokens
                token_debug = {
                    **token_debug,
                    "text_tokens": token_debug["text_tokens"] + len(close_tokens),
                    "prompt_tokens": token_debug["prompt_tokens"] + len(close_tokens),
                }
            if not append_tokens:
                raise RuntimeError("No new input tokens to append for this turn.")
            context_after_append = len(committed_tokens) + len(append_tokens)
            if context_after_append + args.answer_limit >= args.max_length:
                raise RuntimeError(
                    "Not enough max_length budget for this turn: "
                    f"context={context_after_append} limit={args.answer_limit} "
                    f"max_length={args.max_length}"
                )

            print(f"\n\n  Q{i}={question!r}")
            print(f"  templated_Q{i}={prompt!r}")
            print(
                f"  turn={i} committed_tokens={len(committed_tokens)} "
                f"append_tokens={len(append_tokens)} "
                f"context_after_append={context_after_append}"
            )
            print(
                f"  token_lengths: text_only={token_debug['text_tokens']} "
                f"processor_input_ids={token_debug['prompt_tokens']} "
                f"image_est={token_debug['image_tokens_est']}"
            )
            print(f"  decoded_processor_prompt={tokenizer.decode(append_tokens)!r}")

            ttft_start = time.perf_counter()
            if inputs is not None and not committed_tokens:
                generator.set_inputs(inputs)
            else:
                generator.append_tokens(np.array(append_tokens, dtype=np.int32))
            committed_tokens.extend(append_tokens)

            generated, pieces, eos, ttft_ms = generate_up_to(
                generator, tokenizer, args.answer_limit, f"A{i}", ttft_start
            )
            ttft_values.append(ttft_ms)
            if eos:
                effective_generated = generated
                effective_pieces = pieces
            else:
                # OGA keeps the last fixed-count token in the sequence before it
                # has been consumed into KV cache. Drop it before appending Q{i+1}.
                effective_generated = generated[:-1]
                effective_pieces = pieces[:-1]
                rewind_len = len(committed_tokens) + len(effective_generated)
                generator.rewind_to(rewind_len)

            text = "".join(effective_pieces)
            committed_tokens.extend(effective_generated)
            answers.append((effective_generated, text, eos))
            close_previous_assistant = not eos

        print("\n" + "=" * 60)
        return ttft_values
    finally:
        del generator


def run_demo(args: argparse.Namespace) -> int:
    # print(f"Loading model: {args.model_path}")
    if args.image_path and not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting execution provider to {args.execution_provider}...")
            config.append_provider(args.execution_provider)

    model = og.Model(config)
    print(f"Model loaded: {model.type}")
    print(f"Device: {model.device_type}")
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor() if args.image_path else None
    images = og.Images.open(args.image_path) if args.image_path else None
    ttft_matrix: list[list[float]] = []

    try:
        if args.testAll:
            print(f"Running all prompt groups: {len(prompts)}")
            for i, questions in enumerate(prompts, start=1):
                print("\nrun ", i, "test === begin \n")
                ttft_matrix.append(
                    run_questions(
                        args,
                        model,
                        tokenizer,
                        processor,
                        images,
                        questions,
                        f"prompt group {i}",
                    )
                )
                print("\nrun ", i, "test === end \n")
        else:
            ttft_matrix.append(
                run_questions(
                    args,
                    model,
                    tokenizer,
                    processor,
                    images,
                    args.questions,
                    "custom questions",
                )
            )
        print("ttft_matrix_ms=[")
        for row in ttft_matrix:
            print("  " + json.dumps(row) + ",")
        print("]")
        return 0
    finally:
        del model


def extract_bracketed_list(text: str, start: int) -> str:
    list_start = text.find("[", start)
    if list_start < 0:
        raise ValueError("ttft_matrix_ms marker found, but '[' was not found")

    depth = 0
    for idx in range(list_start, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[list_start : idx + 1]

    raise ValueError("ttft_matrix_ms list is not closed")


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(bytes([0xFF, 0xFE])) or data.startswith(bytes([0xFE, 0xFF])):
        return data.decode("utf-16")

    # Windows PowerShell redirection may create UTF-16LE without a visible BOM.
    if len(data) >= 4 and data[1::2].count(0) > len(data) // 4:
        return data.decode("utf-16-le")
    return data.decode("utf-8", errors="replace")


def check_kvcache_reuse_output(path: Path) -> tuple[bool, list[str]]:
    text = read_text_auto(path)
    reasons: list[str] = []

    q2_pos = text.find("Q2")
    if q2_pos < 0:
        reasons.append("can't find Q2")

    berlin_pos = text.find(" Berlin", q2_pos + len("Q2")) if q2_pos >= 0 else -1
    if q2_pos >= 0 and berlin_pos < 0:
        reasons.append("can't find Berlin")

    ttft_pos = text.find("ttft_matrix_ms", berlin_pos if berlin_pos >= 0 else 0)
    if ttft_pos < 0:
        reasons.append("can't find ttft_matrix_ms")
    else:
        matrix_text = extract_bracketed_list(text, ttft_pos)
        matrix = ast.literal_eval(matrix_text)
        first_row = matrix[0]
        if len(first_row) < 2:
            reasons.append("can't find ttft1/ttft2")
        elif first_row[1] >= first_row[0]:
            reasons.append("ttft2 > ttft1")

    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal OGA KV-cache reuse demo")
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Path to the OGA model directory",
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        default="follow_config",
        choices=["cpu", "dml", "follow_config"],
        help="Execution provider (default: follow_config)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=default_questions,
        help="User questions, one item per conversation turn",
    )
    parser.add_argument(
        "--testAll",
        action="store_true",
        help="Run every question group in the built-in prompts list",
    )
    parser.add_argument(
        "--answer_limit",
        type=int,
        default=2000,
        help="Maximum assistant tokens to generate per turn",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20480,
        help="OGA max_length search option",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="OGA repetition_penalty search option",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default=None,
        help="Optional image path to attach to the first prompt",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        default=Path("dump.txt"),
        help="Path to write demo output (default: dump.txt)",
    )
    args = parser.parse_args()

    if args.output_file.parent != Path(""):
        args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as out_file:
        with contextlib.redirect_stdout(out_file):
            exit_code = run_demo(args)

    if not args.testAll:
        try:
            passed, reasons = check_kvcache_reuse_output(args.output_file)
        except Exception as exc:
            passed, reasons = False, [f"check failed: {exc}"]
        if passed:
            print("OK")
        else:
            print("Fail: " + "; ".join(reasons))
            return 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
