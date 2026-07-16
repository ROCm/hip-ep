# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

r"""
Minimal OGA KV-cache reuse demo.

This is a standalone multi-turn OGA append demo:
1. Build each user turn with the model's chat template.
2. Append only the newly required token suffix for that turn.
3. Generate an assistant answer before moving to the next turn.

run environment: use same environment in https://amdcloud.sharepoint.com/sites/AIG/ACAS/AIS/Shared%20Documents/Forms/AllItems.aspx?sortField=Modified&isAscending=false&viewid=a7467c25%2D045c%2D4cfa%2Db7d6%2D6087479176b2&FolderCTID=0x01200030E43EC46C7307458A426ABB9063F706&OR=Teams%2DHL&CT=1758686810028&TeamsCID=985c526a%2D11af%2D4578%2Db989%2D6ca2917610a4&ovuser=3dd8961f%2De488%2D4e60%2D8e11%2Da82d994e183d%2Cshili9%40amd%2Ecom&id=%2Fsites%2FAIG%2FACAS%2FAIS%2FShared%20Documents%2FROCm%20EP%2Fdemo%2Fhipep%20EP%20Python%20Package%20Download%20%20Install%20%20Run%2Epdf&parent=%2Fsites%2FAIG%2FACAS%2FAIS%2FShared%20Documents%2FROCm%20EP%2Fdemo

run command example:  
(.venv) PS C:\a-1\hipep-wheels-release_0_2_0>  python C:\a-1\hipep-wheels-release_0_2_0\examples\kvcache_reuse_demo.py -m    C:\a-1\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml  --testAll   > c:\a-1\dump.txt
this command run the full six group of prompts;
or
(.venv) PS C:\a-1\hipep-wheels-release_0_2_0>  python C:\a-1\hipep-wheels-release_0_2_0\examples\kvcache_reuse_demo.py -m    C:\a-1\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml > c:\a-1\dump.txt
this command run 1 group of prompts;

for vlm model, you can provide picture by parameter of -i

"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


prompts = [
 
]

default_questions = [
  " My name is Alice. I live in Berlin.", 
  " Where do I live?"
]

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
    print("shili inputs ", inputs)
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
    args = parser.parse_args()
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
