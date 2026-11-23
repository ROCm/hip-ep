#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Versioned synthetic conversations with auditable answers and long distractors.

This is a small characterization fixture, not a standardized quality benchmark.
All project names, records and people below are fictional.
"""

VERSION = "project-ledger-v1"
TURNS = [
    (
        "Project Lark has owner Mira, codeword amber-17, and a budget of 120 credits. "
        "Project Heron has owner Sol, codeword violet-42, and budget 90 credits.",
        "What is Lark's codeword? Reply with only the codeword.",
        "amber-17",
        ["amber-17"],
    ),
    (
        "The current Lark budget is revised from 120 to 135 credits. All other facts remain unchanged.",
        "Who owns Heron? Reply with only the name.",
        "Sol",
        ["sol"],
    ),
    (
        "Heron transfers 20 credits to Lark. Apply this transfer after all previous updates.",
        "What is Lark's current budget? Reply with only the number.",
        "155",
        ["155"],
    ),
    (
        "Lark's owner changes from Mira to Neri. Its codeword stays the same.",
        "Give Lark's original owner and current owner, in that order, separated by a comma.",
        "Mira, Neri",
        ["mira", "neri"],
    ),
    (
        "A rumor says Heron's codeword is green-88. This rumor is explicitly false; do not update the ledger.",
        "What is Heron's actual codeword? Reply with only the codeword.",
        "violet-42",
        ["violet-42"],
    ),
    (
        "No record has ever specified a launch date for Lark or Heron.",
        "On what date will Lark launch? If it is unknown, reply UNKNOWN.",
        "UNKNOWN",
        ["unknown"],
    ),
    (
        "Heron's owner remains Sol. Lark's original codeword has never changed.",
        "Reply as JSON with keys owner and codeword for Lark's current owner and codeword.",
        '{"owner":"Neri","codeword":"amber-17"}',
        ["neri", "amber-17"],
    ),
    (
        "Final audit: no additional budget transfers or revisions have occurred.",
        "Reply as JSON with keys lark_budget, heron_budget, total_budget, and lark_owner, using the current ledger.",
        '{"lark_budget":155,"heron_budget":70,"total_budget":225,"lark_owner":"Neri"}',
        ["155", "70", "225", "neri"],
    ),
]


def make_fixture(tokenizer, turns=8, prompt_tokens=4096):
    if not 1 <= turns <= len(TURNS):
        raise ValueError("Fixture supports one through eight turns")
    result = []
    for turn, (facts, question, expected, required) in enumerate(TURNS[:turns]):
        prefix = (
            "<|im_start|>system\nTrack the project ledger across turns. "
            "New explicit updates override earlier values. Archive records concern "
            "unrelated projects. Answer the final question concisely.\n<|im_end|>\n"
            if turn == 0
            else "\n"
        )
        prefix += "<|im_start|>user\nLedger update: " + facts + "\nUnrelated archive:\n"
        # Follow the supplied Qwen template's enable_thinking=False branch;
        # a short answer budget must not be consumed by hidden reasoning text.
        suffix = (
            "\nEnd archive.\n"
            + question
            + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        # Padding is meaningful fictional records; only its final record may
        # be truncated. Never truncate the update, question or chat delimiters.
        fillers = "".join(
            f"Archive {turn + 1}-{i}: unrelated project Reed-{i} completed a routine "
            f"inventory inspection of {10 + i % 13} boxes; no ledger update.\n"
            for i in range(prompt_tokens)
        )
        lead, tail = list(tokenizer.encode(prefix)), list(tokenizer.encode(suffix))
        need = prompt_tokens - len(lead) - len(tail)
        if need < 1:
            raise ValueError("Prompt budget too small to preserve the task")
        padding = list(tokenizer.encode(fillers))[:need]
        ids = [int(token) for token in lead + padding + tail]
        if len(ids) != prompt_tokens:
            raise ValueError("Insufficient fixture padding")
        result.append(
            {
                "turn": turn + 1,
                "tokens": ids,
                "text": tokenizer.decode(ids),
                "facts": facts,
                "question": question,
                "expected": expected,
                "required": required,
            }
        )
    return {"version": VERSION, "prompt_tokens": prompt_tokens, "turns": result}


def score_answer(answer, turn):
    # Diagnostic exact-content score; an LLM judge must assess contradictions,
    # unsupported extra assertions, JSON compliance and temporal ordering.
    found = [item.lower() in answer.lower() for item in turn["required"]]
    return {
        "required_items_found": sum(found),
        "required_items": len(found),
        "all_required_items_found": all(found),
    }
