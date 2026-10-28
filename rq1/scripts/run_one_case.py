"""Run and score exactly one RQ1 case against one local LLM profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import query_llms  # noqa: E402


def build_prompt(case: dict) -> str:
    task = case["task"]
    nodes = case["query_nodes"]
    if task == "edge_existence":
        question = f"Is there an edge between node {nodes[0]} and node {nodes[1]}?"
        answer_format = "Answer with exactly one word: Yes or No."
    elif task == "node_degree":
        question = f"What is the degree of node {nodes[0]}?"
        answer_format = "Answer with exactly one integer and no explanation."
    elif task == "cycle_detection":
        question = "Does this graph contain at least one cycle?"
        answer_format = "Answer with exactly one word: Yes or No."
    else:
        raise ValueError(f"unsupported smoke-test task: {task}")

    return (
        "You are solving a basic graph reasoning question. The graph is undirected. "
        "Read the complete graph description and answer the question. "
        f"{answer_format}\n\n"
        "Graph description:\n"
        f"{case['graph_text']}\n\n"
        f"Question: {question}\nAnswer:"
    )


def normalize_answer(task: str, text: str):
    if task in {"edge_existence", "cycle_detection"}:
        match = re.search(r"\b(yes|no)\b", text.lower())
        return None if match is None else match.group(1) == "yes"
    match = re.search(r"(?<![\w-])-?\d+(?![\w-])", text)
    return None if match is None else int(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="rq1/manifests/smoke_manifest.jsonl",
        help="JSONL manifest relative to the project root",
    )
    parser.add_argument("--index", type=int, default=0, help="zero-based case index")
    parser.add_argument("--profile", default="codex", choices=sorted(query_llms.PROFILES))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", help="optional result path relative to the project root")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    cases = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if not 0 <= args.index < len(cases):
        raise IndexError(f"case index {args.index} is outside 0..{len(cases) - 1}")

    case = cases[args.index]
    prompt = build_prompt(case)
    response = query_llms.query(
        args.profile,
        prompt,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    parsed = normalize_answer(case["task"], response["text"])
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "case_index": args.index,
        "profile": response["profile"],
        "model": response["model"],
        "case": case,
        "prompt": prompt,
        "raw_answer": response["text"],
        "parsed_answer": parsed,
        "ground_truth": case["answer"],
        "valid": parsed is not None,
        "correct": parsed == case["answer"],
    }

    output = (
        ROOT / args.output
        if args.output
        else ROOT / "rq1" / "results" / f"one_case_{args.profile}_index_{args.index}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"profile={response['profile']} model={response['model']}")
    print(f"task={case['task']} ground_truth={case['answer']} parsed={parsed} correct={result['correct']}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
