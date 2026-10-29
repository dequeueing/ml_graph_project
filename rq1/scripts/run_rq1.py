"""Run RQ1 sequentially for the configured models with resume support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rq1" / "scripts"))

import query_llms  # noqa: E402
from run_one_case import build_prompt, extract_usage, normalize_answer  # noqa: E402


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        case_id = record.get("case_id")
        # A request error is retryable on a later invocation.  Only a record
        # with a successful API response is considered complete.
        if case_id and record.get("status") == "ok":
            completed.add(case_id)
    return completed


def run_profile(profile: str, cases: list[dict], output: Path, max_tokens: int, timeout: float, retries: int) -> None:
    completed = load_completed(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = [case for case in cases if case["case_id"] not in completed]
    print(f"{profile}: {len(completed)} completed, {len(pending)} pending", flush=True)

    with output.open("a", encoding="utf-8") as handle:
        for position, case in enumerate(pending, start=1):
            prompt = build_prompt(case)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            started = time.perf_counter()
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": case["case_id"],
                "profile": profile,
                "request_max_tokens": max_tokens,
                "case": case,
                "prompt_sha256": prompt_hash,
            }
            for attempt in range(retries + 1):
                try:
                    response = query_llms.query(
                        profile, prompt, max_tokens=max_tokens, timeout=timeout
                    )
                    parsed = normalize_answer(case["task"], response["text"])
                    record.update(
                        {
                            "model": response["model"],
                            "raw_answer": response["text"],
                            "usage": extract_usage(response.get("response")),
                            "parsed_answer": parsed,
                            "ground_truth": case["answer"],
                            "valid": parsed is not None,
                            "correct": parsed == case["answer"],
                            "status": "ok",
                            "attempts": attempt + 1,
                        }
                    )
                    record.pop("error", None)
                    break
                except Exception as exc:  # Keep a failed case auditable and continue.
                    record.update(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempts": attempt + 1,
                        }
                    )
                    if attempt < retries:
                        time.sleep(min(2 ** attempt, 30))
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{profile}: {position}/{len(pending)} {case['case_id']} "
                f"status={record['status']} correct={record.get('correct')}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rq1/config/rq1_full.json")
    parser.add_argument("--profile", choices=("claude-glm", "claude-ds"))
    parser.add_argument("--limit", type=int, help="run only the first N cases per profile")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = ROOT / config["output"]
    cases = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        cases = cases[: args.limit]
    profiles = [args.profile] if args.profile else config["profiles"]
    for profile in profiles:
        output = ROOT / "rq1" / "results" / f"rq1_full_{profile}.jsonl"
        run_profile(
            profile,
            cases,
            output,
            max_tokens=int(config["max_tokens"]),
            timeout=float(config["timeout_seconds"]),
            retries=args.retries,
        )


if __name__ == "__main__":
    main()
