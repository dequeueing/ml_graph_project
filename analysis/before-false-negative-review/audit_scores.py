"""Re-score all saved traditional answers without additional model calls.

Run: python3 analysis/audit_scores.py
Verifies that the legacy extraction reproduces every saved final score before
applying the new answer extraction. The benchmark and response texts stay fixed.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from score_answers import adjudicate, normalized


HERE = Path(__file__).resolve().parent


def main():
    input_path = HERE / "scoring_audit_inputs.json.gz"
    data = json.loads(gzip.decompress(input_path.read_bytes()))
    cases = data["cases"]
    assert len(cases) == 17500 and len({c["case_id"] for c in cases}) == 17500
    result = {
        "manifest_sha256": data["manifest_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "analysis/scoring_audit_inputs.json.gz",
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "previous_source": data["source"], "previous_source_sha256": data["source_sha256"],
        "scorer_sha256": hashlib.sha256((HERE / "score_answers.py").read_bytes()).hexdigest(),
        "models": {},
    }
    audit = {"changes": [], "models": {}, "fallback_examples": []}
    for model, model_data in data["models"].items():
        previous = dict(model_data["previous_scores"])
        responses = model_data["responses"]
        assert len(previous) == len(responses) == 17500
        cells = defaultdict(Counter)
        totals = Counter()
        fallback_counts = Counter()
        scores = []
        for case in cases:
            key = case["case_id"]
            record = responses[key]
            legacy, _, legacy_answer = adjudicate(case, record["text"], legacy=True)
            legacy_score = record["release_correct"] if legacy is None else int(legacy)
            assert legacy_score == previous[key], (model, key, legacy_score, previous[key])
            checked, kind, answer = adjudicate(case, record["text"])
            score = record["release_correct"] if checked is None else int(checked)
            counts = dict(n=1, previous=previous[key], score=score,
                          verified=checked is not None, legacy_verified=legacy is not None,
                          gained=score > previous[key], lost=score < previous[key],
                          fallback=checked is None, blank=not record["text"].strip(),
                          repaired=record["repaired"])
            cells[case["task"], case["order"], case["prompt_style"]].update(counts)
            totals.update(counts)
            scores.append((key, score))
            if score != previous[key]:
                audit["changes"].append({
                    "model": model, "case_id": key, "previous": previous[key], "score": score,
                    "extraction": kind, "legacy_verified": legacy is not None,
                    "answer": answer, "legacy_answer": legacy_answer,
                    "final_lines": record["text"].splitlines()[-3:],
                })
            if checked is None:
                group = (case["task"], case["prompt_style"])
                fallback_counts[group] += 1
                if fallback_counts[group] <= 3:
                    audit["fallback_examples"].append({
                        "model": model, "case_id": key, "score": score,
                        "final_lines": record["text"].splitlines()[-5:],
                    })
        assert len(cells) == 125 and all(c["n"] == 140 for c in cells.values())
        result["models"][model] = {
            "repaired": totals["repaired"], "scores": scores,
            "cells": [dict(task=t, order=o, prompt=p, **c)
                      for (t, o, p), c in sorted(cells.items())],
        }
        audit["models"][model] = {"counts": dict(totals),
                                    "fallback_groups": {"/".join(k): v for k, v in fallback_counts.items()}}
        print(model, dict(totals), flush=True)
    (HERE / "audited_final_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "scoring_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Reproduced all 52500 previous scores before re-scoring.")
    print("Changed scores:", len(audit["changes"]))


if __name__ == "__main__":
    main()
