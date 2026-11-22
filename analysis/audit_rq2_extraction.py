"""Read-only server audit of explicit final 'Topological sort:' answers.

Run with the experiment environment, passing this file through SSH stdin.
Only the missing 'sort' answer label is added. Original scores and databases
are not modified. Every changed answer is validated against its graph.
"""

import importlib.util
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path("/mnt/new_ssd_3/taojie/comp5222-graphdo/half-scale")
spec = importlib.util.spec_from_file_location("saved_scorer", ROOT / "repair-glm-4096/score_answers.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
source = json.loads((ROOT / "scratch/glm-final-report-20261122.json").read_text())
cases = [json.loads(line) for line in (ROOT / "benchmark/traditional/manifest.jsonl").open()]
old_cue = scorer.PATH_CUE
new_cue = re.compile(r"\b(?:path|order|ordering|sorting|sort|answer)(?:\s+is)?\s*:\s*", re.I)
result = {"source": "scratch/glm-final-report-20261122.json",
          "manifest_sha256": source["manifest_sha256"],
          "change": "Add sort to the final-line path-label regex, with graph validation unchanged",
          "models": {}}

for model, data in source["models"].items():
    scores = dict(data["scores"])
    original = scorer.read_records(ROOT / "runs/traditional" / f"{model}.sqlite3", source["manifest_sha256"])
    repairs = scorer.read_records(ROOT / "repair-glm-4096/runs/traditional" / f"{model}.sqlite3", source["manifest_sha256"])
    changes = []
    counts = Counter()
    for case in cases:
        if case["task"] != "topology":
            continue
        key = case["case_id"]
        response = repairs.get(key, original[key])
        text = response["text"]
        scorer.PATH_CUE = old_cue
        checked_old = scorer.adjudicate(case, text)
        old = bool(response["correct"]) if checked_old is None else checked_old
        assert int(old) == scores[key]
        scorer.PATH_CUE = new_cue
        checked_new = scorer.adjudicate(case, text)
        new = bool(response["correct"]) if checked_new is None else checked_new
        counts["responses"] += 1
        counts["old_correct"] += old
        counts["new_correct"] += new
        if old != new:
            counts["gained" if new else "lost"] += 1
            counts[case["prompt_style"]] += 1
            changes.append({"case_id": key, "old": int(old), "new": int(new),
                            "last_line": scorer.normalized(text).splitlines()[-1],
                            "old_extracted": checked_old is not None,
                            "new_extracted": checked_new is not None})
    result["models"][model] = {"counts": dict(counts), "changes": changes}

print(json.dumps(result, indent=2))
