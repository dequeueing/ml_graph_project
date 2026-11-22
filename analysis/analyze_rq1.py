"""Recompute all 300 RQ1 pairs from audited, saved traditional scores.

Run locally: python3 analysis/analyze_rq1.py
Node-classification labels are independently extracted for the queried target.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import gzip
import hashlib

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "audited_final_scores.json"
ORDERS = ["bfs", "dfs", "pagerank", "ppr"]
snapshot = json.loads(SOURCE.read_text())
cases = json.loads(gzip.decompress((HERE / "scoring_audit_inputs.json.gz").read_bytes()))["cases"]
comparisons = []
result = {"source": str(SOURCE), "manifest_sha256": snapshot["manifest_sha256"],
          "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "traditional": {}, "node": {}}


def outcomes(rows):
    return dict(Counter("higher" if r["gain"] > r["loss"] else
                        "equal" if r["gain"] == r["loss"] else "lower"
                        for r in rows))


for model, data in snapshot["models"].items():
    scores = dict(data["scores"])
    grouped = defaultdict(lambda: defaultdict(dict))
    for case in cases:
        if case["case_id"] not in scores: continue
        grouped[case["task"], case["prompt_style"]][case["order"]][
            case["graph_id"], case["question_id"], case["repeat"]
        ] = scores[case["case_id"]]
    rows = []
    for (task, prompt), orders in grouped.items():
        base = orders["random"]
        for order in ORDERS:
            assert orders[order].keys() == base.keys()
            gain = sum(orders[order][k] > base[k] for k in base)
            loss = sum(orders[order][k] < base[k] for k in base)
            n = gain + loss
            p = min(1, 2 * sum(math.comb(n, i) for i in range(min(gain, loss) + 1))
                    / 2 ** n) if n else 1.0
            rows.append(dict(model=model, task=task, prompt=prompt, order=order,
                             n=len(base), baseline_correct=sum(base.values()),
                             correct=sum(orders[order].values()), gain=gain, loss=loss,
                             delta_pp=100 * (gain - loss) / len(base), p=p))
    comparisons.extend(rows)
    tasks = sorted({r["task"] for r in rows})
    result["traditional"][model] = {
        "higher_equal_lower": outcomes(rows),
        "outcomes_by_task": {t: outcomes([r for r in rows if r["task"] == t]) for t in tasks},
        "mean_delta_pp": {o: sum(r["delta_pp"] for r in rows if r["order"] == o) / 25
                          for o in ORDERS},
        "nonbinary_mean_delta_pp": {
            o: sum(r["delta_pp"] for r in rows if r["order"] == o
                   and r["task"] not in ["connectivity", "cycle"]) / 15 for o in ORDERS},
        "ppr_shortest_path_by_prompt": [r for r in rows if r["task"] == "shortest_path"
                                         and r["order"] == "ppr"],
        "paired_changes": {"gain": sum(r["gain"] for r in rows),
                           "loss": sum(r["loss"] for r in rows),
                           "pairs": sum(r["n"] for r in rows)},
        "equal_accuracy_with_changed_correctness": sorted(
            [r for r in rows if r["gain"] == r["loss"] and r["gain"] > 0],
            key=lambda r: r["gain"], reverse=True),
        "ppr_net_correct_by_task": {
            t: sum(r["gain"] - r["loss"] for r in rows if r["order"] == "ppr"
                   and r["task"] == t) for t in tasks},
    }

ordered = sorted(comparisons, key=lambda r: r["p"])
adjusted = 0
for i, row in enumerate(ordered):
    adjusted = max(adjusted, min(1, row["p"] * (len(ordered) - i)))
    row["holm_p"] = adjusted
result["traditional_summary"] = {
    "comparisons": len(comparisons), "higher_equal_lower": outcomes(comparisons),
    "holm_significant": sum(r["holm_p"] < .05 for r in comparisons),
    "minimum_adjusted_test": ordered[0],
}
result["paired_comparisons"] = comparisons

node_source = HERE / "node_score_audit.json"
node_data=json.loads(node_source.read_text())
for model, data in node_data["models"].items():
    cells={(r["group"],r["order"]):r for r in data["cells"]}
    result["node"][model]={}
    for order in ORDERS:
        rows=[]
        for group in sorted({g for g,o in cells}):
            a,b=cells[group,"random"],cells[group,order]
            rows.append(dict(group=group,baseline_correct=a["correct"],correct=b["correct"],n=b["n"],delta_pp=100*(b["correct"]-a["correct"])/b["n"]))
        result["node"][model][order]=rows
result["node_source"]=str(node_source.relative_to(HERE.parent))
result["node_source_sha256"]=hashlib.sha256(node_source.read_bytes()).hexdigest()
(HERE / "rq1_insights.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["traditional_summary"], indent=2))
for model, data in result["traditional"].items():
    print(model, data["higher_equal_lower"], data["mean_delta_pp"])
