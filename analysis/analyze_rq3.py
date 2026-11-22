"""Analyze order preferences from the accepted case-level scores.

No answer extraction, rescoring, or model calls. Frozen inputs remain unchanged.
Run with python analysis/analyze_rq3.py to refresh the derived RQ3 evidence.
"""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
RELEASE = HERE / "releases/scoring-v1-20261122"
MODELS = ["deepseek-v4.1-flash", "glm-5.3", "gpt-6-luna"]
TASKS = ["connectivity", "cycle", "shortest_path", "hamilton", "topology"]
PROMPTS = ["zero_shot", "zero_shot_cot", "few_shot", "cot", "cot_bag"]
ORDERS = ["random", "bfs", "dfs", "pagerank", "ppr"]
GROUPS = [f"{dataset}/{sampler}" for dataset in ["cora", "citeseer", "pubmed"]
          for sampler in ["ego", "forest_fire"]]


def leaders(values):
    maximum = max(values.values())
    return [key for key in ORDERS if key in values and
            math.isclose(values[key], maximum, abs_tol=1e-10)]


def main():
    manifest = json.loads((RELEASE / "MANIFEST.json").read_text())
    hashes = {}

    def read(name):
        relative = f"analysis/{name}"
        raw = (RELEASE / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        assert hashes[relative] == manifest["files_sha256"][relative]
        return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)

    traditional = read("audited_final_scores.json")
    node = read("node_score_audit.json")
    inputs = read("verification_prompt_inputs.json.gz")
    metrics = read("report_metrics.json")
    rq2 = read("rq2_insights.json")
    previous_cells = {(r["model"], r["task"], r["prompt"]): r for r in rq2["cells"]}
    grouped = defaultdict(dict)
    for model in MODELS:
        scores = traditional["models"][model]["scores"]
        assert len(scores) == len(dict(scores)) == 15950
        for case_id, correct in scores:
            task, question, order, prompt, repeat = case_id.split("/")
            assert correct in (0, 1)
            grouped[model, task, prompt, order][question, repeat] = int(correct)

    cells = []
    cell_map = {}
    task_means = []
    traversal = []
    for model in MODELS:
        for task in TASKS:
            rows = []
            for prompt in PROMPTS:
                groups = {order: grouped[model, task, prompt, order] for order in ORDERS}
                keys = set(groups["random"])
                n = len(keys)
                assert n == (78 if task == "hamilton" else 140)
                assert all(set(group) == keys for group in groups.values())
                counts = {order: sum(group.values()) for order, group in groups.items()}
                assert counts == previous_cells[model, task, prompt]["correct_by_order"]
                accuracy = {order: 100 * count / n for order, count in counts.items()}
                row = dict(model=model, task=task, prompt=prompt, n=n,
                           correct=counts, accuracy_pct=accuracy, leaders=leaders(counts))
                rows.append(row)
                cells.append(row)
                cell_map[model, task, prompt] = row
            means = {order: mean(row["accuracy_pct"][order] for row in rows) for order in ORDERS}
            assert all(math.isclose(means[order], metrics["models"][model]["tasks"][task][order],
                                    abs_tol=1e-10) for order in ORDERS)
            overall_leaders = leaders(means)
            consistency = {order: {
                "unique_first": sum(row["leaders"] == [order] for row in rows),
                "tied_first": sum(order in row["leaders"] and len(row["leaders"]) > 1 for row in rows),
                "first_including_ties": sum(order in row["leaders"] for row in rows),
            } for order in ORDERS}
            task_means.append(dict(model=model, task=task, accuracy_pct=means,
                                   leaders=overall_leaders, prompt_support=consistency,
                                   lead_over_runner_up_pp=sorted(means.values(), reverse=True)[0]
                                   - sorted(means.values(), reverse=True)[1]))
            differences = {row["prompt"]: row["accuracy_pct"]["dfs"] - row["accuracy_pct"]["bfs"]
                           for row in rows}
            traversal.append(dict(model=model, task=task,
                                  dfs_minus_bfs_pp=means["dfs"] - means["bfs"],
                                  prompt_differences_pp=differences,
                                  dfs_higher=sum(v > 1e-10 for v in differences.values()),
                                  tied=sum(abs(v) < 1e-10 for v in differences.values()),
                                  bfs_higher=sum(v < -1e-10 for v in differences.values())))

    # Reconstruct node groups from case metadata rather than relying on row ordering.
    group_by_question = {}
    for case in inputs["node_cases"]:
        task, question, order, prompt, repeat = case["case_id"].split("/")
        assert task == "node_classification"
        if question in group_by_question:
            assert group_by_question[question] == case["group"]
        group_by_question[question] = case["group"]
    assert len(group_by_question) == 150
    node_rows = []
    node_summaries = []
    for model in [MODELS[0], MODELS[2]]:
        grouped_node = defaultdict(dict)
        model_data = node["models"][model]
        assert len(model_data["scores"]) == 2250
        saved = {(r["group"], r["order"]): r for r in model_data["cells"]}
        for case_id, correct in model_data["scores"]:
            task, question, order, prompt, repeat = case_id.split("/")
            grouped_node[group_by_question[question], order][question, repeat] = int(correct)
        model_rows = []
        for group in GROUPS:
            samples = {order: grouped_node[group, order] for order in ORDERS}
            keys = set(samples["random"])
            assert len(keys) == 75 and all(set(v) == keys for v in samples.values())
            counts = {order: sum(v.values()) for order, v in samples.items()}
            assert all(counts[order] == saved[group, order]["correct"] for order in ORDERS)
            accuracy = {order: count / 75 * 100 for order, count in counts.items()}
            probability_best = max(accuracy[order] for order in ["pagerank", "ppr"])
            traversal_best = max(accuracy[order] for order in ["bfs", "dfs"])
            row = dict(model=model, group=group, n=75, correct=counts,
                       accuracy_pct=accuracy, leaders=leaders(counts),
                       best_probability_pct=probability_best, best_traversal_pct=traversal_best,
                       probability_minus_traversal_pp=probability_best - traversal_best,
                       pr_minus_ppr_pp=accuracy["pagerank"] - accuracy["ppr"])
            node_rows.append(row)
            model_rows.append(row)
        node_summaries.append(dict(
            model=model,
            mean_accuracy_pct={order: mean(row["accuracy_pct"][order] for row in model_rows)
                               for order in ORDERS},
            leading_group_counts={order: sum(order in row["leaders"] for row in model_rows)
                                  for order in ORDERS},
            probability_family_higher=sum(row["probability_minus_traversal_pp"] > 1e-10 for row in model_rows),
            traversal_family_higher=sum(row["probability_minus_traversal_pp"] < -1e-10 for row in model_rows),
            pr_higher_than_ppr=sum(row["pr_minus_ppr_pp"] > 1e-10 for row in model_rows),
            ppr_higher_than_pr=sum(row["pr_minus_ppr_pp"] < -1e-10 for row in model_rows),
        ))

    controls = []
    for model in MODELS:
        scores = dict(traditional["models"][model]["scores"])
        outcomes = Counter()
        pair_ids = []
        for case_id, pr in scores.items():
            if not (case_id.startswith("hamilton/") and "/pagerank/" in case_id):
                continue
            ppr_id = case_id.replace("/pagerank/", "/ppr/")
            assert inputs["traditional_prompts"][case_id] == inputs["traditional_prompts"][ppr_id]
            ppr = scores[ppr_id]
            outcomes["both_correct" if pr and ppr else "pr_only" if pr else "ppr_only" if ppr else "both_wrong"] += 1
            pair_ids.append([case_id, ppr_id])
        assert len(pair_ids) == 390
        n = len(pair_ids)
        discordant = outcomes["pr_only"] + outcomes["ppr_only"]
        assert discordant == metrics["models"][model]["identical_prompt_disagreements"]
        controls.append(dict(
            model=model, n=n, identical_prompt_pairs=n,
            outcomes={key: outcomes[key] for key in ["both_correct", "pr_only", "ppr_only", "both_wrong"]},
            discordant=discordant, discordant_pct=100 * discordant / n,
            pr_accuracy_pct=100 * (outcomes["both_correct"] + outcomes["pr_only"]) / n,
            ppr_accuracy_pct=100 * (outcomes["both_correct"] + outcomes["ppr_only"]) / n,
            pr_minus_ppr_pp=100 * (outcomes["pr_only"] - outcomes["ppr_only"]) / n,
        ))

    output = dict(
        provenance=dict(release="scoring-v1-20261122", files_sha256=hashes),
        scope=dict(traditional_scores=47850, node_scores=4500,
                   leader_definition="Maximum observed accuracy, retaining all ties.",
                   node_family_comparison="Best observed PR/PPR accuracy minus best observed BFS/DFS accuracy within each group.",
                   inference="Descriptive comparisons. Prompt-style counts are not independent replications. No new hypothesis tests."),
        traditional_cells=cells, traditional_task_means=task_means,
        dfs_vs_bfs=traversal, node_groups=node_rows, node_summary=node_summaries,
        identical_prompt_controls=controls,
    )
    destination = HERE / "rq3_insights.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)
    print("Cross-checked all 375 traditional and 60 node accuracy cells.")
    for row in traversal:
        if row["task"] in ["hamilton", "topology"]:
            print("DFS-BFS", row["model"], row["task"], round(row["dfs_minus_bfs_pp"], 4))
    for row in task_means:
        if row["task"] == "shortest_path":
            print("Shortest path", row["model"], row["leaders"], row["prompt_support"]["ppr"])
    for row in node_summaries:
        print("Node", row)
    for row in controls:
        print("Identical prompts", row)


if __name__ == "__main__":
    main()
