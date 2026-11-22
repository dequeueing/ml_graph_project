"""Reproduce the half-scale GraphDO report statistics from the saved responses."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
from statistics import pvariance
import sys

import networkx as nx
from scipy.stats import binomtest

sys.path.insert(0, str(Path.cwd()))
from prepare import hamilton_path
from run import load_benchmark
from run_experiment import evaluate_answer
from evaluator import build_graph

LEADING_LABEL = re.compile(r"^\s*[*_]{0,2}(yes|no)\b", re.I)
INTEGER_LIST = re.compile(r"\d+(?:\s*,\s*\d+)*")


def graph_facts(case):
    data = case["graph_data"]
    task = case["task"]
    if task == "node_classification":
        return {}
    graph = build_graph(data, case["is_weighted"], case["is_directed"])
    if task == "cycle":
        return {"cyclic": bool(nx.cycle_basis(graph))}
    if task == "hamilton":
        return {"has_hamilton_path": bool(hamilton_path(graph))}
    if task == "topology":
        return {"isolates": set(nx.isolates(graph))}
    if task == "connectivity":
        source, target = (case["query_params"][key] for key in ("source", "target"))
        return {"reachable": nx.has_path(graph, source, target)}
    return {}


def audit_answer(case, answer, facts, counts):
    task, text, official = case["task"], answer["text"], answer["correct"]
    if task in ("cycle", "hamilton", "connectivity"):
        leading = LEADING_LABEL.match(text)
        yes = leading[1].lower() == "yes" if leading else None
        if task == "hamilton":
            if not facts["has_hamilton_path"]:
                counts["no_path_answers"] += 1
                counts["official_no_path_credited"] += official
                counts["leading_no_on_no_path"] += yes is False
        else:
            truth = facts["cyclic"] if task == "cycle" else facts["reachable"]
            counts["label_covered"] += yes is not None
            counts["label_correct"] += yes == truth
            counts["label_correct_official_wrong"] += yes == truth and not official
            counts["label_wrong_official_correct"] += yes is not None and yes != truth and official
            return yes == truth if yes is not None else None
    elif task == "topology":
        body = text.strip()
        if body.startswith("[") and body.endswith("]"):
            body = body[1:-1].strip()
        if INTEGER_LIST.fullmatch(body):
            nodes = [int(value) for value in body.split(",")]
            expected = set(range(case["graph_data"]["num_nodes"]))
            positions = {node: i for i, node in enumerate(nodes)}
            complete = len(nodes) == len(expected) and positions.keys() == expected
            valid = complete and all(positions[u] < positions[v] for u, v in case["graph_data"]["edges"])
            counts["list_covered"] += 1
            counts["valid_list"] += valid
            counts["invalid_list_official_correct"] += not valid and official
            counts["invalid_list_missing_only_isolates"] += (
                not valid and official and set(positions) < expected
                and len(nodes) == len(positions) and expected - set(positions) <= facts["isolates"]
                and all(positions[u] < positions[v] for u, v in case["graph_data"]["edges"])
            )
            counts["valid_list_official_wrong"] += valid and not official
    return None


def summarize(root):
    config = json.loads((root / "experiment.json").read_text())
    orders = config["orders"]
    audit = defaultdict(Counter)
    runtime = defaultdict(Counter)
    scores = defaultdict(dict)
    prefix_scores = defaultdict(dict)
    samples = defaultdict(Counter)
    group_totals = Counter()
    for section in ("traditional", "node"):
        provenance, cases = load_benchmark(root / "benchmark" / section)
        by_id = {case["case_id"]: case for case in cases}
        graphs = {(case["group"], case["graph_id"]): case for case in cases}
        facts = {k: graph_facts(v) for k, v in graphs.items()}
        signatures = defaultdict(set)
        for (group, graph_id), case in graphs.items():
            fact = facts[group, graph_id]
            samples[group]["graphs"] += 1
            for name in ("cyclic", "has_hamilton_path", "isolates", "reachable"):
                if name in fact:
                    samples[group][name] += bool(fact[name])
            if section == "node":
                graph = case["graph_data"]
                signature = (int(graph["test_node"]),
                             tuple(sorted(tuple(sorted(edge)) for edge in graph["edges"])),
                             tuple(sorted(graph["node_labels"].items())))
                signatures[group].add(signature)
        for group, unique in signatures.items():
            samples[group]["unique_node_questions"] = len(unique)

        for model in config["models"]:
            path = root / "runs" / section / f"{model}.sqlite3"
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                assert db.execute("SELECT manifest_hash FROM metadata").fetchone() == (provenance["manifest_sha256"],)
                rows = db.execute("SELECT case_id,model,status,payload FROM results").fetchall()
            assert len(rows) == len(by_id) and {row[0] for row in rows} == by_id.keys()
            for case_id, stored_model, status, payload in rows:
                answer = json.loads(payload)
                assert answer["model"] == stored_model == model and answer["case_id"] == case_id
                assert answer["status"] == status == "ok", (model, case_id, status)
                case = by_id[case_id]
                score = evaluate_answer(case["task"], answer["text"], case["graph_data"],
                                        case["is_weighted"], case["is_directed"], case["query_params"])
                assert score in (0, 1) and score == answer["correct"], (model, case_id)
                group, style, order = (case[k] for k in ("group", "prompt_style", "order"))
                graph_id, repeat = case["graph_id"], case["repeat"]
                obs = graph_id, repeat
                cell = model, group, style, order
                assert obs not in scores[cell]
                scores[cell][obs] = score
                counts = audit[cell]
                counts["n"] += 1
                counts["official_correct"] += score
                counts["blank"] += not answer["text"].strip()
                counts["max_tokens"] += answer.get("finish_reason") == "max_tokens"
                label_score = audit_answer(case, answer, facts[group, graph_id], counts)
                if label_score is not None:
                    prefix_scores[cell][obs] = int(label_score)

                observed = runtime[model, group, style]
                observed["n"] += 1
                observed["blank"] += not answer["text"].strip()
                observed["max_tokens"] += answer.get("finish_reason") == "max_tokens"
                observed["reported_requested_model"] += answer.get("returned_model") == model
                observed["attempts_in_final_pass"] += answer.get("attempts", 1)
                group_totals[section] += 1

    cells, comparisons, variances, paired_prefix = [], [], [], []
    by_model_group = defaultdict(Counter)
    n_per_order = defaultdict(Counter)
    strata = sorted({key[:3] for key in scores})
    for model, group, style in strata:
        node = "/" in group
        n = 75 if node else 140
        base = scores[model, group, style, "random"]
        assert len(base) == n
        order_counts = {}
        for order in orders:
            key = model, group, style, order
            vals = scores[key]
            assert vals.keys() == base.keys()
            k = sum(vals.values())
            order_counts[order] = k
            by_model_group[model, group][order] += k
            n_per_order[model, group][order] += n
            if node:
                by_model_group[model, "all node groups"][order] += k
                n_per_order[model, "all node groups"][order] += n
            cells.append(dict(model=model, group=group, style=style, order=order, n=n, correct=k,
                              audit=dict(audit[key])))
            if order == "random":
                continue
            improve = sum(vals[g] > base[g] for g in base)
            worsen = sum(vals[g] < base[g] for g in base)
            item = dict(model=model, group=group, style=style, order=order, n=n, correct=k,
                        random_correct=sum(base.values()), improve=improve, worsen=worsen,
                        delta_pp=100 * (improve-worsen)/n)
            baseline = item["random_correct"]
            item["relative_gain_pct"] = 100 * (k - baseline) / baseline if baseline else None
            if not node:
                item["p_exact"] = binomtest(improve, improve+worsen, 0.5).pvalue if improve+worsen else 1.0
            comparisons.append(item)
        variances.append(dict(model=model, group=group, style=style,
                              variance=pvariance([100 * n_correct/n for n_correct in order_counts.values()])))
        if group == "cycle":
            matched = set.intersection(*(set(prefix_scores[model, group, style, order]) for order in orders))
            paired_prefix.append(dict(model=model, style=style, n=len(matched), by_order={
                order: {"leading_label_correct": sum(prefix_scores[model, group, style, order][g] for g in matched),
                        "official_correct": sum(scores[model, group, style, order][g] for g in matched)}
                for order in orders
            }))

    family = sorted((item for item in comparisons if "p_exact" in item), key=lambda item: item["p_exact"])
    assert len(family) == 300 and len(comparisons) == 372 and len(cells) == 465
    running_max = 0.0
    for rank, item in enumerate(family):
        running_max = max(running_max, (len(family)-rank)*item["p_exact"])
        item["p_holm"] = min(1.0, running_max)
    rankings = []
    for (model, group), correct in sorted(by_model_group.items()):
        totals = n_per_order[model, group]
        assert len(set(totals.values())) == 1
        n = totals[orders[0]]
        rankings.append(dict(model=model, group=group, n_per_order=n, correct=dict(correct),
                             accuracy_pct={order: 100 * correct[order] / n for order in orders},
                             best_orders=[order for order in orders if correct[order] == max(correct.values())],
                             relative_gain_pct={
                                 reference: {
                                     order: (100 * (correct[order] - correct[reference]) / correct[reference]
                                             if correct[reference] else None)
                                     for order in orders if order != reference
                                 }
                                 for reference in orders
                             }))
    assert group_totals == {"traditional": 52500, "node": 6750}
    return dict(config=config, totals=group_totals, samples=samples, cells=cells,
                comparisons=comparisons, variances=variances, rankings=rankings, paired_cycle_labels=paired_prefix,
                runtime=[dict(model=m, group=g, style=s, **dict(counts)) for (m,g,s), counts in runtime.items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
