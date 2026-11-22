"""Run or summarize the frozen GraphDO benchmark on the server."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
import fcntl
import hashlib
from itertools import batched
import json
from pathlib import Path
import sqlite3
from statistics import mean, pvariance
import sys
from time import monotonic

from model_adapters import ADAPTERS, ModelError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GraphDO"))
from evaluator import evaluate_answer


def load_benchmark(path):
    config = json.loads((path / "provenance.json").read_text())
    raw = (path / "manifest.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Benchmark manifest differs from its recorded checksum")
    cases = [json.loads(line) for line in raw.splitlines()]
    if len(cases) != config["prompt_count"] or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Incorrect case count or duplicate case IDs")
    return config, cases


def evaluate(job):
    case, model = job
    started = monotonic()
    try:
        reply = ADAPTERS[model](case["prompt"])
        correct = evaluate_answer(case["task"], reply.text, case["graph_data"],
                                  case["is_weighted"], case["is_directed"], case["query_params"])
        if correct not in (0, 1):
            raise ValueError("Unexpected upstream score")
        result = dict(asdict(reply), status="ok", correct=correct)
    except ModelError as error:
        result = {"status": "error", "error": str(error)}
    except OSError as error:
        result = {"status": "error", "error": f"{type(error).__name__} (errno {error.errno})"}
    except Exception as error:
        result = {"status": "error", "error": type(error).__name__}
    return {
        "case_id": case["case_id"], "model": model,
        "elapsed_seconds": round(monotonic() - started, 3), **result,
    }


def prepare_database(database, manifest_hash):
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE IF NOT EXISTS metadata (manifest_hash TEXT PRIMARY KEY)")
    database.execute("""CREATE TABLE IF NOT EXISTS results (
        case_id TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
        PRIMARY KEY (case_id, model))""")
    row = database.execute("SELECT manifest_hash FROM metadata").fetchone()
    if row is not None and row[0] != manifest_hash:
        raise ValueError("Use a separate results database for a different benchmark")
    if row is None:
        database.execute("INSERT INTO metadata VALUES (?)", (manifest_hash,))
    database.commit()


def run(args, config, cases):
    models = args.models or config["models"]
    orders = args.orders or config["orders"]
    if not set(models) <= set(config["models"]) or not set(orders) <= set(config["orders"]):
        raise ValueError("Selected model or order is absent from the benchmark")
    if not 1 <= args.graphs_per_task <= config["graphs_per_task"] or args.workers < 1:
        raise ValueError("Invalid graph limit or worker count")
    jobs = [(case, model) for case in cases
            if case["order"] in orders and int(case["graph_id"]) < args.graphs_per_task
            for model in models]
    print(f"Selected {len(jobs)} requests across {len(models)} models", flush=True)
    if args.dry_run:
        return True

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with args.db.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(sqlite3.connect(args.db)) as database:
            prepare_database(database, config["manifest_sha256"])
            done = set(database.execute("SELECT case_id, model FROM results WHERE status='ok'"))
            pending = [job for job in jobs if (job[0]["case_id"], job[1]) not in done]
            print(f"{len(pending)} pending, {len(jobs) - len(pending)} already complete", flush=True)
            failures = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for batch in batched(pending, args.workers):
                    for result in pool.map(evaluate, batch):
                        with database:
                            database.execute(
                                """INSERT INTO results VALUES (?, ?, ?, ?)
                                ON CONFLICT(case_id, model) DO UPDATE SET
                                status=excluded.status, payload=excluded.payload""",
                                (result["case_id"], result["model"], result["status"], json.dumps(result)),
                            )
                        if result["status"] == "error":
                            failures += 1
                        detail = result.get("error", f"score={result.get('correct')}")
                        print(f"{result['model']} {result['case_id']}: {result['status']} {detail}", flush=True)
            print(f"Attempted {len(pending)} requests with {failures} errors", flush=True)
    return failures == 0


def relative_gain(score, baseline):
    return 100 * (score - baseline) / baseline if baseline else None


def summarize(args, config, cases):
    with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        if database.execute("SELECT manifest_hash FROM metadata").fetchone() != (config["manifest_sha256"],):
            raise ValueError("Results belong to a different benchmark")
        rows = [json.loads(row[0]) for row in database.execute("SELECT payload FROM results")]
    by_case = {case["case_id"]: case for case in cases}
    counts = defaultdict(lambda: {"correct": 0, "ok": 0, "error": 0})
    for row in rows:
        case = by_case[row["case_id"]]
        cell = counts[row["model"], case["task"], case["prompt_style"], case["order"]]
        cell[row["status"]] += 1
        cell["correct"] += row.get("correct", 0)
    cells = []
    rq1 = []
    rq2 = []
    rq3 = []
    n = config["graphs_per_task"]
    for model in config["models"]:
        for task in config["tasks"]:
            style_scores = {}
            for style in config["prompt_styles"]:
                scores = {}
                for order in config["orders"]:
                    cell = counts[model, task, style, order]
                    accuracy = 100 * cell["correct"] / n if cell["ok"] == n else None
                    cells.append(dict(model=model, task=task, style=style, order=order,
                                      expected=n, accuracy_pct=accuracy, **cell))
                    if accuracy is not None:
                        scores[order] = accuracy
                if "random" in scores:
                    baseline = scores["random"]
                    for order, score in scores.items():
                        if order != "random":
                            rq1.append(dict(model=model, task=task, style=style, order=order,
                                            difference_pp=score-baseline, relative_gain_pct=relative_gain(score, baseline)))
                if len(scores) == len(config["orders"]):
                    style_scores[style] = scores
                    rq2.append(dict(model=model, task=task, style=style,
                                    accuracy_variance=pvariance(scores.values())))
            if len(style_scores) == len(config["prompt_styles"]):
                average = {order: mean(scores[order] for scores in style_scores.values())
                           for order in config["orders"]}
                comparisons = [dict(order=order, baseline=baseline,
                                    relative_gain_pct=relative_gain(average[order], average[baseline]))
                               for order in average for baseline in average if order != baseline]
                rq3.append(dict(model=model, task=task, mean_accuracy_pct=average,
                                ranking=sorted(average, key=average.get, reverse=True),
                                best_orders=[order for order, value in average.items() if value == max(average.values())],
                                pairwise_relative_gain=comparisons))
    document = {"cells": cells, "rq1": rq1, "rq2": rq2, "rq3": rq3}
    encoded = json.dumps(document, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)
    print(f"{sum(row['status'] == 'ok' for row in rows)} successful, "
          f"{sum(row['status'] == 'error' for row in rows)} failed, "
          f"{sum(cell['accuracy_pct'] is not None for cell in cells)}/{len(cells)} complete cells",
          file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark" / "subset")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("--db", type=Path, default=ROOT / "runs" / "formal.sqlite3")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--graphs-per-task", type=int)
    run_parser.add_argument("--models", nargs="+", choices=ADAPTERS)
    run_parser.add_argument("--orders", nargs="+")
    run_parser.add_argument("--workers", type=int, default=3)
    summary_parser = subcommands.add_parser("summary")
    summary_parser.add_argument("--db", type=Path, default=ROOT / "runs" / "formal.sqlite3")
    summary_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, cases = load_benchmark(args.benchmark)
    if args.command == "run":
        if args.graphs_per_task is None:
            args.graphs_per_task = config["graphs_per_task"]
        if not run(args, config, cases):
            sys.exit(1)
    else:
        summarize(args, config, cases)


if __name__ == "__main__":
    main()
