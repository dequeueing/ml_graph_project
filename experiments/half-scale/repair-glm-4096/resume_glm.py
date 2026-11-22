"""Retry the capped GLM responses against the original prompt manifest."""

import argparse
from contextlib import closing
import importlib.util
import inspect
import json
from pathlib import Path
import signal
import sqlite3
import sys
from threading import Event

DESTINATION = Path(__file__).resolve().parent
HALF = DESTINATION.parent
MODEL = "glm-5.3"
OUTPUT_LIMIT = 4096
RETRY_LIMIT = 8192

sys.path.insert(0, str(HALF.parent))
import model_adapters
from run_experiment import load_benchmark

spec = importlib.util.spec_from_file_location("half_scale_runner", HALF / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def adapter_at(limit):
    source = inspect.getsource(model_adapters.invoke_glm)
    marker = '"max_tokens": 400,'
    if source.count(marker) != 1:
        raise RuntimeError("The GLM adapter's original token limit changed")
    namespace = vars(model_adapters).copy()
    exec(compile(source.replace(marker, f'"max_tokens": {limit},'),
                 f"<GLM adapter with {limit} output tokens>", "exec"), namespace)
    return namespace["invoke_glm"]


def main():
    config, cases = load_benchmark(HALF / "benchmark" / "traditional")
    old_file = (HALF / "runs" / "traditional" / f"{MODEL}.sqlite3").resolve()
    old_db = sqlite3.connect(old_file.as_uri() + "?mode=ro", uri=True)
    with closing(old_db):
        assert old_db.execute("SELECT manifest_hash FROM metadata").fetchone() == (
            config["manifest_sha256"],
        )
        rows = old_db.execute("SELECT case_id, payload FROM results WHERE status='ok'")
        to_retry = {}
        for case_id, payload in rows:
            row = json.loads(payload)
            blank = not row["text"].strip()
            if row.get("finish_reason") == "max_tokens" or blank:
                to_retry[case_id] = blank
    planned = [case for case in cases if case["case_id"] in to_retry]
    assert len(planned) == len(to_retry) == 5991
    plan = {
        "model": MODEL, "manifest_sha256": config["manifest_sha256"],
        "selected": len(planned), "previously_blank": sum(to_retry.values()),
        "output_limit": OUTPUT_LIMIT, "fallback_limit": RETRY_LIMIT,
        "max_concurrency": 128,
    }
    (DESTINATION / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan), flush=True)

    invoke = adapter_at(OUTPUT_LIMIT)
    retry = adapter_at(RETRY_LIMIT)

    def invoke_repair(prompt):
        reply = invoke(prompt)
        if reply.finish_reason == "max_tokens" or not reply.text.strip():
            reply = retry(prompt)
        if reply.finish_reason == "max_tokens" or not reply.text.strip():
            raise model_adapters.ModelError("GLM still incomplete at 8192 output tokens")
        return reply

    runner.ADAPTERS[MODEL] = invoke_repair
    runner.HERE = DESTINATION
    destination_db = DESTINATION / "runs" / "traditional" / f"{MODEL}.sqlite3"
    stop = Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    if not destination_db.exists():
        blanks = [case for case in planned if to_retry[case["case_id"]]]
        first = list(runner.prioritise(blanks, seed=42))[:16]
        smoke = argparse.Namespace(group="traditional", model=MODEL, initial=16, maximum=16)
        print("Checking 16 previously blank prompts at the higher token limit", flush=True)
        runner.run(smoke, first, config["manifest_sha256"], stop)
        with closing(sqlite3.connect(destination_db)) as db:
            recovered = db.execute("SELECT count(*) FROM results WHERE status='ok'").fetchone()[0]
        print(f"Preflight recovered {recovered}/{len(first)} visible responses", flush=True)
        if not recovered:
            raise RuntimeError("No preflight prompts completed at the higher token limit")

    if stop.is_set():
        return 1
    args = argparse.Namespace(group="traditional", model=MODEL, initial=128, maximum=128)
    return 0 if runner.run(args, planned, config["manifest_sha256"], stop) else 1


if __name__ == "__main__":
    sys.exit(main())
