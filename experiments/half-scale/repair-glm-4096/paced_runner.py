"""Run one model on the paired half-scale benchmark with bounded parallelism."""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import fcntl
import json
from pathlib import Path
import random
import re
import signal
import sqlite3
import sys
from threading import Event
from time import monotonic, sleep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_experiment import ADAPTERS, evaluate, load_benchmark, prepare_database

CASE_COUNTS = {"traditional": 17500, "node": 2250}
RATE_ERROR = re.compile(r"\b(?:429|503|529)\b|rate limit|overloaded", re.I)
AUTH_ERROR = re.compile(r"\b401\b|unauthorized|model_not_found|unexpected argument|not supported", re.I)
FORBIDDEN_ERROR = re.compile(r"\b403\b")


def prioritise(cases, seed):
    """Try all task and prompt settings early, then mix tasks throughout the run."""
    rng = random.Random(seed)
    rng.shuffle(cases)
    seen = set()
    first, rest = [], []
    for case in cases:
        key = (case["group"], case["prompt_style"], case["order"])
        (rest if key in seen else first).append(case)
        seen.add(key)
    return deque(first + rest)


def report(model, completed, errors, in_flight, parallel, total):
    print(f"{model}: {completed}/{total} ok, {errors} terminal errors, "
          f"{in_flight} in flight, concurrency {parallel}", flush=True)


def run(args, cases, manifest_hash, stop):
    output = HERE / "runs" / args.group
    output.mkdir(parents=True, exist_ok=True)
    destination = output / (args.model + ".sqlite3")
    with destination.with_suffix(".lock").open("a") as lock, closing(sqlite3.connect(destination)) as db:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_database(db, manifest_hash)
        existing = dict(db.execute("SELECT case_id, status FROM results WHERE model=?", (args.model,)))
        already_ok = {case_id for case_id, status in existing.items() if status == "ok"}
        retry = []  # The queued final retry handles previously failed cases.
        pending = deque(retry)
        pending.extend(prioritise([case for case in cases if case["case_id"] not in existing], seed=42))
        attempts = Counter()
        error_kinds = Counter()
        errors = 0
        successes = len(already_ok)
        limit = args.initial
        pause_until = 0.0
        next_start = 0.0
        last_rate_cut = -1.0
        next_increase = monotonic() + 25
        next_report = monotonic() + 20
        successes_since_increase = 0
        consecutive_failures = 0
        consecutive_forbidden = 0
        consecutive_timeouts = 0
        last_success = monotonic()
        halting = False
        pause = 15
        total = len(cases)
        report(args.model, successes, errors, 0, limit, total)

        with ThreadPoolExecutor(max_workers=args.maximum) as pool:
            running = {}
            while running or (pending and not halting):
                now = monotonic()
                if stop.is_set() and not halting:
                    print(f"{args.model}: draining {len(running)} requests", flush=True)
                    halting = True
                if not halting and now >= next_increase and successes_since_increase >= limit and now >= pause_until:
                    increased = min(args.maximum, limit + 2)
                    if increased > limit:
                        print(f"{args.model}: concurrency {limit} -> {increased}", flush=True)
                    limit = increased
                    pause = 15
                    successes_since_increase = 0
                    next_increase = now + 25
                if pending and len(running) < limit and now >= max(pause_until, next_start) and not halting:
                    case = pending.popleft()
                    attempts[case["case_id"]] += 1
                    running[pool.submit(evaluate, (case, args.model))] = (case, now)
                    next_start = now + 0.45

                if now >= next_report:
                    report(args.model, successes, errors, len(running), limit, total)
                    next_report = now + 20
                if not running:
                    if pending and not halting:
                        sleep(min(2, max(0.05, pause_until - now)))
                    continue
                timeout = 2.0
                if pending and not halting and len(running) < limit:
                    timeout = max(0.03, min(timeout, max(next_start, pause_until) - monotonic()))
                finished, _ = wait(running, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in finished:
                    case, started = running.pop(future)
                    result = future.result()
                    case_id = case["case_id"]
                    tries = attempts.pop(case_id)
                    if result["status"] == "ok":
                        successes += 1
                        successes_since_increase += 1
                        consecutive_failures = 0
                        consecutive_forbidden = 0
                        consecutive_timeouts = 0
                        last_success = monotonic()
                    else:
                        detail = result["error"]
                        error_kinds[detail] += 1
                        rate_error = RATE_ERROR.search(detail)
                        forbidden = FORBIDDEN_ERROR.search(detail)
                        timeout = detail == "TimeoutExpired"
                        if not (rate_error or timeout):
                            consecutive_failures += 1
                        if timeout:
                            consecutive_timeouts += 1
                        if forbidden:
                            consecutive_forbidden += 1
                        now = monotonic()
                        fresh_rate_error = rate_error and started > last_rate_cut
                        if (fresh_rate_error or forbidden or consecutive_timeouts >= 4) and now >= pause_until:
                            if rate_error:
                                last_rate_cut = now
                            limit = max(1, int(limit * 0.7))
                            pause_until = now + pause
                            next_increase = pause_until + 45
                            successes_since_increase = 0
                            print(f"{args.model}: {detail}, cooling down {pause}s, concurrency {limit}", flush=True)
                            pause = min(120, pause * 2)
                        if (AUTH_ERROR.search(detail) or consecutive_forbidden >= 4
                                or consecutive_failures >= 24 or now - last_success > 600):
                            if not halting:
                                print(f"{args.model}: pausing this model after {detail}; other models continue", flush=True)
                            halting = True
                        if tries < 3 and not halting:
                            attempts[case_id] = tries
                            pending.appendleft(case)
                            continue
                        errors += 1
                    result["attempts"] = tries
                    with db:
                        db.execute("""INSERT INTO results VALUES (?, ?, ?, ?)
                            ON CONFLICT(case_id, model) DO UPDATE SET
                            status=excluded.status, payload=excluded.payload""",
                            (case_id, args.model, result["status"], json.dumps(result)))
        report(args.model, successes, errors, 0, limit, total)
        if error_kinds:
            print(f"{args.model}: request errors {dict(error_kinds)}", flush=True)
        return successes == total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=CASE_COUNTS, required=True)
    parser.add_argument("--model", choices=ADAPTERS, required=True)
    parser.add_argument("--initial", type=int, required=True)
    parser.add_argument("--maximum", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.initial <= args.maximum:
        raise ValueError("Invalid concurrency range")
    config, cases = load_benchmark(HERE / "benchmark" / args.group)
    if len(cases) != CASE_COUNTS[args.group]:
        raise ValueError("The selected benchmark is missing cases")
    manifest_hash = config["manifest_sha256"]
    if args.dry_run:
        print(f"{args.model}: checked {len(cases)} unique planned requests, manifest {manifest_hash}")
        return 0
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return 0 if run(args, cases, manifest_hash, stop) else 1


if __name__ == "__main__":
    sys.exit(main())
