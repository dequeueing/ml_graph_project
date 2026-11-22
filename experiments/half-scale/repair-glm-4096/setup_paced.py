"""Install a paced resume of the existing GLM recovery, using the same results DB."""

from pathlib import Path

root = Path.cwd()
runner = (root.parent / "run.py").read_text()
resume = (root / "resume_glm.py").read_text()


def substitute(source, patches):
    for old, new in patches:
        if source.count(old) != 1:
            raise ValueError(f"Expected one match for {old}")
        source = source.replace(old, new, 1)
    return source


paced_runner = substitute(runner, [
    ("        pause_until = 0.0",
     "        pause_until = 0.0\n        next_start = 0.0\n        last_rate_cut = -1.0"),
    ("limit + max(2, limit // 3)", "limit + 2"),
    ("                while pending and len(running) < limit and now >= pause_until and not halting:",
     "                if pending and len(running) < limit and now >= max(pause_until, next_start) and not halting:"),
    ("                    running[pool.submit(evaluate, (case, args.model))] = case",
     "                    running[pool.submit(evaluate, (case, args.model))] = (case, now)\n"
     "                    next_start = now + 0.45"),
    ("                finished, _ = wait(running, timeout=2, return_when=FIRST_COMPLETED)",
     "                timeout = 2.0\n"
     "                if pending and not halting and len(running) < limit:\n"
     "                    timeout = max(0.03, min(timeout, max(next_start, pause_until) - monotonic()))\n"
     "                finished, _ = wait(running, timeout=timeout, return_when=FIRST_COMPLETED)"),
    ("                    case = running.pop(future)", "                    case, started = running.pop(future)"),
    ("                        if (rate_error or forbidden or consecutive_timeouts >= 4) and now >= pause_until:\n"
     "                            limit = max(1, limit // 2)",
     "                        fresh_rate_error = rate_error and started > last_rate_cut\n"
     "                        if (fresh_rate_error or forbidden or consecutive_timeouts >= 4) and now >= pause_until:\n"
     "                            if rate_error:\n"
     "                                last_rate_cut = now\n"
     "                            limit = max(1, int(limit * 0.7))"),
    ("        retry = [case for case in cases if existing.get(case[\"case_id\"]) == \"error\"]",
     "        retry = []  # The queued final retry handles previously failed cases."),
])
paced_resume = substitute(resume, [
    ('HALF / "run.py"', 'DESTINATION / "paced_runner.py"'),
    ('"max_concurrency": 128,','"max_concurrency": 28,\n        "start_spacing_seconds": 0.45,'),
    ('"plan.json"', '"paced-plan.json"'),
    ("initial=128, maximum=128", "initial=24, maximum=28"),
])
for file_name, source in (("paced_runner.py", paced_runner), ("resume_glm_paced.py", paced_resume)):
    compile(source, file_name, "exec")
    (root / file_name).write_text(source)

(root / "switch_to_paced.py").write_text('''from pathlib import Path
import os
import signal
import subprocess
import time

root = Path.cwd()


def read_active(file_name, expected):
    pid = int((root / file_name).read_text())
    proc = Path(f"/proc/{pid}")
    args = (proc / "cmdline").read_bytes().split(bytes([0]))
    if expected.encode() not in args:
        raise RuntimeError(f"{file_name} does not own the expected process")
    return pid


def active(pid):
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


previous = read_active("run.pid", "resume_glm.py")
watcher = read_active("final.pid", "finalize.sh")
if (root / "final.log").stat().st_size:
    raise RuntimeError("The original final retry already started")
os.kill(watcher, signal.SIGTERM)
cutoff = time.monotonic() + 12
while active(watcher):
    if time.monotonic() >= cutoff:
        raise RuntimeError("The old finalizer did not exit")
    time.sleep(0.25)

print(f"Stopping submissions from {previous} and draining in-flight requests.", flush=True)
if active(previous):
    os.kill(previous, signal.SIGTERM)
while active(previous):
    time.sleep(1)

with (root / "run.log").open("ab") as run_log:
    run_log.write(b"\\nResuming with paced starts, initial concurrency 24, ceiling 28.\\n")
    run_log.flush()
    run = subprocess.Popen(
        ["../../.venv/bin/python", "-u", "resume_glm_paced.py"],
        stdin=subprocess.DEVNULL, stdout=run_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
(root / "run.pid").write_text(f"{run.pid}\\n")
with (root / "final.log").open("ab") as final_log:
    final = subprocess.Popen(
        ["bash", "finalize.sh"], stdin=subprocess.DEVNULL,
        stdout=final_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
(root / "final.pid").write_text(f"{final.pid}\\n")
print(f"Paced runner {run.pid}, final retry watcher {final.pid}.", flush=True)
''')
print("Paced resume installed.")
