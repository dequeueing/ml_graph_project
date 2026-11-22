from pathlib import Path
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
    run_log.write(b"\nResuming with paced starts, initial concurrency 24, ceiling 28.\n")
    run_log.flush()
    run = subprocess.Popen(
        ["../../.venv/bin/python", "-u", "resume_glm_paced.py"],
        stdin=subprocess.DEVNULL, stdout=run_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
(root / "run.pid").write_text(f"{run.pid}\n")
with (root / "final.log").open("ab") as final_log:
    final = subprocess.Popen(
        ["bash", "finalize.sh"], stdin=subprocess.DEVNULL,
        stdout=final_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
(root / "final.pid").write_text(f"{final.pid}\n")
print(f"Paced runner {run.pid}, final retry watcher {final.pid}.", flush=True)
