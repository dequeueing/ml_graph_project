#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
group=$1
case "$group" in
    traditional) start=(44 12 54); maximum=(56 26 76) ;;
    node) start=(14 6 4); maximum=(24 8 6) ;;
    *) exit 2 ;;
esac
models=(deepseek-v4.1-flash glm-5.3 gpt-6-luna)
root="runs/$group"
mkdir -p "$root"
exec 9>"$root/launch.lock"
flock -n 9
mkdir -p -m 700 "$PWD/scratch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$PWD/scratch" TOKIO_WORKER_THREADS=8 RAYON_NUM_THREADS=8
../.venv/bin/python run.py --group "$group" --model "$models" --initial 1 --maximum 1 --dry-run
for index in 0 1 2; do
    model=${models[index]}
    if [[ -s "$root/$model.pid" ]] && kill -0 "$(cat "$root/$model.pid")" 2>/dev/null; then
        printf '%s already active, PID %s\n' "$model" "$(cat "$root/$model.pid")"
        continue
    fi
    nohup ../.venv/bin/python -u run.py --group "$group" --model "$model" \
        --initial "${start[index]}" --maximum "${maximum[index]}" \
        </dev/null >>"$root/$model.log" 2>&1 9>&- &
    pid=$!
    printf '%s\n' "$pid" > "$root/$model.pid"
    printf '%s PID %s, concurrency %s to %s\n' "$model" "$pid" "${start[index]}" "${maximum[index]}"
done
