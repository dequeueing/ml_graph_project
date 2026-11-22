#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
previous=$(cat final.pid)
while read -r state < <(ps -o stat= -p "$previous") && [[ "$state" != Z* ]]; do
    sleep 4
done
count=$(../../.venv/bin/python -c 'import sqlite3; db=sqlite3.connect("file:runs/traditional/glm-5.3.sqlite3?mode=ro",uri=True); print(db.execute("select count(*) from results where status=? ",("ok",)).fetchone()[0])') || exit 1
if [[ "$count" == 5991 ]]; then
    printf 'All cases recovered by the preceding stage.\n'
    exit 0
fi
printf 'Starting targeted recovery of %s remaining prompts.\n' "$((5991-count))"
../../.venv/bin/python -u retry_last.py
retry_status=$?
../../.venv/bin/python -u score_answers.py
score_status=$?
printf 'Targeted retry exit %s. Scoring audit exit %s.\n' "$retry_status" "$score_status"
((score_status == 0))
