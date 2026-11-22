#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
first_pid=$(cat run.pid)
while read -r status < <(ps -o stat= -p "$first_pid") && [[ "$status" != Z* ]]; do
    sleep 10
done
printf 'Retrying cases that did not return a complete answer in the first pass.
'
../../.venv/bin/python -u retry_remaining.py
retry_status=$?
../../.venv/bin/python -u score_answers.py
audit_status=$?
printf 'Retry exit %s. Scoring audit exit %s.
' "$retry_status" "$audit_status"
(( retry_status == 0 && audit_status == 0 ))
