#!/usr/bin/env bash
# Run the seed experiment one Kaggle session at a time. Stops the chain if a kernel ends in ERROR/CANCEL (fail-fast aborts show up as ERROR).
# Usage: SEEDS="0 1 2" prosqa_monitor/run_seeds.sh        (only run after the user has said yes to these pushes)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
for sd in ${SEEDS:-0 1 2}; do
  slug="prosqa-seeds-s$sd"
  echo "$(date +%H:%M) pushing seed $sd"
  NB=prosqa_monitor/notebooks/prosqa_monitor_seeds_seed$sd SLUG=$slug TITLE="prosqa seeds s$sd" shared/push_to_kaggle.sh push 2>&1 | tail -2 || { echo "push failed"; exit 1; }
  last=""
  while true; do
    st=$(NB=x SLUG=$slug shared/push_to_kaggle.sh status 2>&1 | tail -1)
    [ "$st" != "$last" ] && echo "$(date +%H:%M) seed $sd: $st" && last="$st"
    case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "seed $sd ended abnormally: stopping the chain"; NB=x SLUG=$slug shared/push_to_kaggle.sh output >/dev/null 2>&1; exit 2;; esac
    sleep 60
  done
  NB=x SLUG=$slug shared/push_to_kaggle.sh output >/dev/null 2>&1
  echo "$(date +%H:%M) seed $sd downloaded: $(python3 -c "import json;print(json.load(open('kaggle_output/$slug/s${sd}_run_meta.json'))['status'])" 2>/dev/null || echo 'no run_meta')"
done
echo "chain finished"
