#!/usr/bin/env bash
# Replay every reported Code to Control champion (Atari, Flappy Bird, MuJoCo)
# and compare to the paper. `./reproduce.sh --all` also reruns the fits (see below).
# Zero LLM calls: each controller is loaded and played unchanged.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
OUT="$(mktemp -d)"
printf "%-14s %10s %10s  %s\n" game reported replayed result
fail=0
for g in pong pong_best spaceinvaders asterix breakout fishingderby freeway flappy; do
  read -r game paper < <("$PY" -c "import json;d=json.load(open('champions/$g.json'));print(d['game'],d['reported_score'])")
  read -r -a rails < <("$PY" -c "import json;r=json.load(open('champions/$g.json'))['rails'];print(' '.join(f'{k} {v}' for k,v in r.items()))")
  got=$("$PY" -u scripts/run_controller_flappy.py --game "$game" --mode llm \
          --replay-src "champions/$g.json" --max-episodes 1 "${rails[@]}" \
          --out "$OUT/$g.json" 2>&1 | grep -oE "best=[-0-9.]+" | head -1 | cut -d= -f2 || true)
  if [ "$got" = "$paper" ]; then r=match; else r=MISMATCH; fail=1; fi
  printf "%-14s %10s %10s  %s\n" "$g" "$paper" "${got:-error}" "$r"
done
echo
"$PY" scripts/replay_mujoco.py || fail=1

# --all: also rerun Figure 3B (Flappy Bird under changed physics, with refitting) and refit
# every MuJoCo controller from scratch, checking its weights. About 20 minutes.
if [ "${1:-}" = "--all" ]; then
  echo
  "$PY" scripts/flappy_transfer.py || fail=1
  for f in champions/mujoco/*.json; do
    echo
    "$PY" scripts/fit_mujoco.py "$(basename "$f" .json)" --check | grep -v "iteration" || fail=1
  done
fi
exit $fail
