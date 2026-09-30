"""Synthesize a MuJoCo controller from scratch: feature maps from the language model, then fits.

This is the continuous-control pipeline the paper ran for each task:

  1. Play 240 steps of random actions and record the transitions.
  2. Show 40 of them to the language model with the prompt in
     abstraction_prompts/general/mujoco_feature_map.txt, and ask for a feature map of 14 terms.
     Six independent maps are requested; these are the six calls per task.
  3. Fit each map's weight matrix with random search (scripts/fit_mujoco.py) and score it on
     the ten test episodes.
  4. Refit the best map three times, with fit seeds 0, 1 and 2, and report the median.

It needs an Anthropic API key. Everything is written to runs/mujoco/<task>/.

    export ANTHROPIC_API_KEY=...
    python scripts/synthesize_mujoco.py halfcheetah
    python scripts/synthesize_mujoco.py halfcheetah --dry-run   # build the prompt, call nothing

A new run samples the language model afresh, so its feature maps and scores will differ from
the paper's. The paper used claude-opus-4-8 with extended thinking off.
"""
import argparse
import json
import os
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_mujoco as F  # noqa: E402
import replay_mujoco as R  # noqa: E402

ROOT = R.ROOT
sys.path.insert(0, ROOT)
PROMPT = open(os.path.join(ROOT, "abstraction_prompts", "general", "mujoco_feature_map.txt")).read()

N_TERMS, LIBS, COLLECT, SHOWN, MAX_PROMPT_CHARS = 14, 6, 240, 40, 300000
SEED, COLLECT_SEED = 0, 900


def collect(task):
    """Random-action transitions, as JSON lines for the prompt, and how many are shown."""
    spec = F.TASKS[task]
    en = R.Episode(spec["env_id"], spec["labels"], COLLECT_SEED)
    rng = np.random.default_rng(SEED)
    lo, hi = [float(x) for x in en.low], [float(x) for x in en.high]
    ctx, prev = [], 0.0
    for _ in range(COLLECT):
        s = en.state()
        act = [float(x) for x in rng.uniform(lo, hi)]
        en.step(act)
        nx = en.state()
        sc = float(nx["score"][0])
        ctx.append({"state": {k: round(v[0], 4) for k, v in s.items()
                              if isinstance(v, list) and k != "score"},
                    "action": [round(x, 3) for x in act],
                    "next": {k: round(v[0], 4) for k, v in nx.items()
                             if isinstance(v, list) and k != "score"},
                    "reward": round(sc - prev, 4)})
        prev = sc
    # Spread the shown transitions across the whole window. Halve the count until the block
    # fits the character budget: the humanoids have 376 observation fields.
    shown = SHOWN
    while shown > 1:
        step = max(1, len(ctx) // max(shown, 1))
        samples = "\n".join(json.dumps(c) for c in ctx[::step][:shown])
        if len(samples) <= MAX_PROMPT_CHARS:
            break
        shown //= 2
    step = max(1, len(ctx) // max(shown, 1))
    samples = "\n".join(json.dumps(c) for c in ctx[::step][:shown])
    return samples, min(shown, len(ctx[::step]))


def build_prompt(task):
    spec = F.TASKS[task]
    samples, shown = collect(task)
    first = R.Episode(spec["env_id"], spec["labels"], SEED).state()
    example = next((k for k in first if isinstance(first[k], list) and k != "score"), "key_a")
    return PROMPT.format(n=N_TERMS, samples=samples, example_key=example), shown


def extract(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return m[0] if m else (text or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=sorted(F.TASKS))
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--query-mode", default="anthropic")
    ap.add_argument("--libs", type=int, default=LIBS, help="feature maps to request")
    ap.add_argument("--dry-run", action="store_true", help="write the prompt and stop")
    a = ap.parse_args()

    say = lambda m: print(m, flush=True)  # noqa: E731
    out = os.path.join(ROOT, "runs", "mujoco", a.task)
    os.makedirs(out, exist_ok=True)
    prompt, shown = build_prompt(a.task)
    open(os.path.join(out, "prompt.txt"), "w").write(prompt)
    say(f"[{a.task}] {COLLECT} random-action transitions collected, {shown} shown; "
        f"prompt written to {os.path.relpath(out, ROOT)}/prompt.txt ({len(prompt):,} chars)")
    if a.dry_run:
        return

    from theorycoder.llm_client import LLMClient, save_usage
    cl = LLMClient(query_mode=a.query_mode, language_model=a.model, max_tokens=4000)
    spec = F.TASKS[a.task]
    first_state = R.Episode(spec["env_id"], spec["labels"], SEED).state()
    drawn = []
    for L in range(a.libs):
        try:
            text, _ = cl.query(prompt, label=f"lib{L}")
        except Exception as exc:
            say(f"  map {L}: language-model call failed ({type(exc).__name__}: {exc})")
            continue
        path = os.path.join(out, f"map{L}.py")
        open(path, "w").write(extract(text).rstrip() + "\n")
        try:
            mod = R.load_module(path, f"map{L}_{a.task}")
            n = len(mod.terms(first_state, 0))
            if n != N_TERMS:
                raise ValueError(f"{n} terms, expected {N_TERMS}")
        except Exception as exc:
            say(f"  map {L}: rejected, {str(exc)[:80]}")
            continue
        drawn.append((L, path, mod))
    save_usage(cl, os.path.join(out, "usage.json"))
    if not drawn:
        raise SystemExit("no feature map survived synthesis")

    # Fit every map once, then keep the best on the test episodes.
    fits = []
    for ci, (L, path, mod) in enumerate(drawn):
        say(f"  map {L}: fitting")
        rec = F.fit_once(a.task, path, (SEED, ci), mod=mod)
        fits.append((rec["reported_return"], L, path))
        say(f"  map {L}: test return {rec['reported_return']:.2f}, "
            f"displacement {rec['displacement_m']:.2f} m")
    best_return, best_L, best_path = max(fits, key=lambda f: f[0])   # first best, as in the runs
    say(f"best map: {best_L} ({best_return:.2f}). Refitting it with fit seeds 0, 1, 2.")

    refits = []
    for s in range(3):
        rec = F.fit_once(a.task, best_path, (F.REFIT_SEED_BASE, s))
        refits.append(rec)
        say(f"  fit seed {s}: test return {rec['reported_return']:.2f}, "
            f"displacement {rec['displacement_m']:.2f} m")
    median = sorted(refits, key=lambda r: r["reported_return"])[1]
    json.dump(median, open(os.path.join(out, "controller.json"), "w"), indent=1)
    json.dump({"task": a.task, "model": a.model, "maps": [f[1] for f in fits],
               "map_test_returns": [f[0] for f in fits], "best_map": best_L,
               "refit_returns": [r["reported_return"] for r in refits],
               "median_return": statistics.median(r["reported_return"] for r in refits)},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    say(f"median of three fits: {median['reported_return']:.2f}. The median controller is in "
        f"{os.path.relpath(out, ROOT)}/controller.json; replay it with "
        f"scripts/replay_mujoco.py after copying it and map{best_L}.py into champions/mujoco/.")


if __name__ == "__main__":
    main()
